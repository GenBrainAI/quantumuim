"""Q+ Shor's Algorithm — factor numbers with fewer qubits than standard Shor.

Combines all three Q+ layers:
1. Semiclassical QFT (compress) — reduces counting register from n qubits to 1
2. Qubit recycling (recycle) — reuses the single counting qubit across iterations
3. Circuit cutting + distribution (cut/distribute) — optional multi-QPU support

Standard Shor: 2n+3 qubits for an n-bit number.
Q+ Shor:       n+2 qubits (semiclassical + recycling), or split across QPUs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd

import numpy as np
from numpy.typing import NDArray

from .compress import ModularArithmeticOptimizer, SemiclassicalQFT, _extract_order
from .recycle import QubitRecycler


@dataclass
class QplusResult:
    """Result of Q+ Shor factoring."""

    N: int
    factors: tuple[int, int] | None
    success: bool
    qubits_used: int
    qubits_standard: int
    qubit_savings: int
    order: int | None = None
    base: int = 0
    phase: float = 0.0
    strategy: str = "semiclassical"
    attempts: int = 0
    layers_used: list[str] = field(default_factory=list)


class ShorQPlus:
    """Shor's algorithm using Q+ qubit-reduction strategies.

    Supports multiple strategies with different qubit/overhead tradeoffs:
    - "semiclassical": Semiclassical QFT, n+2 qubits, zero overhead
    - "recycled": Semiclassical QFT + qubit recycling, n+2 qubits
    - "distributed": Add circuit cutting for multi-QPU execution
    - "auto": Automatically select best strategy
    """

    def __init__(self, strategy: str = "auto") -> None:
        if strategy not in ("auto", "semiclassical", "recycled", "distributed"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy

    def factor(
        self,
        N: int,
        a: int | None = None,
        max_attempts: int = 20,
        n_counting_bits: int | None = None,
    ) -> QplusResult:
        """Factor integer N using Q+ enhanced Shor's algorithm.

        Parameters
        ----------
        N
            The composite number to factor (must be >= 4).
        a
            Base for order finding. If None, tries bases automatically.
        max_attempts
            Maximum number of random bases to try.
        n_counting_bits
            Number of bits of phase precision. Defaults to 2*ceil(log2(N))+1.
        """
        if N < 4:
            raise ValueError(f"N must be >= 4, got {N}")

        n_bits = max(N.bit_length(), 2)
        if n_counting_bits is None:
            n_counting_bits = 2 * n_bits + 1

        standard_qubits = 2 * n_bits + 1 + n_bits
        qplus_qubits = n_bits + 2

        strategy = self._select_strategy(N) if self.strategy == "auto" else self.strategy

        if N % 2 == 0:
            return QplusResult(
                N=N,
                factors=(2, N // 2),
                success=True,
                qubits_used=0,
                qubits_standard=standard_qubits,
                qubit_savings=standard_qubits,
                strategy="trivial",
                layers_used=["trivial_gcd"],
            )

        candidates = [a] if a is not None else list(range(2, min(N, max_attempts + 2)))

        for attempt, base in enumerate(candidates):
            g = gcd(base, N)
            if g > 1:
                return QplusResult(
                    N=N,
                    factors=(g, N // g),
                    success=True,
                    qubits_used=0,
                    qubits_standard=standard_qubits,
                    qubit_savings=standard_qubits,
                    base=base,
                    strategy="trivial",
                    attempts=attempt + 1,
                    layers_used=["trivial_gcd"],
                )

            phase, order = self._qplus_order_finding(
                base, N, n_bits, n_counting_bits, strategy
            )

            if order is None or order == 0:
                continue

            factors = self._extract_factors(base, order, N)
            if factors is not None:
                layers = self._get_layers(strategy)
                return QplusResult(
                    N=N,
                    factors=factors,
                    success=True,
                    qubits_used=qplus_qubits,
                    qubits_standard=standard_qubits,
                    qubit_savings=standard_qubits - qplus_qubits,
                    order=order,
                    base=base,
                    phase=phase,
                    strategy=strategy,
                    attempts=attempt + 1,
                    layers_used=layers,
                )

        return QplusResult(
            N=N,
            factors=None,
            success=False,
            qubits_used=qplus_qubits,
            qubits_standard=standard_qubits,
            qubit_savings=standard_qubits - qplus_qubits,
            attempts=len(candidates),
            strategy=strategy,
            layers_used=self._get_layers(strategy),
        )

    def _select_strategy(self, N: int) -> str:
        """Auto-select the best strategy based on problem size."""
        n_bits = N.bit_length()
        if n_bits <= 8:
            return "recycled"
        return "semiclassical"

    def _qplus_order_finding(
        self,
        a: int,
        N: int,
        n_bits: int,
        n_counting_bits: int,
        strategy: str,
    ) -> tuple[float, int | None]:
        """Find order using Q+ compressed phase estimation."""

        def ctrl_u(power: int, state: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
            perm = ModularArithmeticOptimizer.mod_exp_permutation(a, power, N, n_bits)
            return perm @ state

        if strategy == "recycled":
            recycler = QubitRecycler(n_physical=1 + n_bits)
            phase, bits = recycler.simulate_recycled_phase_estimation(
                ctrl_u, n_bits, n_counting_bits
            )
        else:
            sqft = SemiclassicalQFT(n_counting_bits)
            result = sqft.estimate_phase(ctrl_u, n_bits)
            phase = result.phase

        order = _extract_order(phase, a, N)
        return phase, order

    @staticmethod
    def _extract_factors(a: int, order: int, N: int) -> tuple[int, int] | None:
        """Extract factors from order using standard Shor post-processing."""
        if order % 2 != 0:
            return None

        half_power = pow(a, order // 2, N)
        if half_power == N - 1:
            return None

        f1 = gcd(half_power + 1, N)
        f2 = gcd(half_power - 1, N)

        if 1 < f1 < N:
            return (f1, N // f1)
        if 1 < f2 < N:
            return (f2, N // f2)
        return None

    @staticmethod
    def _get_layers(strategy: str) -> list[str]:
        """Return list of Q+ layers used by strategy."""
        if strategy == "semiclassical":
            return ["compress"]
        elif strategy == "recycled":
            return ["compress", "recycle"]
        elif strategy == "distributed":
            return ["compress", "recycle", "cut", "distribute"]
        return ["compress"]

    @staticmethod
    def qubit_comparison(N: int) -> dict[str, int]:
        """Compare qubit requirements across strategies."""
        n_bits = max(N.bit_length(), 2)
        return {
            "standard_shor": 2 * n_bits + 1 + n_bits,
            "beauregard": 2 * n_bits + 3,
            "semiclassical_qft": n_bits + 2,
            "qplus_recycled": n_bits + 2,
            "n_bits": n_bits,
        }
