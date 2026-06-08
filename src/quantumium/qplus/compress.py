"""Layer 1: Algorithmic compression for qubit reduction.

Implements semiclassical QFT (Griffiths-Niu 1996) and modular arithmetic
optimization (Beauregard 2002, Häner 2017) to reduce qubit requirements
for algorithms like Shor's from 2n+3 down to n+1 qubits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SemiclassicalQFTResult:
    """Result of semiclassical QFT phase estimation."""

    phase: float
    measured_bits: list[int]
    n_qubits_used: int


class SemiclassicalQFT:
    """Semiclassical QFT — replaces n-qubit QFT with sequential single-qubit
    measurements and classically-controlled phase corrections.

    Instead of building the full n-qubit QFT circuit (requiring n qubits
    simultaneously), this measures each qubit sequentially and uses the
    classical measurement outcomes to apply corrective phase rotations
    to subsequent qubits. This reduces the QFT from n qubits to 1 reusable
    qubit (when combined with qubit recycling).

    Reference: Griffiths & Niu, PRL 76, 3228 (1996).
    """

    def __init__(self, n_bits: int) -> None:
        if n_bits < 1:
            raise ValueError("n_bits must be >= 1")
        self.n_bits = n_bits

    def estimate_phase(
        self,
        controlled_unitary_fn: Callable[
            [int, NDArray[np.complexfloating]], NDArray[np.complexfloating]
        ],
        n_work: int,
        initial_work_state: NDArray[np.complexfloating] | None = None,
    ) -> SemiclassicalQFTResult:
        """Estimate phase using semiclassical QFT with 1 counting qubit.

        Each iteration: prepare |+⟩ on counting qubit, apply controlled-U^(2^k),
        apply classically-controlled Z-rotations from prior measurements, measure.

        Parameters
        ----------
        controlled_unitary_fn
            Callable(power, work_state) -> work_state that applies controlled-U^power
            to the work register when control is |1⟩. Takes the power and current
            work register statevector, returns updated work statevector.
        n_work
            Number of qubits in the work register.
        initial_work_state
            Initial state of work register. Defaults to |1⟩ = |00...01⟩.
        """
        dim_work = 2**n_work
        if initial_work_state is None:
            work_state = np.zeros(dim_work, dtype=complex)
            work_state[1] = 1.0
        else:
            work_state = np.array(initial_work_state, dtype=complex)

        measured_bits: list[int] = []

        for k in range(self.n_bits - 1, -1, -1):
            power = 2**k
            full_state = np.zeros(2 * dim_work, dtype=complex)
            full_state[:dim_work] = work_state / np.sqrt(2)
            ctrl1_work = controlled_unitary_fn(power, work_state)
            full_state[dim_work:] = ctrl1_work / np.sqrt(2)

            phase_correction = 0.0
            for j, bit in enumerate(measured_bits):
                if bit:
                    phase_correction -= np.pi / (2 ** (j + 1))

            if abs(phase_correction) > 1e-15:
                full_state[dim_work:] *= np.exp(1j * phase_correction)

            prob_0 = np.sum(np.abs(full_state[:dim_work]) ** 2)
            prob_1 = np.sum(np.abs(full_state[dim_work:]) ** 2)

            total = prob_0 + prob_1
            if total > 0:
                prob_0 /= total
                prob_1 /= total

            bit = int(np.random.random() >= prob_0)
            measured_bits.append(bit)

            if bit == 0 and prob_0 > 0:
                work_state = full_state[:dim_work] / np.sqrt(prob_0)
            elif prob_1 > 0:
                work_state = full_state[dim_work:] / np.sqrt(prob_1)

        measured_bits.reverse()
        phase = sum(b * 2 ** -(i + 1) for i, b in enumerate(measured_bits))

        return SemiclassicalQFTResult(
            phase=phase,
            measured_bits=measured_bits,
            n_qubits_used=1 + n_work,
        )


@dataclass
class CompressedModExpConfig:
    """Configuration for compressed modular exponentiation."""

    a: int
    N: int
    n_bits: int
    strategy: str


class ModularArithmeticOptimizer:
    """Optimized modular exponentiation for Shor's algorithm.

    Implements Beauregard's approach (2002) for in-place modular multiplication
    using 2n+3 qubits, and further optimizations from Häner et al. (2017).

    For simulation, provides efficient classical computation of modular
    exponentiation that tracks the qubit reduction achieved.
    """

    @staticmethod
    def qubit_count(n_bits: int, strategy: str = "beauregard") -> int:
        """Return qubit count for factoring an n-bit number.

        Parameters
        ----------
        n_bits
            Bit length of the number to factor.
        strategy
            "standard" (2n+3 counting + n work = ~3n),
            "beauregard" (n+1 counting via semiclassical + n+1 work + 1 ancilla = 2n+3),
            "haner" (further reduced, ~2n+2).
        """
        if strategy == "standard":
            return 2 * n_bits + 1 + n_bits
        elif strategy == "beauregard":
            return 2 * n_bits + 3
        elif strategy == "haner":
            return 2 * n_bits + 2
        elif strategy == "semiclassical":
            return n_bits + 2
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def mod_exp_permutation(a: int, power: int, N: int, n_bits: int) -> NDArray[np.complexfloating]:
        """Build modular exponentiation as a permutation matrix.

        |x⟩ → |a^power · x mod N⟩ for x < N, identity for x ≥ N.
        """
        dim = 2**n_bits
        matrix = np.zeros((dim, dim), dtype=complex)
        a_pow = pow(a, power, N)
        for x in range(dim):
            if x < N:
                y = (a_pow * x) % N
                matrix[y, x] = 1.0
            else:
                matrix[x, x] = 1.0
        return matrix

    @staticmethod
    def controlled_mod_multiply(
        a: int,
        N: int,
        n_bits: int,
        work_state: NDArray[np.complexfloating],
    ) -> NDArray[np.complexfloating]:
        """Apply controlled modular multiplication to work register state.

        Simulates |x⟩ → |a·x mod N⟩ for computational basis states.
        """
        perm = ModularArithmeticOptimizer.mod_exp_permutation(a, 1, N, n_bits)
        return perm @ work_state

    @staticmethod
    def create_config(a: int, N: int, strategy: str = "beauregard") -> CompressedModExpConfig:
        """Create a configuration for compressed modular exponentiation."""
        n_bits = max(N.bit_length(), 1)
        return CompressedModExpConfig(
            a=a,
            N=N,
            n_bits=n_bits,
            strategy=strategy,
        )

    @staticmethod
    def compressed_order_finding(
        a: int,
        N: int,
        n_counting: int | None = None,
        strategy: str = "semiclassical",
    ) -> tuple[float, int | None, int]:
        """Find order of a mod N using compressed circuits.

        Returns (phase, candidate_order, qubits_used).
        """
        n_bits = max(N.bit_length(), 1)
        if n_counting is None:
            n_counting = 2 * n_bits + 1

        qubits_used = ModularArithmeticOptimizer.qubit_count(n_bits, strategy)

        def ctrl_u(power: int, state: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
            perm = ModularArithmeticOptimizer.mod_exp_permutation(a, power, N, n_bits)
            return perm @ state

        sqft = SemiclassicalQFT(n_counting)
        result = sqft.estimate_phase(ctrl_u, n_bits)

        order = _extract_order(result.phase, a, N)

        return result.phase, order, qubits_used


def _extract_order(phase: float, a: int, N: int) -> int | None:
    """Extract order from phase using continued fractions."""
    if abs(phase) < 1e-10:
        return None

    x = phase
    convergents: list[tuple[int, int]] = []
    a_coeffs: list[int] = []

    for _ in range(30):
        a_k = int(x)
        a_coeffs.append(a_k)
        frac = x - a_k
        if abs(frac) < 1e-10:
            break
        x = 1.0 / frac

    h_prev, h_curr = 0, 1
    k_prev, k_curr = 1, 0
    for coeff in a_coeffs:
        h_prev, h_curr = h_curr, coeff * h_curr + h_prev
        k_prev, k_curr = k_curr, coeff * k_curr + k_prev
        if k_curr > 0 and k_curr <= N:
            convergents.append((h_curr, k_curr))

    for _, denom in convergents:
        if 0 < denom <= N and pow(a, denom, N) == 1:
            return denom

    for _, denom in convergents:
        for mult in range(2, N):
            candidate = denom * mult
            if candidate > N:
                break
            if pow(a, candidate, N) == 1:
                return candidate

    return None
