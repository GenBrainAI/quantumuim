#!/usr/bin/env python3
"""Getting Started with Q+ — Semiclassical QFT tutorial.

Shows how Q+ compresses a quantum phase estimation circuit from n counting
qubits down to 1 reusable qubit, achieving the same result with far fewer
quantum resources.

Usage:
    python src/quantumium/qplus/examples/getting_started.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantumium.core.gate import Rz
from quantumium.core.circuit import QuantumCircuit
from quantumium.circuits.templates import qft
from quantumium.qplus.compress import SemiclassicalQFT, ModularArithmeticOptimizer


def step1_standard_qft_cost() -> None:
    """Show how many qubits standard QFT-based phase estimation needs."""
    print("STEP 1: Standard QFT qubit costs")
    print("-" * 50)

    for n_bits in [4, 6, 8]:
        qft_circuit = qft(n_bits)
        print(
            f"  {n_bits}-bit QFT: {qft_circuit.num_qubits} counting qubits, "
            f"circuit depth = {qft_circuit.depth}"
        )

    print("\n  For 8-bit Shor: need 8 counting + 8 work = 16+ qubits")
    print()


def step2_semiclassical_compression() -> None:
    """Demonstrate semiclassical QFT reducing counting qubits to 1."""
    print("STEP 2: Semiclassical QFT — 1 counting qubit instead of n")
    print("-" * 50)

    # Build a controlled-Rz unitary to estimate its phase
    theta = np.pi / 4  # phase = 1/8
    qc = QuantumCircuit(1)
    qc.apply(Rz(theta), 0)
    rz_matrix = Rz(theta).matrix

    def ctrl_rz(power: int, state: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        u: NDArray[np.complexfloating] = np.linalg.matrix_power(rz_matrix, power)
        return u @ state

    sqft = SemiclassicalQFT(n_bits=8)
    initial = np.array([0, 1], dtype=complex)  # eigenstate |1⟩

    phases: list[float] = []
    for _ in range(30):
        result = sqft.estimate_phase(ctrl_rz, 1, initial)
        phases.append(result.phase)

    expected_phase = theta / (2 * np.pi)
    mean_phase = np.mean(phases)

    print(f"  Target phase: {expected_phase:.4f} (θ = π/4)")
    print(f"  Estimated:    {mean_phase:.4f} (mean over 30 trials)")
    print(f"  Qubits used:  {result.n_qubits_used} (1 counting + 1 work)")
    print(f"  Savings:      {8 - 1} fewer counting qubits vs standard QFT")
    print()


def step3_qubit_strategy_comparison() -> None:
    """Compare qubit counts across optimization strategies."""
    print("STEP 3: Strategy comparison for 8-bit numbers")
    print("-" * 50)

    print(f"  {'Strategy':<20} {'Qubits':>8}  Description")
    print(f"  {'─' * 20} {'─' * 8}  {'─' * 30}")

    strategies = [
        ("standard", "Textbook Shor (3n+1)"),
        ("beauregard", "Beauregard (2n+3)"),
        ("haner", "Häner et al. (2n+2)"),
        ("semiclassical", "Q+ semiclassical (n+2)"),
    ]
    for strategy, desc in strategies:
        qubits = ModularArithmeticOptimizer.qubit_count(8, strategy)
        print(f"  {strategy:<20} {qubits:>8}  {desc}")

    std = ModularArithmeticOptimizer.qubit_count(8, "standard")
    qp = ModularArithmeticOptimizer.qubit_count(8, "semiclassical")
    print(f"\n  Reduction: {std} → {qp} qubits ({(1 - qp / std) * 100:.0f}% fewer)")
    print()


def step4_full_factoring() -> None:
    """Factor a small number using the full Q+ pipeline."""
    print("STEP 4: Factor 15 with Q+")
    print("-" * 50)

    from quantumium.qplus.shor import ShorQPlus

    shor = ShorQPlus(strategy="semiclassical")
    comp = ShorQPlus.qubit_comparison(15)

    print(f"  Standard Shor: {comp['standard_shor']} qubits")
    print(f"  Q+ approach:   {comp['semiclassical_qft']} qubits")
    print()

    for trial in range(1, 6):
        result = shor.factor(15, max_attempts=10)
        if result.success and result.factors:
            p, q = result.factors
            print(
                f"  Trial {trial}: 15 = {p} × {q} "
                f"(used {result.qubits_used} qubits, layers: {result.layers_used})"
            )
            break
    else:
        print("  (probabilistic — may need multiple trials)")
    print()


def main() -> None:
    print("=" * 60)
    print("  GETTING STARTED WITH Q+")
    print("  Beyond-qubit-count quantum computing")
    print("=" * 60)
    print()

    step1_standard_qft_cost()
    step2_semiclassical_compression()
    step3_qubit_strategy_comparison()
    step4_full_factoring()

    print("=" * 60)
    print("  Next: Try multi_qpu_example.py for distributed execution")
    print("=" * 60)


if __name__ == "__main__":
    main()
