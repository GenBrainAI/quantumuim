#!/usr/bin/env python3
"""Circuit Cutting Example — Splitting large circuits into smaller subcircuits.

Demonstrates the Q+ circuit cutting layer (Peng et al. 2020):
- Wire cutting with Pauli-basis decomposition
- Subcircuit evaluation and recombination
- Overhead analysis for different numbers of cuts

Usage:
    python src/quantumium/qplus/examples/circuit_cutting_example.py
"""

from __future__ import annotations

import numpy as np

from quantumium.core.gate import H, CNOT
from quantumium.core.circuit import QuantumCircuit
from quantumium.qplus.cut import CircuitCutter, CutRecombiner, WireCut


def _circuit_unitary(qc: QuantumCircuit) -> np.ndarray:
    n = qc.num_qubits
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for inst in qc.instructions:
        full = qc._expand_gate(inst.gate.matrix, inst.target_qubits, n)
        U = full @ U
    return U


def demo_wire_cut_basics() -> None:
    """Explain wire cutting and show Pauli decomposition."""
    print("1. WIRE CUTTING BASICS")
    print("-" * 50)
    print("  A wire cut replaces a quantum wire with classical")
    print("  communication: the circuit is split into two halves,")
    print("  each evaluated with 6 Pauli-basis preparations/")
    print("  measurements, then classically recombined.")
    print()

    cutter = CircuitCutter([WireCut(qubit=0, after_gate_index=0)])
    configs = cutter.decompose(n_qubits=2, gates=[])

    print(f"  1 wire cut → {len(configs)} subcircuit configurations:")
    for i, (prep, meas, coeff) in enumerate(configs[:6]):
        print(f"    Config {i}: prep={prep}, meas={meas}, coeff={coeff:+.4f}")
    if len(configs) > 6:
        print(f"    ... ({len(configs)} total)")
    print()


def demo_overhead_scaling() -> None:
    """Show how overhead scales with number of cuts."""
    print("2. OVERHEAD SCALING (O(4^K) sampling, 6^K evaluations)")
    print("-" * 50)

    print(f"  {'Cuts':>6} {'Evaluations':>14} {'Overhead':>10} {'Practical?':>12}")
    print(f"  {'─' * 6} {'─' * 14} {'─' * 10} {'─' * 12}")

    cutter = CircuitCutter()
    for k in range(1, 7):
        res = cutter.estimate_resources(n_cuts=k)
        evals = int(res["n_subcircuit_evals"])
        overhead = int(res["overhead_factor"])
        practical = "✓" if k <= 3 else "expensive" if k <= 5 else "✗"
        print(f"  {k:>6} {evals:>14,} {overhead:>10,}x {practical:>12}")

    print()
    print("  Rule of thumb: keep K ≤ 3 for practical overhead.")
    print()


def demo_cut_and_recombine() -> None:
    """Cut a Bell state circuit and recombine the results."""
    print("3. CUT AND RECOMBINE — BELL STATE")
    print("-" * 50)

    # Build Bell state: H on q0, CNOT(q0, q1)
    qc = QuantumCircuit(2)
    qc.apply(H, 0)
    qc.apply(CNOT, 0, 1)
    U = _circuit_unitary(qc)

    # Ideal result
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0
    final = U @ state
    ideal_p00 = float(np.abs(final[0]) ** 2)
    print(f"  Ideal |⟨00|Bell⟩|² = {ideal_p00:.4f}")

    # Cut the circuit at wire 0, after gate 0
    cutter = CircuitCutter([WireCut(qubit=0, after_gate_index=0)])
    identity = np.eye(4, dtype=complex)

    reconstructed = cutter.evaluate_cut_circuit(identity, U, 0, 2)
    print(f"  Reconstructed value  = {reconstructed:.4f}")
    print()

    # Verify reconstruction
    is_close = CutRecombiner.verify_reconstruction(
        ideal_p00, reconstructed, tolerance=0.15
    )
    print(f"  Verification (tol=0.15): {'PASS' if is_close else 'FAIL'}")
    print()


def demo_recombine_subcircuits() -> None:
    """Show classical recombination of subcircuit expectations."""
    print("4. CLASSICAL RECOMBINATION")
    print("-" * 50)

    # Simulate subcircuit results from a 1-cut decomposition
    subcircuit_results: list[tuple[float, float]] = [
        (0.5, 0.85),   # Pauli I preparation
        (-0.5, 0.15),  # Pauli X preparation
        (0.5, 0.65),   # Pauli Y preparation
        (-0.5, 0.35),  # Pauli Z preparation
        (0.25, 0.90),  # mixed term 1
        (-0.25, 0.50), # mixed term 2
    ]

    recombined = CutRecombiner.recombine(subcircuit_results)
    print(f"  Subcircuit expectations: {[f'{e:.2f}' for _, e in subcircuit_results]}")
    print(f"  Coefficients:            {[f'{c:+.2f}' for c, _ in subcircuit_results]}")
    print(f"  Recombined value: {recombined:.4f}")
    print()

    # Probability recombination
    prob_results: list[tuple[float, dict[str, float]]] = [
        (0.5, {"00": 0.9, "01": 0.1}),
        (0.5, {"00": 0.7, "01": 0.3}),
        (-0.5, {"00": 0.5, "01": 0.5}),
        (0.5, {"00": 0.6, "01": 0.4}),
    ]

    probs = CutRecombiner.recombine_probabilities(prob_results)
    print(f"  Recombined probability distribution:")
    for bitstring, prob in sorted(probs.items()):
        bar = "█" * int(prob * 40)
        print(f"    |{bitstring}⟩: {prob:.4f} {bar}")
    print(f"  Sum: {sum(probs.values()):.4f}")
    print()


def demo_multi_cut() -> None:
    """Show cutting a larger circuit at multiple points."""
    print("5. MULTIPLE CUTS — LARGER CIRCUIT")
    print("-" * 50)

    # 4-qubit GHZ state
    qc = QuantumCircuit(4)
    qc.apply(H, 0)
    for i in range(3):
        qc.apply(CNOT, i, i + 1)
    U = _circuit_unitary(qc)

    print(f"  4-qubit GHZ circuit: H + 3 CNOTs")
    print()

    for n_cuts in [1, 2, 3]:
        cuts = [WireCut(qubit=i, after_gate_index=i) for i in range(n_cuts)]
        cutter = CircuitCutter(cuts)
        res = cutter.estimate_resources(n_cuts=n_cuts)
        print(f"  {n_cuts} cut(s): {int(res['n_subcircuit_evals'])} evals, "
              f"{int(res['overhead_factor'])}x overhead")

    print()


def main() -> None:
    print("=" * 60)
    print("  CIRCUIT CUTTING WITH Q+")
    print("  Splitting circuits for distributed execution")
    print("=" * 60)
    print()

    demo_wire_cut_basics()
    demo_overhead_scaling()
    demo_cut_and_recombine()
    demo_recombine_subcircuits()
    demo_multi_cut()

    print("=" * 60)
    print("  See also: shor_benchmark.py for full Q+ stack in action")
    print("=" * 60)


if __name__ == "__main__":
    main()
