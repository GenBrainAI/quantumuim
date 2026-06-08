#!/usr/bin/env python3
"""Multi-QPU Distribution Example — Running circuits across multiple QPUs.

Demonstrates QPUOrchestrator with all three distribution strategies:
- classical_only: wire cutting (no quantum link needed)
- locc: local operations + classical communication
- teleport: gate teleportation via pre-shared Bell pairs

Usage:
    python src/quantumium/qplus/examples/multi_qpu_example.py
"""

from __future__ import annotations

import numpy as np

from quantumium.core.gate import H, CNOT
from quantumium.core.circuit import QuantumCircuit
from quantumium.qplus.distribute import QPUOrchestrator


def _circuit_unitary(qc: QuantumCircuit) -> np.ndarray:
    n = qc.num_qubits
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for inst in qc.instructions:
        full = qc._expand_gate(inst.gate.matrix, inst.target_qubits, n)
        U = full @ U
    return U


def demo_partition_planning() -> None:
    """Show how QPUOrchestrator partitions circuits across QPUs."""
    print("1. CIRCUIT PARTITIONING")
    print("-" * 50)

    for n_qubits in [4, 6, 8]:
        for n_qpus in [2, 4]:
            if n_qpus > n_qubits:
                continue
            orch = QPUOrchestrator(n_qpus=n_qpus, strategy="classical_only")
            plan = orch.partition_circuit(n_qubits)
            partitions = [p.qubit_indices for p in plan.partitions]
            print(
                f"  {n_qubits} qubits / {n_qpus} QPUs: "
                f"partitions={partitions}, "
                f"cuts={len(plan.cuts)}, "
                f"overhead={plan.estimated_overhead:.0f}x"
            )

    print()


def demo_strategy_comparison() -> None:
    """Compare all three distribution strategies on the same circuit."""
    print("2. STRATEGY COMPARISON")
    print("-" * 50)

    qc = QuantumCircuit(4)
    qc.apply(H, 0)
    for i in range(3):
        qc.apply(CNOT, i, i + 1)
    U = _circuit_unitary(qc)

    print("  Circuit: 4-qubit GHZ state (H + 3 CNOTs)")
    print(f"  {'Strategy':<18} {'Value':>8} {'Overhead':>10} {'QPUs':>6} {'Cuts':>6}")
    print(f"  {'─' * 18} {'─' * 8} {'─' * 10} {'─' * 6} {'─' * 6}")

    for strategy in ["classical_only", "locc", "teleport"]:
        orch = QPUOrchestrator(n_qpus=2, strategy=strategy)
        result = orch.execute_distributed(U, 4)
        print(
            f"  {strategy:<18} {result.recombined_value:>8.4f} "
            f"{result.overhead:>10.1f}x "
            f"{result.n_qpus:>6} "
            f"{result.n_cuts:>6}"
        )

    print()


def demo_strategy_recommendation() -> None:
    """Show automatic strategy selection based on hardware capabilities."""
    print("3. STRATEGY RECOMMENDATIONS")
    print("-" * 50)

    scenarios = [
        (4, 2, False, "Small circuit, no quantum link"),
        (8, 4, False, "Large circuit, no quantum link"),
        (8, 4, True, "Large circuit, with quantum link"),
        (16, 8, True, "Very large circuit, with quantum link"),
    ]

    for n_qubits, n_qpus, quantum_link, desc in scenarios:
        rec = QPUOrchestrator.recommend_strategy(n_qubits, n_qpus, quantum_link)
        print(f"  {desc}")
        print(f"    → {rec}")

    print()


def demo_resource_estimation() -> None:
    """Estimate resources needed for distributing different circuit sizes."""
    print("4. RESOURCE ESTIMATION")
    print("-" * 50)

    print(f"  {'Qubits':>8} {'Gates':>8} {'QPUs':>6} {'Cuts':>6} {'Overhead':>10} {'Strategy':<15}")
    print(f"  {'─' * 8} {'─' * 8} {'─' * 6} {'─' * 6} {'─' * 10} {'─' * 15}")

    configs = [
        (4, 6, 2, "classical_only"),
        (6, 10, 2, "classical_only"),
        (8, 15, 4, "locc"),
        (12, 25, 4, "teleport"),
    ]

    for n_q, n_g, n_qpus, strategy in configs:
        orch = QPUOrchestrator(n_qpus=n_qpus, strategy=strategy)
        res = orch.estimate_resources(n_q, n_g)
        print(
            f"  {n_q:>8} {n_g:>8} {int(res['n_qpus']):>6} "
            f"{int(res['n_cuts']):>6} {res['overhead_factor']:>10.1f}x "
            f"{strategy:<15}"
        )

    print()


def demo_distributed_execution() -> None:
    """Run a Bell state circuit distributed across 2 QPUs."""
    print("5. DISTRIBUTED BELL STATE EXECUTION")
    print("-" * 50)

    qc = QuantumCircuit(2)
    qc.apply(H, 0)
    qc.apply(CNOT, 0, 1)
    U = _circuit_unitary(qc)

    state = np.zeros(4, dtype=complex)
    state[0] = 1.0
    final = U @ state
    ideal_value = float(np.abs(final[0]) ** 2)

    print(f"  Ideal ⟨0|ψ⟩² = {ideal_value:.4f}")
    print()

    for n_qpus in [1, 2]:
        orch = QPUOrchestrator(n_qpus=n_qpus, strategy="classical_only")
        result = orch.execute_distributed(U, 2)
        error = abs(result.recombined_value - ideal_value)
        print(
            f"  {n_qpus} QPU(s): value={result.recombined_value:.4f}, "
            f"overhead={result.overhead:.1f}x, "
            f"error={error:.4f}"
        )

    print()


def main() -> None:
    print("=" * 60)
    print("  MULTI-QPU DISTRIBUTION WITH Q+")
    print("  Distributing quantum circuits across multiple QPUs")
    print("=" * 60)
    print()

    demo_partition_planning()
    demo_strategy_comparison()
    demo_strategy_recommendation()
    demo_resource_estimation()
    demo_distributed_execution()

    print("=" * 60)
    print("  Next: Try circuit_cutting_example.py for circuit splitting")
    print("=" * 60)


if __name__ == "__main__":
    main()
