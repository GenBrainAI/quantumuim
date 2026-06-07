#!/usr/bin/env python3
"""Q+ Algorithm Demo — showcasing each layer independently.

Demonstrates:
1. Semiclassical QFT phase estimation (Layer 1: Compress)
2. Qubit recycling with feedforward (Layer 2: Recycle)
3. Circuit cutting decomposition (Layer 3a: Cut)
4. Multi-QPU distribution planning (Layer 3b: Distribute)
"""

import numpy as np

from quantumium.qplus.compress import SemiclassicalQFT, ModularArithmeticOptimizer
from quantumium.qplus.recycle import QubitRecycler
from quantumium.qplus.cut import CircuitCutter, CutRecombiner, WireCut
from quantumium.qplus.distribute import QPUOrchestrator


def demo_layer1_compress():
    """Layer 1: Semiclassical QFT reduces counting qubits to 1."""
    print("=" * 60)
    print("LAYER 1: Algorithmic Compression (Semiclassical QFT)")
    print("=" * 60)

    print("\nPhase estimation for Z gate (eigenvalue -1, phase = 0.5):")

    def ctrl_z(power, state):
        result = state.copy()
        result[1] *= (-1) ** power
        return result

    sqft = SemiclassicalQFT(n_bits=8)
    initial = np.array([0, 1], dtype=complex)

    phases = []
    for trial in range(10):
        result = sqft.estimate_phase(ctrl_z, 1, initial)
        phases.append(result.phase)
        if trial < 3:
            print(f"  Trial {trial + 1}: phase = {result.phase:.4f}, "
                  f"qubits used = {result.n_qubits_used}")

    print(f"  Mean phase: {np.mean(phases):.4f} (expected: 0.5)")
    print(f"  Qubits used: {result.n_qubits_used} vs standard {8 + 1} = {8 + 1}")

    print("\nQubit counts for modular exponentiation strategies:")
    for strategy in ["standard", "beauregard", "haner", "semiclassical"]:
        qubits = ModularArithmeticOptimizer.qubit_count(8, strategy)
        print(f"  {strategy:15s}: {qubits} qubits for 8-bit number")


def demo_layer2_recycle():
    """Layer 2: Qubit recycling reuses measured qubits."""
    print("\n" + "=" * 60)
    print("LAYER 2: Qubit Recycling")
    print("=" * 60)

    recycler = QubitRecycler(n_physical=5)
    print(f"\nPhysical qubits: {recycler.n_physical}")

    n_bits = 4

    def ctrl_u(power, state):
        perm = ModularArithmeticOptimizer.mod_exp_permutation(7, power, 15, n_bits)
        return perm @ state

    print("Finding order of 7 mod 15 with recycled counting qubit:")
    successes = 0
    for trial in range(10):
        qr = QubitRecycler(n_bits + 1)
        phase, bits = qr.simulate_recycled_phase_estimation(ctrl_u, n_bits, 9)
        from quantumium.qplus.compress import _extract_order
        order = _extract_order(phase, 7, 15)
        if order == 4:
            successes += 1
        if trial < 3:
            print(f"  Trial {trial + 1}: phase={phase:.4f}, order={order}")

    print(f"  Success rate: {successes}/10 (order=4)")
    print(f"  Qubits: {n_bits + 1} (1 recycled counting + {n_bits} work)")


def demo_layer3a_cut():
    """Layer 3a: Circuit cutting for distributing across QPUs."""
    print("\n" + "=" * 60)
    print("LAYER 3a: Circuit Cutting")
    print("=" * 60)

    cutter = CircuitCutter([WireCut(0, 0)])
    configs = cutter.decompose(2, [])
    print(f"\n1 wire cut -> {len(configs)} subcircuit configurations")
    print(f"Overhead factor: {cutter.overhead}")

    for k in range(1, 5):
        resources = cutter.estimate_resources(n_cuts=k)
        print(f"  {k} cuts: {int(resources['n_subcircuit_evals'])} evaluations, "
              f"{int(resources['overhead_factor'])}x overhead")

    print("\nCut circuit evaluation (identity):")
    n = 2
    dim = 2**n
    identity = np.eye(dim, dtype=complex)
    result = cutter.evaluate_cut_circuit(identity, identity, 0, n)
    print(f"  Reconstructed expectation: {result:.4f} (expected: 1.0)")


def demo_layer3b_distribute():
    """Layer 3b: Multi-QPU orchestration."""
    print("\n" + "=" * 60)
    print("LAYER 3b: Multi-QPU Distribution")
    print("=" * 60)

    for n_qpus in [1, 2, 3]:
        orch = QPUOrchestrator(n_qpus=n_qpus)
        plan = orch.partition_circuit(6)
        print(f"\n{n_qpus} QPU(s), 6 qubits:")
        print(f"  Partitions: {[p.qubit_indices for p in plan.partitions]}")
        print(f"  Cuts: {len(plan.cuts)}, Overhead: {plan.estimated_overhead}x")

    print("\nStrategy recommendations:")
    for n_q, n_qpu, qlink in [(4, 2, False), (8, 4, False), (8, 4, True)]:
        rec = QPUOrchestrator.recommend_strategy(n_q, n_qpu, qlink)
        print(f"  {n_q} qubits, {n_qpu} QPUs, quantum_link={qlink}: {rec}")


def main():
    print("Q+ ALGORITHM DEMO")
    print("Beyond-qubit-count quantum computing framework\n")

    demo_layer1_compress()
    demo_layer2_recycle()
    demo_layer3a_cut()
    demo_layer3b_distribute()

    print("\n" + "=" * 60)
    print("All Q+ layers demonstrated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
