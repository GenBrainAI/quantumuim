"""Quantum networking simulation demo.

Demonstrates:
1. Network topology construction (chain, star, mesh)
2. Entanglement generation with channel loss
3. Entanglement swapping through repeater chain
4. BBPSSW purification improving fidelity
5. Quantum teleportation over noisy channel
6. Full pipeline: distribute_entanglement one-liner
"""

import numpy as np

from quantumium.networking import (
    QuantumNetwork,
    generate_entanglement,
    entanglement_swap,
    purify_bbpssw,
    teleport,
    distribute_entanglement,
)
from quantumium.networking.purification import purify_rounds
from quantumium.networking.teleportation import teleportation_fidelity


def main():
    print("=" * 60)
    print("Quantum Networking Simulation Demo")
    print("=" * 60)

    # 1. Repeater chain
    print("\n1. Repeater Chain — Entanglement Distribution")
    print("-" * 40)
    net = QuantumNetwork.chain(
        ["Alice", "R1", "R2", "Bob"],
        segment_distance_km=50,
        loss_db_per_km=0.2,
    )
    print(f"   Network: {net.num_nodes} nodes, {net.num_channels} channels")
    print(f"   Topology: Alice --50km-- R1 --50km-- R2 --50km-- Bob")

    pair_ar = generate_entanglement(net, "Alice", "R1")
    pair_r1r2 = generate_entanglement(net, "R1", "R2")
    pair_r2b = generate_entanglement(net, "R2", "Bob")
    print(f"   Segment fidelities: {pair_ar.fidelity:.4f}, "
          f"{pair_r1r2.fidelity:.4f}, {pair_r2b.fidelity:.4f}")

    pair_ar2 = entanglement_swap(pair_ar, pair_r1r2)
    pair_ab = entanglement_swap(pair_ar2, pair_r2b)
    print(f"   After swapping: Alice-Bob fidelity = {pair_ab.fidelity:.4f}")

    # 2. BBPSSW purification
    print("\n2. BBPSSW Purification")
    print("-" * 40)
    from quantumium.networking.entanglement import EntangledPair
    pair_low = EntangledPair("A", "B", 0.75)
    print(f"   Initial fidelity: {pair_low.fidelity:.4f}")
    for rounds in [1, 2, 5, 10, 20]:
        result = purify_rounds(pair_low, rounds + 1)
        print(f"   After {rounds:2d} round(s): F = {result.fidelity:.6f}")

    # 3. Teleportation
    print("\n3. Quantum Teleportation")
    print("-" * 40)
    for f_ch in [1.0, 0.9, 0.75, 0.6]:
        pair_tel = EntangledPair("A", "B", f_ch)
        result = teleport([1, 0], pair_tel)
        theory = teleportation_fidelity(f_ch)
        print(f"   Channel F={f_ch:.2f} -> Teleport F={result.fidelity:.4f} "
              f"(theory: {theory:.4f})")

    # 4. Star topology
    print("\n4. Star Topology")
    print("-" * 40)
    star = QuantumNetwork.star("Hub", ["A", "B", "C", "D"], distance_km=20)
    print(f"   Network: {star.num_nodes} nodes, {star.num_channels} channels")
    pair_star = distribute_entanglement(star, "A", "C")
    print(f"   A-C via Hub: fidelity = {pair_star.fidelity:.4f}")

    # 5. Mesh topology
    print("\n5. Mesh Topology (3x3)")
    print("-" * 40)
    mesh = QuantumNetwork.mesh(3, 3, distance_km=10)
    print(f"   Network: {mesh.num_nodes} nodes, {mesh.num_channels} channels")
    pair_mesh = distribute_entanglement(mesh, "N0_0", "N2_2")
    print(f"   Corner-to-corner: fidelity = {pair_mesh.fidelity:.4f}")

    # 6. Full pipeline
    print("\n6. Full Pipeline — distribute_entanglement()")
    print("-" * 40)
    net6 = QuantumNetwork.chain(
        ["Alice", "R1", "Bob"],
        segment_distance_km=5,
    )
    pair_raw = distribute_entanglement(net6, "Alice", "Bob", purification_rounds=0)
    pair_p3 = distribute_entanglement(net6, "Alice", "Bob", purification_rounds=3)
    pair_p10 = distribute_entanglement(net6, "Alice", "Bob", purification_rounds=10)
    print(f"   Chain: Alice --5km-- R1 --5km-- Bob")
    print(f"   Raw:             F = {pair_raw.fidelity:.4f}")
    print(f"   + 3 purif rounds: F = {pair_p3.fidelity:.4f}")
    print(f"   +10 purif rounds: F = {pair_p10.fidelity:.4f}")

    tel = teleport(np.array([1, 1]) / np.sqrt(2), pair_p10)
    print(f"   Teleport |+> over purified link: F = {tel.fidelity:.4f}")

    print("\n" + "=" * 60)
    print("Networking demo complete.")
    print("Key insight: Repeaters + purification enable high-fidelity")
    print("entanglement over long distances for quantum internet.")
    print("=" * 60)


if __name__ == "__main__":
    main()
