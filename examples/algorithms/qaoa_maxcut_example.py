"""QAOA for MaxCut — solving graph partitioning with quantum optimization.

Demonstrates the Quantum Approximate Optimization Algorithm on small graphs.
"""

import numpy as np

from quantumium.algorithms.qaoa import qaoa_maxcut, maxcut_cost_operator


def main():
    print("=" * 60)
    print("QAOA for MaxCut — Quantumium")
    print("=" * 60)

    # Example 1: Triangle graph
    print("\n1. Triangle graph (3 nodes, 3 edges)")
    print("   Edges: (0,1), (1,2), (0,2)")
    print("   Optimal MaxCut = 2")

    result = qaoa_maxcut(
        num_qubits=3,
        edges=[(0, 1), (1, 2), (0, 2)],
        p=2,
        num_starts=10,
        max_iter=100,
    )
    print(f"   QAOA result: {result.optimal_value:.4f}")
    print(f"   Best bitstring: {result.optimal_bitstring}")
    print(f"   Optimization steps: {len(result.history)}")

    # Example 2: 4-node path graph
    print("\n2. Path graph (4 nodes)")
    print("   Edges: (0,1), (1,2), (2,3)")
    print("   Optimal MaxCut = 2 (alternating partition)")

    result2 = qaoa_maxcut(
        num_qubits=4,
        edges=[(0, 1), (1, 2), (2, 3)],
        p=2,
        num_starts=10,
        max_iter=100,
    )
    print(f"   QAOA result: {result2.optimal_value:.4f}")
    print(f"   Best bitstring: {result2.optimal_bitstring}")

    # Example 3: 4-node complete graph
    print("\n3. Complete graph K4 (4 nodes, 6 edges)")
    edges_k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    print(f"   Edges: {edges_k4}")
    print("   Optimal MaxCut = 4")

    result3 = qaoa_maxcut(
        num_qubits=4,
        edges=edges_k4,
        p=2,
        num_starts=10,
        max_iter=100,
    )
    print(f"   QAOA result: {result3.optimal_value:.4f}")
    print(f"   Best bitstring: {result3.optimal_bitstring}")

    # Show cost landscape for single edge
    print("\n4. Cost landscape analysis (single edge)")
    cost = maxcut_cost_operator(2, [(0, 1)])
    print(f"   Cost vector: {cost}")
    print(f"   |00⟩→{cost[0]}, |01⟩→{cost[1]}, |10⟩→{cost[2]}, |11⟩→{cost[3]}")

    print("\n" + "=" * 60)
    print("All QAOA demonstrations complete.")


if __name__ == "__main__":
    main()
