"""Quantum walks demo — discrete and continuous walks on various graphs.

Demonstrates:
1. Hadamard walk on a line (ballistic spreading vs diffusive)
2. Grover walk on a complete graph
3. Continuous-time walk on a cycle (perfect state transfer)
4. Walk on a hypercube
"""

import numpy as np
from quantumium.quantum_walks import (
    hadamard_walk_line,
    grover_walk,
    discrete_walk,
    continuous_walk,
    cycle_graph,
    complete_graph,
    hypercube_graph,
    line_graph,
)

print("=" * 60)
print("Quantum Walks Demo")
print("=" * 60)

# --- 1. Hadamard Walk on a Line ---
print("\n1. Hadamard Walk on a Line (41 positions, 20 steps)")
print("-" * 40)

result = hadamard_walk_line(
    num_positions=41,
    num_steps=20,
    initial_coin_state=np.array([1, 1j], dtype=complex) / np.sqrt(2),
)

center = 20
probs = result.position_probabilities
mean = sum(i * p for i, p in enumerate(probs))
variance = sum((i - mean) ** 2 * p for i, p in enumerate(probs))
print(f"  Mean position: {mean:.2f} (center={center})")
print(f"  Variance: {variance:.2f}")
print(f"  Std dev: {np.sqrt(variance):.2f}")
print(f"  Top positions: ", end="")
top = sorted(range(41), key=lambda i: -probs[i])[:5]
for i in top:
    print(f"  pos {i}: {probs[i]:.4f}", end="")
print()

# Compare: classical random walk variance = n, quantum ~ n^2
print(f"  Quantum advantage: variance ~ O(n^2) vs classical O(n)")
print(f"  After 20 steps: quantum std={np.sqrt(variance):.1f} "
      f"vs classical std~{np.sqrt(20):.1f}")

# --- 2. Grover Walk on Complete Graph ---
print("\n2. Grover Walk on K_5 (10 steps)")
print("-" * 40)

A = complete_graph(5)
result = grover_walk(A, num_steps=10, initial_vertex=0)
print(f"  Probability distribution: {np.round(result.position_probabilities, 4)}")
print(f"  Starting vertex probability: {result.position_probabilities[0]:.4f}")

# --- 3. Continuous Walk on Cycle ---
print("\n3. Continuous-Time Walk on Cycle (8 vertices)")
print("-" * 40)

A = cycle_graph(8)
for t in [0.0, 0.5, 1.0, 2.0, np.pi]:
    result = continuous_walk(A, time=t, initial_vertex=0)
    p0 = result.position_probabilities[0]
    p4 = result.position_probabilities[4]
    print(f"  t={t:5.2f}: P(start)={p0:.4f}, P(opposite)={p4:.4f}, "
          f"sum={np.sum(result.position_probabilities):.6f}")

# --- 4. Continuous Walk on Hypercube ---
print("\n4. Continuous-Time Walk on 4D Hypercube (16 vertices)")
print("-" * 40)

A = hypercube_graph(4)
result = continuous_walk(A, time=np.pi / 4, initial_vertex=0)
print(f"  At t=π/4:")
print(f"    P(vertex 0) = {result.position_probabilities[0]:.4f}")
print(f"    P(vertex 15) = {result.position_probabilities[15]:.4f}")
print(f"    Antipodal transfer fidelity: {result.position_probabilities[15]:.4f}")

# --- 5. Comparison: Discrete vs Continuous ---
print("\n5. Discrete vs Continuous Walk Spreading (line, 21 positions)")
print("-" * 40)

center = 10
A = line_graph(21)

# Discrete
result_d = hadamard_walk_line(
    num_positions=21, num_steps=10,
    initial_coin_state=np.array([1, 1j], dtype=complex) / np.sqrt(2),
)
var_d = sum((i - center) ** 2 * p
            for i, p in enumerate(result_d.position_probabilities))

# Continuous
result_c = continuous_walk(A, time=10.0, initial_vertex=center)
var_c = sum((i - center) ** 2 * p
            for i, p in enumerate(result_c.position_probabilities))

print(f"  Discrete (10 steps): variance = {var_d:.2f}")
print(f"  Continuous (t=10):   variance = {var_c:.2f}")

print("\n" + "=" * 60)
print("Done!")
