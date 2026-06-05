"""Quantum counting and amplitude estimation demo.

Demonstrates:
1. Quantum counting — estimating the number of marked items
2. Canonical amplitude estimation (QPE-based)
3. Iterative amplitude estimation (QPE-free)
"""

import math
from quantumium.algorithms.quantum_counting import (
    quantum_counting,
    amplitude_estimation,
    iterative_amplitude_estimation,
)
from quantumium.algorithms.grover import grover_oracle
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import Gate

print("=" * 60)
print("Quantum Counting & Amplitude Estimation")
print("=" * 60)

# --- 1. Quantum Counting ---
print("\n1. Quantum Counting")
print("-" * 40)

for n_qubits, marked, desc in [
    (2, [0], "1 out of 4"),
    (2, [0, 1], "2 out of 4"),
    (3, [0, 1, 2], "3 out of 8"),
    (3, [5], "1 out of 8"),
]:
    N = 2**n_qubits
    M_true = len(marked)
    result = quantum_counting(n_qubits, marked, num_counting_qubits=5)
    print(f"  {desc} (N={N}): M_est={result.estimated_count:.2f} "
          f"(true={M_true}), CI=[{result.confidence_interval[0]:.2f}, "
          f"{result.confidence_interval[1]:.2f}]")

# Precision vs counting qubits
print("\n  Precision scaling (1 out of 4):")
for nc in [3, 4, 5, 6]:
    r = quantum_counting(2, [0], num_counting_qubits=nc)
    print(f"    {nc} counting qubits: M_est={r.estimated_count:.4f}, "
          f"error={abs(r.estimated_count - 1):.4f}")

# --- 2. Canonical Amplitude Estimation ---
print("\n2. Canonical Amplitude Estimation (QPE-based)")
print("-" * 40)

for marked, desc in [([0], "1/4"), ([0, 1], "2/4"), ([0, 1, 2], "3/4")]:
    true_prob = len(marked) / 4
    oracle = QuantumCircuit(2)
    oracle.apply(Gate(grover_oracle(2, marked).matrix, "O"), 0, 1)
    result = amplitude_estimation(oracle, num_evaluation_qubits=5)
    print(f"  P(good)={desc}: estimated={result.estimated_probability:.4f} "
          f"(true={true_prob:.4f}), theta={result.theta:.4f}")

# --- 3. Iterative Amplitude Estimation ---
print("\n3. Iterative Amplitude Estimation (QPE-free)")
print("-" * 40)

for marked, desc in [([0], "1/4"), ([0, 1], "2/4")]:
    true_prob = len(marked) / 4
    result = iterative_amplitude_estimation(2, marked, num_rounds=6)
    print(f"  P(good)={desc}: estimated={result.estimated_probability:.4f} "
          f"(true={true_prob:.4f}), oracle_calls={result.total_oracle_calls}")

# Scaling with rounds
print("\n  Convergence with rounds (2 out of 4):")
for nr in [2, 3, 4, 5, 6]:
    r = iterative_amplitude_estimation(2, [0, 1], num_rounds=nr)
    print(f"    {nr} rounds: P_est={r.estimated_probability:.6f} "
          f"(true=0.500000), error={abs(r.estimated_probability - 0.5):.6f}, "
          f"calls={r.total_oracle_calls}")

print("\n" + "=" * 60)
print("Done!")
