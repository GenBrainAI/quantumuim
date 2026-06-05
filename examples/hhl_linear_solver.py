"""Example: HHL quantum linear systems solver.

Demonstrates solving Ax = b using the HHL algorithm, showing
how quantum phase estimation and eigenvalue inversion produce
a state proportional to A^{-1}|b>.
"""

import numpy as np

from quantumium.algorithms.hhl import hhl_solve

print("=" * 55)
print("HHL Quantum Linear Systems Solver")
print("=" * 55)

# --- Example 1: Diagonal system ---
print("\n--- Example 1: Diagonal A = diag(1, 2) ---")
A1 = np.array([[1, 0], [0, 2]], dtype=complex)
b1 = np.array([1, 1], dtype=complex)
print(f"A = {A1.real}")
print(f"b = {b1.real}")

result1 = hhl_solve(A1, b1, num_clock=3)
print(f"Classical x = A^-1 b (normalized): {result1.classical_solution.real}")
print(f"HHL solution state:                {result1.solution_state.real}")
print(f"Fidelity: {result1.fidelity:.6f}")
print(f"Success probability: {result1.success_probability:.4f}")

# --- Example 2: Non-diagonal Hermitian ---
print("\n--- Example 2: Hermitian A = [[2,1],[1,3]] ---")
A2 = np.array([[2, 1], [1, 3]], dtype=complex)
b2 = np.array([1, 0], dtype=complex)
print(f"A = {A2.real}")
print(f"b = {b2.real}")

result2 = hhl_solve(A2, b2, num_clock=4)
x_exact = np.linalg.solve(A2, b2)
print(f"Exact solution x: {x_exact.real}")
print(f"Classical x (normalized): {result2.classical_solution.real}")
print(f"HHL solution state:       {result2.solution_state.real}")
print(f"Fidelity: {result2.fidelity:.6f}")
print(f"Success probability: {result2.success_probability:.4f}")

# --- Example 3: Precision scaling ---
print("\n--- Example 3: Fidelity vs clock qubits ---")
A3 = np.array([[3, 1], [1, 2]], dtype=complex)
b3 = np.array([1, 1], dtype=complex)
print(f"A = {A3.real}, b = {b3.real}")
for nc in [2, 3, 4]:
    result = hhl_solve(A3, b3, num_clock=nc)
    print(f"  num_clock={nc}: fidelity={result.fidelity:.6f}, "
          f"success_prob={result.success_probability:.4f}")

print("\n" + "=" * 55)
