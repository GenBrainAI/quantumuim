"""Benchmarking demo — characterizing quantum hardware quality.

Demonstrates:
  1. Randomized benchmarking for gate error rates
  2. Quantum volume estimation
  3. State and process fidelity metrics
"""
from __future__ import annotations

import numpy as np

from quantumium.benchmarks import (
    randomized_benchmarking,
    quantum_volume,
    state_fidelity,
    process_fidelity,
    average_gate_fidelity,
)
from quantumium.core.gate import H, X


def rb_demo():
    print("=" * 60)
    print("  Randomized Benchmarking — Single-Qubit Gate Errors")
    print("=" * 60)

    # Run RB with different error rates
    for err in [0.001, 0.01, 0.05]:
        result = randomized_benchmarking(
            sequence_lengths=[1, 2, 4, 8, 16, 32, 64, 128],
            num_samples=30,
            error_rate=err,
            seed=42,
        )
        print(f"\n  Error rate = {err:.3f}:")
        print(f"    Depolarizing parameter α = {result.alpha:.4f}")
        print(f"    Error per Clifford (EPC)  = {result.error_per_clifford:.5f}")
        print(f"    Fit: p(m) = {result.offset_A:.3f} × {result.alpha:.4f}^m + {result.offset_B:.3f}")
        print(f"    Survival at m=1:  {result.survival_probabilities[0]:.4f}")
        print(f"    Survival at m=64: {result.survival_probabilities[-2]:.4f}")


def qv_demo():
    print("\n" + "=" * 60)
    print("  Quantum Volume Estimation")
    print("=" * 60)

    for n in [2, 3, 4]:
        result = quantum_volume(
            num_qubits=n,
            num_circuits=50,
            error_rate=0.02,
            seed=42,
        )
        print(f"\n  n = {n} qubits:")
        print(f"    Quantum volume  = {result.quantum_volume}")
        print(f"    Heavy output prob = {result.heavy_output_prob:.4f} (threshold: 2/3)")
        print(f"    Passed: {'Yes' if result.passed else 'No'}")


def fidelity_demo():
    print("\n" + "=" * 60)
    print("  Fidelity Metrics")
    print("=" * 60)

    # State fidelity: compare |0> with |+>
    zero = np.array([1, 0], dtype=complex)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    one = np.array([0, 1], dtype=complex)

    f_zero_plus = state_fidelity(zero, plus)
    f_zero_zero = state_fidelity(zero, zero)
    f_zero_one = state_fidelity(zero, one)

    print(f"\n  State fidelity:")
    print(f"    F(|0>, |0>) = {f_zero_zero:.4f}  (identical)")
    print(f"    F(|0>, |+>) = {f_zero_plus:.4f}  (50% overlap)")
    print(f"    F(|0>, |1>) = {f_zero_one:.4f}  (orthogonal)")

    # Process fidelity: compare H gate with itself and with X
    f_HH = process_fidelity(H.matrix, H.matrix)
    f_HX = process_fidelity(H.matrix, X.matrix)

    print(f"\n  Process fidelity:")
    print(f"    F(H, H) = {f_HH:.4f}  (identical)")
    print(f"    F(H, X) = {f_HX:.4f}  (different gates)")

    # Average gate fidelity
    f_avg_HH = average_gate_fidelity(H.matrix, H.matrix)
    f_avg_HX = average_gate_fidelity(H.matrix, X.matrix)

    print(f"\n  Average gate fidelity:")
    print(f"    F_avg(H, H) = {f_avg_HH:.4f}")
    print(f"    F_avg(H, X) = {f_avg_HX:.4f}")


if __name__ == "__main__":
    rb_demo()
    qv_demo()
    fidelity_demo()
    print("\nDone!")
