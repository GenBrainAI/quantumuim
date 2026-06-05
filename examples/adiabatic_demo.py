"""Adiabatic quantum computing demo — solving MaxCut via quantum annealing.

Demonstrates:
  1. Building a MaxCut problem Hamiltonian
  2. Adiabatic evolution from transverse field to problem Hamiltonian
  3. Spectral gap analysis along the anneal path
  4. Comparing linear vs polynomial schedules
"""
from __future__ import annotations

import numpy as np

from quantumium.adiabatic import (
    adiabatic_evolve,
    spectral_gap,
    linear_schedule,
    polynomial_schedule,
    transverse_field_hamiltonian,
    maxcut_hamiltonian,
    ising_hamiltonian,
)


def maxcut_demo():
    print("=" * 60)
    print("  Adiabatic Quantum Computing — MaxCut on a 4-node graph")
    print("=" * 60)

    # Define a 4-node graph: a square with one diagonal
    #   0 — 1
    #   |   |
    #   3 — 2
    #   0 — 2 (diagonal)
    num_qubits = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

    H_init = transverse_field_hamiltonian(num_qubits)
    H_final = maxcut_hamiltonian(num_qubits, edges)

    # Analyze the spectral gap
    s_vals, gaps = spectral_gap(H_init, H_final, num_points=50)
    min_gap = float(np.min(gaps))
    min_gap_s = float(s_vals[np.argmin(gaps)])
    print(f"\nSpectral gap analysis:")
    print(f"  Minimum gap: {min_gap:.4f} at s = {min_gap_s:.3f}")
    if min_gap > 1e-10:
        print(f"  Rule of thumb: T >> 1/gap^2 = {1/min_gap**2:.1f}")
    else:
        print(f"  Gap closes — degenerate ground state (common for MaxCut)")

    # Run with linear schedule
    print(f"\n--- Linear schedule (T=20, 1000 steps) ---")
    result_lin = adiabatic_evolve(H_init, H_final, total_time=20.0, num_steps=1000)
    print(f"  Ground state probability: {result_lin.ground_state_prob:.4f}")
    print(f"  Optimal bitstring: {result_lin.optimal_bitstring}")

    # Interpret the bitstring as a partition
    bs = result_lin.optimal_bitstring
    set_0 = [i for i, b in enumerate(bs) if b == "0"]
    set_1 = [i for i, b in enumerate(bs) if b == "1"]
    cut_value = sum(1 for u, v in edges if bs[u] != bs[v])
    print(f"  Partition: {set_0} | {set_1}")
    print(f"  Cut value: {cut_value} / {len(edges)} edges")

    # Run with polynomial schedule (slower start, faster end)
    print(f"\n--- Polynomial schedule (T=20, power=3) ---")
    schedule_poly = lambda t, T: polynomial_schedule(t, T, power=3.0)
    result_poly = adiabatic_evolve(
        H_init, H_final, total_time=20.0, num_steps=1000, schedule=schedule_poly
    )
    print(f"  Ground state probability: {result_poly.ground_state_prob:.4f}")
    print(f"  Optimal bitstring: {result_poly.optimal_bitstring}")

    # Compare anneal times
    print(f"\n--- Effect of anneal time (linear schedule) ---")
    for T in [1, 5, 10, 20, 50]:
        r = adiabatic_evolve(H_init, H_final, total_time=T, num_steps=max(100, T * 50))
        print(f"  T={T:3d}: ground state prob = {r.ground_state_prob:.4f}, "
              f"bitstring = {r.optimal_bitstring}")


def ising_demo():
    print("\n" + "=" * 60)
    print("  Ising Model — Finding the ground state")
    print("=" * 60)

    num_qubits = 3
    J = np.array([
        [0, 1.0, 0.5],
        [0, 0,   1.0],
        [0, 0,   0],
    ])
    h = np.array([0.2, -0.3, 0.1])

    H_init = transverse_field_hamiltonian(num_qubits)
    H_final = ising_hamiltonian(num_qubits, J=J, h=h)

    result = adiabatic_evolve(H_init, H_final, total_time=30.0, num_steps=1500)
    print(f"\n  Ground state probability: {result.ground_state_prob:.4f}")
    print(f"  Optimal spin config: {result.optimal_bitstring}")
    print(f"  (0 = spin up, 1 = spin down)")

    # Verify by brute force
    eigenvalues = np.linalg.eigvalsh(H_final)
    print(f"  Exact ground state energy: {eigenvalues[0]:.4f}")
    print(f"  Adiabatic final energy:    {result.energy_history[-1]:.4f}")


if __name__ == "__main__":
    maxcut_demo()
    ising_demo()
    print("\nDone!")
