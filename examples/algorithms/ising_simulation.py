"""Hamiltonian Simulation — Transverse-Field Ising Model.

Simulates quantum spin dynamics using Trotter-Suzuki decomposition.
This is the original application Feynman envisioned for quantum computers.
"""

import numpy as np

from quantumium.algorithms.hamiltonian_sim import (
    exact_evolution,
    trotter_error,
    simulate_evolution,
    transverse_field_ising,
)


def main():
    print("=" * 60)
    print("Hamiltonian Simulation — Ising Model — Quantumium")
    print("=" * 60)

    # 3-qubit transverse-field Ising model
    n_qubits = 3
    J, h = 1.0, 0.5
    terms = transverse_field_ising(n_qubits, J=J, h=h)
    H_full = sum(terms)

    print(f"\nSystem: {n_qubits}-qubit transverse-field Ising model")
    print(f"Parameters: J={J}, h={h}")
    print(f"Hamiltonian terms: {len(terms)} ({n_qubits-1} ZZ + {n_qubits} X)")

    # Energy spectrum
    eigenvalues = np.sort(np.linalg.eigvalsh(H_full))
    print(f"\nEnergy spectrum: {np.round(eigenvalues, 4)}")
    print(f"Ground state energy: {eigenvalues[0]:.4f}")
    print(f"Energy gap: {eigenvalues[1] - eigenvalues[0]:.4f}")

    # Trotter error convergence
    t = 1.0
    print(f"\nTrotter error convergence (t={t}):")
    print(f"{'Steps':>8} {'1st order':>12} {'2nd order':>12}")
    print("-" * 35)
    for steps in [1, 2, 5, 10, 20, 50]:
        err1 = trotter_error(terms, t, steps, order=1)
        err2 = trotter_error(terms, t, steps, order=2)
        print(f"{steps:>8} {err1:>12.6f} {err2:>12.6f}")

    # Time evolution simulation
    print(f"\nTime evolution from |000⟩:")
    init = np.zeros(2**n_qubits, dtype=complex)
    init[0] = 1.0

    for t in [0.0, 0.5, 1.0, 2.0]:
        U = exact_evolution(H_full, t)
        state = U @ init
        probs = np.abs(state) ** 2
        top_states = sorted(range(len(probs)), key=lambda i: -probs[i])[:4]
        print(f"  t={t:.1f}: ", end="")
        for s in top_states:
            if probs[s] > 0.01:
                print(f"|{s:0{n_qubits}b}⟩:{probs[s]:.3f} ", end="")
        print()

    # Circuit-based simulation
    print(f"\nCircuit simulation (t=1.0, 20 Trotter steps, 2nd order):")
    qc = simulate_evolution(n_qubits, terms, time=1.0, steps=20, order=2)
    counts = qc.measure_all(shots=2000)
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:5]
    for bitstring, count in sorted_counts:
        print(f"  |{bitstring}⟩: {count} ({count/2000*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Simulation complete — quantum dynamics on a classical simulator.")


if __name__ == "__main__":
    main()
