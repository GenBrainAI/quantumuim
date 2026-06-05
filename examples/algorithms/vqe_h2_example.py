"""VQE for H2 molecule — finding ground state energy with a quantum computer.

Demonstrates the Variational Quantum Eigensolver on the simplest molecular
system: the hydrogen molecule in STO-3G basis.
"""

import numpy as np

from quantumium.algorithms.vqe import vqe, h2_hamiltonian


def main():
    print("=" * 60)
    print("VQE for H2 Molecule — Quantumium")
    print("=" * 60)

    H = h2_hamiltonian()

    # Exact solution via classical diagonalization
    eigenvalues = np.linalg.eigvalsh(H)
    exact_ground = eigenvalues[0]
    print(f"\nExact ground state energy: {exact_ground:.6f} Ha")
    print(f"All eigenvalues: {np.round(eigenvalues, 4)}")

    # VQE optimization
    print("\nRunning VQE (2 qubits, depth=2, 5 random starts)...")
    result = vqe(
        num_qubits=2,
        hamiltonian=H,
        ansatz_depth=2,
        max_iter=200,
        learning_rate=0.05,
        num_starts=5,
    )

    print(f"\nVQE ground state energy: {result.optimal_energy:.6f} Ha")
    print(f"Error from exact: {abs(result.optimal_energy - exact_ground):.6f} Ha")
    print(f"Chemical accuracy (1.6 mHa): {'YES' if abs(result.optimal_energy - exact_ground) < 0.0016 else 'NO'}")
    print(f"Optimization iterations: {result.num_iterations}")

    # Energy convergence
    if len(result.history) > 10:
        print(f"\nConvergence (first 5 → last 5):")
        print(f"  Start: {np.mean(result.history[:5]):.4f} Ha")
        print(f"  End:   {np.mean(result.history[-5:]):.4f} Ha")

    print("\n" + "=" * 60)
    print("VQE demonstration complete.")
    print("This is the foundation for quantum chemistry on real hardware.")


if __name__ == "__main__":
    main()
