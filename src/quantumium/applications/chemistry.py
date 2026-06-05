"""Quantum chemistry applications.

Provides molecular Hamiltonians and tools for computing ground state
energies and molecular properties using VQE and other algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..algorithms.vqe import vqe, h2_hamiltonian, hamiltonian_expectation, VQEResult


@dataclass
class MoleculeResult:
    """Result from a molecular simulation."""
    molecule: str
    ground_energy: float
    exact_energy: float
    error: float
    chemical_accuracy: bool
    vqe_result: VQEResult


def _build_hamiltonian(
    coefficients: dict[str, float],
    num_qubits: int,
) -> NDArray[np.complexfloating]:
    """Build a qubit Hamiltonian from Pauli string coefficients.

    Args:
        coefficients: Dict mapping Pauli strings to coefficients.
            e.g. {"II": -0.5, "ZI": 0.3, "IZ": -0.2, "ZZ": 0.1, "XX": 0.05}
        num_qubits: Number of qubits.
    """
    dim = 2**num_qubits
    H = np.zeros((dim, dim), dtype=complex)

    pauli_matrices = {
        'I': np.eye(2, dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    }

    for pauli_string, coeff in coefficients.items():
        if len(pauli_string) != num_qubits:
            raise ValueError(f"Pauli string '{pauli_string}' length doesn't match {num_qubits} qubits")
        term = np.array([[1]], dtype=complex)
        for char in pauli_string:
            term = np.kron(term, pauli_matrices[char])
        H += coeff * term

    return H


# Molecular Hamiltonian library (Jordan-Wigner transformed, STO-3G basis)

MOLECULAR_HAMILTONIANS = {
    "H2": {
        "description": "Hydrogen molecule at equilibrium (0.735 Å)",
        "num_qubits": 2,
        "coefficients": {
            "II": -0.8105,
            "ZI": +0.1721,
            "IZ": -0.2257,
            "ZZ": +0.1709,
            "XX": +0.0453,
            "YY": +0.0453,
        },
        "exact_energy": None,  # computed dynamically
        "bond_length": 0.735,
    },
    "LiH": {
        "description": "Lithium hydride at equilibrium (1.546 Å), frozen core",
        "num_qubits": 4,
        "coefficients": {
            "IIII": -7.4983,
            "ZIII": +0.3924,
            "IZII": -0.3924,
            "IIZI": -0.0112,
            "IIIZ": +0.0112,
            "ZZII": +0.1762,
            "IIZZ": +0.1210,
            "ZIZI": +0.1659,
            "IZIZ": +0.1205,
            "ZIIZ": +0.1224,
            "IZZI": +0.1659,
            "XXII": +0.0453,
            "YYII": +0.0453,
            "IIXX": +0.0145,
            "IIYY": +0.0145,
            "XZXI": +0.0116,
            "YZYI": +0.0116,
            "XIIX": +0.0116,
            "YIIY": +0.0116,
        },
        "exact_energy": None,
        "bond_length": 1.546,
    },
    "HeH+": {
        "description": "Helium hydride cation at equilibrium (0.775 Å)",
        "num_qubits": 2,
        "coefficients": {
            "II": -2.1433,
            "ZI": +0.3690,
            "IZ": -0.5819,
            "ZZ": +0.1981,
            "XX": +0.0513,
            "YY": +0.0513,
        },
        "exact_energy": None,
        "bond_length": 0.775,
    },
}


def get_molecular_hamiltonian(molecule: str) -> tuple[NDArray[np.complexfloating], int]:
    """Get the qubit Hamiltonian for a molecule.

    Returns:
        Tuple of (hamiltonian_matrix, num_qubits).
    """
    if molecule not in MOLECULAR_HAMILTONIANS:
        available = ", ".join(MOLECULAR_HAMILTONIANS.keys())
        raise ValueError(f"Unknown molecule '{molecule}'. Available: {available}")

    mol = MOLECULAR_HAMILTONIANS[molecule]
    H = _build_hamiltonian(mol["coefficients"], mol["num_qubits"])
    return H, mol["num_qubits"]


def simulate_molecule(
    molecule: str,
    ansatz_depth: int = 2,
    max_iter: int = 200,
    num_starts: int = 5,
) -> MoleculeResult:
    """Run VQE to find the ground state energy of a molecule.

    Args:
        molecule: Molecule name (e.g., "H2", "LiH", "HeH+").
        ansatz_depth: Number of ansatz layers.
        max_iter: Maximum VQE iterations.
        num_starts: Number of random restarts.
    """
    H, num_qubits = get_molecular_hamiltonian(molecule)
    exact = float(np.linalg.eigvalsh(H)[0])

    result = vqe(
        num_qubits=num_qubits,
        hamiltonian=H,
        ansatz_depth=ansatz_depth,
        max_iter=max_iter,
        num_starts=num_starts,
    )

    error = abs(result.optimal_energy - exact)
    chemical_accuracy = error < 0.0016  # 1.6 mHa = 1 kcal/mol

    return MoleculeResult(
        molecule=molecule,
        ground_energy=result.optimal_energy,
        exact_energy=exact,
        error=error,
        chemical_accuracy=chemical_accuracy,
        vqe_result=result,
    )


def potential_energy_surface(
    molecule_coeffs_fn,
    distances: list[float],
    num_qubits: int,
    ansatz_depth: int = 2,
    max_iter: int = 150,
    num_starts: int = 3,
) -> list[tuple[float, float, float]]:
    """Compute potential energy surface by scanning bond distances.

    Args:
        molecule_coeffs_fn: Function that takes a distance and returns
            a dict of Pauli string coefficients.
        distances: List of bond distances to scan.
        num_qubits: Number of qubits.
        ansatz_depth: Ansatz layers.
        max_iter: VQE iterations per point.
        num_starts: Random restarts per point.

    Returns:
        List of (distance, vqe_energy, exact_energy) tuples.
    """
    results = []
    for d in distances:
        coeffs = molecule_coeffs_fn(d)
        H = _build_hamiltonian(coeffs, num_qubits)
        exact = float(np.linalg.eigvalsh(H)[0])

        vqe_result = vqe(
            num_qubits=num_qubits,
            hamiltonian=H,
            ansatz_depth=ansatz_depth,
            max_iter=max_iter,
            num_starts=num_starts,
        )

        results.append((d, vqe_result.optimal_energy, exact))

    return results
