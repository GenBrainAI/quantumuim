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
    """Result from a molecular ground-state energy simulation.

    Encapsulates both the VQE-computed energy and the exact
    diagonalization reference, along with accuracy metrics.

    Attributes:
        molecule: Name of the simulated molecule (e.g., ``"H2"``).
        ground_energy: Ground-state energy found by VQE (Hartree).
        exact_energy: Exact ground-state energy from diagonalization (Hartree).
        error: Absolute difference between VQE and exact energies.
        chemical_accuracy: ``True`` if the error is below 1.6 mHa
            (1 kcal/mol), the standard chemical accuracy threshold.
        vqe_result: Full :class:`VQEResult` object from the underlying
            VQE run, containing optimal parameters and convergence history.
    """
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
    """Build a qubit Hamiltonian matrix from Pauli string coefficients.

    Constructs the full 2^n x 2^n Hamiltonian matrix as a sum of
    tensor-product Pauli terms weighted by the given coefficients.

    Args:
        coefficients: Dict mapping Pauli strings to their real-valued
            coefficients. Each key is a string of ``'I'``, ``'X'``, ``'Y'``,
            ``'Z'`` characters with length equal to *num_qubits*.
            Example: ``{"II": -0.5, "ZI": 0.3, "XX": 0.05}``.
        num_qubits: Number of qubits in the system.

    Returns:
        NDArray[np.complexfloating]: The Hamiltonian as a dense complex
            matrix of shape ``(2**num_qubits, 2**num_qubits)``.

    Raises:
        ValueError: If any Pauli string length does not match *num_qubits*.
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
    """Get the qubit Hamiltonian for a known molecule.

    Looks up pre-computed Jordan-Wigner-transformed Pauli coefficients
    (STO-3G basis) and builds the full Hamiltonian matrix.

    Args:
        molecule: Molecule identifier. Must be one of the keys in
            :data:`MOLECULAR_HAMILTONIANS` (e.g., ``"H2"``, ``"LiH"``,
            ``"HeH+"``).

    Returns:
        tuple[NDArray[np.complexfloating], int]: A 2-tuple of
            ``(hamiltonian_matrix, num_qubits)`` where the matrix has
            shape ``(2**num_qubits, 2**num_qubits)``.

    Raises:
        ValueError: If *molecule* is not in the library.
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
    """Run VQE to find the ground-state energy of a molecule.

    Retrieves the pre-built qubit Hamiltonian for the requested molecule,
    computes the exact ground-state energy via diagonalization, and then
    runs VQE to find the variational minimum. The result includes an
    accuracy check against chemical accuracy (1.6 mHa).

    Args:
        molecule: Molecule name. Must be a key in
            :data:`MOLECULAR_HAMILTONIANS` (e.g., ``"H2"``, ``"LiH"``,
            ``"HeH+"``).
        ansatz_depth: Number of layers in the hardware-efficient ansatz.
        max_iter: Maximum number of VQE optimizer iterations per start.
        num_starts: Number of random parameter initializations to try.

    Returns:
        MoleculeResult: Contains the VQE energy, exact energy, absolute
            error, chemical-accuracy flag, and the full VQE result object.

    Raises:
        ValueError: If *molecule* is not in the library.
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
    molecule_coeffs_fn: "Callable[[float], dict[str, float]]",
    distances: list[float],
    num_qubits: int,
    ansatz_depth: int = 2,
    max_iter: int = 150,
    num_starts: int = 3,
) -> list[tuple[float, float, float]]:
    """Compute a potential energy surface by scanning bond distances.

    For each bond distance, generates the molecular Hamiltonian via the
    user-supplied coefficient function, computes the exact ground-state
    energy, and runs VQE to obtain the variational estimate.

    Args:
        molecule_coeffs_fn: Callable that accepts a bond distance (float)
            and returns a ``dict[str, float]`` mapping Pauli strings to
            their coefficients at that geometry.
        distances: Bond distances (in Angstroms) to evaluate.
        num_qubits: Number of qubits in the Hamiltonian.
        ansatz_depth: Number of variational ansatz layers.
        max_iter: Maximum VQE optimizer iterations per distance point.
        num_starts: Number of random restarts per distance point.

    Returns:
        list[tuple[float, float, float]]: A list of
            ``(distance, vqe_energy, exact_energy)`` tuples, one per
            entry in *distances*.
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
