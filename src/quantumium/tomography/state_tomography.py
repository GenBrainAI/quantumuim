"""Quantum state tomography for single and multi-qubit systems.

Reconstructs the density matrix from measurements in multiple Pauli bases.
Implements both linear inversion and maximum likelihood estimation (MLE).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as iterproduct

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H, X, Ry
from ..core.noise import NoiseModel


I2 = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = {"I": I2, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}

S_DAG = Gate(np.array([[1, 0], [0, -1j]], dtype=complex), "Sdg")


@dataclass
class TomographyResult:
    """Result from quantum state tomography."""
    density_matrix: NDArray[np.complexfloating]
    fidelity: float | None
    num_qubits: int
    method: str
    measurements: dict[str, dict[str, int]]


def _measurement_bases(num_qubits: int) -> list[str]:
    """Generate all Pauli measurement bases for n qubits (excluding I)."""
    return ["".join(combo) for combo in iterproduct("XYZ", repeat=num_qubits)]


def _add_basis_rotation(
    qc: QuantumCircuit,
    basis: str,
) -> QuantumCircuit:
    """Add gates to rotate from a Pauli basis to the Z (computational) basis."""
    for i, b in enumerate(basis):
        if b == "X":
            qc.apply(H, i)
        elif b == "Y":
            qc.apply(S_DAG, i)
            qc.apply(H, i)
    return qc


def collect_tomography_data(
    circuit: QuantumCircuit,
    shots_per_basis: int = 4096,
    noise_model: NoiseModel | None = None,
) -> dict[str, dict[str, int]]:
    """Run a circuit in all Pauli bases and collect measurement counts."""
    n = circuit.num_qubits
    bases = _measurement_bases(n)
    results: dict[str, dict[str, int]] = {}

    for basis in bases:
        qc = QuantumCircuit(n)
        for inst in circuit.instructions:
            qc.apply(inst.gate, *inst.target_qubits)
        _add_basis_rotation(qc, basis)

        if noise_model is not None:
            counts = qc.noisy_measure(noise_model, shots=shots_per_basis)
        else:
            counts = qc.measure_all(shots=shots_per_basis)

        results[basis] = counts

    return results


def _expectation_from_counts(
    counts: dict[str, int],
    qubit_indices: list[int],
) -> float:
    """Compute expectation value of Z^⊗n from measurement counts."""
    total = sum(counts.values())
    expectation = 0.0
    for bitstring, count in counts.items():
        parity = sum(int(bitstring[i]) for i in qubit_indices) % 2
        expectation += (1 - 2 * parity) * count / total
    return expectation


def linear_inversion(
    measurements: dict[str, dict[str, int]],
    num_qubits: int,
) -> NDArray[np.complexfloating]:
    """Reconstruct density matrix via linear inversion.

    Uses the relation: rho = sum_P <P> * P / 2^n
    where the sum is over all n-qubit Pauli operators.
    """
    dim = 2**num_qubits
    rho = np.zeros((dim, dim), dtype=complex)

    pauli_labels = ["I", "X", "Y", "Z"]
    for combo in iterproduct(pauli_labels, repeat=num_qubits):
        label = "".join(combo)

        if all(c == "I" for c in label):
            pauli_matrix = np.eye(dim, dtype=complex)
            expectation = 1.0
        else:
            pauli_matrix = np.array([[1]], dtype=complex)
            for c in label:
                pauli_matrix = np.kron(pauli_matrix, PAULIS[c])

            measure_basis = label.replace("I", "Z")
            if measure_basis in measurements:
                non_i_indices = [i for i, c in enumerate(label) if c != "I"]
                expectation = _expectation_from_counts(
                    measurements[measure_basis], non_i_indices
                )
            else:
                expectation = 0.0

        rho += expectation * pauli_matrix

    rho /= dim
    return rho


def _project_to_physical(rho: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Project a matrix to the closest valid density matrix (positive semidefinite, trace 1)."""
    rho = (rho + rho.conj().T) / 2

    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)

    if eigvals.sum() > 0:
        eigvals /= eigvals.sum()

    return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T


def maximum_likelihood(
    measurements: dict[str, dict[str, int]],
    num_qubits: int,
    max_iter: int = 100,
) -> NDArray[np.complexfloating]:
    """Reconstruct density matrix via iterative maximum likelihood estimation.

    Uses the R*rho*R algorithm (diluted MLE) which guarantees physicality.
    """
    dim = 2**num_qubits
    rho = np.eye(dim, dtype=complex) / dim

    pauli_bases = _measurement_bases(num_qubits)
    basis_operators: list[list[NDArray[np.complexfloating]]] = []
    basis_frequencies: list[list[float]] = []

    for basis in pauli_bases:
        if basis not in measurements:
            continue
        counts = measurements[basis]
        total = sum(counts.values())

        rotation = np.array([[1]], dtype=complex)
        for b in basis:
            if b == "X":
                rotation = np.kron(rotation, (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex))
            elif b == "Y":
                rotation = np.kron(rotation, (1 / np.sqrt(2)) * np.array([[1, -1j], [1, 1j]], dtype=complex))
            elif b == "Z":
                rotation = np.kron(rotation, np.eye(2, dtype=complex))

        ops = []
        freqs = []
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            proj = np.zeros((dim, dim), dtype=complex)
            basis_vec = rotation[:, idx]
            proj = np.outer(basis_vec, basis_vec.conj())
            ops.append(proj)
            freqs.append(count / total)

        basis_operators.append(ops)
        basis_frequencies.append(freqs)

    for _ in range(max_iter):
        R = np.zeros((dim, dim), dtype=complex)
        for ops, freqs in zip(basis_operators, basis_frequencies):
            for proj, freq in zip(ops, freqs):
                p_measured = max(float(np.real(np.trace(proj @ rho))), 1e-15)
                R += (freq / p_measured) * proj

        R /= len(basis_operators) if basis_operators else 1
        rho_new = R @ rho @ R
        rho_new = (rho_new + rho_new.conj().T) / 2
        trace = np.trace(rho_new)
        if abs(trace) > 1e-15:
            rho_new /= trace
        rho = rho_new

    return rho


def state_fidelity(
    rho: NDArray[np.complexfloating],
    sigma: NDArray[np.complexfloating],
) -> float:
    """Compute fidelity between two density matrices: F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    eigvals, eigvecs = np.linalg.eigh(rho)
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
    sqrt_rho = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.conj().T

    product = sqrt_rho @ sigma @ sqrt_rho
    eigvals_prod = np.linalg.eigvalsh(product)
    return float(np.sum(np.sqrt(np.maximum(eigvals_prod, 0)))) ** 2


def tomography(
    circuit: QuantumCircuit,
    method: str = "linear",
    shots_per_basis: int = 4096,
    noise_model: NoiseModel | None = None,
    mle_iterations: int = 100,
) -> TomographyResult:
    """Perform full quantum state tomography on a circuit.

    Args:
        circuit: The quantum circuit to characterize.
        method: "linear" for linear inversion, "mle" for maximum likelihood.
        shots_per_basis: Number of measurement shots per Pauli basis.
        noise_model: Optional noise model for simulation.
        mle_iterations: Number of MLE iterations (only for method="mle").

    Returns:
        TomographyResult with reconstructed density matrix and fidelity.
    """
    n = circuit.num_qubits
    measurements = collect_tomography_data(circuit, shots_per_basis, noise_model)

    if method == "linear":
        rho_raw = linear_inversion(measurements, n)
        rho = _project_to_physical(rho_raw)
    elif method == "mle":
        rho = maximum_likelihood(measurements, n, max_iter=mle_iterations)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'linear' or 'mle'.")

    rho_ideal = circuit.density_matrix()
    fid = state_fidelity(rho, rho_ideal)

    return TomographyResult(
        density_matrix=rho,
        fidelity=fid,
        num_qubits=n,
        method=method,
        measurements=measurements,
    )
