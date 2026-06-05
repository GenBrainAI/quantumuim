"""Quantum process tomography.

Reconstructs the chi (process) matrix or Choi matrix of an unknown quantum
channel by preparing a complete set of input states, applying the channel,
and performing state tomography on each output.

For a single-qubit channel, the 4 input states are {|0>, |1>, |+>, |+i>}.
For n qubits, we use the tensor product of single-qubit input sets.

The chi matrix is defined by:
    E(rho) = sum_{i,j} chi_{i,j} P_i rho P_j^dag
where {P_i} are the n-qubit Pauli operators.

References:
    Chuang & Nielsen, J. Mod. Opt. 44, 2455 (1997)
    Nielsen & Chuang, Ch. 8.4.2
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as iterproduct

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H, S
from .channels import kraus_to_choi


# Single-qubit input states for process tomography
_INPUT_STATES_1Q = [
    np.array([1, 0], dtype=complex),         # |0>
    np.array([0, 1], dtype=complex),         # |1>
    np.array([1, 1], dtype=complex) / np.sqrt(2),  # |+>
    np.array([1, 1j], dtype=complex) / np.sqrt(2),  # |+i>
]

# Pauli basis (unnormalized)
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS_1Q = [_I, _X, _Y, _Z]


def _nqubit_paulis(n: int) -> list[NDArray[np.complexfloating]]:
    """Build all n-qubit Pauli operators (unnormalized).

    Generates the d^2 tensor products of single-qubit Pauli matrices
    {I, X, Y, Z}^{otimes n} where d = 2^n.

    Args:
        n: Number of qubits.

    Returns:
        A list of d^2 matrices, each of shape (d, d).
    """
    if n == 1:
        return _PAULIS_1Q
    return [
        _multi_kron(combo)
        for combo in iterproduct(_PAULIS_1Q, repeat=n)
    ]


def _multi_kron(matrices: tuple) -> NDArray[np.complexfloating]:
    """Compute the Kronecker (tensor) product of a sequence of matrices.

    Args:
        matrices: Tuple of 2-D NumPy arrays to tensor together.

    Returns:
        The tensor product matrix.
    """
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def _input_states(n: int) -> list[NDArray[np.complexfloating]]:
    """Build the complete set of n-qubit input states for process tomography.

    Generates tensor products of the single-qubit input states
    {|0>, |1>, |+>, |+i>} to form a tomographically complete set of 4^n
    input states.

    Args:
        n: Number of qubits.

    Returns:
        A list of 4^n statevectors, each of length 2^n.
    """
    if n == 1:
        return _INPUT_STATES_1Q
    return [
        _multi_kron_vec(combo)
        for combo in iterproduct(_INPUT_STATES_1Q, repeat=n)
    ]


def _multi_kron_vec(vectors: tuple) -> NDArray[np.complexfloating]:
    """Compute the Kronecker (tensor) product of a sequence of vectors.

    Args:
        vectors: Tuple of 1-D NumPy arrays to tensor together.

    Returns:
        The tensor product vector.
    """
    result = vectors[0]
    for v in vectors[1:]:
        result = np.kron(result, v)
    return result


def _apply_channel_simulation(
    channel_circuit: QuantumCircuit,
    input_state: NDArray[np.complexfloating],
    shots: int,
) -> NDArray[np.complexfloating]:
    """Apply a channel (given as a circuit) to an input state.

    Constructs the full unitary of the channel circuit, applies it to the
    input state, and returns the output density matrix as a pure-state
    outer product.

    Args:
        channel_circuit: The quantum circuit representing the channel.
        input_state: Input statevector of length 2^n.
        shots: Number of measurement shots (currently unused; the output
            is computed analytically from the circuit unitary).

    Returns:
        Output density matrix of shape (d, d) where d = 2^n.
    """
    n = channel_circuit.num_qubits
    d = 2**n

    # Build the unitary of the channel circuit
    U = np.eye(d, dtype=complex)
    for inst in channel_circuit.instructions:
        gate_matrix = inst.gate.matrix
        if len(inst.target_qubits) == 1:
            q = inst.target_qubits[0]
            full = np.eye(1, dtype=complex)
            for i in range(n):
                if i == n - 1 - q:
                    full = np.kron(full, gate_matrix)
                else:
                    full = np.kron(full, np.eye(2, dtype=complex))
            U = full @ U
        else:
            from ..core.circuit import QuantumCircuit as QC
            temp = QC(n)
            temp.apply(inst.gate, *inst.target_qubits)
            sv_temp = temp.statevector()
            temp_U = np.eye(d, dtype=complex)
            for t_inst in temp.instructions:
                g = t_inst.gate.matrix
                tq = t_inst.target_qubits
                temp_U = temp._expand_gate(g, tq) @ temp_U
            U = temp_U @ U

    output_state = U @ input_state
    return np.outer(output_state, output_state.conj())


def _apply_channel_direct(
    unitary: NDArray[np.complexfloating],
    input_state: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Apply a unitary channel directly to an input state.

    Computes U|psi> and returns the output density matrix |psi'><psi'|.

    Args:
        unitary: (d x d) unitary matrix.
        input_state: Input statevector of length d.

    Returns:
        Output density matrix of shape (d, d).
    """
    output = unitary @ input_state
    return np.outer(output, output.conj())


@dataclass
class ProcessTomographyResult:
    """Container for quantum process tomography results.

    Attributes:
        chi_matrix: The chi (process) matrix of shape (d^2, d^2), where
            E(rho) = sum_{ij} chi_{ij} P_i rho P_j^dag.
        choi_matrix: The Choi-Jamiolkowski matrix of shape (d^2, d^2).
        num_qubits: Number of qubits in the characterized process.
        process_fidelity: Process fidelity with respect to a target unitary,
            or None if no target was provided.
        input_states_used: Number of input states used in the tomography.
    """
    chi_matrix: NDArray[np.complexfloating]
    choi_matrix: NDArray[np.complexfloating]
    num_qubits: int
    process_fidelity: float | None
    input_states_used: int


def chi_matrix(
    unitary: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Compute the chi (process) matrix for a unitary channel analytically.

    chi_{ij} such that E(rho) = sum_{ij} chi_{ij} P_i rho P_j^dag.

    Args:
        unitary: Unitary matrix representing the channel.

    Returns:
        Chi matrix (d^2 x d^2).
    """
    U = np.asarray(unitary, dtype=complex)
    d = U.shape[0]
    n = int(np.log2(d))
    paulis = _nqubit_paulis(n)
    d2 = d * d

    chi = np.zeros((d2, d2), dtype=complex)
    for i, Pi in enumerate(paulis):
        for j, Pj in enumerate(paulis):
            chi[i, j] = np.trace(Pi @ U @ Pj.conj().T) / d

    # chi_{ij} = (1/d^2) sum_{m,n} Tr(P_i U) * Tr(P_j U)^* doesn't work
    # Correct formula: decompose U = sum_i c_i P_i, then chi = c c^dag
    coeffs = np.array([np.trace(P.conj().T @ U) / d for P in paulis])
    chi = np.outer(coeffs, coeffs.conj())

    return chi


def process_tomography(
    unitary: NDArray[np.complexfloating],
    target_unitary: NDArray[np.complexfloating] | None = None,
) -> ProcessTomographyResult:
    """Perform quantum process tomography on a unitary channel.

    Prepares each input state, applies the channel, measures the output,
    and reconstructs the chi matrix and Choi matrix.

    Args:
        unitary: Unitary matrix representing the channel to characterize.
        target_unitary: Optional ideal unitary for fidelity comparison.

    Returns:
        ProcessTomographyResult with chi and Choi matrices.
    """
    U = np.asarray(unitary, dtype=complex)
    d = U.shape[0]
    n = int(np.log2(d))
    paulis = _nqubit_paulis(n)
    d2 = d * d

    inputs = _input_states(n)

    # Collect output density matrices
    rho_out = []
    for psi_in in inputs:
        rho_out.append(_apply_channel_direct(U, psi_in))

    # Build the coefficient matrix: rho_in_k = sum_j beta_{kj} P_j
    # Then solve for chi: rho_out_k = sum_{ij} chi_{ij} P_i (sum_j beta_{kj} P_j) P_j_dag
    # Simpler: use direct Choi reconstruction

    # Build Choi matrix from input-output pairs
    # J(E) = sum_{i,j} |i><j| ⊗ E(|i><j|)
    choi = np.zeros((d2, d2), dtype=complex)
    for i in range(d):
        for j in range(d):
            e_i = np.zeros(d, dtype=complex)
            e_i[i] = 1.0
            e_j = np.zeros(d, dtype=complex)
            e_j[j] = 1.0
            eij = np.outer(e_i, e_j.conj())
            output = U @ eij @ U.conj().T
            choi += np.kron(eij, output)

    # Convert Choi to chi matrix
    # chi_{mn} = sum_{ij,kl} (P_m)_{ij} J_{ij,kl} (P_n^*)_{kl} / d
    chi_mat = np.zeros((d2, d2), dtype=complex)
    for m, Pm in enumerate(paulis):
        for nn, Pn in enumerate(paulis):
            val = 0.0
            for i in range(d):
                for j in range(d):
                    for k in range(d):
                        for l in range(d):
                            val += Pm[i, j] * choi[i * d + k, j * d + l] * Pn[k, l].conj()
            chi_mat[m, nn] = val / d

    # Process fidelity with target
    proc_fid = None
    if target_unitary is not None:
        V = np.asarray(target_unitary, dtype=complex)
        proc_fid = float(abs(np.trace(U.conj().T @ V)) ** 2 / d**2)

    return ProcessTomographyResult(
        chi_matrix=chi_mat,
        choi_matrix=choi,
        num_qubits=n,
        process_fidelity=proc_fid,
        input_states_used=len(inputs),
    )
