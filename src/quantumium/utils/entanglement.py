"""Entanglement measures and quantum information utilities.

Provides partial trace, von Neumann entropy, concurrence, negativity,
and purity for analyzing quantum states.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def partial_trace(
    rho: NDArray[np.complexfloating],
    keep: list[int],
    num_qubits: int,
) -> NDArray[np.complexfloating]:
    """Trace out qubits not in `keep` from a density matrix.

    Args:
        rho: Density matrix of the full system (2^n x 2^n).
        keep: List of qubit indices to keep (0-indexed).
        num_qubits: Total number of qubits.

    Returns:
        Reduced density matrix of the kept subsystem.
    """
    if rho.shape[0] != 2**num_qubits:
        raise ValueError(f"Density matrix size {rho.shape[0]} doesn't match {num_qubits} qubits")

    dims = [2] * num_qubits
    rho_tensor = rho.reshape(dims + dims)

    # Qubit q maps to tensor axis (num_qubits - 1 - q) due to LSB=qubit0 convention
    trace_out_qubits = sorted(set(range(num_qubits)) - set(keep))
    trace_out_axes = sorted([num_qubits - 1 - q for q in trace_out_qubits])

    for offset, ax in enumerate(trace_out_axes):
        ax_row = ax - offset
        ax_col = ax_row + (num_qubits - offset)
        rho_tensor = np.trace(rho_tensor, axis1=ax_row, axis2=ax_col)

    n_keep = len(keep)
    return rho_tensor.reshape(2**n_keep, 2**n_keep)


def von_neumann_entropy(rho: NDArray[np.complexfloating]) -> float:
    """Compute the von Neumann entropy S(rho) = -Tr(rho log2 rho).

    Returns 0 for pure states, log2(d) for maximally mixed states.
    """
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    return float(-np.sum(eigvals * np.log2(eigvals)))


def purity(rho: NDArray[np.complexfloating]) -> float:
    """Compute the purity Tr(rho^2). Returns 1 for pure states, 1/d for maximally mixed."""
    return float(np.real(np.trace(rho @ rho)))


def concurrence(rho: NDArray[np.complexfloating]) -> float:
    """Compute the concurrence of a two-qubit density matrix.

    C(rho) = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4)
    where lambda_i are the square roots of eigenvalues of rho * tilde(rho)
    in decreasing order, and tilde(rho) = (Y⊗Y) rho* (Y⊗Y).
    """
    if rho.shape != (4, 4):
        raise ValueError("Concurrence is defined for two-qubit states (4x4 density matrix)")

    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    YY = np.kron(Y, Y)

    rho_tilde = YY @ rho.conj() @ YY
    product = rho @ rho_tilde

    eigvals = np.linalg.eigvals(product)
    eigvals = np.real(eigvals)
    eigvals = np.sqrt(np.maximum(eigvals, 0))
    eigvals = np.sort(eigvals)[::-1]

    return float(max(0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3]))


def negativity(
    rho: NDArray[np.complexfloating],
    subsystem_a: list[int],
    num_qubits: int,
) -> float:
    """Compute the negativity of a bipartite state under partial transpose.

    N(rho) = (||rho^{T_A}||_1 - 1) / 2
    where rho^{T_A} is the partial transpose with respect to subsystem A.

    Args:
        rho: Full density matrix.
        subsystem_a: Qubit indices forming subsystem A.
        num_qubits: Total number of qubits.
    """
    dims = [2] * num_qubits
    rho_tensor = rho.reshape(dims + dims)

    perm = list(range(2 * num_qubits))
    for q in subsystem_a:
        ax = num_qubits - 1 - q
        perm[ax], perm[ax + num_qubits] = perm[ax + num_qubits], perm[ax]

    rho_pt = rho_tensor.transpose(perm).reshape(rho.shape)

    eigvals = np.linalg.eigvalsh(rho_pt)
    trace_norm = np.sum(np.abs(eigvals))
    return float((trace_norm - 1) / 2)


def entanglement_entropy(
    rho: NDArray[np.complexfloating],
    subsystem: list[int],
    num_qubits: int,
) -> float:
    """Compute the entanglement entropy of a subsystem.

    For a pure state, this is the von Neumann entropy of the reduced density matrix.
    For mixed states, this gives the entropy of the reduced state (not a true
    entanglement measure, but still useful).

    Args:
        rho: Full system density matrix.
        subsystem: Qubit indices of subsystem A.
        num_qubits: Total number of qubits.
    """
    rho_reduced = partial_trace(rho, subsystem, num_qubits)
    return von_neumann_entropy(rho_reduced)


def schmidt_decomposition(
    statevector: NDArray[np.complexfloating],
    subsystem_a: list[int],
    num_qubits: int,
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Compute the Schmidt decomposition of a bipartite pure state.

    Args:
        statevector: State vector of the full system.
        subsystem_a: Qubit indices forming subsystem A.
        num_qubits: Total number of qubits.

    Returns:
        (coefficients, u_vectors, v_vectors) where
        |psi> = sum_i coefficients[i] * |u_i> |v_i>
    """
    subsystem_b = sorted(set(range(num_qubits)) - set(subsystem_a))
    n_a = len(subsystem_a)
    n_b = len(subsystem_b)

    psi = statevector.reshape([2] * num_qubits)
    axes_a = sorted([num_qubits - 1 - q for q in subsystem_a])
    axes_b = sorted([num_qubits - 1 - q for q in subsystem_b])
    psi = psi.transpose(axes_a + axes_b).reshape(2**n_a, 2**n_b)

    U, s, Vh = np.linalg.svd(psi, full_matrices=False)
    mask = s > 1e-15
    return s[mask], U[:, mask], Vh[mask, :]
