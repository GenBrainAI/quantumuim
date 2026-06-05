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
    """Trace out qubits not in the keep list from a density matrix.

    Computes the reduced density matrix by performing a partial trace
    over all qubits not specified in `keep`.

    Args:
        rho (NDArray[np.complexfloating]): Density matrix of the full
            system, of shape (2^n, 2^n) where n = num_qubits.
        keep (list[int]): List of qubit indices to retain (0-indexed).
            The remaining qubits are traced out.
        num_qubits (int): Total number of qubits in the system.

    Returns:
        NDArray[np.complexfloating]: Reduced density matrix of the
            kept subsystem, of shape (2^k, 2^k) where k = len(keep).

    Raises:
        ValueError: If the density matrix size does not match num_qubits.
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
    """Compute the von Neumann entropy of a density matrix.

    Defined as S(rho) = -Tr(rho * log2(rho)). Returns 0 for pure
    states and log2(d) for the maximally mixed state of dimension d.

    Args:
        rho (NDArray[np.complexfloating]): Density matrix (d x d).

    Returns:
        float: Entropy in bits (base-2 logarithm).
    """
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    return float(-np.sum(eigvals * np.log2(eigvals)))


def purity(rho: NDArray[np.complexfloating]) -> float:
    """Compute the purity of a density matrix.

    Purity is defined as Tr(rho^2). It equals 1 for pure states and
    1/d for the maximally mixed state of dimension d.

    Args:
        rho (NDArray[np.complexfloating]): Density matrix (d x d).

    Returns:
        float: Purity value in [1/d, 1].
    """
    return float(np.real(np.trace(rho @ rho)))


def concurrence(rho: NDArray[np.complexfloating]) -> float:
    """Compute the concurrence of a two-qubit density matrix.

    The concurrence C(rho) = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4)
    where lambda_i are the square roots of eigenvalues of rho * tilde(rho)
    in decreasing order, and tilde(rho) = (Y x Y) rho* (Y x Y).

    Concurrence ranges from 0 (separable) to 1 (maximally entangled).

    Args:
        rho (NDArray[np.complexfloating]): Two-qubit density matrix
            of shape (4, 4).

    Returns:
        float: Concurrence value in [0, 1].

    Raises:
        ValueError: If rho is not a 4x4 matrix.
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

    Negativity N(rho) = (||rho^{T_A}||_1 - 1) / 2 where rho^{T_A} is
    the partial transpose with respect to subsystem A. It is zero for
    separable states and positive for entangled states (for 2x2 and
    2x3 systems, positivity is also sufficient for entanglement).

    Args:
        rho (NDArray[np.complexfloating]): Full density matrix of shape
            (2^n, 2^n) where n = num_qubits.
        subsystem_a (list[int]): Qubit indices forming subsystem A
            (0-indexed). The partial transpose is taken over this
            subsystem.
        num_qubits (int): Total number of qubits in the system.

    Returns:
        float: Negativity value (non-negative; 0 for separable states).
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

    For a pure bipartite state, this is the von Neumann entropy of the
    reduced density matrix and serves as a true entanglement measure.
    For mixed states, it gives the entropy of the reduced state, which
    is not a proper entanglement measure but is still diagnostically
    useful.

    Args:
        rho (NDArray[np.complexfloating]): Full system density matrix
            of shape (2^n, 2^n) where n = num_qubits.
        subsystem (list[int]): Qubit indices of subsystem A (0-indexed).
            The von Neumann entropy is computed for this reduced subsystem.
        num_qubits (int): Total number of qubits in the system.

    Returns:
        float: Entanglement entropy in bits (base-2 logarithm).
    """
    rho_reduced = partial_trace(rho, subsystem, num_qubits)
    return von_neumann_entropy(rho_reduced)


def schmidt_decomposition(
    statevector: NDArray[np.complexfloating],
    subsystem_a: list[int],
    num_qubits: int,
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Compute the Schmidt decomposition of a bipartite pure state.

    Decomposes |psi> = sum_i lambda_i |u_i> |v_i> where lambda_i are
    the Schmidt coefficients and |u_i>, |v_i> are orthonormal bases
    for subsystems A and B respectively. Only terms with non-negligible
    coefficients (> 1e-15) are returned.

    Args:
        statevector (NDArray[np.complexfloating]): State vector of the
            full system, of length 2^num_qubits.
        subsystem_a (list[int]): Qubit indices forming subsystem A
            (0-indexed). Subsystem B consists of all remaining qubits.
        num_qubits (int): Total number of qubits in the system.

    Returns:
        tuple[NDArray[np.floating], NDArray[np.complexfloating],
            NDArray[np.complexfloating]]: A tuple (coefficients,
            u_vectors, v_vectors) where:
            - coefficients: Schmidt coefficients (positive, descending),
              array of shape (r,) where r is the Schmidt rank.
            - u_vectors: Orthonormal basis vectors for subsystem A,
              array of shape (2^n_a, r).
            - v_vectors: Orthonormal basis vectors for subsystem B,
              array of shape (r, 2^n_b).
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
