"""Problem Hamiltonians for adiabatic quantum computing.

Provides standard initial and final Hamiltonians for adiabatic
algorithms: transverse field (initial), Ising model, and MaxCut.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def transverse_field_hamiltonian(
    num_qubits: int,
) -> NDArray[np.complexfloating]:
    """Transverse field Hamiltonian H_init = -Σ X_i.

    The ground state is |+...+>, the uniform superposition.
    Used as the initial Hamiltonian in adiabatic evolution.
    """
    dim = 2**num_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for q in range(num_qubits):
        H -= _embed_op(SX, q, num_qubits)
    return H


def ising_hamiltonian(
    num_qubits: int,
    J: NDArray[np.floating] | None = None,
    h: NDArray[np.floating] | None = None,
) -> NDArray[np.complexfloating]:
    """Ising model Hamiltonian H = -Σ J_{ij} Z_i Z_j - Σ h_i Z_i.

    Args:
        num_qubits: Number of qubits.
        J: Coupling matrix (num_qubits × num_qubits). Only upper triangle used.
        h: Local field strengths (num_qubits,).

    Returns:
        Ising Hamiltonian as a matrix.
    """
    dim = 2**num_qubits
    H = np.zeros((dim, dim), dtype=complex)

    if J is not None:
        J = np.asarray(J, dtype=float)
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                if abs(J[i, j]) > 1e-15:
                    ZiZj = _embed_op(SZ, i, num_qubits) @ _embed_op(SZ, j, num_qubits)
                    H -= J[i, j] * ZiZj

    if h is not None:
        h = np.asarray(h, dtype=float)
        for i in range(num_qubits):
            if abs(h[i]) > 1e-15:
                H -= h[i] * _embed_op(SZ, i, num_qubits)

    return H


def maxcut_hamiltonian(
    num_qubits: int,
    edges: list[tuple[int, int]],
    weights: list[float] | None = None,
) -> NDArray[np.complexfloating]:
    """MaxCut problem Hamiltonian.

    H = Σ_{(i,j)∈E} w_{ij} (I - Z_i Z_j) / 2

    The ground state minimizes H, which corresponds to maximizing
    the cut value. Ground state energy = -(max cut value).

    Args:
        num_qubits: Number of vertices/qubits.
        edges: List of (i, j) edge tuples.
        weights: Optional edge weights (default all 1.0).

    Returns:
        MaxCut Hamiltonian matrix.
    """
    if weights is None:
        weights = [1.0] * len(edges)

    dim = 2**num_qubits
    H = np.zeros((dim, dim), dtype=complex)
    I_full = np.eye(dim, dtype=complex)

    for (i, j), w in zip(edges, weights):
        ZiZj = _embed_op(SZ, i, num_qubits) @ _embed_op(SZ, j, num_qubits)
        H += w * (I_full - ZiZj) / 2

    return H


def _embed_op(
    op: NDArray[np.complexfloating],
    qubit: int,
    num_qubits: int,
) -> NDArray[np.complexfloating]:
    """Embed a single-qubit operator into the full Hilbert space."""
    result = np.array([[1]], dtype=complex)
    for q in range(num_qubits):
        result = np.kron(result, op if q == qubit else I2)
    return result
