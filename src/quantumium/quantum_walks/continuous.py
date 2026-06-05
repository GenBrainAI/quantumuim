"""Continuous-time quantum walk (CTQW).

A continuous-time quantum walk evolves under the Hamiltonian defined by
the graph adjacency matrix (or Laplacian):

    |psi(t)> = exp(-i A t) |psi(0)>

where A is the adjacency matrix. Unlike discrete walks, CTQWs don't
require a coin space — the state lives directly in the vertex space.

References:
    Farhi & Gutmann, "Quantum computation and decision trees"
    (PRA 58, 915, 1998)
    Childs et al., "Exponential algorithmic speedup by quantum walk"
    (STOC 2003)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _matrix_exp(
    A: NDArray[np.complexfloating],
    t: float,
) -> NDArray[np.complexfloating]:
    """Compute exp(-i A t) via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    return eigvecs @ np.diag(np.exp(-1j * eigvals * t)) @ eigvecs.conj().T


@dataclass
class ContinuousWalkResult:
    """Result of a continuous-time quantum walk.

    Attributes:
        position_probabilities (NDArray[np.floating]): Probability of
            finding the walker at each vertex, array of shape (num_vertices,).
        num_vertices (int): Number of vertices in the graph.
        time (float): Evolution time at which probabilities were computed.
        state (NDArray[np.complexfloating]): Full quantum state vector
            in the vertex space, of length num_vertices.
        hamiltonian (str): Label indicating which Hamiltonian was used,
            either "adjacency" or "laplacian".
    """
    position_probabilities: NDArray[np.floating]
    num_vertices: int
    time: float
    state: NDArray[np.complexfloating]
    hamiltonian: str


def continuous_walk(
    adjacency: NDArray[np.floating],
    time: float,
    initial_vertex: int = 0,
    initial_state: NDArray[np.complexfloating] | None = None,
    gamma: float = 1.0,
    use_laplacian: bool = False,
) -> ContinuousWalkResult:
    """Run a continuous-time quantum walk.

    Evolves the state under exp(-i gamma H t) where H is either the
    adjacency matrix or the graph Laplacian.

    Args:
        adjacency: Adjacency matrix of the graph (n x n).
        time: Evolution time.
        initial_vertex: Starting vertex (used if initial_state is None).
        initial_state: Full initial state vector. If None, starts at
            a single vertex.
        gamma: Coupling strength (hopping rate).
        use_laplacian: If True, use the Laplacian L = D - A as the
            Hamiltonian instead of the adjacency matrix.

    Returns:
        ContinuousWalkResult with position probability distribution.
    """
    A = np.asarray(adjacency, dtype=complex)
    n = A.shape[0]

    if use_laplacian:
        D = np.diag(np.sum(np.abs(A), axis=1))
        H = D - A
        ham_label = "laplacian"
    else:
        H = A
        ham_label = "adjacency"

    H = gamma * H

    if initial_state is not None:
        state = np.asarray(initial_state, dtype=complex)
        if len(state) != n:
            raise ValueError(f"State dimension {len(state)} != graph size {n}")
    else:
        state = np.zeros(n, dtype=complex)
        state[initial_vertex] = 1.0

    U = _matrix_exp(H, time)
    final_state = U @ state
    probs = np.abs(final_state) ** 2

    return ContinuousWalkResult(
        position_probabilities=probs,
        num_vertices=n,
        time=time,
        state=final_state,
        hamiltonian=ham_label,
    )
