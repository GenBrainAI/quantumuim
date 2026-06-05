"""Discrete-time quantum walk (DTQW).

A discrete-time quantum walk uses a coin operator and a conditional shift
operator. The walker lives in a Hilbert space H_position ⊗ H_coin, where
the coin space has dimension equal to the maximum vertex degree.

The walk step is: U = S · (I_pos ⊗ C)

where C is the coin operator and S is the shift operator that moves the
walker based on the coin state.

References:
    Aharonov et al., "Quantum random walks" (PRA 48, 1687, 1993)
    Kempe, "Quantum random walks: An introductory overview"
    (Contemporary Physics 44, 307, 2003)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .graphs import line_graph, cycle_graph


def _hadamard_coin() -> NDArray[np.complexfloating]:
    """Hadamard coin for 2-dimensional coin space."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _grover_coin(d: int) -> NDArray[np.complexfloating]:
    """Grover diffusion operator as a coin for d-dimensional coin space.

    G = 2|s><s| - I where |s> = (1/sqrt(d)) sum_i |i>
    """
    s = np.ones(d, dtype=complex) / np.sqrt(d)
    return 2 * np.outer(s, s.conj()) - np.eye(d, dtype=complex)


def _dft_coin(d: int) -> NDArray[np.complexfloating]:
    """Discrete Fourier Transform coin for d-dimensional coin space."""
    omega = np.exp(2j * np.pi / d)
    return np.array(
        [[omega ** (j * k) for k in range(d)] for j in range(d)],
        dtype=complex,
    ) / np.sqrt(d)


def _build_shift_operator(
    adjacency: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    """Build the shift operator from an adjacency matrix.

    For each vertex v, the shift maps |v, k> to |neighbor_k(v), k'>,
    where k indexes the neighbors of v. The coin dimension equals the
    maximum degree.

    The shift operator acts as:
        S|v, k> = |w, k'> where w is the k-th neighbor of v
        and k' is the index of v in w's neighbor list.
    """
    n = adjacency.shape[0]
    degrees = np.sum(adjacency > 0, axis=1).astype(int)
    d_coin = int(max(degrees))

    dim = n * d_coin
    S = np.zeros((dim, dim), dtype=complex)

    neighbors = []
    for v in range(n):
        nbrs = [j for j in range(n) if adjacency[v, j] > 0]
        while len(nbrs) < d_coin:
            nbrs.append(v)
        neighbors.append(nbrs)

    for v in range(n):
        for k in range(d_coin):
            w = neighbors[v][k]
            k_prime = next(
                (idx for idx, u in enumerate(neighbors[w]) if u == v),
                k % d_coin,
            )
            S[w * d_coin + k_prime, v * d_coin + k] = 1.0

    return S


@dataclass
class DiscreteWalkResult:
    """Result of a discrete-time quantum walk."""
    position_probabilities: NDArray[np.floating]
    num_vertices: int
    num_steps: int
    coin_dimension: int
    full_state: NDArray[np.complexfloating]


def discrete_walk(
    adjacency: NDArray[np.floating],
    num_steps: int,
    initial_vertex: int = 0,
    initial_coin_state: NDArray[np.complexfloating] | None = None,
    coin: NDArray[np.complexfloating] | str = "grover",
) -> DiscreteWalkResult:
    """Run a discrete-time quantum walk on an arbitrary graph.

    Args:
        adjacency: Adjacency matrix of the graph (n x n).
        num_steps: Number of walk steps to apply.
        initial_vertex: Starting vertex index.
        initial_coin_state: Initial state of the coin register.
            If None, uses uniform superposition.
        coin: Coin operator matrix or one of "hadamard", "grover", "dft".
            "hadamard" only works for 2-regular graphs.

    Returns:
        DiscreteWalkResult with position probability distribution.
    """
    n = adjacency.shape[0]
    degrees = np.sum(adjacency > 0, axis=1).astype(int)
    d_coin = int(max(degrees))
    dim = n * d_coin

    if isinstance(coin, str):
        coin_ops = {
            "hadamard": lambda: _hadamard_coin() if d_coin == 2 else _dft_coin(d_coin),
            "grover": lambda: _grover_coin(d_coin),
            "dft": lambda: _dft_coin(d_coin),
        }
        if coin not in coin_ops:
            raise ValueError(f"Unknown coin '{coin}'. Use 'hadamard', 'grover', or 'dft'.")
        C = coin_ops[coin]()
    else:
        C = np.asarray(coin, dtype=complex)

    if C.shape[0] != d_coin:
        raise ValueError(
            f"Coin dimension {C.shape[0]} doesn't match graph max degree {d_coin}"
        )

    S = _build_shift_operator(adjacency)
    coin_op = np.kron(np.eye(n, dtype=complex), C)
    U = S @ coin_op

    state = np.zeros(dim, dtype=complex)
    if initial_coin_state is not None:
        cs = np.asarray(initial_coin_state, dtype=complex)
        if len(cs) != d_coin:
            raise ValueError(f"Coin state dimension {len(cs)} != {d_coin}")
        state[initial_vertex * d_coin:(initial_vertex + 1) * d_coin] = cs
    else:
        cs = np.ones(d_coin, dtype=complex) / np.sqrt(d_coin)
        state[initial_vertex * d_coin:(initial_vertex + 1) * d_coin] = cs

    for _ in range(num_steps):
        state = U @ state

    probs = np.zeros(n)
    for v in range(n):
        segment = state[v * d_coin:(v + 1) * d_coin]
        probs[v] = float(np.sum(np.abs(segment) ** 2))

    return DiscreteWalkResult(
        position_probabilities=probs,
        num_vertices=n,
        num_steps=num_steps,
        coin_dimension=d_coin,
        full_state=state,
    )


def hadamard_walk_line(
    num_positions: int,
    num_steps: int,
    initial_position: int | None = None,
    initial_coin_state: NDArray[np.complexfloating] | None = None,
) -> DiscreteWalkResult:
    """Hadamard walk on a line (1D lattice with absorbing boundaries).

    The classic quantum walk: uses a Hadamard coin on a line graph.
    Produces the characteristic asymmetric distribution.

    Args:
        num_positions: Number of positions on the line.
        num_steps: Number of steps.
        initial_position: Starting position (default: center).
        initial_coin_state: Initial coin state (default: |0>).

    Returns:
        DiscreteWalkResult.
    """
    if initial_position is None:
        initial_position = num_positions // 2

    A = line_graph(num_positions)

    if initial_coin_state is None:
        initial_coin_state = np.array([1, 0], dtype=complex)

    return discrete_walk(
        adjacency=A,
        num_steps=num_steps,
        initial_vertex=initial_position,
        initial_coin_state=initial_coin_state,
        coin="hadamard",
    )


def grover_walk(
    adjacency: NDArray[np.floating],
    num_steps: int,
    initial_vertex: int = 0,
    initial_coin_state: NDArray[np.complexfloating] | None = None,
) -> DiscreteWalkResult:
    """Grover walk on an arbitrary graph.

    Uses the Grover diffusion operator as the coin. The Grover walk
    exhibits different mixing properties compared to Hadamard walks
    and is related to Grover's search algorithm.

    Args:
        adjacency: Graph adjacency matrix.
        num_steps: Number of walk steps.
        initial_vertex: Starting vertex.
        initial_coin_state: Initial coin state (default: uniform).

    Returns:
        DiscreteWalkResult.
    """
    return discrete_walk(
        adjacency=adjacency,
        num_steps=num_steps,
        initial_vertex=initial_vertex,
        initial_coin_state=initial_coin_state,
        coin="grover",
    )
