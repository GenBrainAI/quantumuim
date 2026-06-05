"""Graph adjacency matrices for quantum walks.

Provides standard graph topologies as adjacency matrices suitable for
both discrete and continuous quantum walks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def line_graph(n: int) -> NDArray[np.floating]:
    """Adjacency matrix of a path graph (line) with n vertices.

    Vertex i connects to i-1 and i+1 (no wraparound).
    Boundary vertices have degree 1.
    """
    if n < 2:
        raise ValueError("Line graph requires at least 2 vertices")
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def cycle_graph(n: int) -> NDArray[np.floating]:
    """Adjacency matrix of a cycle graph with n vertices.

    Vertex i connects to (i-1) mod n and (i+1) mod n.
    All vertices have degree 2.
    """
    if n < 3:
        raise ValueError("Cycle graph requires at least 3 vertices")
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[(i + 1) % n, i] = 1.0
    return A


def complete_graph(n: int) -> NDArray[np.floating]:
    """Adjacency matrix of the complete graph K_n.

    Every vertex connects to every other vertex. All vertices have degree n-1.
    """
    if n < 2:
        raise ValueError("Complete graph requires at least 2 vertices")
    return np.ones((n, n), dtype=float) - np.eye(n, dtype=float)


def hypercube_graph(dim: int) -> NDArray[np.floating]:
    """Adjacency matrix of the dim-dimensional hypercube graph.

    Has 2^dim vertices. Vertices connect if their binary labels differ
    in exactly one bit. All vertices have degree dim.
    """
    if dim < 1:
        raise ValueError("Hypercube dimension must be at least 1")
    n = 2**dim
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for bit in range(dim):
            j = i ^ (1 << bit)
            A[i, j] = 1.0
    return A


def grid_graph(rows: int, cols: int) -> NDArray[np.floating]:
    """Adjacency matrix of a 2D grid graph (rows x cols).

    Vertex (r, c) maps to index r * cols + c.
    No periodic boundary conditions.
    """
    if rows < 1 or cols < 1:
        raise ValueError("Grid dimensions must be at least 1")
    n = rows * cols
    A = np.zeros((n, n), dtype=float)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if c + 1 < cols:
                A[idx, idx + 1] = 1.0
                A[idx + 1, idx] = 1.0
            if r + 1 < rows:
                A[idx, idx + cols] = 1.0
                A[idx + cols, idx] = 1.0
    return A
