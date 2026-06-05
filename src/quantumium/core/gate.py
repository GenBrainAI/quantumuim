"""Quantum gate definitions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Gate:
    """A quantum gate represented as a unitary matrix."""

    def __init__(self, matrix: NDArray[np.complexfloating], name: str = "") -> None:
        matrix = np.asarray(matrix, dtype=complex)
        n = matrix.shape[0]
        if matrix.shape != (n, n):
            raise ValueError("Gate matrix must be square")
        identity = np.eye(n, dtype=complex)
        if not np.allclose(matrix @ matrix.conj().T, identity, atol=1e-10):
            raise ValueError("Gate matrix must be unitary")
        self._matrix = matrix
        self.name = name or f"Gate({n}x{n})"

    @property
    def matrix(self) -> NDArray[np.complexfloating]:
        return self._matrix.copy()

    @property
    def num_qubits(self) -> int:
        return int(np.log2(self._matrix.shape[0]))

    def __repr__(self) -> str:
        return self.name


# Standard single-qubit gates
I = Gate(np.eye(2, dtype=complex), "I")
X = Gate(np.array([[0, 1], [1, 0]], dtype=complex), "X")
Y = Gate(np.array([[0, -1j], [1j, 0]], dtype=complex), "Y")
Z = Gate(np.array([[1, 0], [0, -1]], dtype=complex), "Z")
H = Gate(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2), "H")
S = Gate(np.array([[1, 0], [0, 1j]], dtype=complex), "S")
T = Gate(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex), "T")

# Standard two-qubit gates
CNOT = Gate(np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex), "CNOT")

SWAP = Gate(np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=complex), "SWAP")


def Rx(theta: float) -> Gate:
    """Rotation around X-axis."""
    return Gate(np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)],
    ], dtype=complex), f"Rx({theta:.3f})")


def Ry(theta: float) -> Gate:
    """Rotation around Y-axis."""
    return Gate(np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)],
    ], dtype=complex), f"Ry({theta:.3f})")


def Rz(theta: float) -> Gate:
    """Rotation around Z-axis."""
    return Gate(np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)],
    ], dtype=complex), f"Rz({theta:.3f})")
