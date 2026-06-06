"""Rotated planar surface code lattice construction.

The rotated surface code arranges data qubits on a d×d grid with X-type and
Z-type stabilizer plaquettes. For code distance d, there are d² data qubits,
(d²-1)/2 X stabilizers, and (d²-1)/2 Z stabilizers.

Boundary convention:
    - Z stabilizers have rough boundaries on LEFT and RIGHT
    - X stabilizers have rough boundaries on TOP and BOTTOM
    - Logical X is a vertical chain (column 0)
    - Logical Z is a horizontal chain (row 0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SurfaceCodeLattice:
    """Rotated planar surface code lattice for a given code distance.

    Attributes:
        d: Code distance (must be odd, >= 3).
        num_data: Number of data qubits (d²).
        data_qubits: List of (row, col) coordinates for data qubits.
        x_stabilizers: List of X-type stabilizer plaquettes, each a list of
            data qubit indices.
        z_stabilizers: List of Z-type stabilizer plaquettes, each a list of
            data qubit indices.
        data_index: Mapping from (row, col) to qubit index.
    """

    def __init__(self, d: int) -> None:
        if d < 3 or d % 2 == 0:
            raise ValueError(f"Code distance must be odd and >= 3, got {d}")
        self.d = d
        self.num_data = d * d

        self.data_qubits: list[tuple[int, int]] = []
        self.data_index: dict[tuple[int, int], int] = {}
        idx = 0
        for r in range(d):
            for c in range(d):
                self.data_qubits.append((r, c))
                self.data_index[(r, c)] = idx
                idx += 1

        self.x_stabilizers: list[list[int]] = []
        self.z_stabilizers: list[list[int]] = []
        self._build_stabilizers()

    def _build_stabilizers(self) -> None:
        d = self.d

        for r in range(d - 1):
            for c in range(d - 1):
                corners = [
                    self.data_index[(r, c)],
                    self.data_index[(r, c + 1)],
                    self.data_index[(r + 1, c)],
                    self.data_index[(r + 1, c + 1)],
                ]
                if (r + c) % 2 == 0:
                    self.x_stabilizers.append(corners)
                else:
                    self.z_stabilizers.append(corners)

        for r in range(0, d - 1, 2):
            self.z_stabilizers.append([
                self.data_index[(r, 0)],
                self.data_index[(r + 1, 0)],
            ])

        for r in range(1, d - 1, 2):
            self.z_stabilizers.append([
                self.data_index[(r, d - 1)],
                self.data_index[(r + 1, d - 1)],
            ])

        for c in range(1, d - 1, 2):
            self.x_stabilizers.append([
                self.data_index[(0, c)],
                self.data_index[(0, c + 1)],
            ])

        for c in range(0, d - 1, 2):
            self.x_stabilizers.append([
                self.data_index[(d - 1, c)],
                self.data_index[(d - 1, c + 1)],
            ])

    @property
    def num_x_stabilizers(self) -> int:
        return len(self.x_stabilizers)

    @property
    def num_z_stabilizers(self) -> int:
        return len(self.z_stabilizers)

    def stabilizer_matrix_x(self) -> NDArray[np.int8]:
        """Return the X-stabilizer parity check matrix (num_x_stab x num_data)."""
        H = np.zeros((self.num_x_stabilizers, self.num_data), dtype=np.int8)
        for i, stab in enumerate(self.x_stabilizers):
            for q in stab:
                H[i, q] = 1
        return H

    def stabilizer_matrix_z(self) -> NDArray[np.int8]:
        """Return the Z-stabilizer parity check matrix (num_z_stab x num_data)."""
        H = np.zeros((self.num_z_stabilizers, self.num_data), dtype=np.int8)
        for i, stab in enumerate(self.z_stabilizers):
            for q in stab:
                H[i, q] = 1
        return H
