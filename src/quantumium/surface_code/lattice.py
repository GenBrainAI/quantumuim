"""Rotated planar surface code lattice construction."""

from __future__ import annotations

import numpy as np


class SurfaceCode:
    """Rotated planar surface code for a given odd distance d >= 3.

    Data qubits sit on a d x d grid. X stabilizers (plaquettes) detect
    Z errors; Z stabilizers (vertices) detect X errors. Encodes 1 logical qubit.
    """

    def __init__(self, distance: int) -> None:
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance must be an odd integer >= 3")
        self.distance = distance
        self.n_data = distance * distance
        self._build_stabilizers()
        self.error_x = np.zeros(self.n_data, dtype=int)
        self.error_z = np.zeros(self.n_data, dtype=int)

    def _build_stabilizers(self) -> None:
        d = self.distance
        x_stabs: list[list[int]] = []
        z_stabs: list[list[int]] = []

        # Interior stabilizers from the (d-1) x (d-1) grid of cells.
        # Cell (r,c) covers data qubits at grid positions
        # (r,c), (r,c+1), (r+1,c), (r+1,c+1).
        for r in range(d - 1):
            for c in range(d - 1):
                qubits = [r * d + c, r * d + c + 1, (r + 1) * d + c, (r + 1) * d + c + 1]
                if (r + c) % 2 == 0:
                    x_stabs.append(qubits)
                else:
                    z_stabs.append(qubits)

        # Boundary stabilizers (weight 2).
        # Top boundary: missing row above row 0.
        for c in range(1, d - 1, 2):
            x_stabs.append([c, c + 1])

        # Bottom boundary: missing row below row d-1.
        for c in range(0, d - 1, 2):
            x_stabs.append([(d - 1) * d + c, (d - 1) * d + c + 1])

        # Left boundary: missing column left of col 0.
        for r in range(0, d - 1, 2):
            z_stabs.append([r * d, (r + 1) * d])

        # Right boundary: missing column right of col d-1.
        for r in range(1, d - 1, 2):
            z_stabs.append([r * d + d - 1, (r + 1) * d + d - 1])

        self.x_stabilizers = x_stabs
        self.z_stabilizers = z_stabs

        self.h_x = np.zeros((len(x_stabs), self.n_data), dtype=int)
        for i, stab in enumerate(x_stabs):
            for q in stab:
                self.h_x[i, q] = 1

        self.h_z = np.zeros((len(z_stabs), self.n_data), dtype=int)
        for i, stab in enumerate(z_stabs):
            for q in stab:
                self.h_z[i, q] = 1

    def reset(self) -> None:
        """Clear all errors."""
        self.error_x = np.zeros(self.n_data, dtype=int)
        self.error_z = np.zeros(self.n_data, dtype=int)

    @property
    def n_x_stabilizers(self) -> int:
        return len(self.x_stabilizers)

    @property
    def n_z_stabilizers(self) -> int:
        return len(self.z_stabilizers)
