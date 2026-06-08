"""Logical operators for the surface code."""

from __future__ import annotations

import numpy as np

from quantumium.surface_code.lattice import SurfaceCode


def logical_x(code: SurfaceCode) -> np.ndarray:
    """Return the logical X operator as a binary vector over data qubits.

    Logical X is a chain of X operators along column 0 (left-to-right rough boundary).
    """
    op = np.zeros(code.n_data, dtype=int)
    d = code.distance
    for r in range(d):
        op[r * d] = 1
    return op


def logical_z(code: SurfaceCode) -> np.ndarray:
    """Return the logical Z operator as a binary vector over data qubits.

    Logical Z is a chain of Z operators along row 0 (top-to-bottom smooth boundary).
    """
    op = np.zeros(code.n_data, dtype=int)
    d = code.distance
    for c in range(d):
        op[c] = 1
    return op


def has_logical_error(
    code: SurfaceCode, x_correction: np.ndarray, z_correction: np.ndarray
) -> bool:
    """Check if the net effect of errors + correction is a non-trivial logical operator."""
    lx = logical_x(code)
    lz = logical_z(code)

    # Total X operator = original X errors XOR X correction.
    total_x = (code.error_x + x_correction) % 2
    # Total Z operator = original Z errors XOR Z correction.
    total_z = (code.error_z + z_correction) % 2

    # A logical X error: total_x anticommutes with logical Z.
    # Anticommutation check: dot product of total_x with logical_z support (mod 2).
    logical_x_error = (np.dot(total_x, lz) % 2) == 1
    # A logical Z error: total_z anticommutes with logical X.
    logical_z_error = (np.dot(total_z, lx) % 2) == 1

    return logical_x_error or logical_z_error
