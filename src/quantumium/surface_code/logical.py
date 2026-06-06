"""Logical operators for the surface code.

For the rotated planar surface code of distance d:
- Logical X (X_L): a chain of X operators along a row (top-to-bottom path).
- Logical Z (Z_L): a chain of Z operators along a column (left-to-right path).

These operators commute with all stabilizers but anticommute with each other.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .lattice import SurfaceCodeLattice


def logical_x(lattice: SurfaceCodeLattice) -> NDArray[np.int8]:
    """Return the logical X operator as a binary vector.

    The logical X operator is a vertical chain of X operators along column 0.

    Returns:
        Binary vector of length num_data with 1s on the logical X support.
    """
    op = np.zeros(lattice.num_data, dtype=np.int8)
    for r in range(lattice.d):
        op[lattice.data_index[(r, 0)]] = 1
    return op


def logical_z(lattice: SurfaceCodeLattice) -> NDArray[np.int8]:
    """Return the logical Z operator as a binary vector.

    The logical Z operator is a horizontal chain of Z operators along row 0.

    Returns:
        Binary vector of length num_data with 1s on the logical Z support.
    """
    op = np.zeros(lattice.num_data, dtype=np.int8)
    for c in range(lattice.d):
        op[lattice.data_index[(0, c)]] = 1
    return op


def apply_logical_x(
    error: NDArray[np.int8], lattice: SurfaceCodeLattice
) -> NDArray[np.int8]:
    """Apply a logical X to an existing X-error vector.

    Returns a new error vector with the logical X toggled.
    """
    lx = logical_x(lattice)
    return ((error + lx) % 2).astype(np.int8)


def apply_logical_z(
    error: NDArray[np.int8], lattice: SurfaceCodeLattice
) -> NDArray[np.int8]:
    """Apply a logical Z to an existing Z-error vector.

    Returns a new error vector with the logical Z toggled.
    """
    lz = logical_z(lattice)
    return ((error + lz) % 2).astype(np.int8)


def has_logical_error(
    error: NDArray[np.int8],
    correction: NDArray[np.int8],
    lattice: SurfaceCodeLattice,
    error_type: str,
) -> bool:
    """Check whether the residual error (error + correction) is a logical error.

    A logical error occurs when the residual is a nontrivial logical operator,
    i.e., it commutes with all stabilizers but is not itself a stabilizer.
    We check this by computing the overlap with the conjugate logical operator.

    For X errors: residual is a logical X error if it has odd overlap with
    the logical Z chain (anticommutation check).
    For Z errors: residual is a logical Z error if it has odd overlap with
    the logical X chain.

    Args:
        error: Original error vector.
        correction: Decoder-proposed correction vector.
        lattice: Surface code lattice.
        error_type: "X" or "Z".

    Returns:
        True if a logical error occurred.
    """
    residual = (error + correction) % 2

    if error_type == "X":
        check = logical_z(lattice)
    elif error_type == "Z":
        check = logical_x(lattice)
    else:
        raise ValueError(f"error_type must be 'X' or 'Z', got '{error_type}'")

    overlap = int(np.sum(residual * check)) % 2
    return overlap == 1
