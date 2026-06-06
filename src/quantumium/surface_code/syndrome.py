"""Syndrome extraction and error application for surface codes.

Works with classical error vectors (binary arrays) rather than full
statevectors, since surface code simulation at scale requires Pauli frame
tracking rather than exponential-size state vectors.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .lattice import SurfaceCodeLattice


def apply_error(
    lattice: SurfaceCodeLattice,
    error_type: str,
    qubit_indices: list[int] | int,
) -> NDArray[np.int8]:
    """Create a Pauli error vector on specified data qubits.

    Args:
        lattice: The surface code lattice.
        error_type: "X" or "Z" — the type of Pauli error.
        qubit_indices: Index or list of indices of data qubits to flip.

    Returns:
        Binary error vector of length num_data.
    """
    if error_type not in ("X", "Z"):
        raise ValueError(f"error_type must be 'X' or 'Z', got '{error_type}'")

    if isinstance(qubit_indices, int):
        qubit_indices = [qubit_indices]

    error = np.zeros(lattice.num_data, dtype=np.int8)
    for q in qubit_indices:
        if q < 0 or q >= lattice.num_data:
            raise ValueError(f"Qubit index {q} out of range [0, {lattice.num_data})")
        error[q] ^= 1
    return error


def extract_syndrome(
    lattice: SurfaceCodeLattice,
    error: NDArray[np.int8],
    error_type: str,
) -> NDArray[np.int8]:
    """Extract the syndrome from a Pauli error vector.

    X errors are detected by Z stabilizers and vice versa.

    Args:
        lattice: The surface code lattice.
        error: Binary error vector of length num_data.
        error_type: "X" or "Z" — the type of error that was applied.

    Returns:
        Binary syndrome vector. Length = num_z_stabilizers for X errors,
        num_x_stabilizers for Z errors.
    """
    if error_type == "X":
        H = lattice.stabilizer_matrix_z()
    elif error_type == "Z":
        H = lattice.stabilizer_matrix_x()
    else:
        raise ValueError(f"error_type must be 'X' or 'Z', got '{error_type}'")

    syndrome = (H @ error) % 2
    return syndrome.astype(np.int8)


def random_error(
    lattice: SurfaceCodeLattice,
    error_type: str,
    p: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int8]:
    """Generate a random i.i.d. Pauli error on each data qubit with probability p.

    Args:
        lattice: The surface code lattice.
        error_type: "X" or "Z".
        p: Per-qubit error probability.
        rng: Optional random number generator for reproducibility.

    Returns:
        Binary error vector of length num_data.
    """
    if rng is None:
        rng = np.random.default_rng()
    error = (rng.random(lattice.num_data) < p).astype(np.int8)
    return error
