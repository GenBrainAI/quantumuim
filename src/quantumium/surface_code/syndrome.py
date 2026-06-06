"""Syndrome extraction and error injection for surface codes."""

from __future__ import annotations

import numpy as np

from quantumium.surface_code.lattice import SurfaceCode


def inject_error(code: SurfaceCode, error_type: str, qubit: int) -> None:
    """Apply a Pauli error to a data qubit.

    Args:
        error_type: 'X', 'Z', or 'Y' (Y = X then Z).
    """
    if qubit < 0 or qubit >= code.n_data:
        raise ValueError(f"Qubit index {qubit} out of range [0, {code.n_data})")
    if error_type in ("X", "Y"):
        code.error_x[qubit] ^= 1
    if error_type in ("Z", "Y"):
        code.error_z[qubit] ^= 1
    if error_type not in ("X", "Y", "Z"):
        raise ValueError(f"Unknown error type: {error_type}")


def inject_random_errors(
    code: SurfaceCode, p: float, rng: np.random.Generator | None = None,
) -> None:
    """Apply independent depolarizing noise at rate p to every data qubit."""
    if rng is None:
        rng = np.random.default_rng()
    for q in range(code.n_data):
        r = rng.random()
        if r < p / 3:
            code.error_x[q] ^= 1
        elif r < 2 * p / 3:
            code.error_z[q] ^= 1
        elif r < p:
            code.error_x[q] ^= 1
            code.error_z[q] ^= 1


def measure_syndrome(code: SurfaceCode) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_syndrome, z_syndrome) binary vectors.

    x_syndrome[i] = 1 means X-stabilizer i is triggered (detected a Z error).
    z_syndrome[i] = 1 means Z-stabilizer i is triggered (detected an X error).
    """
    x_syn = (code.h_x @ code.error_z) % 2
    z_syn = (code.h_z @ code.error_x) % 2
    return x_syn.astype(int), z_syn.astype(int)
