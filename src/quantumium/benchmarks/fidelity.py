"""Fidelity metrics for quantum states and processes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def state_fidelity(
    state1: NDArray[np.complexfloating],
    state2: NDArray[np.complexfloating],
) -> float:
    """Compute fidelity between two pure quantum states.

    F(psi, phi) = |<psi|phi>|^2

    Args:
        state1: First statevector.
        state2: Second statevector.

    Returns:
        Fidelity in [0, 1].
    """
    state1 = np.asarray(state1, dtype=complex)
    state2 = np.asarray(state2, dtype=complex)
    if len(state1) != len(state2):
        raise ValueError("States must have the same dimension")
    return float(abs(np.dot(state1.conj(), state2)) ** 2)


def process_fidelity(
    U_ideal: NDArray[np.complexfloating],
    U_actual: NDArray[np.complexfloating],
) -> float:
    """Compute process fidelity between two unitary operations.

    F_proc(U, V) = |Tr(U^dag V)|^2 / d^2

    where d is the dimension of the Hilbert space.

    Args:
        U_ideal: Ideal unitary matrix.
        U_actual: Actual (noisy) unitary matrix.

    Returns:
        Process fidelity in [0, 1].
    """
    U_ideal = np.asarray(U_ideal, dtype=complex)
    U_actual = np.asarray(U_actual, dtype=complex)
    if U_ideal.shape != U_actual.shape:
        raise ValueError("Matrices must have the same shape")
    d = U_ideal.shape[0]
    return float(abs(np.trace(U_ideal.conj().T @ U_actual)) ** 2 / d**2)


def average_gate_fidelity(
    U_ideal: NDArray[np.complexfloating],
    U_actual: NDArray[np.complexfloating],
) -> float:
    """Compute average gate fidelity between two unitaries.

    F_avg = (d * F_proc + 1) / (d + 1)

    where F_proc is the process fidelity and d is the dimension.

    Args:
        U_ideal: Ideal unitary matrix.
        U_actual: Actual unitary matrix.

    Returns:
        Average gate fidelity in [0, 1].
    """
    d = U_ideal.shape[0]
    f_proc = process_fidelity(U_ideal, U_actual)
    return (d * f_proc + 1) / (d + 1)
