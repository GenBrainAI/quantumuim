"""Fidelity functions and time-evolution propagators.

Provides gate fidelity metrics and piecewise-constant propagator
computation for evaluating control pulse performance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .hamiltonians import ControlHamiltonian
from .pulses import PulseSequence


def propagator(
    hamiltonian: ControlHamiltonian,
    pulse: PulseSequence,
) -> NDArray[np.complexfloating]:
    """Compute the time-evolution unitary from a pulse sequence.

    Uses piecewise-constant approximation:
    U = U_N · U_{N-1} · ... · U_1
    where U_k = exp(-i · H(t_k) · dt)

    Args:
        hamiltonian: Control Hamiltonian system.
        pulse: Time-discretized pulse sequence.

    Returns:
        Unitary propagator matrix.
    """
    dim = hamiltonian.dim
    U = np.eye(dim, dtype=complex)

    for step in range(pulse.num_steps):
        H_t = hamiltonian.total_hamiltonian(pulse.amplitudes[step])
        U_step = _matrix_exp(-1j * H_t * pulse.dt)
        U = U_step @ U

    return U


def propagator_sequence(
    hamiltonian: ControlHamiltonian,
    pulse: PulseSequence,
) -> list[NDArray[np.complexfloating]]:
    """Compute the propagator at each time step (for GRAPE gradients).

    Returns:
        List of N unitaries [U_1, U_2, ..., U_N] where
        U_k = exp(-i H(t_k) dt).
    """
    propagators = []
    for step in range(pulse.num_steps):
        H_t = hamiltonian.total_hamiltonian(pulse.amplitudes[step])
        U_step = _matrix_exp(-1j * H_t * pulse.dt)
        propagators.append(U_step)
    return propagators


def gate_fidelity(
    U_actual: NDArray[np.complexfloating],
    U_target: NDArray[np.complexfloating],
) -> float:
    """Compute the gate fidelity between two unitaries.

    F = |Tr(U_target† · U_actual)|² / d²

    This is the average fidelity over all input states.

    Args:
        U_actual: Realized unitary.
        U_target: Target unitary.

    Returns:
        Fidelity in [0, 1].
    """
    d = U_actual.shape[0]
    overlap = np.trace(U_target.conj().T @ U_actual)
    return float(abs(overlap)**2 / d**2)


def process_fidelity(
    U_actual: NDArray[np.complexfloating],
    U_target: NDArray[np.complexfloating],
) -> float:
    """Compute the process fidelity (entanglement fidelity).

    F_pro = |Tr(U_target† · U_actual)|² / d²

    Related to average gate fidelity by:
    F_avg = (d · F_pro + 1) / (d + 1)

    Args:
        U_actual: Realized unitary.
        U_target: Target unitary.

    Returns:
        Process fidelity in [0, 1].
    """
    return gate_fidelity(U_actual, U_target)


def _matrix_exp(A: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Compute matrix exponential via eigendecomposition.

    For Hermitian arguments (like -iHdt), this is exact and efficient.
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    exp_diag = np.diag(np.exp(eigenvalues))
    return eigenvectors @ exp_diag @ np.linalg.inv(eigenvectors)
