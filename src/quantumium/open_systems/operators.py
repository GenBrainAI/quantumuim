"""Common collapse (Lindblad) operators for open quantum systems.

Provides standard operators for modeling T1 decay, T2 dephasing,
thermal transitions, and bosonic mode dissipation.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def sigma_minus(num_qubits: int = 1, qubit: int = 0) -> NDArray[np.complexfloating]:
    """Qubit lowering operator σ- = |0><1|.

    Models spontaneous emission / amplitude damping.
    """
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    if num_qubits == 1:
        return sm
    return _embed(sm, num_qubits, qubit)


def sigma_plus(num_qubits: int = 1, qubit: int = 0) -> NDArray[np.complexfloating]:
    """Qubit raising operator σ+ = |1><0|."""
    sp = np.array([[0, 0], [1, 0]], dtype=complex)
    if num_qubits == 1:
        return sp
    return _embed(sp, num_qubits, qubit)


def decay_operator(
    gamma: float,
    num_qubits: int = 1,
    qubit: int = 0,
) -> tuple[float, NDArray[np.complexfloating]]:
    """T1 decay (amplitude damping) collapse operator.

    Returns (rate, operator) pair for the Lindblad equation.
    Rate γ = 1/T1 gives the energy relaxation rate.
    """
    return gamma, sigma_minus(num_qubits, qubit)


def dephasing_operator(
    gamma: float,
    num_qubits: int = 1,
    qubit: int = 0,
) -> tuple[float, NDArray[np.complexfloating]]:
    """Pure dephasing collapse operator.

    Uses L = σz/2. The pure dephasing rate γ_φ relates to T2 by:
    1/T2 = 1/(2T1) + γ_φ

    Returns (rate, operator) pair.
    """
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    if num_qubits == 1:
        op = sz
    else:
        op = _embed(sz, num_qubits, qubit)
    return gamma, op


def thermal_operator(
    gamma: float,
    n_thermal: float,
    num_qubits: int = 1,
    qubit: int = 0,
) -> list[tuple[float, NDArray[np.complexfloating]]]:
    """Thermal bath operators for a qubit at finite temperature.

    Returns two (rate, operator) pairs:
    - Emission: rate = γ(1 + n̄), operator = σ-
    - Absorption: rate = γn̄, operator = σ+

    Args:
        gamma: Base transition rate.
        n_thermal: Mean thermal photon number n̄ = 1/(exp(ℏω/kT) - 1).
    """
    ops = [(gamma * (1 + n_thermal), sigma_minus(num_qubits, qubit))]
    if n_thermal > 0:
        ops.append((gamma * n_thermal, sigma_plus(num_qubits, qubit)))
    return ops


def annihilation_operator(dim: int) -> NDArray[np.complexfloating]:
    """Bosonic annihilation operator a for a truncated Fock space.

    a|n> = √n |n-1>
    """
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n - 1, n] = math.sqrt(n)
    return a


def creation_operator(dim: int) -> NDArray[np.complexfloating]:
    """Bosonic creation operator a† for a truncated Fock space."""
    return annihilation_operator(dim).conj().T


def number_operator(dim: int) -> NDArray[np.complexfloating]:
    """Number operator n = a†a for a truncated Fock space."""
    return np.diag(np.arange(dim, dtype=complex))


def _embed(
    op: NDArray[np.complexfloating],
    num_qubits: int,
    qubit: int,
) -> NDArray[np.complexfloating]:
    """Embed a single-qubit operator into a multi-qubit Hilbert space."""
    I2 = np.eye(2, dtype=complex)
    result = np.array([[1]], dtype=complex)
    for q in range(num_qubits):
        result = np.kron(result, op if q == qubit else I2)
    return result
