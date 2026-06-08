"""Common collapse (Lindblad) operators for open quantum systems.

Provides standard operators for modeling T1 decay, T2 dephasing,
thermal transitions, and bosonic mode dissipation.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def sigma_minus(num_qubits: int = 1, qubit: int = 0) -> NDArray[np.complexfloating]:
    """Construct the qubit lowering operator sigma_minus = |0><1|.

    Models spontaneous emission and amplitude damping transitions.
    For multi-qubit systems, the operator acts on the specified qubit
    and identity on all others.

    Args:
        num_qubits (int): Total number of qubits in the system. Defaults to 1.
        qubit (int): Index of the qubit on which to apply the operator
            (0-indexed). Defaults to 0.

    Returns:
        NDArray[np.complexfloating]: The lowering operator matrix of shape
            (2^num_qubits, 2^num_qubits).
    """
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    if num_qubits == 1:
        return sm
    return _embed(sm, num_qubits, qubit)


def sigma_plus(num_qubits: int = 1, qubit: int = 0) -> NDArray[np.complexfloating]:
    """Construct the qubit raising operator sigma_plus = |1><0|.

    Models excitation and absorption transitions. For multi-qubit systems,
    the operator acts on the specified qubit and identity on all others.

    Args:
        num_qubits (int): Total number of qubits in the system. Defaults to 1.
        qubit (int): Index of the qubit on which to apply the operator
            (0-indexed). Defaults to 0.

    Returns:
        NDArray[np.complexfloating]: The raising operator matrix of shape
            (2^num_qubits, 2^num_qubits).
    """
    sp = np.array([[0, 0], [1, 0]], dtype=complex)
    if num_qubits == 1:
        return sp
    return _embed(sp, num_qubits, qubit)


def decay_operator(
    gamma: float,
    num_qubits: int = 1,
    qubit: int = 0,
) -> tuple[float, NDArray[np.complexfloating]]:
    """Construct a T1 decay (amplitude damping) collapse operator.

    Returns a (rate, operator) pair suitable for the Lindblad master
    equation. The rate gamma = 1/T1 gives the energy relaxation rate.

    Args:
        gamma (float): Decay rate (inverse of T1 relaxation time).
        num_qubits (int): Total number of qubits. Defaults to 1.
        qubit (int): Index of the qubit subject to decay (0-indexed).
            Defaults to 0.

    Returns:
        tuple[float, NDArray[np.complexfloating]]: A (rate, operator)
            pair where the operator is sigma_minus acting on the
            specified qubit.
    """
    return gamma, sigma_minus(num_qubits, qubit)


def dephasing_operator(
    gamma: float,
    num_qubits: int = 1,
    qubit: int = 0,
) -> tuple[float, NDArray[np.complexfloating]]:
    """Construct a pure dephasing collapse operator.

    Uses L = sigma_z / 2. The pure dephasing rate gamma_phi relates
    to the T2 decoherence time by: 1/T2 = 1/(2*T1) + gamma_phi.

    Args:
        gamma (float): Pure dephasing rate gamma_phi.
        num_qubits (int): Total number of qubits. Defaults to 1.
        qubit (int): Index of the qubit subject to dephasing (0-indexed).
            Defaults to 0.

    Returns:
        tuple[float, NDArray[np.complexfloating]]: A (rate, operator)
            pair where the operator is sigma_z/2 acting on the
            specified qubit.
    """
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    if num_qubits == 1:
        op = sz
    else:
        op = _embed(sz, num_qubits, qubit)  # type: ignore[arg-type, assignment]
    return gamma, op  # type: ignore[return-value]


def thermal_operator(
    gamma: float,
    n_thermal: float,
    num_qubits: int = 1,
    qubit: int = 0,
) -> list[tuple[float, NDArray[np.complexfloating]]]:
    """Construct thermal bath collapse operators for a qubit at finite temperature.

    Returns up to two (rate, operator) pairs modeling stimulated and
    spontaneous transitions:
    - Emission: rate = gamma * (1 + n_bar), operator = sigma_minus
    - Absorption: rate = gamma * n_bar, operator = sigma_plus
      (only included when n_thermal > 0)

    Args:
        gamma (float): Base transition rate.
        n_thermal (float): Mean thermal photon number
            n_bar = 1 / (exp(hbar*omega / (k_B*T)) - 1).
        num_qubits (int): Total number of qubits. Defaults to 1.
        qubit (int): Index of the qubit coupled to the thermal bath
            (0-indexed). Defaults to 0.

    Returns:
        list[tuple[float, NDArray[np.complexfloating]]]: List of
            (rate, operator) pairs for use in the Lindblad equation.
    """
    ops = [(gamma * (1 + n_thermal), sigma_minus(num_qubits, qubit))]
    if n_thermal > 0:
        ops.append((gamma * n_thermal, sigma_plus(num_qubits, qubit)))
    return ops


def annihilation_operator(dim: int) -> NDArray[np.complexfloating]:
    """Construct the bosonic annihilation operator for a truncated Fock space.

    The annihilation operator satisfies a|n> = sqrt(n) |n-1> for n >= 1,
    and a|0> = 0. The Fock space is truncated to the first `dim` levels.

    Args:
        dim (int): Dimension of the truncated Fock space (number of
            levels to include).

    Returns:
        NDArray[np.complexfloating]: The annihilation operator matrix
            of shape (dim, dim).
    """
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n - 1, n] = math.sqrt(n)
    return a


def creation_operator(dim: int) -> NDArray[np.complexfloating]:
    """Construct the bosonic creation operator for a truncated Fock space.

    The creation operator satisfies a_dag|n> = sqrt(n+1) |n+1>. It is
    the conjugate transpose of the annihilation operator.

    Args:
        dim (int): Dimension of the truncated Fock space (number of
            levels to include).

    Returns:
        NDArray[np.complexfloating]: The creation operator matrix
            of shape (dim, dim).
    """
    return annihilation_operator(dim).conj().T  # type: ignore[no-any-return]


def number_operator(dim: int) -> NDArray[np.complexfloating]:
    """Construct the photon number operator for a truncated Fock space.

    The number operator n = a_dag * a satisfies n|k> = k|k>. It is
    represented as a diagonal matrix with entries 0, 1, ..., dim-1.

    Args:
        dim (int): Dimension of the truncated Fock space (number of
            levels to include).

    Returns:
        NDArray[np.complexfloating]: The number operator matrix
            of shape (dim, dim).
    """
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
