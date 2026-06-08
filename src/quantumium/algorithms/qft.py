"""Quantum Fourier Transform (QFT) implementation.

The QFT maps computational basis states to Fourier basis states:
|j> -> (1/sqrt(N)) * sum_k exp(2*pi*i*j*k/N) |k>

It is the quantum analogue of the discrete Fourier transform and serves as
a key subroutine in quantum phase estimation, Shor's algorithm, and many
other quantum algorithms.

This module provides:

- :func:`qft` -- applies the QFT to specified qubits of a circuit.
- :func:`inverse_qft` -- applies the inverse QFT (QFT-dagger).

The implementation uses O(n^2) controlled phase rotation gates.
"""

from __future__ import annotations

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H, SWAP


def _controlled_phase(k: int) -> Gate:
    """Build the controlled rotation gate CR_k.

    Applies a phase shift of 2*pi/2^k to the target qubit when the control
    qubit is in state |1>.

    Parameters
    ----------
    k : int
        The rotation index. The applied phase is 2*pi / 2^k.

    Returns
    -------
    Gate
        A two-qubit controlled phase gate.
    """
    angle = 2 * np.pi / (2**k)
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * angle)],
        ],
        dtype=complex,
    )
    return Gate(matrix, f"CR{k}")


def qft(circuit: QuantumCircuit, qubits: list[int]) -> QuantumCircuit:
    """Apply the Quantum Fourier Transform to specified qubits of a circuit.

    Transforms computational basis states to Fourier basis states:
    |j> -> (1/sqrt(N)) * sum_k exp(2*pi*i*j*k/N) |k>, where N = 2^n.

    The implementation uses O(n^2) controlled phase rotation gates and
    a final layer of SWAP gates to produce the standard QFT output ordering.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to which the QFT gates are appended (modified in place).
    qubits : list[int]
        Indices of the qubits to transform. The QFT is applied across these
        qubits in the order given.

    Returns
    -------
    QuantumCircuit
        The same circuit object, with QFT gates appended (returned for chaining).
    """
    n = len(qubits)
    for i in range(n):
        circuit.apply(H, qubits[i])
        for j in range(i + 1, n):
            cr = _controlled_phase(j - i + 1)
            circuit.apply(cr, qubits[j], qubits[i])

    for i in range(n // 2):
        circuit.apply(SWAP, qubits[i], qubits[n - 1 - i])

    return circuit


def inverse_qft(circuit: QuantumCircuit, qubits: list[int]) -> QuantumCircuit:
    """Apply the inverse Quantum Fourier Transform (QFT-dagger) to specified qubits.

    This is the Hermitian conjugate of the QFT, mapping Fourier basis states
    back to computational basis states. Used extensively in quantum phase
    estimation and related algorithms.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to which the inverse QFT gates are appended (modified in place).
    qubits : list[int]
        Indices of the qubits to transform.

    Returns
    -------
    QuantumCircuit
        The same circuit object, with inverse QFT gates appended.
    """
    n = len(qubits)

    for i in range(n // 2):
        circuit.apply(SWAP, qubits[i], qubits[n - 1 - i])

    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            cr_dag = _controlled_phase_dag(j - i + 1)
            circuit.apply(cr_dag, qubits[j], qubits[i])
        circuit.apply(H, qubits[i])

    return circuit


def _controlled_phase_dag(k: int) -> Gate:
    """Build the inverse (conjugate) of the controlled rotation gate CR_k.

    Applies a phase shift of -2*pi/2^k to the target qubit when the control
    qubit is in state |1>.

    Parameters
    ----------
    k : int
        The rotation index. The applied phase is -2*pi / 2^k.

    Returns
    -------
    Gate
        A two-qubit controlled phase gate (inverse of CR_k).
    """
    angle = -2 * np.pi / (2**k)
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * angle)],
        ],
        dtype=complex,
    )
    return Gate(matrix, f"CR{k}†")
