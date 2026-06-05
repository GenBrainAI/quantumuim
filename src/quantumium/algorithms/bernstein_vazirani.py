"""Bernstein-Vazirani algorithm implementation.

The Bernstein-Vazirani algorithm determines a secret string s in {0,1}^n
hidden inside a linear Boolean function f(x) = s . x (mod 2) using a
single quantum query. Classically, n queries are required.

This module provides:

- :func:`bv_oracle` -- constructs the oracle for f(x) = s . x mod 2.
- :func:`bernstein_vazirani` -- builds the complete circuit.
- :func:`find_secret` -- runs the circuit and extracts the secret string.

Example::

    >>> from quantumium.algorithms.bernstein_vazirani import find_secret
    >>> secret = find_secret(num_qubits=4, secret=0b1010)
    >>> assert secret == 0b1010
"""

from __future__ import annotations

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import Gate, H, X


def _bit_reverse(val: int, n: int) -> int:
    """Reverse the bit-order of an n-bit integer.

    Parameters
    ----------
    val : int
        The integer whose bits are to be reversed.
    n : int
        The number of bits to consider.

    Returns
    -------
    int
        The bit-reversed integer.
    """
    result = 0
    for i in range(n):
        if val & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bv_oracle(num_qubits: int, secret: int) -> Gate:
    """Build a Bernstein-Vazirani oracle for f(x) = s . x mod 2.

    Constructs the unitary |x>|y> -> |x>|y XOR (s . x mod 2)> on
    ``num_qubits + 1`` qubits (n input qubits plus 1 ancilla).

    Parameters
    ----------
    num_qubits : int
        Number of input qubits.
    secret : int
        The secret integer s, where 0 <= s < 2^num_qubits.

    Returns
    -------
    Gate
        A unitary gate on ``num_qubits + 1`` qubits implementing the oracle.

    Raises
    ------
    ValueError
        If ``secret`` is negative or >= 2^num_qubits.
    """
    if secret < 0 or secret >= 2**num_qubits:
        raise ValueError(f"Secret {secret} out of range for {num_qubits} qubits")

    n = num_qubits + 1
    dim = 2**n

    matrix = np.zeros((dim, dim), dtype=complex)

    for x in range(2**num_qubits):
        dot = bin(x & secret).count("1") % 2
        x_rev = _bit_reverse(x, num_qubits)
        for y in range(2):
            input_idx = (x_rev << 1) | y
            output_y = y ^ dot
            output_idx = (x_rev << 1) | output_y
            matrix[output_idx, input_idx] = 1

    return Gate(matrix, f"BV_Oracle(s={secret})")


def bernstein_vazirani(num_qubits: int, secret: int) -> QuantumCircuit:
    """Build a complete Bernstein-Vazirani circuit to find the secret string s.

    The circuit applies Hadamards, queries the oracle, and applies Hadamards
    again. Measuring the first ``num_qubits`` qubits directly reveals the
    binary representation of the secret integer s.

    Parameters
    ----------
    num_qubits : int
        Number of input qubits.
    secret : int
        The secret integer s, where f(x) = s . x mod 2. Must satisfy
        0 <= s < 2^num_qubits.

    Returns
    -------
    QuantumCircuit
        A circuit on ``num_qubits + 1`` qubits. Measuring the first
        ``num_qubits`` qubits reveals the secret string s in binary.
    """
    total_qubits = num_qubits + 1
    oracle = bv_oracle(num_qubits, secret)

    qc = QuantumCircuit(total_qubits)

    qc.apply(X, num_qubits)

    for q in range(total_qubits):
        qc.apply(H, q)

    qc.apply(oracle, *range(total_qubits))

    for q in range(num_qubits):
        qc.apply(H, q)

    return qc


def find_secret(num_qubits: int, secret: int, shots: int = 1024) -> int:
    """Run the Bernstein-Vazirani algorithm and extract the secret string.

    Builds and executes the circuit, then reads the most frequent measurement
    outcome on the input register to determine the secret.

    Parameters
    ----------
    num_qubits : int
        Number of input qubits.
    secret : int
        The secret integer s, where f(x) = s . x mod 2.
    shots : int, optional
        Number of measurement shots (default 1024).

    Returns
    -------
    int
        The recovered secret integer. Should equal ``secret`` with high
        probability.
    """
    qc = bernstein_vazirani(num_qubits, secret)
    counts = qc.measure_all(shots=shots)

    best_bitstring = max(counts, key=counts.get)  # type: ignore[arg-type]
    input_bits = best_bitstring[1:]
    return int(input_bits, 2)
