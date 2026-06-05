"""Bernstein-Vazirani algorithm implementation."""

from __future__ import annotations

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import Gate, H, X


def _bit_reverse(val: int, n: int) -> int:
    result = 0
    for i in range(n):
        if val & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bv_oracle(num_qubits: int, secret: int) -> Gate:
    """Build a Bernstein-Vazirani oracle for f(x) = s*x mod 2.

    Operates on num_qubits+1 qubits (n input + 1 ancilla).
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
    """Build a Bernstein-Vazirani circuit to find the secret string s.

    Args:
        num_qubits: Number of input qubits.
        secret: The secret integer s where f(x) = s*x mod 2.

    Returns:
        Circuit on num_qubits+1 qubits. Measuring the first n qubits
        reveals the secret string s.
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
    """Run the algorithm and extract the secret string."""
    qc = bernstein_vazirani(num_qubits, secret)
    counts = qc.measure_all(shots=shots)

    best_bitstring = max(counts, key=counts.get)  # type: ignore[arg-type]
    input_bits = best_bitstring[1:]
    return int(input_bits, 2)
