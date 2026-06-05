"""Deutsch-Jozsa algorithm implementation."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import Gate, H, X


def _bit_reverse(val: int, n: int) -> int:
    result = 0
    for i in range(n):
        if val & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def deutsch_jozsa_oracle(
    num_qubits: int,
    f: Callable[[int], int],
) -> Gate:
    """Build a Deutsch-Jozsa oracle from a classical function f: {0,1}^n -> {0,1}.

    The oracle implements |x>|y> -> |x>|y XOR f(x)> as a unitary on n+1 qubits.
    """
    n = num_qubits + 1
    dim = 2**n

    matrix = np.zeros((dim, dim), dtype=complex)

    for x in range(2**num_qubits):
        fx = f(x)
        if fx not in (0, 1):
            raise ValueError(f"f({x}) = {fx}, must be 0 or 1")
        x_rev = _bit_reverse(x, num_qubits)
        for y in range(2):
            input_idx = (x_rev << 1) | y
            output_y = y ^ fx
            output_idx = (x_rev << 1) | output_y
            matrix[output_idx, input_idx] = 1

    return Gate(matrix, "DJ_Oracle")


def deutsch_jozsa(
    num_qubits: int,
    f: Callable[[int], int],
) -> QuantumCircuit:
    """Build a Deutsch-Jozsa circuit.

    Args:
        num_qubits: Number of input qubits (the function domain is {0,1}^n).
        f: Classical function f: {0,1}^n -> {0,1}, must be constant or balanced.

    Returns:
        A circuit on num_qubits+1 qubits. Measuring the first n qubits:
        all zeros -> f is constant, otherwise -> f is balanced.
    """
    total_qubits = num_qubits + 1
    oracle = deutsch_jozsa_oracle(num_qubits, f)

    qc = QuantumCircuit(total_qubits)

    qc.apply(X, num_qubits)

    for q in range(total_qubits):
        qc.apply(H, q)

    qc.apply(oracle, *range(total_qubits))

    for q in range(num_qubits):
        qc.apply(H, q)

    return qc


def classify_function(
    num_qubits: int,
    f: Callable[[int], int],
    shots: int = 1024,
) -> Literal["constant", "balanced"]:
    """Run Deutsch-Jozsa and classify the function."""
    qc = deutsch_jozsa(num_qubits, f)
    counts = qc.measure_all(shots=shots)

    input_counts: dict[str, int] = {}
    for bitstring, count in counts.items():
        input_bits = bitstring[1:]
        input_counts[input_bits] = input_counts.get(input_bits, 0) + count

    all_zeros = "0" * num_qubits
    if input_counts.get(all_zeros, 0) > shots * 0.9:
        return "constant"

    return "balanced"
