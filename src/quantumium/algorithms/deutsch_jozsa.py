"""Deutsch-Jozsa algorithm implementation.

The Deutsch-Jozsa algorithm determines with certainty whether a Boolean
function f: {0,1}^n -> {0,1} is constant (same output for all inputs) or
balanced (returns 0 for exactly half of the inputs and 1 for the other half),
using a single query to the oracle. Classically, this requires up to
2^(n-1) + 1 queries in the worst case.

This module provides:

- :func:`deutsch_jozsa_oracle` -- constructs the quantum oracle from a classical function.
- :func:`deutsch_jozsa` -- builds the complete Deutsch-Jozsa circuit.
- :func:`classify_function` -- runs the circuit and returns ``"constant"`` or ``"balanced"``.

Example::

    >>> from quantumium.algorithms.deutsch_jozsa import classify_function
    >>> f_const = lambda x: 0   # constant function
    >>> classify_function(3, f_const)
    'constant'
"""

from __future__ import annotations

from typing import Callable, Literal

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


def deutsch_jozsa_oracle(
    num_qubits: int,
    f: Callable[[int], int],
) -> Gate:
    """Build a Deutsch-Jozsa oracle from a classical Boolean function.

    Constructs the unitary transformation |x>|y> -> |x>|y XOR f(x)>
    on ``num_qubits + 1`` qubits, where the first ``num_qubits`` qubits
    encode the input x and the last qubit is the ancilla y.

    Parameters
    ----------
    num_qubits : int
        Number of input qubits (the domain of f is {0, 1}^num_qubits).
    f : Callable[[int], int]
        A classical function mapping integers in [0, 2^num_qubits) to {0, 1}.
        Must be either constant or balanced for the algorithm to work correctly.

    Returns
    -------
    Gate
        A unitary gate on ``num_qubits + 1`` qubits implementing the oracle.

    Raises
    ------
    ValueError
        If ``f(x)`` returns a value other than 0 or 1 for any input.
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
    """Build a complete Deutsch-Jozsa circuit.

    Constructs the quantum circuit that determines whether ``f`` is constant
    or balanced. The circuit uses ``num_qubits + 1`` qubits: the first
    ``num_qubits`` are the input register and the last qubit is the ancilla
    initialized to |1>.

    Parameters
    ----------
    num_qubits : int
        Number of input qubits (the function domain is {0, 1}^num_qubits).
    f : Callable[[int], int]
        Classical function f: {0,1}^n -> {0,1}. Must be constant or balanced.

    Returns
    -------
    QuantumCircuit
        A circuit on ``num_qubits + 1`` qubits. After execution, measuring the
        first ``num_qubits`` qubits yields all zeros if and only if ``f`` is
        constant; any non-zero outcome indicates ``f`` is balanced.
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
    """Run the Deutsch-Jozsa algorithm and classify the function as constant or balanced.

    Builds and executes the Deutsch-Jozsa circuit, then inspects the measurement
    results on the input register to determine the function type.

    Parameters
    ----------
    num_qubits : int
        Number of input qubits (the function domain is {0, 1}^num_qubits).
    f : Callable[[int], int]
        Classical function f: {0,1}^n -> {0,1}. Must be constant or balanced.
    shots : int, optional
        Number of measurement shots to perform (default 1024). More shots
        improve statistical confidence.

    Returns
    -------
    Literal["constant", "balanced"]
        ``"constant"`` if the all-zeros outcome dominates (> 90% of shots),
        ``"balanced"`` otherwise.
    """
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
