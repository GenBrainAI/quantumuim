"""Grover's search algorithm implementation.

Grover's algorithm provides a quadratic speedup for unstructured search.
Given a search space of N = 2^n items with M marked (target) items,
it finds a marked item in O(sqrt(N/M)) oracle queries, compared to
O(N/M) classically.

This module provides:

- :func:`grover_oracle` -- constructs the phase oracle that flips marked states.
- :func:`grover_diffuser` -- constructs the diffusion (inversion about mean) operator.
- :func:`grover_search` -- builds the complete Grover search circuit.

Example::

    >>> from quantumium.algorithms.grover import grover_search
    >>> qc = grover_search(num_qubits=3, marked_states=[5])
    >>> counts = qc.measure_all(shots=1024)
    >>> # State |5> = |101> should dominate the measurement results
"""

from __future__ import annotations

import math

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import Gate, H


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


def grover_oracle(num_qubits: int, marked_states: list[int]) -> Gate:
    """Build a phase oracle that flips the sign of marked computational basis states.

    The oracle implements the transformation |x> -> (-1)^{f(x)} |x>,
    where f(x) = 1 for marked states and 0 otherwise.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the search register (search space size N = 2^num_qubits).
    marked_states : list[int]
        List of integer indices of the marked (target) states. Each must satisfy
        0 <= state < 2^num_qubits.

    Returns
    -------
    Gate
        A diagonal unitary gate with -1 entries for marked states and +1 elsewhere.

    Raises
    ------
    ValueError
        If any marked state is out of range for the given number of qubits.
    """
    dim = 2**num_qubits
    matrix = np.eye(dim, dtype=complex)
    for state in marked_states:
        if state < 0 or state >= dim:
            raise ValueError(f"Marked state {state} out of range for {num_qubits} qubits")
        idx = _bit_reverse(state, num_qubits)
        matrix[idx, idx] = -1
    return Gate(matrix, f"Oracle({marked_states})")


def grover_diffuser(num_qubits: int) -> Gate:
    """Build the Grover diffusion operator (inversion about the mean).

    Constructs the operator D = 2|s><s| - I, where |s> is the uniform
    superposition state (1/sqrt(N)) * sum_x |x>. This operator amplifies
    the amplitude of marked states when applied after the oracle.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the search register.

    Returns
    -------
    Gate
        The diffusion operator as a unitary gate.
    """
    dim = 2**num_qubits
    s = np.ones(dim, dtype=complex) / math.sqrt(dim)
    matrix = 2 * np.outer(s, s.conj()) - np.eye(dim, dtype=complex)
    return Gate(matrix, f"Diffuser({num_qubits})")


def grover_search(
    num_qubits: int,
    marked_states: list[int],
    num_iterations: int | None = None,
) -> QuantumCircuit:
    """Build a complete Grover search circuit.

    Constructs a quantum circuit that applies Grover's algorithm: an initial
    layer of Hadamard gates to create a uniform superposition, followed by
    repeated applications of the oracle and diffusion operator.

    Parameters
    ----------
    num_qubits : int
        Number of search qubits. The search space has N = 2^num_qubits states.
    marked_states : list[int]
        List of integer indices identifying the target states to search for.
    num_iterations : int or None, optional
        Number of Grover iterations (oracle + diffuser applications). If ``None``,
        uses the theoretically optimal count ``floor(pi/4 * sqrt(N/M))`` where
        M is the number of marked states.

    Returns
    -------
    QuantumCircuit
        A quantum circuit on ``num_qubits`` qubits. Measuring all qubits after
        execution yields a marked state with high probability.

    Raises
    ------
    ValueError
        If ``marked_states`` is empty or contains all states in the search space.

    Example
    -------
    >>> qc = grover_search(3, [5])
    >>> counts = qc.measure_all(shots=1024)
    >>> # The bitstring '101' (state 5) should appear with highest frequency.
    """
    n_total = 2**num_qubits
    n_marked = len(marked_states)

    if n_marked == 0:
        raise ValueError("Must mark at least one state")
    if n_marked >= n_total:
        raise ValueError("Cannot mark all states")

    if num_iterations is None:
        num_iterations = max(1, math.floor(math.pi / 4 * math.sqrt(n_total / n_marked)))

    oracle = grover_oracle(num_qubits, marked_states)
    diffuser = grover_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits)
    qubits = tuple(range(num_qubits))

    for q in range(num_qubits):
        qc.apply(H, q)

    for _ in range(num_iterations):
        qc.apply(oracle, *qubits)
        qc.apply(diffuser, *qubits)

    return qc
