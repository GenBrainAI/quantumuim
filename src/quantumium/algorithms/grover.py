"""Grover's search algorithm implementation."""

from __future__ import annotations

import math

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import Gate, H


def _bit_reverse(val: int, n: int) -> int:
    result = 0
    for i in range(n):
        if val & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def grover_oracle(num_qubits: int, marked_states: list[int]) -> Gate:
    """Build an oracle that flips the phase of the marked states."""
    dim = 2**num_qubits
    matrix = np.eye(dim, dtype=complex)
    for state in marked_states:
        if state < 0 or state >= dim:
            raise ValueError(f"Marked state {state} out of range for {num_qubits} qubits")
        idx = _bit_reverse(state, num_qubits)
        matrix[idx, idx] = -1
    return Gate(matrix, f"Oracle({marked_states})")


def grover_diffuser(num_qubits: int) -> Gate:
    """Build the Grover diffusion operator (2|s⟩⟨s| - I)."""
    dim = 2**num_qubits
    s = np.ones(dim, dtype=complex) / math.sqrt(dim)
    matrix = 2 * np.outer(s, s.conj()) - np.eye(dim, dtype=complex)
    return Gate(matrix, f"Diffuser({num_qubits})")


def grover_search(
    num_qubits: int,
    marked_states: list[int],
    num_iterations: int | None = None,
) -> QuantumCircuit:
    """Build a Grover search circuit.

    Args:
        num_qubits: Number of search qubits.
        marked_states: List of integer indices to search for.
        num_iterations: Number of Grover iterations. If None, uses the
            optimal count floor(pi/4 * sqrt(N/M)).
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
