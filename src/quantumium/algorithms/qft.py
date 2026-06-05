"""Quantum Fourier Transform implementation."""

from __future__ import annotations

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H, SWAP


def _controlled_phase(k: int) -> Gate:
    """Controlled rotation gate CR_k: phase shift of 2π/2^k on target when control is |1⟩."""
    angle = 2 * np.pi / (2**k)
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * angle)],
    ], dtype=complex)
    return Gate(matrix, f"CR{k}")


def qft(circuit: QuantumCircuit, qubits: list[int]) -> QuantumCircuit:
    """Apply Quantum Fourier Transform to the specified qubits of a circuit.

    The QFT maps computational basis states to Fourier basis states:
    |j⟩ → (1/√N) Σₖ e^(2πijk/N) |k⟩

    Uses O(n²) gates where n = len(qubits).
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
    """Apply inverse QFT (QFT†) to the specified qubits."""
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
    """Inverse of controlled rotation gate CR_k."""
    angle = -2 * np.pi / (2**k)
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * angle)],
    ], dtype=complex)
    return Gate(matrix, f"CR{k}†")
