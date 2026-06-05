"""Pre-built quantum circuit templates for common patterns."""

from __future__ import annotations

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import Gate, H, X, Z, CNOT, SWAP, Rz


def bell_pair(qubit_a: int = 0, qubit_b: int = 1) -> QuantumCircuit:
    """Create a Bell pair (|00> + |11>) / sqrt(2) on two qubits."""
    n = max(qubit_a, qubit_b) + 1
    qc = QuantumCircuit(n)
    qc.apply(H, qubit_a)
    qc.apply(CNOT, qubit_a, qubit_b)
    return qc


def ghz_state(num_qubits: int) -> QuantumCircuit:
    """Create a GHZ state (|00...0> + |11...1>) / sqrt(2)."""
    if num_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits")
    qc = QuantumCircuit(num_qubits)
    qc.apply(H, 0)
    for i in range(num_qubits - 1):
        qc.apply(CNOT, i, i + 1)
    return qc


def quantum_teleportation() -> QuantumCircuit:
    """Quantum teleportation circuit on 3 qubits.

    Qubit 0: state to teleport (prepared externally)
    Qubit 1: Alice's half of Bell pair
    Qubit 2: Bob's half of Bell pair (receives teleported state)

    Returns the circuit with Bell pair creation and teleportation gates.
    The classical correction (X and Z conditioned on measurement) is
    included as controlled gates for simulation purposes.
    """
    qc = QuantumCircuit(3)
    qc.apply(H, 1)
    qc.apply(CNOT, 1, 2)
    qc.apply(CNOT, 0, 1)
    qc.apply(H, 0)
    qc.apply(CNOT, 1, 2)

    cz = Gate(np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ], dtype=complex), "CZ")
    qc.apply(cz, 0, 2)

    return qc


def qft(num_qubits: int) -> QuantumCircuit:
    """Quantum Fourier Transform circuit."""
    if num_qubits < 1:
        raise ValueError("QFT requires at least 1 qubit")
    qc = QuantumCircuit(num_qubits)

    for i in range(num_qubits):
        qc.apply(H, i)
        for j in range(i + 1, num_qubits):
            k = j - i + 1
            theta = 2 * np.pi / (2**k)
            cp = _controlled_phase(theta)
            qc.apply(cp, j, i)

    for i in range(num_qubits // 2):
        qc.apply(SWAP, i, num_qubits - 1 - i)

    return qc


def inverse_qft(num_qubits: int) -> QuantumCircuit:
    """Inverse Quantum Fourier Transform circuit."""
    if num_qubits < 1:
        raise ValueError("Inverse QFT requires at least 1 qubit")
    qc = QuantumCircuit(num_qubits)

    for i in range(num_qubits // 2):
        qc.apply(SWAP, i, num_qubits - 1 - i)

    for i in range(num_qubits - 1, -1, -1):
        for j in range(num_qubits - 1, i, -1):
            k = j - i + 1
            theta = -2 * np.pi / (2**k)
            cp = _controlled_phase(theta)
            qc.apply(cp, j, i)
        qc.apply(H, i)

    return qc


def _controlled_phase(theta: float) -> Gate:
    """Two-qubit controlled phase gate."""
    return Gate(np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * theta)],
    ], dtype=complex), f"CP({theta:.3f})")
