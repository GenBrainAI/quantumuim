"""Pre-built quantum circuit templates for common patterns."""

from __future__ import annotations

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import Gate, H, CNOT, SWAP


def bell_pair(qubit_a: int = 0, qubit_b: int = 1) -> QuantumCircuit:
    """Create a circuit that prepares a Bell pair (|00> + |11>) / sqrt(2).

    Applies a Hadamard gate to qubit_a followed by a CNOT from
    qubit_a to qubit_b, producing the maximally entangled Bell state.

    Args:
        qubit_a (int): Index of the first qubit (Hadamard target).
            Defaults to 0.
        qubit_b (int): Index of the second qubit (CNOT target).
            Defaults to 1.

    Returns:
        QuantumCircuit: A circuit that produces the Bell state when
            applied to |00>.
    """
    n = max(qubit_a, qubit_b) + 1
    qc = QuantumCircuit(n)
    qc.apply(H, qubit_a)
    qc.apply(CNOT, qubit_a, qubit_b)
    return qc


def ghz_state(num_qubits: int) -> QuantumCircuit:
    """Create a circuit that prepares the GHZ state (|00...0> + |11...1>) / sqrt(2).

    Applies a Hadamard to qubit 0, then a cascade of CNOT gates from
    qubit i to qubit i+1, producing a maximally entangled n-qubit state.

    Args:
        num_qubits (int): Number of qubits. Must be at least 2.

    Returns:
        QuantumCircuit: A circuit that produces the GHZ state when
            applied to |00...0>.

    Raises:
        ValueError: If num_qubits < 2.
    """
    if num_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits")
    qc = QuantumCircuit(num_qubits)
    qc.apply(H, 0)
    for i in range(num_qubits - 1):
        qc.apply(CNOT, i, i + 1)
    return qc


def quantum_teleportation() -> QuantumCircuit:
    """Create a quantum teleportation circuit on 3 qubits.

    Qubit 0: The state to teleport (must be prepared externally).
    Qubit 1: Alice's half of the shared Bell pair.
    Qubit 2: Bob's half of the shared Bell pair (receives the
    teleported state).

    The circuit includes Bell pair creation between qubits 1 and 2,
    followed by Alice's Bell-basis measurement (CNOT + H on qubits
    0 and 1), and classical corrections implemented as controlled
    gates for unitary simulation purposes.

    Returns:
        QuantumCircuit: A 3-qubit teleportation circuit.
    """
    qc = QuantumCircuit(3)
    qc.apply(H, 1)
    qc.apply(CNOT, 1, 2)
    qc.apply(CNOT, 0, 1)
    qc.apply(H, 0)
    qc.apply(CNOT, 1, 2)

    cz = Gate(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1],
            ],
            dtype=complex,
        ),
        "CZ",
    )
    qc.apply(cz, 0, 2)

    return qc


def qft(num_qubits: int) -> QuantumCircuit:
    """Create a Quantum Fourier Transform (QFT) circuit.

    Implements the standard QFT using Hadamard gates and controlled
    phase rotations, followed by qubit reversal (SWAP gates). The QFT
    maps computational basis states to their Fourier-transformed
    representation.

    Args:
        num_qubits (int): Number of qubits. Must be at least 1.

    Returns:
        QuantumCircuit: A circuit implementing the QFT.

    Raises:
        ValueError: If num_qubits < 1.
    """
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
    """Create an inverse Quantum Fourier Transform (QFT-dagger) circuit.

    Implements the inverse of the QFT by reversing the order of
    operations and negating the phase rotation angles. Used as a
    subroutine in quantum phase estimation and other algorithms.

    Args:
        num_qubits (int): Number of qubits. Must be at least 1.

    Returns:
        QuantumCircuit: A circuit implementing the inverse QFT.

    Raises:
        ValueError: If num_qubits < 1.
    """
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
    """Construct a two-qubit controlled phase gate.

    Applies a phase of e^{i*theta} to the |11> state and acts as
    identity on all other computational basis states.

    Args:
        theta (float): Phase angle in radians.

    Returns:
        Gate: A 4x4 controlled phase gate.
    """
    return Gate(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * theta)],
            ],
            dtype=complex,
        ),
        f"CP({theta:.3f})",
    )
