"""Quantum Phase Estimation algorithm."""

from __future__ import annotations

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import Gate, H


def _bit_reverse(val: int, n: int) -> int:
    result = 0
    for i in range(n):
        if val & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def controlled_unitary(unitary: Gate) -> Gate:
    """Create a controlled version of a unitary gate."""
    n = unitary.matrix.shape[0]
    dim = 2 * n
    matrix = np.eye(dim, dtype=complex)
    matrix[n:, n:] = unitary.matrix
    return Gate(matrix, f"C-{unitary.name}")


def unitary_power(unitary: Gate, power: int) -> Gate:
    """Compute U^power for a unitary gate."""
    result = np.linalg.matrix_power(unitary.matrix, power)
    return Gate(result, f"{unitary.name}^{power}")


def qpe_circuit(
    num_counting_qubits: int,
    unitary: Gate,
) -> QuantumCircuit:
    """Build a Quantum Phase Estimation circuit.

    Args:
        num_counting_qubits: Number of qubits for phase precision.
        unitary: The unitary gate whose eigenvalue phase to estimate.
            Must be a single-qubit gate. The eigenstate should be
            prepared on the target qubit externally.

    Returns:
        Circuit with num_counting_qubits + 1 total qubits.
        Counting qubits are 0..n-1, target qubit is n.
        After measurement, the counting qubits encode the phase as
        theta ~= measured_value / 2^n.
    """
    if unitary.num_qubits != 1:
        raise ValueError("QPE currently supports single-qubit unitaries only")

    total_qubits = num_counting_qubits + 1
    target_qubit = num_counting_qubits
    qc = QuantumCircuit(total_qubits)

    for q in range(num_counting_qubits):
        qc.apply(H, q)

    for q in range(num_counting_qubits):
        power = 2 ** (num_counting_qubits - 1 - q)
        cu = controlled_unitary(unitary_power(unitary, power))
        qc.apply(cu, q, target_qubit)

    from quantumium.circuits.templates import inverse_qft as build_iqft
    iqft = build_iqft(num_counting_qubits)
    for inst in iqft.instructions:
        qc.apply(inst.gate, *inst.target_qubits)

    return qc


def estimate_phase(
    num_counting_qubits: int,
    unitary: Gate,
    shots: int = 4096,
    prepare_one: bool = True,
) -> float:
    """Run QPE and return the estimated phase theta in [0, 1).

    The eigenvalue is exp(2*pi*i*theta). By default, prepares the target
    qubit in |1> (eigenstate with non-trivial phase for diagonal gates).
    """
    from quantumium.core.gate import X as X_gate

    total_qubits = num_counting_qubits + 1
    target = num_counting_qubits
    qc = QuantumCircuit(total_qubits)

    if prepare_one:
        qc.apply(X_gate, target)

    qpe = qpe_circuit(num_counting_qubits, unitary)
    for inst in qpe.instructions:
        qc.apply(inst.gate, *inst.target_qubits)

    counts = qc.measure_all(shots=shots)

    input_counts: dict[str, int] = {}
    for bitstring, count in counts.items():
        counting_bits = bitstring[1:]
        input_counts[counting_bits] = input_counts.get(counting_bits, 0) + count

    best = max(input_counts, key=input_counts.get)  # type: ignore[arg-type]
    measured_int = int(best[::-1], 2)
    return measured_int / (2**num_counting_qubits)
