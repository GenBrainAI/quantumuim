"""Quantum Phase Estimation (QPE) algorithm.

QPE estimates the eigenvalue phase of a unitary operator. Given a unitary U
with eigenvector |u> such that U|u> = exp(2*pi*i*theta)|u>, QPE estimates
theta to n-bit precision using n counting qubits.

This module provides:

- :func:`controlled_unitary` -- wraps a gate with a control qubit.
- :func:`unitary_power` -- computes U^power via matrix exponentiation.
- :func:`qpe_circuit` -- builds the QPE circuit for a single-qubit unitary.
- :func:`estimate_phase` -- runs QPE and returns the estimated phase value.

Example::

    >>> import numpy as np
    >>> from quantumium.core.gate import Gate
    >>> from quantumium.algorithms.qpe import estimate_phase
    >>> # T gate has eigenvalue exp(2*pi*i/8) on |1>, so theta = 1/8
    >>> T = Gate(np.diag([1, np.exp(1j * np.pi / 4)]), "T")
    >>> phase = estimate_phase(4, T)
    >>> abs(phase - 0.125) < 0.01
    True
"""

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
    """Create a controlled version of a unitary gate.

    Constructs a two-register gate where the first qubit acts as the
    control and the remaining qubits are the target. The unitary is
    applied to the target register only when the control qubit is |1>.

    Parameters
    ----------
    unitary : Gate
        The gate to be controlled. Can operate on any number of qubits.

    Returns
    -------
    Gate
        A controlled gate on (1 + unitary.num_qubits) qubits. The matrix
        is block-diagonal: identity on the |0>-control subspace and
        ``unitary.matrix`` on the |1>-control subspace.
    """
    n = unitary.matrix.shape[0]
    dim = 2 * n
    matrix = np.eye(dim, dtype=complex)
    matrix[n:, n:] = unitary.matrix
    return Gate(matrix, f"C-{unitary.name}")


def unitary_power(unitary: Gate, power: int) -> Gate:
    """Compute U^power for a unitary gate via matrix exponentiation.

    Parameters
    ----------
    unitary : Gate
        The base unitary gate.
    power : int
        The integer exponent.

    Returns
    -------
    Gate
        A gate whose matrix is ``unitary.matrix`` raised to the given power.
    """
    result = np.linalg.matrix_power(unitary.matrix, power)
    return Gate(result, f"{unitary.name}^{power}")


def qpe_circuit(
    num_counting_qubits: int,
    unitary: Gate,
) -> QuantumCircuit:
    """Build a Quantum Phase Estimation circuit.

    Constructs the standard QPE circuit: Hadamard gates on the counting
    register, controlled-U^(2^k) operations, and an inverse QFT on the
    counting register.

    Parameters
    ----------
    num_counting_qubits : int
        Number of counting qubits, determining the phase precision to
        n bits (precision ~ 1/2^n).
    unitary : Gate
        The unitary gate whose eigenvalue phase is to be estimated.
        Must be a single-qubit gate. The eigenstate should be prepared
        on the target qubit externally before combining with this circuit.

    Returns
    -------
    QuantumCircuit
        A circuit with ``num_counting_qubits + 1`` total qubits.
        Qubits 0 through n-1 are counting qubits; qubit n is the target.
        After measurement, the counting qubits encode the phase as
        theta approximately equal to measured_value / 2^n.

    Raises
    ------
    ValueError
        If ``unitary`` is not a single-qubit gate.
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

    Builds and executes the full QPE circuit, then extracts the most likely
    phase value from measurement statistics. The eigenvalue of the unitary
    is exp(2*pi*i*theta).

    Parameters
    ----------
    num_counting_qubits : int
        Number of counting qubits for phase precision.
    unitary : Gate
        Single-qubit unitary gate whose eigenvalue phase to estimate.
    shots : int, optional
        Number of measurement shots (default 4096).
    prepare_one : bool, optional
        If True (default), prepares the target qubit in |1>, which is the
        eigenstate with non-trivial phase for diagonal gates. Set to False
        to leave the target in |0>.

    Returns
    -------
    float
        The estimated phase theta in the range [0, 1). The corresponding
        eigenvalue is exp(2*pi*i*theta).
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
    return measured_int / (2**num_counting_qubits)  # type: ignore[no-any-return]
