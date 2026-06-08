"""Pauli twirling for converting coherent errors to stochastic noise.

Pauli twirling inserts random Pauli gates before and after two-qubit
gates. This doesn't reduce total error rate but converts hard-to-handle
coherent errors into easier-to-handle stochastic (Pauli) channels,
improving the effectiveness of other mitigation techniques.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import I, X, Y, Z, CNOT

PAULIS = [I, X, Y, Z]

CNOT_TWIRL_TABLE: list[tuple[int, int, int, int]] = [
    (0, 0, 0, 0),  # I⊗I → I⊗I
    (0, 1, 0, 1),  # I⊗X → I⊗X
    (0, 2, 2, 2),  # I⊗Y → Z⊗Y  (CNOT propagation)
    (0, 3, 2, 3),  # I⊗Z → Z⊗Z
    (1, 0, 1, 1),  # X⊗I → X⊗X
    (1, 1, 1, 0),  # X⊗X → X⊗I
    (1, 2, 3, 3),  # X⊗Y → (-) Y⊗Z → need sign correction
    (1, 3, 3, 2),  # X⊗Z → Y⊗Y
    (2, 0, 2, 1),  # Y⊗I → Y⊗X
    (2, 1, 2, 0),  # Y⊗X → Y⊗I
    (2, 2, 0, 3),  # Y⊗Y → (-) I⊗Z → with correction
    (2, 3, 0, 2),  # Y⊗Z → I⊗Y → with correction
    (3, 0, 3, 0),  # Z⊗I → Z⊗I
    (3, 1, 3, 1),  # Z⊗X → Z⊗X
    (3, 2, 1, 2),  # Z⊗Y → X⊗Y
    (3, 3, 1, 3),  # Z⊗Z → X⊗Z
]


def pauli_twirl_cnot(
    circuit: QuantumCircuit,
    rng: np.random.Generator | None = None,
) -> QuantumCircuit:
    """Apply one random Pauli twirl to all CNOT gates in the circuit.

    For each CNOT gate, a random entry from the CNOT twirl table is
    selected. The corresponding Pauli gates P_a, P_b are inserted before
    the CNOT and the correction Paulis P_c, P_d are inserted after it,
    preserving the ideal unitary while randomizing coherent errors.

    Args:
        circuit: The quantum circuit containing CNOT gates to twirl.
        rng: Optional NumPy random generator for reproducibility.
            If None, a default generator is created.

    Returns:
        A new QuantumCircuit with Pauli twirling applied to every CNOT gate.
    """
    if rng is None:
        rng = np.random.default_rng()

    qc = QuantumCircuit(circuit.num_qubits)

    for inst in circuit.instructions:
        if inst.gate.name == "CNOT" and len(inst.target_qubits) == 2:
            idx = rng.integers(0, len(CNOT_TWIRL_TABLE))
            pa, pb, pc, pd = CNOT_TWIRL_TABLE[idx]

            ctrl, tgt = inst.target_qubits
            if pa != 0:
                qc.apply(PAULIS[pa], ctrl)
            if pb != 0:
                qc.apply(PAULIS[pb], tgt)

            qc.apply(CNOT, ctrl, tgt)

            if pc != 0:
                qc.apply(PAULIS[pc], ctrl)
            if pd != 0:
                qc.apply(PAULIS[pd], tgt)
        else:
            qc.apply(inst.gate, *inst.target_qubits)

    return qc


def twirled_circuit_ensemble(
    circuit: QuantumCircuit,
    num_samples: int = 20,
    seed: int | None = None,
) -> list[QuantumCircuit]:
    """Generate an ensemble of independently Pauli-twirled circuits.

    Each circuit in the ensemble has a different random Pauli twirl
    applied to its CNOT gates. Averaging results over the ensemble
    converts coherent errors into stochastic (Pauli) noise.

    Args:
        circuit: The base quantum circuit to twirl.
        num_samples: Number of twirled circuit copies to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        A list of ``num_samples`` twirled QuantumCircuit instances.
    """
    rng = np.random.default_rng(seed)
    return [pauli_twirl_cnot(circuit, rng) for _ in range(num_samples)]


def twirled_expectation(
    circuit: QuantumCircuit,
    observable_fn: Callable[[NDArray[np.complexfloating]], float],
    num_samples: int = 20,
    seed: int | None = None,
) -> float:
    """Estimate expectation value using Pauli twirling.

    Generates multiple twirled versions of the circuit, evaluates the
    observable on each, and returns the mean. For an ideal simulator
    this should match the untwirled result; on noisy hardware it
    converts coherent errors to stochastic noise.
    """
    ensemble = twirled_circuit_ensemble(circuit, num_samples, seed)
    values = [observable_fn(qc.statevector()) for qc in ensemble]
    return float(np.mean(values))
