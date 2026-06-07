"""Layer 3a: Circuit cutting — decompose circuits across wire cuts.

Implements Pauli-basis wire cutting (Peng et al. 2020) that decomposes a
circuit with K wire cuts into O(4^K) subcircuit evaluations. Each cut wire
is replaced by Pauli-basis state preparations and measurements, and the
full expectation value is reconstructed via classical postprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H, S, X, I as I_gate


PAULI_LABELS = ["I", "X", "Y", "Z"]

PAULI_MATRICES: dict[str, NDArray[np.complexfloating]] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

PAULI_PREP_STATES: dict[str, NDArray[np.complexfloating]] = {
    "0": np.array([1, 0], dtype=complex),
    "1": np.array([0, 1], dtype=complex),
    "+": np.array([1, 1], dtype=complex) / np.sqrt(2),
    "-": np.array([1, -1], dtype=complex) / np.sqrt(2),
    "+i": np.array([1, 1j], dtype=complex) / np.sqrt(2),
    "-i": np.array([1, -1j], dtype=complex) / np.sqrt(2),
}

PAULI_MEASURE_BASES = ["Z", "X", "Y"]


@dataclass
class WireCut:
    """Specification of a wire cut location."""

    qubit: int
    after_gate_index: int
    label: str = ""


@dataclass
class Subcircuit:
    """A subcircuit fragment produced by wire cutting."""

    circuit_id: int
    n_qubits: int
    gates: list[tuple[str, tuple[int, ...]]]
    prep_labels: dict[int, str] = field(default_factory=dict)
    meas_bases: dict[int, str] = field(default_factory=dict)


@dataclass
class CutResult:
    """Result of circuit cutting decomposition."""

    n_cuts: int
    n_subcircuits: int
    n_evaluations: int
    overhead_factor: float
    subcircuit_results: list[dict[str, float]] = field(default_factory=list)


class CircuitCutter:
    """Decomposes circuits by cutting wires using Pauli-basis decomposition.

    For each wire cut, the identity channel on the cut wire is decomposed:
    I = Σ_P (1/2) · P_prep ⊗ P_meas

    This yields 4 terms per cut (I, X, Y, Z basis), so K cuts produce
    4^K subcircuit configurations. Each configuration prepares a Pauli
    eigenstate on one side and measures in the corresponding basis on the other.
    """

    def __init__(self, cuts: list[WireCut] | None = None) -> None:
        self._cuts = cuts or []

    @property
    def n_cuts(self) -> int:
        return len(self._cuts)

    @property
    def overhead(self) -> float:
        return float(4**self.n_cuts)

    def add_cut(self, qubit: int, after_gate_index: int, label: str = "") -> None:
        """Add a wire cut specification."""
        self._cuts.append(WireCut(qubit, after_gate_index, label))

    def decompose(
        self,
        n_qubits: int,
        gates: list[tuple[Gate, tuple[int, ...]]],
    ) -> list[tuple[dict[int, str], dict[int, str], float]]:
        """Decompose circuit into subcircuit configurations.

        Returns list of (prep_config, meas_config, coefficient) tuples.
        Each configuration specifies what to prepare/measure at each cut point.
        """
        if not self._cuts:
            return [({}, {}, 1.0)]

        configs: list[tuple[dict[int, str], dict[int, str], float]] = []

        pauli_decomp = [
            ("0", "Z", 0.5),
            ("1", "Z", -0.5),
            ("+", "X", 0.5),
            ("-", "X", -0.5),
            ("+i", "Y", 0.5),
            ("-i", "Y", -0.5),
        ]

        cut_options = [pauli_decomp] * self.n_cuts

        for combo in product(*cut_options):
            prep_config: dict[int, str] = {}
            meas_config: dict[int, str] = {}
            coeff = 1.0

            for cut_idx, (prep, meas, c) in enumerate(combo):
                cut = self._cuts[cut_idx]
                prep_config[cut_idx] = prep
                meas_config[cut_idx] = meas
                coeff *= c

            configs.append((prep_config, meas_config, coeff))

        return configs

    def evaluate_cut_circuit(
        self,
        unitary_before: NDArray[np.complexfloating],
        unitary_after: NDArray[np.complexfloating],
        cut_qubit: int,
        n_qubits: int,
        observable: NDArray[np.complexfloating] | None = None,
    ) -> float:
        """Evaluate a circuit with one wire cut, summing over Pauli decomposition.

        Parameters
        ----------
        unitary_before
            Unitary of the circuit portion before the cut.
        unitary_after
            Unitary of the circuit portion after the cut.
        cut_qubit
            Which qubit is being cut.
        n_qubits
            Total number of qubits.
        observable
            Observable to measure (defaults to |0...0⟩⟨0...0|).
        """
        dim = 2**n_qubits
        if observable is None:
            observable = np.zeros((dim, dim), dtype=complex)
            observable[0, 0] = 1.0

        result = 0.0

        for label, pauli in PAULI_MATRICES.items():
            expanded_pauli = _expand_single_qubit_op(pauli, cut_qubit, n_qubits)

            rho_init = np.zeros((dim, dim), dtype=complex)
            rho_init[0, 0] = 1.0

            rho_after_first = unitary_before @ rho_init @ unitary_before.conj().T

            rho_with_pauli = expanded_pauli @ rho_after_first @ expanded_pauli.conj().T

            rho_final = unitary_after @ rho_with_pauli @ unitary_after.conj().T

            expectation = np.real(np.trace(observable @ rho_final))
            result += 0.5 * expectation

        return float(result)

    def estimate_resources(self, n_cuts: int | None = None) -> dict[str, float]:
        """Estimate computational overhead for circuit cutting."""
        k = n_cuts if n_cuts is not None else self.n_cuts
        return {
            "n_cuts": float(k),
            "n_subcircuit_evals": float(6**k),
            "overhead_factor": float(4**k),
            "classical_postprocessing": float(4**k),
        }


class CutRecombiner:
    """Recombines subcircuit evaluation results to reconstruct the full expectation value."""

    @staticmethod
    def recombine(
        subcircuit_expectations: list[tuple[float, float]],
    ) -> float:
        """Recombine subcircuit results.

        Parameters
        ----------
        subcircuit_expectations
            List of (coefficient, expectation_value) pairs from each
            subcircuit configuration.

        Returns
        -------
        Reconstructed full-circuit expectation value.
        """
        return sum(coeff * exp_val for coeff, exp_val in subcircuit_expectations)

    @staticmethod
    def recombine_probabilities(
        subcircuit_probs: list[tuple[float, dict[str, float]]],
    ) -> dict[str, float]:
        """Recombine probability distributions from subcircuits.

        Parameters
        ----------
        subcircuit_probs
            List of (coefficient, probability_dict) pairs.

        Returns
        -------
        Reconstructed probability distribution.
        """
        combined: dict[str, float] = {}
        for coeff, probs in subcircuit_probs:
            for bitstring, prob in probs.items():
                combined[bitstring] = combined.get(bitstring, 0.0) + coeff * prob

        total = sum(abs(v) for v in combined.values())
        if total > 1e-15:
            combined = {k: max(0.0, v) for k, v in combined.items()}
            total_pos = sum(combined.values())
            if total_pos > 1e-15:
                combined = {k: v / total_pos for k, v in combined.items()}

        return combined

    @staticmethod
    def verify_reconstruction(
        original: float,
        reconstructed: float,
        tolerance: float = 0.1,
    ) -> bool:
        """Check if reconstruction matches original within tolerance."""
        return abs(original - reconstructed) <= tolerance


def _expand_single_qubit_op(
    op: NDArray[np.complexfloating],
    qubit: int,
    n_qubits: int,
) -> NDArray[np.complexfloating]:
    """Expand a single-qubit operator to the full Hilbert space."""
    result = np.array([[1.0]], dtype=complex)
    for q in range(n_qubits):
        if q == qubit:
            result = np.kron(result, op)
        else:
            result = np.kron(result, np.eye(2, dtype=complex))
    return result
