"""CircuitAnalyzer — extract logical resource counts from quantum circuits."""

from __future__ import annotations

import math
import re

from quantumium.core.circuit import QuantumCircuit
from quantumium.resource_estimation.resources import LogicalResources

T_GATES = {"T", "Tdg", "T†"}
CLIFFORD_GATES = {"I", "X", "Y", "Z", "H", "S", "Sdg", "S†", "CNOT", "CZ", "SWAP"}
ROTATION_PATTERN = re.compile(r"^C?R[xyz]\(")
TOFFOLI_T_COUNT = 7


class CircuitAnalyzer:
    """Extract logical resource counts from a quantum circuit."""

    def __init__(
        self,
        rotation_precision: float = 1e-10,
    ) -> None:
        self.rotation_precision = rotation_precision

    def _rotation_t_cost(self) -> int:
        return max(1, round(3.0 * math.log2(1.0 / self.rotation_precision)))

    def _is_t_gate(self, name: str) -> bool:
        return name in T_GATES

    def _is_clifford(self, name: str) -> bool:
        return name in CLIFFORD_GATES

    def _is_rotation(self, name: str) -> bool:
        return bool(ROTATION_PATTERN.match(name))

    def _is_toffoli(self, name: str) -> bool:
        return name in ("CCX", "Toffoli", "CCZ")

    def t_count(self, circuit: QuantumCircuit) -> int:
        count = 0
        for inst in circuit.instructions:
            name = inst.gate.name
            if self._is_t_gate(name):
                count += 1
            elif self._is_toffoli(name):
                count += TOFFOLI_T_COUNT
        return count

    def rotation_count(self, circuit: QuantumCircuit) -> int:
        count = 0
        for inst in circuit.instructions:
            if self._is_rotation(inst.gate.name):
                count += 1
        return count

    def t_depth(self, circuit: QuantumCircuit) -> int:
        n = circuit.num_qubits
        qubit_t_depth = [0] * n
        for inst in circuit.instructions:
            name = inst.gate.name
            is_t = self._is_t_gate(name) or self._is_toffoli(name)
            if is_t:
                max_depth = max(qubit_t_depth[q] for q in inst.target_qubits)
                new_depth = max_depth + 1
                for q in inst.target_qubits:
                    qubit_t_depth[q] = new_depth
            else:
                shared = max(qubit_t_depth[q] for q in inst.target_qubits)
                for q in inst.target_qubits:
                    qubit_t_depth[q] = shared
        return max(qubit_t_depth) if qubit_t_depth else 0

    def clifford_count(self, circuit: QuantumCircuit) -> dict[str, int]:
        counts: dict[str, int] = {}
        for inst in circuit.instructions:
            name = inst.gate.name
            if self._is_clifford(name):
                counts[name] = counts.get(name, 0) + 1
        return counts

    def gate_counts(self, circuit: QuantumCircuit) -> dict[str, int]:
        counts: dict[str, int] = {}
        for inst in circuit.instructions:
            name = inst.gate.name
            counts[name] = counts.get(name, 0) + 1
        return counts

    def analyze(self, circuit: QuantumCircuit) -> LogicalResources:
        counts = self.gate_counts(circuit)
        n_t = self.t_count(circuit)
        n_rot = self.rotation_count(circuit)
        rot_cost = n_rot * self._rotation_t_cost()
        n_cliff = sum(c for name, c in counts.items() if self._is_clifford(name))
        n_meas = counts.get("MEASURE", 0)

        return LogicalResources(
            n_qubits=circuit.num_qubits,
            n_t_gates=n_t,
            t_depth=self.t_depth(circuit),
            n_clifford=n_cliff,
            n_measurements=n_meas,
            circuit_depth=circuit.depth,
            gate_counts=counts,
            n_rotation_gates=n_rot,
            rotation_t_cost=rot_cost,
        )
