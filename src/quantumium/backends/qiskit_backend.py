"""Qiskit backend — translate Quantumium circuits to Qiskit and execute."""

from __future__ import annotations

import numpy as np

from ..core.circuit import QuantumCircuit
from .base import Backend, ExecutionResult


class QiskitBackend(Backend):
    """Execute Quantumium circuits via IBM Qiskit.

    Supports both Aer local simulator and IBM Quantum cloud hardware.
    Requires: pip install qiskit qiskit-aer qiskit-ibm-runtime
    """

    def __init__(self, use_hardware: bool = False, backend_name: str | None = None):
        self._use_hardware = use_hardware
        self._backend_name = backend_name
        self._qiskit = None
        self._aer = None

    @property
    def name(self) -> str:
        if self._use_hardware:
            return f"qiskit-ibm-{self._backend_name or 'auto'}"
        return "qiskit-aer-simulator"

    def is_available(self) -> bool:
        try:
            import qiskit  # noqa: F401
            return True
        except ImportError:
            return False

    def _translate_circuit(self, circuit: QuantumCircuit):
        """Convert a Quantumium circuit to a Qiskit QuantumCircuit."""
        from qiskit import QuantumCircuit as QiskitCircuit
        from qiskit.quantum_info import Operator

        qc = QiskitCircuit(circuit.num_qubits, circuit.num_qubits)

        for inst in circuit.instructions:
            gate = inst.gate
            qubits = list(inst.target_qubits)
            op = Operator(gate.matrix)
            qc.unitary(op, qubits, label=gate.name)

        qc.measure(range(circuit.num_qubits), range(circuit.num_qubits))
        return qc

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> ExecutionResult:
        if not self.is_available():
            raise RuntimeError("Qiskit is not installed. Run: pip install qiskit qiskit-aer")

        qiskit_circuit = self._translate_circuit(circuit)

        if self._use_hardware:
            counts = self._run_hardware(qiskit_circuit, shots)
        else:
            counts = self._run_simulator(qiskit_circuit, shots)

        return ExecutionResult(
            counts=dict(sorted(counts.items())),
            shots=shots,
            backend_name=self.name,
        )

    def _run_simulator(self, qiskit_circuit, shots: int) -> dict[str, int]:
        from qiskit_aer import AerSimulator
        simulator = AerSimulator()
        result = simulator.run(qiskit_circuit, shots=shots).result()
        return dict(result.get_counts())

    def _run_hardware(self, qiskit_circuit, shots: int) -> dict[str, int]:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        service = QiskitRuntimeService()
        backend = service.least_busy(
            simulator=False,
            min_num_qubits=qiskit_circuit.num_qubits,
        ) if not self._backend_name else service.backend(self._backend_name)

        sampler = SamplerV2(backend)
        job = sampler.run([qiskit_circuit], shots=shots)
        result = job.result()
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
        return dict(counts)
