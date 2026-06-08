"""Qiskit backend — translate Quantumium circuits to Qiskit and execute."""

from __future__ import annotations


from ..core.circuit import QuantumCircuit
from .base import Backend, ExecutionResult


class QiskitBackend(Backend):
    """Execute Quantumium circuits via IBM Qiskit.

    Supports both Aer local simulator and IBM Quantum cloud hardware.
    Requires: pip install qiskit qiskit-aer qiskit-ibm-runtime
    """

    def __init__(self, use_hardware: bool = False, backend_name: str | None = None):
        """Initialize the Qiskit backend.

        Args:
            use_hardware (bool): If ``True``, execute on IBM Quantum cloud
                hardware via ``qiskit-ibm-runtime``. If ``False`` (default),
                use the local Aer simulator.
            backend_name (str | None): Explicit IBM backend name
                (e.g., ``"ibm_brisbane"``). When *None* and *use_hardware*
                is ``True``, the least-busy device with enough qubits is
                selected automatically.
        """
        self._use_hardware = use_hardware
        self._backend_name = backend_name
        self._qiskit = None
        self._aer = None

    @property
    def name(self) -> str:
        """Return the human-readable backend name.

        Returns:
            str: ``"qiskit-aer-simulator"`` for local simulation, or
                ``"qiskit-ibm-<backend>"`` for hardware execution.
        """
        if self._use_hardware:
            return f"qiskit-ibm-{self._backend_name or 'auto'}"
        return "qiskit-aer-simulator"

    def is_available(self) -> bool:
        """Check whether the Qiskit SDK is importable.

        Returns:
            bool: ``True`` if ``qiskit`` can be imported, ``False`` otherwise.
        """
        try:
            import qiskit  # noqa: F401

            return True
        except ImportError:
            return False

    def _translate_circuit(self, circuit: QuantumCircuit):
        """Convert a Quantumium circuit to a Qiskit QuantumCircuit.

        Each Quantumium gate is mapped to a Qiskit ``Operator`` unitary and
        appended to the Qiskit circuit. A full measurement barrier is added
        at the end.

        Args:
            circuit (QuantumCircuit): The Quantumium circuit to translate.

        Returns:
            qiskit.QuantumCircuit: The equivalent Qiskit circuit with
                measurement operations appended on all qubits.
        """
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
        """Execute a Quantumium circuit via Qiskit.

        Translates the circuit to Qiskit format and dispatches to either the
        local Aer simulator or IBM Quantum hardware depending on the
        ``use_hardware`` flag.

        Args:
            circuit (QuantumCircuit): The Quantumium circuit to execute.
            shots (int): Number of measurement repetitions. Defaults to 1024.

        Returns:
            ExecutionResult: Sorted measurement counts and metadata.

        Raises:
            RuntimeError: If the Qiskit SDK is not installed.
        """
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
        """Execute the circuit on the local Aer simulator.

        Args:
            qiskit_circuit: A Qiskit ``QuantumCircuit`` ready for execution.
            shots (int): Number of measurement repetitions.

        Returns:
            dict[str, int]: Mapping of bitstrings to observation counts.
        """
        from qiskit_aer import AerSimulator

        simulator = AerSimulator()
        result = simulator.run(qiskit_circuit, shots=shots).result()
        return dict(result.get_counts())

    def _run_hardware(self, qiskit_circuit, shots: int) -> dict[str, int]:
        """Execute the circuit on IBM Quantum cloud hardware.

        Uses ``qiskit-ibm-runtime`` to submit the circuit via the
        ``SamplerV2`` primitive. If no specific backend was configured,
        the least-busy device with sufficient qubits is selected.

        Args:
            qiskit_circuit: A Qiskit ``QuantumCircuit`` ready for execution.
            shots (int): Number of measurement repetitions.

        Returns:
            dict[str, int]: Mapping of bitstrings to observation counts.
        """
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        service = QiskitRuntimeService()
        backend = (
            service.least_busy(
                simulator=False,
                min_num_qubits=qiskit_circuit.num_qubits,
            )
            if not self._backend_name
            else service.backend(self._backend_name)
        )

        sampler = SamplerV2(backend)
        job = sampler.run([qiskit_circuit], shots=shots)
        result = job.result()
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
        return dict(counts)
