"""Google Cirq backend — translate Quantumium circuits to Cirq and execute."""

from __future__ import annotations

import numpy as np

from ..core.circuit import QuantumCircuit
from .base import Backend, ExecutionResult


class CirqBackend(Backend):
    """Execute Quantumium circuits via Google Cirq.

    Supports Cirq's built-in simulator. For Google Quantum hardware,
    use with Cirq's Engine API (requires Google Cloud credentials).

    Requires: pip install cirq
    """

    def __init__(self, use_hardware: bool = False, processor_id: str | None = None):
        """Initialize the Cirq backend.

        Args:
            use_hardware (bool): If ``True``, execute on Google Quantum
                hardware via the Cirq Engine API. If ``False`` (default),
                use Cirq's built-in local simulator.
            processor_id (str | None): Google Quantum processor name
                (e.g., ``"rainbow"``). When *None* and *use_hardware* is
                ``True``, defaults to ``"rainbow"``.
        """
        self._use_hardware = use_hardware
        self._processor_id = processor_id

    @property
    def name(self) -> str:
        """Return the human-readable backend name.

        Returns:
            str: ``"cirq-simulator"`` for local simulation, or
                ``"cirq-google-<processor_id>"`` for hardware execution.
        """
        if self._use_hardware:
            return f"cirq-google-{self._processor_id or 'auto'}"
        return "cirq-simulator"

    def is_available(self) -> bool:
        """Check whether the Cirq SDK is importable.

        Returns:
            bool: ``True`` if ``cirq`` can be imported, ``False`` otherwise.
        """
        try:
            import cirq  # noqa: F401
            return True
        except ImportError:
            return False

    def _translate_circuit(self, circuit: QuantumCircuit):
        """Convert a Quantumium circuit to a Cirq Circuit.

        Each Quantumium gate is wrapped in a ``cirq.MatrixGate`` and applied
        to the corresponding ``LineQubit`` targets. A measurement gate keyed
        ``'result'`` is appended to all qubits.

        Args:
            circuit (QuantumCircuit): The Quantumium circuit to translate.

        Returns:
            tuple[cirq.Circuit, list[cirq.LineQubit]]: A two-element tuple
                containing the translated Cirq circuit and the list of line
                qubits used.
        """
        import cirq

        qubits = cirq.LineQubit.range(circuit.num_qubits)

        moments = []
        for inst in circuit.instructions:
            gate = inst.gate
            target_qubits = [qubits[q] for q in inst.target_qubits]
            cirq_gate = cirq.MatrixGate(gate.matrix)
            moments.append(cirq_gate.on(*target_qubits))

        cirq_circuit = cirq.Circuit(moments)
        cirq_circuit.append(cirq.measure(*qubits, key='result'))
        return cirq_circuit, qubits

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> ExecutionResult:
        """Execute a Quantumium circuit via Google Cirq.

        Translates the circuit to Cirq format and dispatches to either the
        local Cirq simulator or Google Quantum hardware depending on the
        ``use_hardware`` flag.

        Args:
            circuit (QuantumCircuit): The Quantumium circuit to execute.
            shots (int): Number of measurement repetitions. Defaults to 1024.

        Returns:
            ExecutionResult: Sorted measurement counts and metadata.

        Raises:
            RuntimeError: If the Cirq SDK is not installed.
        """
        if not self.is_available():
            raise RuntimeError("Cirq is not installed. Run: pip install cirq")

        import cirq

        cirq_circuit, qubits = self._translate_circuit(circuit)

        if self._use_hardware:
            counts = self._run_hardware(cirq_circuit, shots)
        else:
            simulator = cirq.Simulator()
            result = simulator.run(cirq_circuit, repetitions=shots)
            counts = self._result_to_counts(result, circuit.num_qubits)

        return ExecutionResult(
            counts=dict(sorted(counts.items())),
            shots=shots,
            backend_name=self.name,
        )

    def _result_to_counts(self, result, num_qubits: int) -> dict[str, int]:
        """Convert a Cirq measurement result to bitstring counts.

        Iterates over each measurement row, converts the binary array to a
        bitstring, and tallies the occurrences.

        Args:
            result: A Cirq ``Result`` object containing measurement data
                under the key ``'result'``.
            num_qubits (int): Number of qubits in the circuit (used for
                context; bitstring length is inferred from the data).

        Returns:
            dict[str, int]: Mapping of bitstrings to observation counts.
        """
        import pandas as pd

        measurements = result.measurements['result']
        counts: dict[str, int] = {}
        for row in measurements:
            bitstring = ''.join(str(int(b)) for b in row)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def _run_hardware(self, cirq_circuit, shots: int) -> dict[str, int]:
        """Execute the circuit on Google Quantum hardware via the Cirq Engine.

        Connects to the Google Quantum Engine, selects the configured
        processor (or defaults to ``"rainbow"``), and submits the circuit.

        Args:
            cirq_circuit: A Cirq ``Circuit`` ready for execution.
            shots (int): Number of measurement repetitions.

        Returns:
            dict[str, int]: Mapping of bitstrings to observation counts.
        """
        import cirq_google

        engine = cirq_google.get_engine()
        processor = engine.get_processor(self._processor_id or 'rainbow')
        result = processor.run(cirq_circuit, repetitions=shots)
        return self._result_to_counts(result, cirq_circuit.num_qubits)
