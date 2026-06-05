"""Amazon Braket backend — translate Quantumium circuits to Braket and execute.

Provides access to multiple quantum hardware vendors (IonQ, Rigetti, OQC, QuEra)
through a single unified interface.

Requires: pip install amazon-braket-sdk
"""

from __future__ import annotations

import numpy as np

from ..core.circuit import QuantumCircuit
from .base import Backend, ExecutionResult


BRAKET_DEVICES = {
    "ionq-aria": "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
    "ionq-forte": "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
    "rigetti-ankaa": "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-2",
    "oqc-lucy": "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy",
    "quera-aquila": "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
    "sv1": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    "dm1": "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
    "tn1": "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
}


class BraketBackend(Backend):
    """Execute Quantumium circuits via Amazon Braket.

    Args:
        device: Device name (e.g., 'ionq-aria', 'sv1') or full ARN.
            Defaults to 'sv1' (state vector simulator).
        s3_bucket: S3 bucket for result storage (required for hardware).
        s3_prefix: S3 prefix for results.
    """

    def __init__(
        self,
        device: str = "sv1",
        s3_bucket: str | None = None,
        s3_prefix: str = "quantumium-results",
    ):
        self._device_name = device
        self._device_arn = BRAKET_DEVICES.get(device, device)
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix

    @property
    def name(self) -> str:
        return f"braket-{self._device_name}"

    def is_available(self) -> bool:
        try:
            import braket  # noqa: F401
            return True
        except ImportError:
            return False

    def _translate_circuit(self, circuit: QuantumCircuit):
        """Convert a Quantumium circuit to a Braket Circuit."""
        from braket.circuits import Circuit as BraketCircuit
        from braket.circuits.gates import Unitary

        braket_circuit = BraketCircuit()

        for inst in circuit.instructions:
            gate = inst.gate
            qubits = list(inst.target_qubits)
            braket_circuit.unitary(matrix=gate.matrix, targets=qubits)

        return braket_circuit

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> ExecutionResult:
        if not self.is_available():
            raise RuntimeError("Amazon Braket SDK is not installed. Run: pip install amazon-braket-sdk")

        from braket.aws import AwsDevice
        from braket.devices import LocalSimulator

        braket_circuit = self._translate_circuit(circuit)

        is_simulator = self._device_name in ("sv1", "dm1", "tn1")
        is_local = self._device_name == "local"

        if is_local:
            device = LocalSimulator()
            task = device.run(braket_circuit, shots=shots)
        elif is_simulator and not self._s3_bucket:
            device = LocalSimulator()
            task = device.run(braket_circuit, shots=shots)
        else:
            device = AwsDevice(self._device_arn)
            s3_location = (self._s3_bucket, self._s3_prefix) if self._s3_bucket else None
            task = device.run(braket_circuit, shots=shots, s3_destination_folder=s3_location)

        result = task.result()
        counts = dict(result.measurement_counts)
        str_counts = {k: int(v) for k, v in counts.items()}

        return ExecutionResult(
            counts=dict(sorted(str_counts.items())),
            shots=shots,
            backend_name=self.name,
            metadata={"device_arn": self._device_arn},
        )
