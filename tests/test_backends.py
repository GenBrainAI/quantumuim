"""Tests for backend abstraction layer."""

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT, X
from quantumium.backends import LocalSimulatorBackend, ExecutionResult
from quantumium.backends.qiskit_backend import QiskitBackend


class TestLocalSimulatorBackend:
    def test_name(self):
        backend = LocalSimulatorBackend()
        assert backend.name == "quantumium-local-sim"

    def test_is_available(self):
        backend = LocalSimulatorBackend()
        assert backend.is_available()

    def test_run_bell_state(self):
        backend = LocalSimulatorBackend()
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        result = backend.run(qc, shots=1000)
        assert isinstance(result, ExecutionResult)
        assert result.shots == 1000
        assert result.backend_name == "quantumium-local-sim"
        assert set(result.counts.keys()).issubset({"00", "01", "10", "11"})
        assert "00" in result.counts or "11" in result.counts

    def test_probabilities(self):
        backend = LocalSimulatorBackend()
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        result = backend.run(qc, shots=100)
        probs = result.probabilities
        assert abs(probs.get("1", 0) - 1.0) < 0.01

    def test_most_frequent(self):
        backend = LocalSimulatorBackend()
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        result = backend.run(qc, shots=100)
        assert result.most_frequent == "1"


class TestQiskitBackend:
    def test_availability_check(self):
        backend = QiskitBackend()
        available = backend.is_available()
        assert isinstance(available, bool)

    def test_name_simulator(self):
        backend = QiskitBackend(use_hardware=False)
        assert backend.name == "qiskit-aer-simulator"

    def test_name_hardware(self):
        backend = QiskitBackend(use_hardware=True, backend_name="ibm_brisbane")
        assert "ibm_brisbane" in backend.name
