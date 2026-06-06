"""Tests for QiskitBackend — skip gracefully if qiskit not installed."""

from __future__ import annotations

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, CNOT
from quantumium.backends import LocalSimulatorBackend, ExecutionResult
from quantumium.backends.qiskit_backend import QiskitBackend

from .conftest import (
    requires_qiskit,
    hardware,
    chi_squared_test,
    hellinger_fidelity,
)


SHOTS = 4096


class TestQiskitBackendMeta:
    def test_name_simulator(self):
        backend = QiskitBackend(use_hardware=False)
        assert backend.name == "qiskit-aer-simulator"

    def test_name_hardware(self):
        backend = QiskitBackend(use_hardware=True, backend_name="ibm_brisbane")
        assert backend.name == "qiskit-ibm-ibm_brisbane"

    def test_name_hardware_auto(self):
        backend = QiskitBackend(use_hardware=True)
        assert backend.name == "qiskit-ibm-auto"

    def test_availability_returns_bool(self):
        backend = QiskitBackend()
        assert isinstance(backend.is_available(), bool)


@requires_qiskit
class TestQiskitSimulator:
    def test_bell_state(self, bell_circuit):
        backend = QiskitBackend()
        result = backend.run(bell_circuit, shots=SHOTS)
        assert isinstance(result, ExecutionResult)
        assert result.shots == SHOTS
        for k in result.counts:
            assert k in {"00", "11"}

    def test_bell_state_chi_squared(self, bell_circuit):
        backend = QiskitBackend()
        result = backend.run(bell_circuit, shots=SHOTS)
        expected = {"00": 0.5, "11": 0.5}
        p_value = chi_squared_test(result.counts, expected, SHOTS)
        assert p_value > 0.01

    def test_ghz3(self, ghz3_circuit):
        backend = QiskitBackend()
        result = backend.run(ghz3_circuit, shots=SHOTS)
        for k in result.counts:
            assert k in {"000", "111"}

    def test_x_gate_deterministic(self, x_gate_circuit):
        backend = QiskitBackend()
        result = backend.run(x_gate_circuit, shots=1000)
        assert result.counts.get("1", 0) == 1000

    def test_qft_roundtrip(self, qft_roundtrip_circuit):
        backend = QiskitBackend()
        result = backend.run(qft_roundtrip_circuit, shots=SHOTS)
        p_zero = result.probabilities.get("000", 0)
        assert p_zero > 0.95

    def test_grover_marked_state(self, grover3_circuit):
        backend = QiskitBackend()
        result = backend.run(grover3_circuit, shots=SHOTS)
        assert result.most_frequent == "101"

    def test_cross_validate_with_local(self, bell_circuit):
        local = LocalSimulatorBackend()
        qiskit = QiskitBackend()
        r_local = local.run(bell_circuit, shots=SHOTS)
        r_qiskit = qiskit.run(bell_circuit, shots=SHOTS)
        fid = hellinger_fidelity(r_local.probabilities, r_qiskit.probabilities)
        assert fid > 0.99, f"Cross-backend fidelity too low: {fid:.4f}"


@requires_qiskit
class TestQiskitTranslation:
    def test_circuit_translation_preserves_qubit_count(self, bell_circuit):
        backend = QiskitBackend()
        qiskit_circuit = backend._translate_circuit(bell_circuit)
        assert qiskit_circuit.num_qubits == 2

    def test_circuit_translation_has_measurements(self, bell_circuit):
        backend = QiskitBackend()
        qiskit_circuit = backend._translate_circuit(bell_circuit)
        from qiskit.circuit import Measure
        has_measure = any(
            isinstance(inst.operation, Measure)
            for inst in qiskit_circuit.data
        )
        assert has_measure


class TestQiskitNotInstalled:
    def test_run_raises_without_sdk(self, bell_circuit):
        backend = QiskitBackend()
        if not backend.is_available():
            with pytest.raises(RuntimeError, match="Qiskit is not installed"):
                backend.run(bell_circuit, shots=100)
