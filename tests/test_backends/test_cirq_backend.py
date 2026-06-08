"""Tests for CirqBackend — skip gracefully if cirq not installed."""

from __future__ import annotations

import pytest

from quantumium.backends import LocalSimulatorBackend, ExecutionResult
from quantumium.backends.cirq_backend import CirqBackend

from .conftest import (
    requires_cirq,
    chi_squared_test,
    hellinger_fidelity,
)


SHOTS = 4096


class TestCirqBackendMeta:
    def test_name_simulator(self):
        backend = CirqBackend(use_hardware=False)
        assert backend.name == "cirq-simulator"

    def test_name_hardware(self):
        backend = CirqBackend(use_hardware=True, processor_id="rainbow")
        assert backend.name == "cirq-google-rainbow"

    def test_name_hardware_auto(self):
        backend = CirqBackend(use_hardware=True)
        assert backend.name == "cirq-google-auto"

    def test_availability_returns_bool(self):
        backend = CirqBackend()
        assert isinstance(backend.is_available(), bool)


@requires_cirq
class TestCirqSimulator:
    def test_bell_state(self, bell_circuit):
        backend = CirqBackend()
        result = backend.run(bell_circuit, shots=SHOTS)
        assert isinstance(result, ExecutionResult)
        assert result.shots == SHOTS
        for k in result.counts:
            assert k in {"00", "11"}

    def test_bell_state_chi_squared(self, bell_circuit):
        backend = CirqBackend()
        result = backend.run(bell_circuit, shots=SHOTS)
        expected = {"00": 0.5, "11": 0.5}
        p_value = chi_squared_test(result.counts, expected, SHOTS)
        assert p_value > 0.01

    def test_ghz3(self, ghz3_circuit):
        backend = CirqBackend()
        result = backend.run(ghz3_circuit, shots=SHOTS)
        for k in result.counts:
            assert k in {"000", "111"}

    def test_x_gate_deterministic(self, x_gate_circuit):
        backend = CirqBackend()
        result = backend.run(x_gate_circuit, shots=1000)
        assert result.counts.get("1", 0) == 1000

    def test_qft_roundtrip(self, qft_roundtrip_circuit):
        backend = CirqBackend()
        result = backend.run(qft_roundtrip_circuit, shots=SHOTS)
        p_zero = result.probabilities.get("000", 0)
        assert p_zero > 0.95

    def test_grover_marked_state(self, grover3_circuit):
        backend = CirqBackend()
        result = backend.run(grover3_circuit, shots=SHOTS)
        assert result.most_frequent == "101"

    def test_cross_validate_with_local(self, bell_circuit):
        local = LocalSimulatorBackend()
        cirq_be = CirqBackend()
        r_local = local.run(bell_circuit, shots=SHOTS)
        r_cirq = cirq_be.run(bell_circuit, shots=SHOTS)
        fid = hellinger_fidelity(r_local.probabilities, r_cirq.probabilities)
        assert fid > 0.99, f"Cross-backend fidelity too low: {fid:.4f}"


@requires_cirq
class TestCirqTranslation:
    def test_circuit_translation_returns_tuple(self, bell_circuit):
        backend = CirqBackend()
        result = backend._translate_circuit(bell_circuit)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_circuit_translation_qubit_count(self, ghz3_circuit):
        backend = CirqBackend()
        cirq_circuit, qubits = backend._translate_circuit(ghz3_circuit)
        assert len(qubits) == 3


class TestCirqNotInstalled:
    def test_run_raises_without_sdk(self, bell_circuit):
        backend = CirqBackend()
        if not backend.is_available():
            with pytest.raises(RuntimeError, match="Cirq is not installed"):
                backend.run(bell_circuit, shots=100)
