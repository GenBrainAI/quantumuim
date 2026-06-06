"""Tests for BraketBackend — skip gracefully if braket not installed."""

from __future__ import annotations

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, CNOT
from quantumium.backends import LocalSimulatorBackend, ExecutionResult
from quantumium.backends.braket_backend import BraketBackend, BRAKET_DEVICES

from .conftest import (
    requires_braket,
    hardware,
    chi_squared_test,
    hellinger_fidelity,
)


SHOTS = 4096


class TestBraketBackendMeta:
    def test_name_default(self):
        backend = BraketBackend()
        assert backend.name == "braket-sv1"

    def test_name_ionq(self):
        backend = BraketBackend(device="ionq-aria")
        assert backend.name == "braket-ionq-aria"

    def test_name_custom_arn(self):
        arn = "arn:aws:braket:::device/custom/test"
        backend = BraketBackend(device=arn)
        assert backend.name == f"braket-{arn}"

    def test_device_arn_lookup(self):
        for alias, arn in BRAKET_DEVICES.items():
            backend = BraketBackend(device=alias)
            assert backend._device_arn == arn

    def test_custom_arn_passthrough(self):
        custom = "arn:aws:braket:::device/custom/my-device"
        backend = BraketBackend(device=custom)
        assert backend._device_arn == custom

    def test_availability_returns_bool(self):
        backend = BraketBackend()
        assert isinstance(backend.is_available(), bool)

    def test_all_known_devices(self):
        expected_devices = {
            "ionq-aria", "ionq-forte", "rigetti-ankaa",
            "oqc-lucy", "quera-aquila", "sv1", "dm1", "tn1",
        }
        assert set(BRAKET_DEVICES.keys()) == expected_devices


@requires_braket
class TestBraketSimulator:
    def test_bell_state(self, bell_circuit):
        backend = BraketBackend(device="local")
        result = backend.run(bell_circuit, shots=SHOTS)
        assert isinstance(result, ExecutionResult)
        assert result.shots == SHOTS
        for k in result.counts:
            assert k in {"00", "11"}

    def test_bell_state_chi_squared(self, bell_circuit):
        backend = BraketBackend(device="local")
        result = backend.run(bell_circuit, shots=SHOTS)
        expected = {"00": 0.5, "11": 0.5}
        p_value = chi_squared_test(result.counts, expected, SHOTS)
        assert p_value > 0.01

    def test_ghz3(self, ghz3_circuit):
        backend = BraketBackend(device="local")
        result = backend.run(ghz3_circuit, shots=SHOTS)
        for k in result.counts:
            assert k in {"000", "111"}

    def test_x_gate_deterministic(self, x_gate_circuit):
        backend = BraketBackend(device="local")
        result = backend.run(x_gate_circuit, shots=1000)
        assert result.counts.get("1", 0) == 1000

    def test_qft_roundtrip(self, qft_roundtrip_circuit):
        backend = BraketBackend(device="local")
        result = backend.run(qft_roundtrip_circuit, shots=SHOTS)
        p_zero = result.probabilities.get("000", 0)
        assert p_zero > 0.95

    def test_grover_marked_state(self, grover3_circuit):
        backend = BraketBackend(device="local")
        result = backend.run(grover3_circuit, shots=SHOTS)
        assert result.most_frequent == "101"

    def test_cross_validate_with_local(self, bell_circuit):
        local = LocalSimulatorBackend()
        braket = BraketBackend(device="local")
        r_local = local.run(bell_circuit, shots=SHOTS)
        r_braket = braket.run(bell_circuit, shots=SHOTS)
        fid = hellinger_fidelity(r_local.probabilities, r_braket.probabilities)
        assert fid > 0.99, f"Cross-backend fidelity too low: {fid:.4f}"

    def test_metadata_has_device_arn(self, bell_circuit):
        backend = BraketBackend(device="local")
        result = backend.run(bell_circuit, shots=100)
        assert "device_arn" in result.metadata


class TestBraketNotInstalled:
    def test_run_raises_without_sdk(self, bell_circuit):
        backend = BraketBackend()
        if not backend.is_available():
            with pytest.raises(RuntimeError, match="Amazon Braket SDK is not installed"):
                backend.run(bell_circuit, shots=100)
