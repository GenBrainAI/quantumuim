"""Tests for circuit JSON serialization."""

import json
import os
import tempfile

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, Y, Z, S, T, CNOT, SWAP, Rx, Ry, Rz, Gate
from quantumium.transpiler.serialization import (
    to_dict, from_dict, to_json, from_json, save, load,
)


class TestToDict:
    def test_basic_circuit(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(CNOT, 0, 1)
        d = to_dict(qc)
        assert d["format"] == "quantumium-v1"
        assert d["num_qubits"] == 2
        assert len(d["instructions"]) == 2
        assert d["instructions"][0]["gate"] == "H"
        assert d["instructions"][1]["gate"] == "CNOT"

    def test_rotation_gate(self):
        qc = QuantumCircuit(1)
        qc.apply(Ry(1.5), 0)
        d = to_dict(qc)
        assert d["instructions"][0]["gate"] == "Ry"
        assert abs(d["instructions"][0]["angle"] - 1.5) < 1e-10

    def test_all_named_gates(self):
        qc = QuantumCircuit(2)
        for g in [H, X, Y, Z, S, T]:
            qc.apply(g, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(SWAP, 0, 1)
        d = to_dict(qc)
        gate_names = [inst["gate"] for inst in d["instructions"]]
        assert "H" in gate_names
        assert "CNOT" in gate_names
        assert "SWAP" in gate_names

    def test_custom_gate(self):
        matrix = np.array([[0, 1j], [1j, 0]], dtype=complex)
        custom = Gate(matrix, "iX")
        qc = QuantumCircuit(1)
        qc.apply(custom, 0)
        d = to_dict(qc)
        assert d["instructions"][0]["gate"] == "custom"
        assert "matrix_real" in d["instructions"][0]
        assert "matrix_imag" in d["instructions"][0]


class TestFromDict:
    def test_basic(self):
        d = {
            "format": "quantumium-v1",
            "num_qubits": 2,
            "instructions": [
                {"gate": "H", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]},
            ],
        }
        qc = from_dict(d)
        assert qc.num_qubits == 2
        assert len(qc.instructions) == 2

    def test_unknown_format(self):
        with pytest.raises(ValueError, match="Unknown format"):
            from_dict({"format": "bad", "num_qubits": 1, "instructions": []})

    def test_unknown_gate(self):
        d = {
            "format": "quantumium-v1",
            "num_qubits": 1,
            "instructions": [{"gate": "NONEXISTENT", "qubits": [0]}],
        }
        with pytest.raises(ValueError, match="Unknown gate"):
            from_dict(d)


class TestJsonRoundTrip:
    def test_bell_state(self):
        original = QuantumCircuit(2)
        original.apply(H, 0).apply(CNOT, 0, 1)
        json_str = to_json(original)
        restored = from_json(json_str)
        np.testing.assert_allclose(
            original.statevector(), restored.statevector(), atol=1e-10
        )

    def test_rotation_circuit(self):
        original = QuantumCircuit(1)
        original.apply(Rx(0.5), 0).apply(Ry(1.2), 0).apply(Rz(2.3), 0)
        json_str = to_json(original)
        restored = from_json(json_str)
        np.testing.assert_allclose(
            original.statevector(), restored.statevector(), atol=1e-10
        )

    def test_custom_gate_roundtrip(self):
        matrix = np.array([[0, 1j], [1j, 0]], dtype=complex)
        custom = Gate(matrix, "iX")
        original = QuantumCircuit(1)
        original.apply(custom, 0)
        json_str = to_json(original)
        restored = from_json(json_str)
        np.testing.assert_allclose(
            original.statevector(), restored.statevector(), atol=1e-10
        )

    def test_valid_json(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(CNOT, 0, 1)
        json_str = to_json(qc)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


class TestFileIO:
    def test_save_and_load(self):
        original = QuantumCircuit(2)
        original.apply(H, 0).apply(CNOT, 0, 1).apply(Ry(0.5), 1)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            save(original, filepath)
            restored = load(filepath)
            np.testing.assert_allclose(
                original.statevector(), restored.statevector(), atol=1e-10
            )
        finally:
            os.unlink(filepath)
