"""Tests for OpenQASM 2.0 export and import."""

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, Y, Z, S, T, CNOT, SWAP, Rx, Ry, Rz
from quantumium.transpiler.qasm import to_qasm, from_qasm


class TestToQasm:
    def test_basic_gates(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(X, 1).apply(CNOT, 0, 1)
        qasm = to_qasm(qc)
        assert "OPENQASM 2.0;" in qasm
        assert "qreg q[2];" in qasm
        assert "h q[0];" in qasm
        assert "x q[1];" in qasm
        assert "cx q[0],q[1];" in qasm

    def test_rotation_gates(self):
        qc = QuantumCircuit(1)
        qc.apply(Rx(1.5), 0)
        qc.apply(Ry(0.7), 0)
        qc.apply(Rz(2.1), 0)
        qasm = to_qasm(qc)
        assert "rx(" in qasm
        assert "ry(" in qasm
        assert "rz(" in qasm

    def test_all_standard_gates(self):
        qc = QuantumCircuit(2)
        for gate in [H, X, Y, Z, S, T]:
            qc.apply(gate, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(SWAP, 0, 1)
        qasm = to_qasm(qc)
        for name in ["h", "x", "y", "z", "s", "t", "cx", "swap"]:
            assert name in qasm

    def test_empty_circuit(self):
        qc = QuantumCircuit(3)
        qasm = to_qasm(qc)
        assert "qreg q[3];" in qasm
        assert "creg c[3];" in qasm

    def test_includes_header(self):
        qc = QuantumCircuit(1)
        qasm = to_qasm(qc)
        assert 'include "qelib1.inc";' in qasm


class TestFromQasm:
    def test_basic_circuit(self):
        qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];

h q[0];
cx q[0],q[1];
"""
        qc = from_qasm(qasm)
        assert qc.num_qubits == 2
        assert len(qc.instructions) == 2
        sv = qc.statevector()
        expected = QuantumCircuit(2)
        expected.apply(H, 0).apply(CNOT, 0, 1)
        np.testing.assert_allclose(sv, expected.statevector(), atol=1e-10)

    def test_rotation_gates(self):
        qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];

ry(1.5707963268) q[0];
"""
        qc = from_qasm(qasm)
        assert len(qc.instructions) == 1
        sv = qc.statevector()
        expected = QuantumCircuit(1)
        expected.apply(Ry(1.5707963268), 0)
        np.testing.assert_allclose(sv, expected.statevector(), atol=1e-6)

    def test_no_qreg_raises(self):
        qasm = """
OPENQASM 2.0;
h q[0];
"""
        with pytest.raises(ValueError, match="No qreg"):
            from_qasm(qasm)

    def test_comments_ignored(self):
        qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
// This is a comment
x q[0];
"""
        qc = from_qasm(qasm)
        assert len(qc.instructions) == 1

    def test_multiple_gates(self):
        qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

h q[0];
h q[1];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
"""
        qc = from_qasm(qasm)
        assert qc.num_qubits == 3
        assert len(qc.instructions) == 5


class TestRoundTrip:
    def test_bell_state_roundtrip(self):
        original = QuantumCircuit(2)
        original.apply(H, 0).apply(CNOT, 0, 1)
        qasm = to_qasm(original)
        restored = from_qasm(qasm)
        np.testing.assert_allclose(
            original.statevector(), restored.statevector(), atol=1e-10
        )

    def test_rotation_roundtrip(self):
        original = QuantumCircuit(1)
        original.apply(Rx(0.5), 0).apply(Ry(1.2), 0).apply(Rz(2.3), 0)
        qasm = to_qasm(original)
        restored = from_qasm(qasm)
        np.testing.assert_allclose(
            original.statevector(), restored.statevector(), atol=1e-6
        )

    def test_ghz_roundtrip(self):
        original = QuantumCircuit(3)
        original.apply(H, 0)
        original.apply(CNOT, 0, 1)
        original.apply(CNOT, 1, 2)
        qasm = to_qasm(original)
        restored = from_qasm(qasm)
        np.testing.assert_allclose(
            original.statevector(), restored.statevector(), atol=1e-10
        )

    def test_complex_circuit_roundtrip(self):
        original = QuantumCircuit(2)
        original.apply(H, 0)
        original.apply(T, 0)
        original.apply(S, 1)
        original.apply(CNOT, 0, 1)
        original.apply(Ry(0.8), 1)
        qasm = to_qasm(original)
        restored = from_qasm(qasm)
        np.testing.assert_allclose(
            original.statevector(), restored.statevector(), atol=1e-6
        )
