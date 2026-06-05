"""Tests for circuit ASCII visualization."""

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, Y, Z, S, T, CNOT, SWAP
from quantumium.utils.visualization import circuit_to_ascii


class TestCircuitVisualization:
    def test_empty_circuit(self):
        qc = QuantumCircuit(2)
        diagram = circuit_to_ascii(qc)
        assert "q0" in diagram
        assert "q1" in diagram

    def test_single_gate(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        diagram = circuit_to_ascii(qc)
        assert "H" in diagram
        assert "q0" in diagram

    def test_multiple_gates(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        qc.apply(X, 0)
        diagram = circuit_to_ascii(qc)
        assert "H" in diagram
        assert "X" in diagram

    def test_cnot(self):
        qc = QuantumCircuit(2)
        qc.apply(CNOT, 0, 1)
        diagram = circuit_to_ascii(qc)
        assert "●" in diagram
        assert "⊕" in diagram

    def test_swap(self):
        qc = QuantumCircuit(2)
        qc.apply(SWAP, 0, 1)
        diagram = circuit_to_ascii(qc)
        assert "✕" in diagram

    def test_bell_circuit(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        diagram = circuit_to_ascii(qc)
        assert "H" in diagram
        assert "●" in diagram
        assert "⊕" in diagram
        lines = diagram.strip().split("\n")
        assert len(lines) == 3

    def test_three_qubit_circuit(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(CNOT, 1, 2)
        diagram = circuit_to_ascii(qc)
        lines = diagram.strip().split("\n")
        assert len(lines) == 5
        assert "q0" in lines[0]
        assert "q1" in lines[2]
        assert "q2" in lines[4]

    def test_various_gates(self):
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        qc.apply(Y, 0)
        qc.apply(Z, 0)
        qc.apply(S, 0)
        qc.apply(T, 0)
        diagram = circuit_to_ascii(qc)
        for gate_name in ["X", "Y", "Z", "S", "T"]:
            assert gate_name in diagram

    def test_vertical_connector(self):
        qc = QuantumCircuit(3)
        qc.apply(CNOT, 0, 2)
        diagram = circuit_to_ascii(qc)
        assert "│" in diagram
