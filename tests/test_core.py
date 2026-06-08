"""Tests for core quantum computing primitives."""

import numpy as np
import pytest

from quantumium.core.qubit import Qubit
from quantumium.core.gate import H, X, CNOT, Rx, Ry, Rz, Gate
from quantumium.core.circuit import QuantumCircuit


class TestQubit:
    def test_zero_state(self):
        q = Qubit.zero()
        assert q.probabilities == pytest.approx((1.0, 0.0))

    def test_one_state(self):
        q = Qubit.one()
        assert q.probabilities == pytest.approx((0.0, 1.0))

    def test_plus_state(self):
        q = Qubit.plus()
        assert q.probabilities == pytest.approx((0.5, 0.5))

    def test_normalization(self):
        q = Qubit(3.0, 4.0)
        p0, p1 = q.probabilities
        assert p0 + p1 == pytest.approx(1.0)

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError):
            Qubit(0.0, 0.0)

    def test_measure_collapses(self):
        q = Qubit.zero()
        result = q.measure()
        assert result == 0
        assert q.probabilities == pytest.approx((1.0, 0.0))


class TestGate:
    def test_pauli_x(self):
        assert X.matrix.shape == (2, 2)
        assert X.num_qubits == 1

    def test_hadamard_is_unitary(self):
        m = H.matrix
        assert np.allclose(m @ m.conj().T, np.eye(2))

    def test_cnot_shape(self):
        assert CNOT.matrix.shape == (4, 4)
        assert CNOT.num_qubits == 2

    def test_non_unitary_raises(self):
        with pytest.raises(ValueError):
            Gate(np.array([[1, 1], [0, 0]], dtype=complex))

    def test_rotation_gates(self):
        for theta in [0, np.pi / 4, np.pi / 2, np.pi]:
            for gate_fn in [Rx, Ry, Rz]:
                g = gate_fn(theta)
                m = g.matrix
                assert np.allclose(m @ m.conj().T, np.eye(2), atol=1e-10)


class TestCircuit:
    def test_single_qubit(self):
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        state = qc.statevector()
        assert np.abs(state[1]) ** 2 == pytest.approx(1.0)

    def test_bell_state(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        state = qc.statevector()
        assert np.abs(state[0]) ** 2 == pytest.approx(0.5, abs=1e-10)
        assert np.abs(state[3]) ** 2 == pytest.approx(0.5, abs=1e-10)

    def test_measure_all(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        counts = qc.measure_all(shots=10000)
        assert set(counts.keys()).issubset({"00", "01", "10", "11"})
        total = sum(counts.values())
        assert total == 10000
        assert counts.get("00", 0) > 3000
        assert counts.get("11", 0) > 3000

    def test_invalid_qubit_index(self):
        qc = QuantumCircuit(2)
        with pytest.raises(IndexError):
            qc.apply(X, 5)

    def test_gate_qubit_mismatch(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError):
            qc.apply(CNOT, 0)

    def test_chaining(self):
        qc = QuantumCircuit(1)
        qc.apply(X, 0).apply(X, 0)
        state = qc.statevector()
        assert np.abs(state[0]) ** 2 == pytest.approx(1.0)
