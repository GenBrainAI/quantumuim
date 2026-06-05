"""Tests for circuit templates."""

import numpy as np
import pytest

from quantumium.circuits.templates import (
    bell_pair,
    ghz_state,
    quantum_teleportation,
    qft,
    inverse_qft,
)
from quantumium.core.gate import H, X


class TestBellPair:
    def test_creates_entangled_state(self):
        qc = bell_pair()
        state = qc.statevector()
        assert np.abs(state[0]) ** 2 == pytest.approx(0.5, abs=1e-10)
        assert np.abs(state[3]) ** 2 == pytest.approx(0.5, abs=1e-10)

    def test_custom_qubits(self):
        qc = bell_pair(1, 2)
        assert qc.num_qubits == 3


class TestGHZ:
    def test_3_qubit(self):
        qc = ghz_state(3)
        state = qc.statevector()
        assert np.abs(state[0]) ** 2 == pytest.approx(0.5, abs=1e-10)
        assert np.abs(state[7]) ** 2 == pytest.approx(0.5, abs=1e-10)
        non_ghz = sum(np.abs(state[i]) ** 2 for i in range(8) if i not in (0, 7))
        assert non_ghz == pytest.approx(0.0, abs=1e-10)

    def test_4_qubit(self):
        qc = ghz_state(4)
        state = qc.statevector()
        assert np.abs(state[0]) ** 2 == pytest.approx(0.5, abs=1e-10)
        assert np.abs(state[15]) ** 2 == pytest.approx(0.5, abs=1e-10)

    def test_minimum_qubits(self):
        with pytest.raises(ValueError):
            ghz_state(1)


class TestTeleportation:
    def test_teleport_zero(self):
        qc = quantum_teleportation()
        state = qc.statevector()
        probs = np.abs(state) ** 2
        p_qubit2_zero = sum(probs[i] for i in range(8) if not (i & 4))
        assert p_qubit2_zero == pytest.approx(1.0, abs=1e-10)

    def test_teleport_one(self):
        qc = quantum_teleportation()
        qc_with_x = type(qc)(3)
        qc_with_x.apply(X, 0)
        for inst in qc.instructions:
            qc_with_x.apply(inst.gate, *inst.target_qubits)
        state = qc_with_x.statevector()
        probs = np.abs(state) ** 2
        p_qubit2_one = sum(probs[i] for i in range(8) if i & 4)
        assert p_qubit2_one == pytest.approx(1.0, abs=1e-10)


class TestQFT:
    def test_single_qubit_is_hadamard(self):
        qc = qft(1)
        state = qc.statevector()
        expected = np.array([1, 1]) / np.sqrt(2)
        assert np.allclose(np.abs(state), np.abs(expected))

    def test_2_qubit_uniform_from_zero(self):
        qc = qft(2)
        state = qc.statevector()
        assert all(np.abs(s) ** 2 == pytest.approx(0.25, abs=1e-10) for s in state)

    def test_qft_inverse_is_identity(self):
        n = 3
        qc_fwd = qft(n)
        qc_inv = inverse_qft(n)
        fwd_state = qc_fwd.statevector()

        combined_matrix = np.eye(2**n, dtype=complex)
        for inst in qc_fwd.instructions:
            expanded = qc_fwd._expand_gate(inst.gate.matrix, inst.target_qubits, n)
            combined_matrix = expanded @ combined_matrix
        for inst in qc_inv.instructions:
            expanded = qc_inv._expand_gate(inst.gate.matrix, inst.target_qubits, n)
            combined_matrix = expanded @ combined_matrix

        assert np.allclose(combined_matrix, np.eye(2**n), atol=1e-10)

    def test_minimum_qubits(self):
        with pytest.raises(ValueError):
            qft(0)
