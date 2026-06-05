"""Tests for extended gate set: CZ, CCX, controlled gates, multi-qubit rotations."""

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import (
    Gate, H, X, Z, S, T, CNOT, CZ, CCX,
    controlled, Rx, Ry, Rz, CRx, CRy, CRz, Rxx, Rzz,
)


class TestCZ:
    def test_unitary(self):
        assert np.allclose(CZ.matrix @ CZ.matrix.conj().T, np.eye(4), atol=1e-10)

    def test_cz_action(self):
        qc = QuantumCircuit(2)
        qc.apply(X, 0).apply(X, 1).apply(CZ, 0, 1)
        sv = qc.statevector()
        expected = np.zeros(4, dtype=complex)
        expected[3] = -1.0
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_cz_no_phase_on_00(self):
        qc = QuantumCircuit(2)
        qc.apply(CZ, 0, 1)
        sv = qc.statevector()
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_cz_symmetric(self):
        qc1 = QuantumCircuit(2)
        qc1.apply(H, 0).apply(H, 1).apply(CZ, 0, 1)
        qc2 = QuantumCircuit(2)
        qc2.apply(H, 0).apply(H, 1).apply(CZ, 1, 0)
        np.testing.assert_allclose(qc1.statevector(), qc2.statevector(), atol=1e-10)


class TestCCX:
    def test_unitary(self):
        m = CCX.matrix
        assert np.allclose(m @ m.conj().T, np.eye(8), atol=1e-10)

    def test_toffoli_flips_target(self):
        qc = QuantumCircuit(3)
        qc.apply(X, 0).apply(X, 1).apply(CCX, 0, 1, 2)
        sv = qc.statevector()
        expected = np.zeros(8, dtype=complex)
        expected[7] = 1.0
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_toffoli_no_flip_one_control(self):
        qc = QuantumCircuit(3)
        qc.apply(X, 0).apply(CCX, 0, 1, 2)
        sv = qc.statevector()
        expected = np.zeros(8, dtype=complex)
        expected[1] = 1.0
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_toffoli_self_inverse(self):
        qc = QuantumCircuit(3)
        qc.apply(X, 0).apply(X, 1)
        qc.apply(CCX, 0, 1, 2).apply(CCX, 0, 1, 2)
        sv = qc.statevector()
        expected = np.zeros(8, dtype=complex)
        expected[3] = 1.0
        np.testing.assert_allclose(sv, expected, atol=1e-10)


class TestControlled:
    def test_controlled_x_is_cnot(self):
        cx = controlled(X)
        np.testing.assert_allclose(cx.matrix, CNOT.matrix, atol=1e-10)

    def test_controlled_z_is_cz(self):
        cz = controlled(Z)
        np.testing.assert_allclose(cz.matrix, CZ.matrix, atol=1e-10)

    def test_controlled_h(self):
        ch = controlled(H, "CH")
        assert ch.num_qubits == 2
        assert ch.name == "CH"
        assert np.allclose(ch.matrix @ ch.matrix.conj().T, np.eye(4), atol=1e-10)
        assert np.allclose(ch.matrix[:2, :2], np.eye(2), atol=1e-10)

    def test_controlled_s(self):
        cs = controlled(S)
        assert np.allclose(cs.matrix[:2, :2], np.eye(2), atol=1e-10)
        np.testing.assert_allclose(cs.matrix[2:, 2:], S.matrix, atol=1e-10)

    def test_rejects_multi_qubit(self):
        with pytest.raises(ValueError, match="single-qubit"):
            controlled(CNOT)

    def test_controlled_gate_action(self):
        ch = controlled(H)
        qc = QuantumCircuit(2)
        qc.apply(X, 0).apply(ch, 0, 1)
        sv = qc.statevector()
        # Control q0=|1⟩ activates H on q1: |01⟩ → (|01⟩+|11⟩)/√2
        expected = np.zeros(4, dtype=complex)
        expected[1] = 1 / np.sqrt(2)  # |01⟩
        expected[3] = 1 / np.sqrt(2)  # |11⟩
        np.testing.assert_allclose(sv, expected, atol=1e-10)


class TestControlledRotations:
    def test_crx_zero_is_identity(self):
        crx = CRx(0.0)
        np.testing.assert_allclose(crx.matrix, np.eye(4), atol=1e-10)

    def test_cry_action(self):
        cry = CRy(np.pi)
        qc = QuantumCircuit(2)
        qc.apply(X, 0).apply(cry, 0, 1)
        sv = qc.statevector()
        expected = np.zeros(4, dtype=complex)
        expected[3] = 1.0
        np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-10)

    def test_crz_is_unitary(self):
        crz = CRz(1.23)
        assert np.allclose(crz.matrix @ crz.matrix.conj().T, np.eye(4), atol=1e-10)

    def test_controlled_rotations_no_action_when_control_0(self):
        for factory in [CRx, CRy, CRz]:
            gate = factory(np.pi / 3)
            qc = QuantumCircuit(2)
            qc.apply(gate, 0, 1)
            sv = qc.statevector()
            expected = np.zeros(4, dtype=complex)
            expected[0] = 1.0
            np.testing.assert_allclose(sv, expected, atol=1e-10)


class TestTwoQubitRotations:
    def test_rxx_unitary(self):
        rxx = Rxx(0.8)
        assert np.allclose(rxx.matrix @ rxx.matrix.conj().T, np.eye(4), atol=1e-10)

    def test_rzz_unitary(self):
        rzz = Rzz(1.5)
        assert np.allclose(rzz.matrix @ rzz.matrix.conj().T, np.eye(4), atol=1e-10)

    def test_rxx_zero_is_identity(self):
        np.testing.assert_allclose(Rxx(0.0).matrix, np.eye(4), atol=1e-10)

    def test_rzz_zero_is_identity(self):
        np.testing.assert_allclose(Rzz(0.0).matrix, np.eye(4), atol=1e-10)

    def test_rxx_pi_creates_entanglement(self):
        qc = QuantumCircuit(2)
        qc.apply(Rxx(np.pi / 2), 0, 1)
        sv = qc.statevector()
        assert abs(sv[0]) > 0.1 and abs(sv[3]) > 0.1

    def test_rzz_diagonal(self):
        rzz = Rzz(1.0)
        off_diag = rzz.matrix.copy()
        np.fill_diagonal(off_diag, 0)
        assert np.allclose(off_diag, 0, atol=1e-10)
