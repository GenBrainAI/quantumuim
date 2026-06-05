"""Tests for quantumium.tensor_networks — MPS representation and operations."""

import math
import numpy as np
import pytest

from quantumium.tensor_networks.mps import (
    MPS,
    mps_zero_state,
    mps_product_state,
    mps_from_statevector,
)
from quantumium.tensor_networks.operations import (
    apply_single_qubit_gate,
    apply_two_qubit_gate,
    measure_qubit,
    entanglement_entropy,
    inner_product,
)


H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)
CNOT_GATE = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)


# ============================================================
# MPS Construction
# ============================================================

class TestMPSZeroState:
    def test_creates_n_tensors(self):
        mps = mps_zero_state(4)
        assert mps.num_qubits == 4
        assert len(mps.tensors) == 4

    def test_tensor_shapes(self):
        mps = mps_zero_state(3)
        for T in mps.tensors:
            assert T.shape == (1, 2, 1)

    def test_statevector(self):
        mps = mps_zero_state(3)
        sv = mps.to_statevector()
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_allclose(sv, expected, atol=1e-12)

    def test_norm(self):
        mps = mps_zero_state(5)
        assert abs(mps.norm() - 1.0) < 1e-12

    def test_bond_dimensions(self):
        mps = mps_zero_state(4)
        assert mps.bond_dimensions() == [1, 1, 1]


class TestMPSProductState:
    def test_plus_state(self):
        plus = np.array([1, 1], dtype=complex) / math.sqrt(2)
        mps = mps_product_state([plus, plus])
        sv = mps.to_statevector()
        expected = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        np.testing.assert_allclose(sv, expected, atol=1e-12)

    def test_mixed_states(self):
        s0 = np.array([1, 0], dtype=complex)
        s1 = np.array([0, 1], dtype=complex)
        mps = mps_product_state([s0, s1])
        sv = mps.to_statevector()
        expected = np.zeros(4, dtype=complex)
        expected[1] = 1.0  # |01⟩ — qubit 0 is MSB, qubit 1 is LSB
        np.testing.assert_allclose(sv, expected, atol=1e-12)

    def test_invalid_state_raises(self):
        with pytest.raises(ValueError):
            mps_product_state([np.array([1, 0, 0])])


class TestMPSFromStatevector:
    def test_zero_state(self):
        sv = np.zeros(8, dtype=complex)
        sv[0] = 1.0
        mps = mps_from_statevector(sv)
        assert mps.num_qubits == 3
        sv_back = mps.to_statevector()
        np.testing.assert_allclose(sv_back, sv, atol=1e-12)

    def test_bell_state(self):
        sv = np.zeros(4, dtype=complex)
        sv[0] = 1 / math.sqrt(2)
        sv[3] = 1 / math.sqrt(2)
        mps = mps_from_statevector(sv)
        sv_back = mps.to_statevector()
        np.testing.assert_allclose(sv_back, sv, atol=1e-12)

    def test_ghz_state(self):
        sv = np.zeros(8, dtype=complex)
        sv[0] = 1 / math.sqrt(2)
        sv[7] = 1 / math.sqrt(2)
        mps = mps_from_statevector(sv)
        sv_back = mps.to_statevector()
        np.testing.assert_allclose(sv_back, sv, atol=1e-12)

    def test_truncation(self):
        sv = np.zeros(4, dtype=complex)
        sv[0] = 1 / math.sqrt(2)
        sv[3] = 1 / math.sqrt(2)
        mps = mps_from_statevector(sv, max_bond_dim=1)
        # With bond dim 1, can't represent entangled state exactly
        assert mps.max_bond() >= 1  # just check it runs
        sv_back = mps.to_statevector()
        assert len(sv_back) == 4

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            mps_from_statevector(np.array([1, 0, 0]))

    def test_roundtrip_random(self):
        rng = np.random.default_rng(42)
        sv = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        sv = sv / np.linalg.norm(sv)
        mps = mps_from_statevector(sv)
        sv_back = mps.to_statevector()
        np.testing.assert_allclose(sv_back, sv, atol=1e-10)


# ============================================================
# Single-Qubit Gates
# ============================================================

class TestSingleQubitGate:
    def test_x_gate(self):
        mps = mps_zero_state(2)
        apply_single_qubit_gate(mps, X_GATE, 0)
        sv = mps.to_statevector()
        expected = np.zeros(4, dtype=complex)
        expected[2] = 1.0  # |10⟩ — qubit 0 is MSB in MPS convention
        np.testing.assert_allclose(sv, expected, atol=1e-12)

    def test_h_gate(self):
        mps = mps_zero_state(1)
        apply_single_qubit_gate(mps, H_GATE, 0)
        sv = mps.to_statevector()
        expected = np.array([1, 1], dtype=complex) / math.sqrt(2)
        np.testing.assert_allclose(sv, expected, atol=1e-12)

    def test_preserves_norm(self):
        mps = mps_zero_state(3)
        apply_single_qubit_gate(mps, H_GATE, 1)
        assert abs(mps.norm() - 1.0) < 1e-10

    def test_invalid_gate_size(self):
        mps = mps_zero_state(2)
        with pytest.raises(ValueError, match="2×2"):
            apply_single_qubit_gate(mps, np.eye(4, dtype=complex), 0)


# ============================================================
# Two-Qubit Gates
# ============================================================

class TestTwoQubitGate:
    def test_cnot_creates_bell(self):
        mps = mps_zero_state(2)
        apply_single_qubit_gate(mps, H_GATE, 0)
        apply_two_qubit_gate(mps, CNOT_GATE, 0, 1)
        sv = mps.to_statevector()
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1 / math.sqrt(2)
        expected[3] = 1 / math.sqrt(2)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_cnot_on_second_qubit(self):
        mps = mps_zero_state(3)
        apply_single_qubit_gate(mps, X_GATE, 1)
        apply_two_qubit_gate(mps, CNOT_GATE, 1, 2)
        sv = mps.to_statevector()
        # Qubit 0=MSB. Start |000⟩, X on q1 → |010⟩, CNOT(1→2) → |011⟩
        expected = np.zeros(8, dtype=complex)
        expected[3] = 1.0  # |011⟩ = index 3
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_preserves_norm(self):
        mps = mps_zero_state(3)
        apply_single_qubit_gate(mps, H_GATE, 0)
        apply_two_qubit_gate(mps, CNOT_GATE, 0, 1)
        assert abs(mps.norm() - 1.0) < 1e-10

    def test_non_adjacent_raises(self):
        mps = mps_zero_state(3)
        with pytest.raises(ValueError, match="adjacent"):
            apply_two_qubit_gate(mps, CNOT_GATE, 0, 2)

    def test_invalid_gate_size(self):
        mps = mps_zero_state(2)
        with pytest.raises(ValueError, match="4×4"):
            apply_two_qubit_gate(mps, np.eye(2, dtype=complex), 0, 1)

    def test_bond_dim_grows(self):
        mps = mps_zero_state(3)
        assert mps.max_bond() == 1
        apply_single_qubit_gate(mps, H_GATE, 0)
        apply_two_qubit_gate(mps, CNOT_GATE, 0, 1)
        assert mps.max_bond() >= 2

    def test_truncation(self):
        mps = mps_zero_state(3, max_bond_dim=1)
        apply_single_qubit_gate(mps, H_GATE, 0)
        apply_two_qubit_gate(mps, CNOT_GATE, 0, 1)
        assert mps.max_bond() <= 1


# ============================================================
# Measurement
# ============================================================

class TestMeasurement:
    def test_measure_zero_state(self):
        mps = mps_zero_state(2)
        result = measure_qubit(mps, 0, rng=np.random.default_rng(42))
        assert result == 0

    def test_measure_one_state(self):
        mps = mps_zero_state(2)
        apply_single_qubit_gate(mps, X_GATE, 0)
        result = measure_qubit(mps, 0, rng=np.random.default_rng(42))
        assert result == 1

    def test_measure_statistics(self):
        outcomes = {0: 0, 1: 0}
        for seed in range(100):
            mps = mps_zero_state(1)
            apply_single_qubit_gate(mps, H_GATE, 0)
            result = measure_qubit(mps, 0, rng=np.random.default_rng(seed))
            outcomes[result] += 1
        assert 30 < outcomes[0] < 70
        assert 30 < outcomes[1] < 70


# ============================================================
# Entanglement Entropy
# ============================================================

class TestEntanglementEntropy:
    def test_product_state_zero_entropy(self):
        mps = mps_zero_state(3)
        assert entanglement_entropy(mps, 1) < 1e-10
        assert entanglement_entropy(mps, 2) < 1e-10

    def test_bell_state_one_ebit(self):
        mps = mps_zero_state(2)
        apply_single_qubit_gate(mps, H_GATE, 0)
        apply_two_qubit_gate(mps, CNOT_GATE, 0, 1)
        S = entanglement_entropy(mps, 1)
        assert abs(S - 1.0) < 1e-10  # 1 ebit

    def test_ghz_state(self):
        mps = mps_zero_state(3)
        apply_single_qubit_gate(mps, H_GATE, 0)
        apply_two_qubit_gate(mps, CNOT_GATE, 0, 1)
        apply_two_qubit_gate(mps, CNOT_GATE, 1, 2)
        S = entanglement_entropy(mps, 1)
        assert abs(S - 1.0) < 1e-10

    def test_invalid_cut_raises(self):
        mps = mps_zero_state(3)
        with pytest.raises(ValueError, match="Cut"):
            entanglement_entropy(mps, 0)
        with pytest.raises(ValueError, match="Cut"):
            entanglement_entropy(mps, 3)


# ============================================================
# Inner Product
# ============================================================

class TestInnerProduct:
    def test_self_overlap(self):
        mps = mps_zero_state(3)
        ip = inner_product(mps, mps)
        assert abs(ip - 1.0) < 1e-12

    def test_orthogonal_states(self):
        mps0 = mps_zero_state(2)
        mps1 = mps_zero_state(2)
        apply_single_qubit_gate(mps1, X_GATE, 0)
        ip = inner_product(mps0, mps1)
        assert abs(ip) < 1e-12

    def test_bell_overlap(self):
        mps1 = mps_zero_state(2)
        apply_single_qubit_gate(mps1, H_GATE, 0)
        apply_two_qubit_gate(mps1, CNOT_GATE, 0, 1)

        mps2 = mps1.copy()
        ip = inner_product(mps1, mps2)
        assert abs(ip - 1.0) < 1e-10

    def test_different_sizes_raises(self):
        mps1 = mps_zero_state(2)
        mps2 = mps_zero_state(3)
        with pytest.raises(ValueError, match="same number"):
            inner_product(mps1, mps2)

    def test_h_state_overlap(self):
        mps1 = mps_zero_state(1)
        apply_single_qubit_gate(mps1, H_GATE, 0)
        mps2 = mps_zero_state(1)
        ip = inner_product(mps1, mps2)
        assert abs(ip - 1 / math.sqrt(2)) < 1e-10


# ============================================================
# MPS Copy
# ============================================================

class TestMPSCopy:
    def test_copy_independent(self):
        mps = mps_zero_state(2)
        mps2 = mps.copy()
        apply_single_qubit_gate(mps, X_GATE, 0)
        sv1 = mps.to_statevector()
        sv2 = mps2.to_statevector()
        assert not np.allclose(sv1, sv2)
