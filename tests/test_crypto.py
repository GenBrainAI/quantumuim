"""Tests for quantum cryptography module."""

import numpy as np

from quantumium.crypto.bb84 import (
    bb84_protocol,
    _prepare_qubit,
    _measure_qubit,
)
from quantumium.crypto.qrng import quantum_random_bits, quantum_random_int


class TestQubitPreparation:
    def test_z_basis_0(self):
        """Z-basis |0> should have amplitude 1 at index 0."""
        sv = _prepare_qubit(0, 0)
        assert abs(sv[0]) > 0.999

    def test_z_basis_1(self):
        """Z-basis |1> should have amplitude 1 at index 1."""
        sv = _prepare_qubit(1, 0)
        assert abs(sv[1]) > 0.999

    def test_x_basis_0(self):
        """|+> should have equal amplitudes."""
        sv = _prepare_qubit(0, 1)
        np.testing.assert_allclose(np.abs(sv), [1 / np.sqrt(2)] * 2, atol=1e-10)

    def test_x_basis_1(self):
        """|-> should have equal absolute amplitudes with sign difference."""
        sv = _prepare_qubit(1, 1)
        np.testing.assert_allclose(np.abs(sv), [1 / np.sqrt(2)] * 2, atol=1e-10)
        assert np.real(sv[0] * sv[1].conj()) < 0  # opposite signs

    def test_normalized(self):
        for bit in [0, 1]:
            for basis in [0, 1]:
                sv = _prepare_qubit(bit, basis)
                assert abs(np.linalg.norm(sv) - 1.0) < 1e-10


class TestMeasurement:
    def test_z_deterministic(self):
        """Measuring |0> in Z basis should always give 0."""
        rng = np.random.default_rng(42)
        sv = _prepare_qubit(0, 0)
        results = [_measure_qubit(sv.copy(), 0, rng) for _ in range(20)]
        assert all(r == 0 for r in results)

    def test_z_deterministic_1(self):
        """Measuring |1> in Z basis should always give 1."""
        rng = np.random.default_rng(42)
        sv = _prepare_qubit(1, 0)
        results = [_measure_qubit(sv.copy(), 0, rng) for _ in range(20)]
        assert all(r == 1 for r in results)

    def test_x_deterministic(self):
        """Measuring |+> in X basis should always give 0."""
        rng = np.random.default_rng(42)
        sv = _prepare_qubit(0, 1)
        results = [_measure_qubit(sv.copy(), 1, rng) for _ in range(20)]
        assert all(r == 0 for r in results)

    def test_wrong_basis_random(self):
        """Measuring |0> in X basis should give roughly 50/50."""
        rng = np.random.default_rng(42)
        sv = _prepare_qubit(0, 0)
        results = [_measure_qubit(sv.copy(), 1, rng) for _ in range(200)]
        frac_0 = results.count(0) / len(results)
        assert 0.3 < frac_0 < 0.7


class TestBB84Protocol:
    def test_no_eavesdropper_keys_match(self):
        """Without Eve, Alice and Bob's keys should match exactly."""
        result = bb84_protocol(num_qubits=500, eavesdropper=False, seed=42)
        assert result.alice_key == result.bob_key
        assert result.qber == 0.0
        assert not result.eavesdropper_detected

    def test_no_eavesdropper_key_length(self):
        """Sifted key should be roughly half the sent qubits."""
        result = bb84_protocol(num_qubits=1000, eavesdropper=False, seed=42)
        assert result.sifted_length > 400
        assert result.sifted_length < 600
        assert result.final_key_length > 0

    def test_eavesdropper_detected(self):
        """With Eve, QBER should be ~25% and she should be detected."""
        result = bb84_protocol(num_qubits=1000, eavesdropper=True, seed=42)
        assert result.qber > 0.15
        assert result.qber < 0.35
        assert result.eavesdropper_detected

    def test_eavesdropper_introduces_errors(self):
        """With Eve, Alice and Bob's keys should differ."""
        result = bb84_protocol(num_qubits=500, eavesdropper=True, seed=42)
        mismatches = sum(
            a != b for a, b in zip(result.alice_key, result.bob_key)
        )
        # Eve causes ~25% errors even in the key portion
        assert mismatches > 0

    def test_result_fields(self):
        result = bb84_protocol(num_qubits=100, seed=0)
        assert result.num_qubits_sent == 100
        assert isinstance(result.alice_key, list)
        assert isinstance(result.bob_key, list)
        assert 0 <= result.qber <= 1
        assert result.detection_threshold == 0.11

    def test_reproducible(self):
        r1 = bb84_protocol(num_qubits=100, seed=123)
        r2 = bb84_protocol(num_qubits=100, seed=123)
        assert r1.alice_key == r2.alice_key
        assert r1.bob_key == r2.bob_key
        assert r1.qber == r2.qber

    def test_minimum_qubits(self):
        try:
            bb84_protocol(num_qubits=2)
            assert False
        except ValueError:
            pass


class TestQuantumRNG:
    def test_bit_values(self):
        """All bits should be 0 or 1."""
        bits = quantum_random_bits(100, seed=42)
        assert all(b in [0, 1] for b in bits)

    def test_length(self):
        bits = quantum_random_bits(50, seed=0)
        assert len(bits) == 50

    def test_distribution(self):
        """Should be roughly 50/50 over many samples."""
        bits = quantum_random_bits(1000, seed=42)
        frac_1 = sum(bits) / len(bits)
        assert 0.4 < frac_1 < 0.6

    def test_reproducible(self):
        b1 = quantum_random_bits(20, seed=99)
        b2 = quantum_random_bits(20, seed=99)
        assert b1 == b2

    def test_minimum(self):
        try:
            quantum_random_bits(0)
            assert False
        except ValueError:
            pass

    def test_random_int_range(self):
        """Random integers should be in [low, high)."""
        for _ in range(20):
            val = quantum_random_int(5, 10, seed=None)
            assert 5 <= val < 10

    def test_random_int_reproducible(self):
        v1 = quantum_random_int(0, 100, seed=42)
        v2 = quantum_random_int(0, 100, seed=42)
        assert v1 == v2

    def test_random_int_invalid(self):
        try:
            quantum_random_int(10, 5)
            assert False
        except ValueError:
            pass
