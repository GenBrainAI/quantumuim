"""Tests for quantum benchmarking module."""

import numpy as np

from quantumium.benchmarks.fidelity import (
    state_fidelity,
    process_fidelity,
    average_gate_fidelity,
)
from quantumium.benchmarks.randomized import (
    randomized_benchmarking,
    _build_single_qubit_cliffords,
    _find_inverse_clifford,
    _compose_cliffords,
)
from quantumium.benchmarks.quantum_volume import quantum_volume


class TestStateFidelity:
    def test_identical(self):
        sv = np.array([1, 0], dtype=complex)
        assert abs(state_fidelity(sv, sv) - 1.0) < 1e-10

    def test_orthogonal(self):
        sv1 = np.array([1, 0], dtype=complex)
        sv2 = np.array([0, 1], dtype=complex)
        assert abs(state_fidelity(sv1, sv2)) < 1e-10

    def test_superposition(self):
        sv1 = np.array([1, 0], dtype=complex)
        sv2 = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert abs(state_fidelity(sv1, sv2) - 0.5) < 1e-10

    def test_range(self):
        sv1 = np.array([1, 0], dtype=complex)
        sv2 = np.array([0.6, 0.8], dtype=complex)
        f = state_fidelity(sv1, sv2)
        assert 0 <= f <= 1

    def test_dimension_mismatch(self):
        try:
            state_fidelity(np.array([1, 0]), np.array([1, 0, 0]))
            assert False
        except ValueError:
            pass


class TestProcessFidelity:
    def test_identical(self):
        U = np.array([[0, 1], [1, 0]], dtype=complex)
        assert abs(process_fidelity(U, U) - 1.0) < 1e-10

    def test_identity_vs_x(self):
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        f = process_fidelity(I, X)
        assert f < 0.01  # Tr(X) = 0, so process fidelity = 0

    def test_close_unitaries(self):
        """Small rotation should have high process fidelity with identity."""
        theta = 0.01
        Rz = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
        I = np.eye(2, dtype=complex)
        f = process_fidelity(I, Rz)
        assert f > 0.99


class TestAverageGateFidelity:
    def test_perfect(self):
        U = np.eye(2, dtype=complex)
        assert abs(average_gate_fidelity(U, U) - 1.0) < 1e-10

    def test_bounds(self):
        U = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        f = average_gate_fidelity(U, X)
        assert 0 <= f <= 1

    def test_relationship(self):
        """F_avg = (d * F_proc + 1) / (d + 1)."""
        U = np.eye(2, dtype=complex)
        theta = 0.1
        V = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=complex
        )
        f_proc = process_fidelity(U, V)
        f_avg = average_gate_fidelity(U, V)
        expected = (2 * f_proc + 1) / 3
        assert abs(f_avg - expected) < 1e-10


class TestCliffordGroup:
    def test_count(self):
        """Single-qubit Clifford group has 24 elements."""
        cliffords = _build_single_qubit_cliffords()
        assert len(cliffords) == 24

    def test_unitary(self):
        """All Cliffords should be unitary."""
        for C in _build_single_qubit_cliffords():
            np.testing.assert_allclose(C @ C.conj().T, np.eye(2), atol=1e-10)

    def test_inverse_exists(self):
        """Every Clifford should have an inverse in the group."""
        cliffords = _build_single_qubit_cliffords()
        for i, C in enumerate(cliffords):
            inv_idx = _find_inverse_clifford(C, cliffords)
            product = cliffords[inv_idx] @ C
            # Should be proportional to identity
            fid = abs(np.trace(product)) ** 2 / 4
            assert fid > 0.99

    def test_compose_identity(self):
        """Composing a Clifford with its inverse should give identity."""
        cliffords = _build_single_qubit_cliffords()
        rng = np.random.default_rng(42)
        for _ in range(5):
            idx = int(rng.integers(0, len(cliffords)))
            total = _compose_cliffords([idx], cliffords)
            inv = _find_inverse_clifford(total, cliffords)
            product = cliffords[inv] @ total
            fid = abs(np.trace(product)) ** 2 / 4
            assert fid > 0.99


class TestRandomizedBenchmarking:
    def test_perfect_gates(self):
        """With no noise, survival should be ~1 for all lengths."""
        result = randomized_benchmarking(
            sequence_lengths=[1, 4, 16],
            num_samples=10,
            error_rate=0.0,
            seed=42,
        )
        for p in result.survival_probabilities:
            assert p > 0.99

    def test_noisy_decay(self):
        """With noise, survival should decay with sequence length."""
        result = randomized_benchmarking(
            sequence_lengths=[1, 8, 32, 64],
            num_samples=10,
            error_rate=0.05,
            seed=42,
        )
        assert result.survival_probabilities[0] > result.survival_probabilities[-1]
        assert 0 < result.alpha < 1
        assert result.error_per_clifford > 0

    def test_result_fields(self):
        result = randomized_benchmarking(sequence_lengths=[1, 4], num_samples=5, seed=0)
        assert len(result.sequence_lengths) == 2
        assert len(result.survival_probabilities) == 2
        assert 0 <= result.alpha <= 1
        assert result.error_per_clifford >= 0


class TestQuantumVolume:
    def test_low_noise_passes(self):
        """Low noise should give high QV."""
        result = quantum_volume(num_qubits=2, num_circuits=20, error_rate=0.001, seed=42)
        assert result.passed
        assert result.quantum_volume == 4

    def test_high_noise_fails(self):
        """Very high noise should fail QV test."""
        result = quantum_volume(num_qubits=3, num_circuits=20, error_rate=0.5, seed=42)
        assert not result.passed

    def test_result_fields(self):
        result = quantum_volume(num_qubits=2, num_circuits=5, seed=0)
        assert result.num_qubits == 2
        assert result.num_circuits == 5
        assert 0 <= result.heavy_output_prob <= 1
        assert result.threshold == 2 / 3

    def test_minimum_qubits(self):
        try:
            quantum_volume(num_qubits=1)
            assert False
        except ValueError:
            pass
