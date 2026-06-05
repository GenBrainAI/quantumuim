"""Tests for quantum counting and amplitude estimation."""

import math

import numpy as np

from quantumium.algorithms.quantum_counting import (
    quantum_counting,
    amplitude_estimation,
    iterative_amplitude_estimation,
)
from quantumium.algorithms.grover import grover_oracle
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H


# ============================================================
# Quantum Counting Tests
# ============================================================

class TestQuantumCounting:
    def test_single_marked_2qubits(self):
        """Count 1 marked state out of 4."""
        result = quantum_counting(
            num_search_qubits=2,
            marked_states=[0],
            num_counting_qubits=4,
        )
        assert abs(result.estimated_count - 1.0) < 0.5
        assert result.total_states == 4

    def test_two_marked_2qubits(self):
        """Count 2 marked states out of 4."""
        result = quantum_counting(
            num_search_qubits=2,
            marked_states=[0, 1],
            num_counting_qubits=4,
        )
        assert abs(result.estimated_count - 2.0) < 0.5

    def test_single_marked_3qubits(self):
        """Count 1 out of 8."""
        result = quantum_counting(
            num_search_qubits=3,
            marked_states=[5],
            num_counting_qubits=5,
        )
        assert abs(result.estimated_count - 1.0) < 1.0

    def test_multiple_marked_3qubits(self):
        """Count 3 out of 8."""
        result = quantum_counting(
            num_search_qubits=3,
            marked_states=[1, 3, 7],
            num_counting_qubits=5,
        )
        assert abs(result.estimated_count - 3.0) < 1.5

    def test_theta_range(self):
        """Theta should be in [0, pi]."""
        result = quantum_counting(
            num_search_qubits=2,
            marked_states=[0],
            num_counting_qubits=3,
        )
        assert 0 <= result.theta <= math.pi

    def test_confidence_interval(self):
        """Confidence interval should contain the true count."""
        result = quantum_counting(
            num_search_qubits=2,
            marked_states=[0, 3],
            num_counting_qubits=5,
        )
        lo, hi = result.confidence_interval
        assert lo <= 2.0 + 0.5  # some slack
        assert hi >= 2.0 - 0.5

    def test_more_counting_qubits_better_precision(self):
        """More counting qubits should give a more accurate estimate."""
        r3 = quantum_counting(2, [0], num_counting_qubits=3)
        r6 = quantum_counting(2, [0], num_counting_qubits=6)
        err3 = abs(r3.estimated_count - 1.0)
        err6 = abs(r6.estimated_count - 1.0)
        assert err6 <= err3 + 0.01

    def test_no_marked_raises(self):
        try:
            quantum_counting(2, [])
            assert False
        except ValueError:
            pass

    def test_all_marked_raises(self):
        try:
            quantum_counting(2, [0, 1, 2, 3])
            assert False
        except ValueError:
            pass

    def test_result_fields(self):
        result = quantum_counting(2, [1], num_counting_qubits=3)
        assert result.total_states == 4
        assert result.num_counting_qubits == 3
        assert result.estimated_count >= 0
        assert len(result.confidence_interval) == 2


# ============================================================
# Canonical Amplitude Estimation Tests
# ============================================================

class TestAmplitudeEstimation:
    def test_quarter_marked(self):
        """1/4 of states marked → probability 0.25."""
        oracle = QuantumCircuit(2)
        oracle_matrix = grover_oracle(2, [0]).matrix
        from quantumium.core.gate import Gate
        oracle.apply(Gate(oracle_matrix, "O"), 0, 1)

        result = amplitude_estimation(
            oracle_circuit=oracle,
            num_evaluation_qubits=4,
        )
        assert abs(result.estimated_probability - 0.25) < 0.15

    def test_half_marked(self):
        """2/4 marked → probability 0.5."""
        oracle = QuantumCircuit(2)
        oracle_matrix = grover_oracle(2, [0, 1]).matrix
        from quantumium.core.gate import Gate
        oracle.apply(Gate(oracle_matrix, "O"), 0, 1)

        result = amplitude_estimation(
            oracle_circuit=oracle,
            num_evaluation_qubits=4,
        )
        assert abs(result.estimated_probability - 0.5) < 0.15

    def test_result_fields(self):
        oracle = QuantumCircuit(2)
        oracle_matrix = grover_oracle(2, [0]).matrix
        from quantumium.core.gate import Gate
        oracle.apply(Gate(oracle_matrix, "O"), 0, 1)

        result = amplitude_estimation(
            oracle_circuit=oracle,
            num_evaluation_qubits=3,
        )
        assert 0 <= result.estimated_amplitude <= 1
        assert 0 <= result.estimated_probability <= 1
        assert 0 <= result.theta <= math.pi
        assert result.num_evaluation_qubits == 3


# ============================================================
# Iterative Amplitude Estimation Tests
# ============================================================

class TestIterativeAE:
    def test_single_marked(self):
        """1 out of 4."""
        result = iterative_amplitude_estimation(
            num_search_qubits=2,
            marked_states=[0],
            num_rounds=5,
        )
        # theta = arcsin(sqrt(1/4)) = pi/6
        expected_prob = 0.25
        assert abs(result.estimated_probability - expected_prob) < 0.15

    def test_two_marked(self):
        """2 out of 4."""
        result = iterative_amplitude_estimation(
            num_search_qubits=2,
            marked_states=[0, 1],
            num_rounds=5,
        )
        assert abs(result.estimated_probability - 0.5) < 0.2

    def test_three_marked_out_of_eight(self):
        """3 out of 8."""
        result = iterative_amplitude_estimation(
            num_search_qubits=3,
            marked_states=[0, 1, 2],
            num_rounds=5,
        )
        assert abs(result.estimated_probability - 3 / 8) < 0.2

    def test_more_rounds_better(self):
        """More rounds → closer estimate."""
        r2 = iterative_amplitude_estimation(2, [0], num_rounds=2)
        r5 = iterative_amplitude_estimation(2, [0], num_rounds=5)
        true_prob = 0.25
        err2 = abs(r2.estimated_probability - true_prob)
        err5 = abs(r5.estimated_probability - true_prob)
        assert err5 <= err2 + 0.01

    def test_result_fields(self):
        result = iterative_amplitude_estimation(2, [0], num_rounds=3)
        assert 0 <= result.estimated_amplitude <= 1
        assert 0 <= result.estimated_probability <= 1
        assert result.num_rounds == 3
        assert result.total_oracle_calls > 0

    def test_no_marked_raises(self):
        try:
            iterative_amplitude_estimation(2, [])
            assert False
        except ValueError:
            pass

    def test_all_marked_raises(self):
        try:
            iterative_amplitude_estimation(2, [0, 1, 2, 3])
            assert False
        except ValueError:
            pass
