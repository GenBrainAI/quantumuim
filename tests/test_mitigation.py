"""Tests for quantum error mitigation techniques."""

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import X, H as Hgate, Z, CNOT
from quantumium.mitigation.zne import (
    noise_scaled_circuits,
    richardson_extrapolate,
    linear_extrapolate,
    zne_expectation,
)
from quantumium.mitigation.measurement import (
    calibration_matrix,
    mitigate_counts,
    mitigate_probabilities,
)
from quantumium.mitigation.twirling import (
    pauli_twirl_cnot,
    twirled_circuit_ensemble,
    twirled_expectation,
)


# --- ZNE Tests ---


class TestNoiseScaledCircuits:
    def test_scale_1_unchanged(self):
        """Scale factor 1 produces the original circuit."""
        qc = QuantumCircuit(1)
        qc.apply(Hgate, 0)
        scaled = noise_scaled_circuits(qc, [1])
        assert len(scaled) == 1
        assert len(scaled[0].instructions) == 1

    def test_scale_3_triples_gates(self):
        """Scale factor 3: G G† G for each gate → 3 instructions per original gate."""
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        scaled = noise_scaled_circuits(qc, [3])
        assert len(scaled[0].instructions) == 3

    def test_scale_5(self):
        """Scale factor 5: G G† G G† G → 5 instructions per gate."""
        qc = QuantumCircuit(1)
        qc.apply(Hgate, 0)
        scaled = noise_scaled_circuits(qc, [5])
        assert len(scaled[0].instructions) == 5

    def test_multiple_scale_factors(self):
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        scaled = noise_scaled_circuits(qc, [1, 3, 5])
        assert len(scaled) == 3
        assert len(scaled[0].instructions) == 1
        assert len(scaled[1].instructions) == 3
        assert len(scaled[2].instructions) == 5

    def test_folded_circuit_preserves_statevector(self):
        """Folded circuits should produce same statevector (ideal simulation)."""
        qc = QuantumCircuit(2)
        qc.apply(Hgate, 0)
        qc.apply(CNOT, 0, 1)
        original_sv = qc.statevector()
        for k in [1, 3, 5]:
            scaled = noise_scaled_circuits(qc, [k])
            np.testing.assert_allclose(
                np.abs(scaled[0].statevector()),
                np.abs(original_sv),
                atol=1e-10,
            )

    def test_even_scale_raises(self):
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        try:
            noise_scaled_circuits(qc, [2])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestExtrapolation:
    def test_richardson_exact_linear(self):
        """Richardson extrapolation on linear data: y = 2 + 0.5x → y(0) = 2."""
        factors = [1.0, 3.0, 5.0]
        values = [2.5, 3.5, 4.5]
        result = richardson_extrapolate(factors, values)
        assert abs(result - 2.0) < 1e-10

    def test_richardson_quadratic(self):
        """Richardson with 3 points fits a quadratic exactly."""
        factors = [1.0, 3.0, 5.0]
        values = [f**2 + 1 for f in factors]  # y = x^2 + 1, y(0) = 1
        result = richardson_extrapolate(factors, values)
        assert abs(result - 1.0) < 1e-8

    def test_linear_extrapolate(self):
        factors = [1.0, 3.0]
        values = [0.9, 0.7]  # slope = -0.1, y(0) = 1.0
        result = linear_extrapolate(factors, values)
        assert abs(result - 1.0) < 1e-10

    def test_zne_ideal_simulator(self):
        """On ideal simulator, ZNE should return same value as direct evaluation."""
        qc = QuantumCircuit(1)
        qc.apply(Hgate, 0)
        qc.apply(Z, 0)

        def z_expectation(sv):
            probs = np.abs(sv) ** 2
            return probs[0] - probs[1]

        direct = z_expectation(qc.statevector())
        zne_val = zne_expectation(qc, z_expectation)
        assert abs(zne_val - direct) < 0.05


# --- Measurement Error Mitigation Tests ---


class TestCalibrationMatrix:
    def test_no_errors(self):
        """With zero error rates, calibration matrix is identity."""
        rates = {0: (0.0, 0.0), 1: (0.0, 0.0)}
        cal = calibration_matrix(rates)
        np.testing.assert_allclose(cal, np.eye(4), atol=1e-10)

    def test_single_qubit(self):
        rates = {0: (0.1, 0.05)}  # p0_given_1=0.1, p1_given_0=0.05
        cal = calibration_matrix(rates)
        expected = np.array([[0.95, 0.1], [0.05, 0.9]])
        np.testing.assert_allclose(cal, expected, atol=1e-10)

    def test_columns_sum_to_one(self):
        """Calibration matrix columns should sum to 1 (stochastic)."""
        rates = {0: (0.05, 0.03), 1: (0.08, 0.02)}
        cal = calibration_matrix(rates)
        col_sums = cal.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones(4), atol=1e-10)

    def test_shape(self):
        rates = {0: (0.05, 0.03), 1: (0.08, 0.02), 2: (0.04, 0.01)}
        cal = calibration_matrix(rates)
        assert cal.shape == (8, 8)


class TestMitigateCounts:
    def test_identity_correction(self):
        """With no readout errors, mitigation is a no-op."""
        cal = np.eye(4)
        counts = {"00": 500, "11": 500}
        result = mitigate_counts(cal, counts, 2)
        assert abs(result.get("00", 0) - 0.5) < 0.01
        assert abs(result.get("11", 0) - 0.5) < 0.01

    def test_corrects_bias(self):
        """Mitigation should correct readout bias toward true distribution."""
        rates = {0: (0.1, 0.1)}
        cal = calibration_matrix(rates)
        # Prepared |0⟩ but 10% reads as 1
        counts = {"0": 900, "1": 100}
        result = mitigate_counts(cal, counts, 1)
        assert result.get("0", 0) > 0.95

    def test_mitigate_probabilities_normalized(self):
        cal = calibration_matrix({0: (0.05, 0.03)})
        noisy = np.array([0.6, 0.4])
        corrected = mitigate_probabilities(cal, noisy)
        assert abs(corrected.sum() - 1.0) < 1e-10
        assert all(c >= 0 for c in corrected)


# --- Pauli Twirling Tests ---


class TestPauliTwirl:
    def test_twirled_circuit_same_structure(self):
        """Twirled circuit should have same number of qubits."""
        qc = QuantumCircuit(2)
        qc.apply(Hgate, 0)
        qc.apply(CNOT, 0, 1)
        twirled = pauli_twirl_cnot(qc, rng=np.random.default_rng(42))
        assert twirled.num_qubits == 2

    def test_ensemble_size(self):
        qc = QuantumCircuit(2)
        qc.apply(CNOT, 0, 1)
        ensemble = twirled_circuit_ensemble(qc, num_samples=10, seed=42)
        assert len(ensemble) == 10

    def test_twirled_expectation_ideal(self):
        """On ideal simulator, twirled expectation should match direct."""
        qc = QuantumCircuit(2)
        qc.apply(Hgate, 0)
        qc.apply(CNOT, 0, 1)

        def z0_expectation(sv):
            n = int(np.log2(len(sv)))
            probs = np.abs(sv) ** 2
            exp = 0.0
            for i, p in enumerate(probs):
                bit = (i >> (n - 1)) & 1
                exp += p * (1 - 2 * bit)
            return exp

        direct = z0_expectation(qc.statevector())
        twirled = twirled_expectation(qc, z0_expectation, num_samples=30, seed=42)
        assert abs(twirled - direct) < 0.15

    def test_no_cnot_unchanged(self):
        """Circuit with no CNOTs should be unchanged by twirling."""
        qc = QuantumCircuit(1)
        qc.apply(Hgate, 0)
        qc.apply(X, 0)
        twirled = pauli_twirl_cnot(qc, rng=np.random.default_rng(42))
        assert len(twirled.instructions) == 2

    def test_deterministic_with_seed(self):
        """Same seed should produce same twirled circuit."""
        qc = QuantumCircuit(2)
        qc.apply(CNOT, 0, 1)
        t1 = twirled_circuit_ensemble(qc, num_samples=5, seed=123)
        t2 = twirled_circuit_ensemble(qc, num_samples=5, seed=123)
        for c1, c2 in zip(t1, t2):
            sv1 = c1.statevector()
            sv2 = c2.statevector()
            np.testing.assert_allclose(sv1, sv2, atol=1e-10)
