"""Tests for density matrix simulation and noisy circuit execution."""

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, CNOT, I
from quantumium.core.noise import (
    NoiseModel,
    bit_flip,
    phase_flip,
    depolarizing,
    amplitude_damping,
)


class TestDensityMatrixNoiseless:
    def test_initial_state(self):
        qc = QuantumCircuit(1)
        rho = qc.density_matrix()
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        np.testing.assert_allclose(rho, expected, atol=1e-10)

    def test_x_gate(self):
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        rho = qc.density_matrix()
        expected = np.array([[0, 0], [0, 1]], dtype=complex)
        np.testing.assert_allclose(rho, expected, atol=1e-10)

    def test_hadamard_creates_superposition(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        rho = qc.density_matrix()
        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        np.testing.assert_allclose(rho, expected, atol=1e-10)

    def test_bell_state(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        rho = qc.density_matrix()
        assert rho.shape == (4, 4)
        assert abs(np.trace(rho) - 1.0) < 1e-10
        sv = qc.statevector()
        expected_rho = np.outer(sv, sv.conj())
        np.testing.assert_allclose(rho, expected_rho, atol=1e-10)

    def test_matches_statevector(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(X, 2)
        rho = qc.density_matrix()
        sv = qc.statevector()
        expected = np.outer(sv, sv.conj())
        np.testing.assert_allclose(rho, expected, atol=1e-10)

    def test_trace_is_one(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(H, 1)
        qc.apply(CNOT, 0, 1)
        rho = qc.density_matrix()
        assert abs(np.trace(rho) - 1.0) < 1e-10

    def test_positive_semidefinite(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        rho = qc.density_matrix()
        eigenvalues = np.linalg.eigvalsh(rho)
        assert all(ev > -1e-10 for ev in eigenvalues)


class TestDensityMatrixNoisy:
    def test_bit_flip_noise_degrades_purity(self):
        noise = NoiseModel(default_single_qubit_noise=bit_flip(0.1))
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        rho = qc.density_matrix(noise_model=noise)
        purity = float(np.real(np.trace(rho @ rho)))
        assert purity < 1.0 - 1e-6

    def test_no_noise_preserves_purity(self):
        noise = NoiseModel()
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        rho = qc.density_matrix(noise_model=noise)
        purity = float(np.real(np.trace(rho @ rho)))
        assert abs(purity - 1.0) < 1e-10

    def test_depolarizing_noise(self):
        noise = NoiseModel(default_single_qubit_noise=depolarizing(0.5))
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        rho = qc.density_matrix(noise_model=noise)
        assert abs(np.trace(rho) - 1.0) < 1e-10
        assert float(np.real(np.trace(rho @ rho))) < 1.0

    def test_amplitude_damping_decays_excited_state(self):
        noise = NoiseModel(default_single_qubit_noise=amplitude_damping(0.5))
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        rho = qc.density_matrix(noise_model=noise)
        assert rho[0, 0].real > 0.2
        assert rho[1, 1].real < 0.8

    def test_phase_flip_destroys_coherence(self):
        noise = NoiseModel(default_single_qubit_noise=phase_flip(0.5))
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        rho = qc.density_matrix(noise_model=noise)
        assert abs(rho[0, 1]) < 1e-10

    def test_gate_specific_noise(self):
        noise = NoiseModel()
        noise.add_gate_noise("X", bit_flip(0.3))
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        rho = qc.density_matrix(noise_model=noise)
        assert rho[0, 0].real > 0.15
        qc2 = QuantumCircuit(1)
        qc2.apply(H, 0)
        rho2 = qc2.density_matrix(noise_model=noise)
        purity = float(np.real(np.trace(rho2 @ rho2)))
        assert abs(purity - 1.0) < 1e-10

    def test_two_qubit_noise(self):
        noise = NoiseModel(
            default_single_qubit_noise=depolarizing(0.05),
            default_two_qubit_noise=depolarizing(0.1),
        )
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        rho = qc.density_matrix(noise_model=noise)
        assert abs(np.trace(rho) - 1.0) < 1e-10
        purity = float(np.real(np.trace(rho @ rho)))
        assert purity < 1.0

    def test_high_noise_approaches_maximally_mixed(self):
        noise = NoiseModel(default_single_qubit_noise=depolarizing(1.0))
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        rho = qc.density_matrix(noise_model=noise)
        np.testing.assert_allclose(rho[0, 0].real, 0.5, atol=0.05)
        np.testing.assert_allclose(rho[1, 1].real, 0.5, atol=0.05)


class TestNoisyMeasure:
    def test_noisy_measure_returns_counts(self):
        noise = NoiseModel(default_single_qubit_noise=bit_flip(0.01))
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        counts = qc.noisy_measure(noise, shots=1000)
        assert sum(counts.values()) == 1000
        assert counts.get("1", 0) > 900

    def test_noisy_measure_all_bitstrings_valid(self):
        noise = NoiseModel(default_single_qubit_noise=depolarizing(0.3))
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        counts = qc.noisy_measure(noise, shots=2000)
        for bitstring in counts:
            assert len(bitstring) == 2
            assert all(c in "01" for c in bitstring)

    def test_heavy_noise_spreads_distribution(self):
        noise = NoiseModel(default_single_qubit_noise=depolarizing(0.8))
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        counts = qc.noisy_measure(noise, shots=5000)
        assert len(counts) >= 2
