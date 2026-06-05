"""Tests for noise models."""

import numpy as np
import pytest

from quantumium.core.noise import (
    bit_flip,
    phase_flip,
    depolarizing,
    amplitude_damping,
    NoiseModel,
)


class TestNoiseChannels:
    def test_bit_flip_no_noise(self):
        ch = bit_flip(0.0)
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = ch.apply(rho)
        assert np.allclose(result, rho)

    def test_bit_flip_full_flip(self):
        ch = bit_flip(1.0)
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = ch.apply(rho)
        expected = np.array([[0, 0], [0, 1]], dtype=complex)
        assert np.allclose(result, expected)

    def test_bit_flip_half(self):
        ch = bit_flip(0.5)
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = ch.apply(rho)
        assert np.allclose(result, np.array([[0.5, 0], [0, 0.5]], dtype=complex))

    def test_phase_flip_preserves_populations(self):
        ch = phase_flip(0.5)
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        result = ch.apply(rho)
        assert result[0, 0] == pytest.approx(0.5)
        assert result[1, 1] == pytest.approx(0.5)

    def test_depolarizing_maximally_mixed(self):
        ch = depolarizing(1.0)
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = ch.apply(rho)
        assert np.allclose(result, np.eye(2) / 2, atol=1e-10)

    def test_amplitude_damping_decay(self):
        ch = amplitude_damping(1.0)
        rho = np.array([[0, 0], [0, 1]], dtype=complex)
        result = ch.apply(rho)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        assert np.allclose(result, expected)

    def test_amplitude_damping_ground_state_stable(self):
        ch = amplitude_damping(0.5)
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = ch.apply(rho)
        assert np.allclose(result, rho)

    def test_invalid_probability(self):
        with pytest.raises(ValueError):
            bit_flip(1.5)
        with pytest.raises(ValueError):
            phase_flip(-0.1)


class TestNoiseModel:
    def test_gate_specific_noise(self):
        model = NoiseModel()
        model.add_gate_noise("X", bit_flip(0.1))
        assert model.get_noise("X", 1) is not None
        assert model.get_noise("H", 1) is None

    def test_default_noise(self):
        model = NoiseModel(default_single_qubit_noise=depolarizing(0.01))
        assert model.get_noise("H", 1) is not None
        assert model.get_noise("CNOT", 2) is None
