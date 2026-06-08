"""Tests for quantum state tomography."""

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, CNOT
from quantumium.core.noise import NoiseModel, depolarizing
from quantumium.tomography.state_tomography import (
    tomography,
    linear_inversion,
    maximum_likelihood,
    collect_tomography_data,
    state_fidelity,
    _project_to_physical,
    _measurement_bases,
)


class TestMeasurementBases:
    def test_single_qubit_bases(self):
        bases = _measurement_bases(1)
        assert set(bases) == {"X", "Y", "Z"}

    def test_two_qubit_bases(self):
        bases = _measurement_bases(2)
        assert len(bases) == 9
        assert "XX" in bases
        assert "ZZ" in bases


class TestProjectToPhysical:
    def test_already_physical(self):
        rho = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
        result = _project_to_physical(rho)
        assert abs(np.trace(result) - 1.0) < 1e-10
        assert all(np.linalg.eigvalsh(result) > -1e-10)

    def test_negative_eigenvalue(self):
        rho = np.array([[1.1, 0], [0, -0.1]], dtype=complex)
        result = _project_to_physical(rho)
        assert abs(np.trace(result) - 1.0) < 1e-10
        assert all(np.linalg.eigvalsh(result) > -1e-10)


class TestStateFidelity:
    def test_same_state(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        assert abs(state_fidelity(rho, rho) - 1.0) < 1e-10

    def test_orthogonal_states(self):
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        rho1 = np.array([[0, 0], [0, 1]], dtype=complex)
        assert state_fidelity(rho0, rho1) < 1e-10

    def test_mixed_state(self):
        rho_pure = np.array([[1, 0], [0, 0]], dtype=complex)
        rho_mixed = np.eye(2, dtype=complex) / 2
        fid = state_fidelity(rho_pure, rho_mixed)
        assert 0.4 < fid < 0.6


class TestLinearInversion:
    def test_zero_state(self):
        qc = QuantumCircuit(1)
        measurements = collect_tomography_data(qc, shots_per_basis=10000)
        rho = linear_inversion(measurements, 1)
        rho = _project_to_physical(rho)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        np.testing.assert_allclose(rho, expected, atol=0.1)

    def test_plus_state(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        measurements = collect_tomography_data(qc, shots_per_basis=10000)
        rho = linear_inversion(measurements, 1)
        rho = _project_to_physical(rho)
        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        np.testing.assert_allclose(rho, expected, atol=0.15)


class TestMaximumLikelihood:
    def test_zero_state_mle(self):
        qc = QuantumCircuit(1)
        measurements = collect_tomography_data(qc, shots_per_basis=10000)
        rho = maximum_likelihood(measurements, 1, max_iter=50)
        assert abs(np.trace(rho) - 1.0) < 1e-6
        assert rho[0, 0].real > 0.85

    def test_mle_is_physical(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        measurements = collect_tomography_data(qc, shots_per_basis=5000)
        rho = maximum_likelihood(measurements, 1, max_iter=50)
        assert abs(np.trace(rho) - 1.0) < 1e-6
        eigvals = np.linalg.eigvalsh(rho)
        assert all(ev > -1e-10 for ev in eigvals)


class TestTomography:
    def test_single_qubit_linear(self):
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        result = tomography(qc, method="linear", shots_per_basis=10000)
        assert result.num_qubits == 1
        assert result.method == "linear"
        assert result.fidelity > 0.9

    def test_single_qubit_mle(self):
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        result = tomography(qc, method="mle", shots_per_basis=10000)
        assert result.fidelity > 0.9

    def test_bell_state_tomography(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        result = tomography(qc, method="linear", shots_per_basis=10000)
        assert result.num_qubits == 2
        assert result.fidelity > 0.8
        rho = result.density_matrix
        assert abs(np.trace(rho) - 1.0) < 1e-6

    def test_noisy_tomography(self):
        noise = NoiseModel(default_single_qubit_noise=depolarizing(0.05))
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        result = tomography(qc, method="linear", shots_per_basis=10000, noise_model=noise)
        assert result.fidelity < 0.99

    def test_invalid_method(self):
        qc = QuantumCircuit(1)
        with pytest.raises(ValueError, match="Unknown method"):
            tomography(qc, method="invalid")

    def test_tomography_has_measurements(self):
        qc = QuantumCircuit(1)
        result = tomography(qc, method="linear", shots_per_basis=1000)
        assert "X" in result.measurements
        assert "Y" in result.measurements
        assert "Z" in result.measurements
