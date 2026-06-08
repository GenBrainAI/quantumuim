"""Cross-backend statistical validation and noise model calibration tests."""

from __future__ import annotations

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H
from quantumium.core.noise import NoiseModel, depolarizing
from quantumium.backends import LocalSimulatorBackend

from .conftest import chi_squared_test, hellinger_fidelity


SHOTS = 8192


class TestCrossBackendValidation:
    """Validate that the local simulator matches ideal quantum mechanics."""

    def test_bell_distribution_matches_theory(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=SHOTS)
        expected = {"00": 0.5, "11": 0.5}
        p_value = chi_squared_test(result.counts, expected, SHOTS)
        assert p_value > 0.01, f"Bell state deviates from theory: p={p_value:.4f}"

    def test_ghz3_distribution_matches_theory(self, local_backend, ghz3_circuit):
        result = local_backend.run(ghz3_circuit, shots=SHOTS)
        expected = {"000": 0.5, "111": 0.5}
        p_value = chi_squared_test(result.counts, expected, SHOTS)
        assert p_value > 0.01

    def test_uniform_superposition_2q(self, local_backend):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(H, 1)
        result = local_backend.run(qc, shots=SHOTS)
        expected = {f"{i:02b}": 0.25 for i in range(4)}
        p_value = chi_squared_test(result.counts, expected, SHOTS)
        assert p_value > 0.01

    def test_uniform_superposition_3q(self, local_backend):
        qc = QuantumCircuit(3)
        for i in range(3):
            qc.apply(H, i)
        result = local_backend.run(qc, shots=SHOTS)
        expected = {f"{i:03b}": 0.125 for i in range(8)}
        p_value = chi_squared_test(result.counts, expected, SHOTS)
        assert p_value > 0.01

    def test_hellinger_fidelity_bell_vs_theory(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=SHOTS)
        theory = {"00": 0.5, "11": 0.5}
        fid = hellinger_fidelity(result.probabilities, theory)
        assert fid > 0.999, f"Hellinger fidelity vs theory: {fid:.6f}"


class TestNoiseModelCalibration:
    """Test that noise models produce expected degradation in results."""

    def test_noiseless_bell_pure(self, bell_circuit):
        qc = bell_circuit
        result = LocalSimulatorBackend().run(qc, shots=SHOTS)
        p_valid = result.probabilities.get("00", 0) + result.probabilities.get("11", 0)
        assert p_valid > 0.99

    def test_depolarizing_degrades_fidelity(self, bell_circuit):
        noise = NoiseModel(default_single_qubit_noise=depolarizing(0.1))
        rho_noisy = bell_circuit.density_matrix(noise_model=noise)
        rho_ideal = bell_circuit.density_matrix()
        fidelity = np.real(np.trace(rho_ideal @ rho_noisy))
        assert 0.5 < fidelity < 1.0, f"Noisy fidelity out of range: {fidelity:.4f}"

    def test_noise_increases_with_error_rate(self, bell_circuit):
        fidelities = []
        for p in [0.01, 0.05, 0.1]:
            noise = NoiseModel(default_single_qubit_noise=depolarizing(p))
            rho_noisy = bell_circuit.density_matrix(noise_model=noise)
            rho_ideal = bell_circuit.density_matrix()
            fid = np.real(np.trace(rho_ideal @ rho_noisy))
            fidelities.append(fid)
        assert fidelities[0] > fidelities[1] > fidelities[2], (
            f"Fidelity not monotonically decreasing: {fidelities}"
        )

    def test_noisy_measurement_adds_errors(self, bell_circuit):
        noise = NoiseModel(default_single_qubit_noise=depolarizing(0.1))
        counts = bell_circuit.noisy_measure(noise, shots=SHOTS)
        p01 = counts.get("01", 0) / SHOTS
        p10 = counts.get("10", 0) / SHOTS
        assert p01 + p10 > 0.01, "Expected some noise-induced errors in 01/10"

    def test_zero_noise_matches_ideal(self, bell_circuit):
        noise = NoiseModel(default_single_qubit_noise=depolarizing(0.0))
        rho_noisy = bell_circuit.density_matrix(noise_model=noise)
        rho_ideal = bell_circuit.density_matrix()
        assert np.allclose(rho_noisy, rho_ideal, atol=1e-10)

    def test_noise_model_gate_specific(self, bell_circuit):
        noise = NoiseModel()
        noise.add_gate_noise("H", depolarizing(0.2))
        rho_noisy = bell_circuit.density_matrix(noise_model=noise)
        rho_ideal = bell_circuit.density_matrix()
        fid = np.real(np.trace(rho_ideal @ rho_noisy))
        assert fid < 1.0, "Gate-specific noise should degrade fidelity"


class TestExecutionResultAPI:
    def test_probabilities_sum_to_one(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=SHOTS)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-10

    def test_most_frequent_deterministic(self, local_backend, x_gate_circuit):
        result = local_backend.run(x_gate_circuit, shots=1000)
        assert result.most_frequent == "1"

    def test_counts_keys_are_strings(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=100)
        for k in result.counts:
            assert isinstance(k, str)

    def test_counts_values_are_ints(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=100)
        for v in result.counts.values():
            assert isinstance(v, int)

    def test_metadata_default_empty(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=100)
        assert result.metadata == {}


class TestExistingTestsUnbroken:
    def test_import_backends(self):
        from quantumium.backends import LocalSimulatorBackend

        assert LocalSimulatorBackend is not None

    def test_import_qiskit_backend(self):
        from quantumium.backends.qiskit_backend import QiskitBackend

        assert QiskitBackend is not None

    def test_import_cirq_backend(self):
        from quantumium.backends.cirq_backend import CirqBackend

        assert CirqBackend is not None

    def test_import_braket_backend(self):
        from quantumium.backends.braket_backend import BraketBackend

        assert BraketBackend is not None
