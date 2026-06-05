"""Tests for VQE implementation."""

import numpy as np

from quantumium.algorithms.vqe import (
    pauli_z_expectation,
    pauli_zz_expectation,
    hamiltonian_expectation,
    hardware_efficient_ansatz,
    h2_hamiltonian,
    vqe,
)
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import X, H as Hgate


class TestPauliExpectations:
    def test_z_on_zero_state(self):
        """⟨0|Z|0⟩ = +1"""
        qc = QuantumCircuit(1)
        sv = qc.statevector()
        assert abs(pauli_z_expectation(sv, 0, 1) - 1.0) < 1e-10

    def test_z_on_one_state(self):
        """⟨1|Z|1⟩ = -1"""
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        sv = qc.statevector()
        assert abs(pauli_z_expectation(sv, 0, 1) - (-1.0)) < 1e-10

    def test_z_on_plus_state(self):
        """⟨+|Z|+⟩ = 0"""
        qc = QuantumCircuit(1)
        qc.apply(Hgate, 0)
        sv = qc.statevector()
        assert abs(pauli_z_expectation(sv, 0, 1)) < 1e-10

    def test_zz_same_state(self):
        """⟨00|Z0Z1|00⟩ = +1"""
        qc = QuantumCircuit(2)
        sv = qc.statevector()
        assert abs(pauli_zz_expectation(sv, 0, 1, 2) - 1.0) < 1e-10

    def test_zz_opposite_state(self):
        """⟨01|Z0Z1|01⟩ = -1"""
        qc = QuantumCircuit(2)
        qc.apply(X, 1)
        sv = qc.statevector()
        assert abs(pauli_zz_expectation(sv, 0, 1, 2) - (-1.0)) < 1e-10


class TestAnsatz:
    def test_correct_param_count(self):
        params = np.zeros(4)  # 2 qubits * 2 params * 1 layer
        qc = hardware_efficient_ansatz(2, params, depth=1)
        assert qc.num_qubits == 2

    def test_wrong_param_count_raises(self):
        try:
            hardware_efficient_ansatz(2, np.zeros(3), depth=1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_output_normalized(self):
        params = np.random.uniform(-np.pi, np.pi, size=8)
        qc = hardware_efficient_ansatz(2, params, depth=2)
        sv = qc.statevector()
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10


class TestH2Hamiltonian:
    def test_shape(self):
        H = h2_hamiltonian()
        assert H.shape == (4, 4)

    def test_hermitian(self):
        H = h2_hamiltonian()
        np.testing.assert_allclose(H, H.conj().T, atol=1e-10)

    def test_ground_state_energy(self):
        """Exact diagonalization should give a reasonable ground state energy."""
        H = h2_hamiltonian()
        eigenvalues = np.linalg.eigvalsh(H)
        ground_energy = eigenvalues[0]
        assert ground_energy < -1.0
        assert ground_energy == min(eigenvalues)


class TestVQE:
    def test_h2_finds_ground_state(self):
        """VQE should find energy close to exact ground state for H2."""
        H = h2_hamiltonian()
        exact = np.linalg.eigvalsh(H)[0]

        result = vqe(
            num_qubits=2,
            hamiltonian=H,
            ansatz_depth=2,
            max_iter=200,
            learning_rate=0.05,
            num_starts=5,
        )

        assert result.optimal_energy < exact + 0.15
        assert len(result.history) > 0
        assert result.num_iterations > 0

    def test_trivial_hamiltonian(self):
        """For H = -Z (1 qubit), ground state energy = -1."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        H = -Z
        result = vqe(
            num_qubits=1,
            hamiltonian=H,
            ansatz_depth=1,
            max_iter=50,
            num_starts=3,
        )
        assert result.optimal_energy < -0.9
