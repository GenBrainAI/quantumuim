"""Tests for Hamiltonian simulation."""

import numpy as np

from quantumium.algorithms.hamiltonian_sim import (
    exact_evolution,
    first_order_trotter,
    trotter_error,
    simulate_evolution,
    transverse_field_ising,
    pauli_x,
    zz_interaction,
)


class TestExactEvolution:
    def test_identity_evolution(self):
        """e^{-i·0·t} = I"""
        H = np.zeros((2, 2), dtype=complex)
        U = exact_evolution(H, 1.0)
        np.testing.assert_allclose(U, np.eye(2, dtype=complex), atol=1e-10)

    def test_pauli_z_evolution(self):
        """e^{-iZt} is diagonal with phases."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        t = np.pi / 4
        U = exact_evolution(Z, t)
        expected = np.diag([np.exp(-1j * t), np.exp(1j * t)])
        np.testing.assert_allclose(U, expected, atol=1e-10)

    def test_unitary_output(self):
        """Evolution operator must be unitary."""
        H = np.array([[1, 0.5], [0.5, -1]], dtype=complex)
        U = exact_evolution(H, 2.0)
        np.testing.assert_allclose(U @ U.conj().T, np.eye(2), atol=1e-10)


class TestTrotterDecomposition:
    def test_single_term_exact(self):
        """With one term, Trotter is exact regardless of steps."""
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        U_exact = exact_evolution(H, 1.0)
        U_trotter = first_order_trotter([H], 1.0, steps=1)
        np.testing.assert_allclose(U_trotter, U_exact, atol=1e-10)

    def test_first_order_converges(self):
        """More Trotter steps → lower error."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        terms = [X, Z]
        err_10 = trotter_error(terms, 1.0, 10, order=1)
        err_100 = trotter_error(terms, 1.0, 100, order=1)
        assert err_100 < err_10

    def test_second_order_better(self):
        """Second-order Trotter should have lower error than first-order."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        terms = [X, Z]
        err_1st = trotter_error(terms, 1.0, 10, order=1)
        err_2nd = trotter_error(terms, 1.0, 10, order=2)
        assert err_2nd < err_1st

    def test_second_order_converges_faster(self):
        """Second-order error scales as O(1/n²) vs O(1/n) for first-order."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        terms = [X, Z]
        err_10 = trotter_error(terms, 1.0, 10, order=2)
        err_20 = trotter_error(terms, 1.0, 20, order=2)
        ratio = err_10 / err_20
        assert ratio > 3.0  # Should be ~4 for O(1/n²)

    def test_trotter_unitary(self):
        """Trotter product must be unitary."""
        terms = transverse_field_ising(2)
        U = first_order_trotter(terms, 1.0, 10)
        np.testing.assert_allclose(U @ U.conj().T, np.eye(4), atol=1e-10)


class TestPauliOperators:
    def test_pauli_x_qubit_0(self):
        """X on qubit 0 of 2-qubit system = X ⊗ I."""
        X0 = pauli_x(0, 2)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        expected = np.kron(X, np.eye(2, dtype=complex))
        np.testing.assert_allclose(X0, expected, atol=1e-10)

    def test_zz_eigenvalues(self):
        """Z_0 Z_1 should have eigenvalues ±1."""
        ZZ = zz_interaction(0, 1, 2)
        eigenvalues = np.sort(np.linalg.eigvalsh(ZZ))
        expected = np.array([-1, -1, 1, 1], dtype=float)
        np.testing.assert_allclose(eigenvalues, expected, atol=1e-10)


class TestIsingModel:
    def test_term_count(self):
        """n-qubit Ising model: (n-1) ZZ terms + n X terms."""
        terms = transverse_field_ising(3)
        assert len(terms) == 5  # 2 ZZ + 3 X

    def test_hermitian_terms(self):
        """All Ising Hamiltonian terms should be Hermitian."""
        for term in transverse_field_ising(3):
            np.testing.assert_allclose(term, term.conj().T, atol=1e-10)


class TestSimulateEvolution:
    def test_circuit_output(self):
        """simulate_evolution should return a valid circuit."""
        terms = transverse_field_ising(2)
        qc = simulate_evolution(2, terms, time=1.0, steps=5)
        sv = qc.statevector()
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    def test_matches_exact(self):
        """With enough steps, circuit should match exact evolution."""
        terms = transverse_field_ising(2)
        H = sum(terms)
        U_exact = exact_evolution(H, 0.5)
        expected_sv = U_exact @ np.array([1, 0, 0, 0], dtype=complex)

        qc = simulate_evolution(2, terms, time=0.5, steps=50, order=2)
        actual_sv = qc.statevector()

        fidelity = abs(np.dot(expected_sv.conj(), actual_sv)) ** 2
        assert fidelity > 0.999
