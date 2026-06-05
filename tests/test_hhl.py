"""Tests for HHL quantum linear systems algorithm."""

import numpy as np

from quantumium.algorithms.hhl import (
    _hermitian_exp,
    hhl_solve,
)


class TestHermitianExp:
    def test_identity(self):
        """e^(i*0*I) = I."""
        I = np.eye(2, dtype=complex)
        result = _hermitian_exp(I, 0)
        np.testing.assert_allclose(result, I, atol=1e-10)

    def test_unitary(self):
        """e^(iAt) should be unitary for Hermitian A."""
        A = np.array([[1, 0.5], [0.5, 2]], dtype=complex)
        U = _hermitian_exp(A, 1.0)
        np.testing.assert_allclose(U @ U.conj().T, np.eye(2), atol=1e-10)

    def test_inverse(self):
        """e^(iAt) * e^(-iAt) = I."""
        A = np.array([[2, 1], [1, 3]], dtype=complex)
        U = _hermitian_exp(A, 0.5)
        U_inv = _hermitian_exp(A, -0.5)
        np.testing.assert_allclose(U @ U_inv, np.eye(2), atol=1e-10)

    def test_pauli_z(self):
        """e^(iZt) for Pauli Z."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        t = np.pi / 4
        result = _hermitian_exp(Z, t)
        expected = np.diag([np.exp(1j * t), np.exp(-1j * t)])
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestHHLSolve:
    def test_diagonal_system(self):
        """Diagonal A — eigenvalues are exact, best case for HHL."""
        A = np.array([[1, 0], [0, 2]], dtype=complex)
        b = np.array([1, 1], dtype=complex)
        result = hhl_solve(A, b, num_clock=3)
        assert result.fidelity > 0.99
        assert result.success_probability > 0.01

    def test_identity_matrix(self):
        """A = I: solution equals b (normalized)."""
        A = np.eye(2, dtype=complex)
        b = np.array([0.6, 0.8], dtype=complex)
        result = hhl_solve(A, b, num_clock=2)
        assert result.fidelity > 0.99

    def test_non_diagonal_hermitian(self):
        """Non-diagonal Hermitian matrix with good precision."""
        A = np.array([[2, 1], [1, 3]], dtype=complex)
        b = np.array([1, 0], dtype=complex)
        result = hhl_solve(A, b, num_clock=4)
        assert result.fidelity > 0.99

    def test_hermitian_required(self):
        A = np.array([[1, 2], [0, 1]], dtype=complex)
        b = np.array([1, 0], dtype=complex)
        try:
            hhl_solve(A, b)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_dimension_mismatch(self):
        A = np.array([[1, 0], [0, 1]], dtype=complex)
        b = np.array([1, 0, 0], dtype=complex)
        try:
            hhl_solve(A, b)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_result_fields(self):
        A = np.array([[2, 0], [0, 3]], dtype=complex)
        b = np.array([1, 0], dtype=complex)
        result = hhl_solve(A, b, num_clock=2)
        assert result.solution_state.shape == (2,)
        assert result.classical_solution.shape == (2,)
        assert 0 <= result.success_probability <= 1
        assert 0 <= result.fidelity <= 1

    def test_solution_normalized(self):
        A = np.array([[1, 0], [0, 2]], dtype=complex)
        b = np.array([1, 1], dtype=complex)
        result = hhl_solve(A, b, num_clock=3)
        if result.success_probability > 1e-10:
            norm = np.linalg.norm(result.solution_state)
            assert abs(norm - 1.0) < 1e-8

    def test_explicit_t(self):
        """User-supplied t value should be respected."""
        A = np.array([[1, 0], [0, 2]], dtype=complex)
        b = np.array([1, 1], dtype=complex)
        result = hhl_solve(A, b, num_clock=3, t=2 * np.pi / 8)
        assert result.fidelity > 0.99

    def test_different_b_vectors(self):
        """Different b vectors should give different solutions."""
        A = np.array([[1, 0], [0, 2]], dtype=complex)
        r1 = hhl_solve(A, np.array([1, 0], dtype=complex), num_clock=3)
        r2 = hhl_solve(A, np.array([0, 1], dtype=complex), num_clock=3)
        assert r1.fidelity > 0.99
        assert r2.fidelity > 0.99
        overlap = abs(np.dot(r1.solution_state.conj(), r2.solution_state))
        assert overlap < 0.5
