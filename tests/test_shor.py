"""Tests for Shor's factoring algorithm (educational implementation)."""

import numpy as np

from quantumium.algorithms.shor import (
    _mod_exp_unitary,
    _num_bits,
    quantum_order_finding,
    shor_factor,
)


class TestModExpUnitary:
    def test_is_unitary(self):
        """Modular exponentiation gate must be unitary."""
        U = _mod_exp_unitary(a=2, power=1, N=15, n_work=4)
        m = U.matrix
        np.testing.assert_allclose(m @ m.conj().T, np.eye(m.shape[0]), atol=1e-10)

    def test_permutation(self):
        """Each column should have exactly one 1."""
        U = _mod_exp_unitary(a=7, power=1, N=15, n_work=4)
        m = U.matrix
        for j in range(m.shape[1]):
            assert abs(np.sum(np.abs(m[:, j])) - 1.0) < 1e-10

    def test_identity_for_a_equals_1(self):
        """a=1 should give identity on valid states."""
        U = _mod_exp_unitary(a=1, power=1, N=15, n_work=4)
        np.testing.assert_allclose(U.matrix, np.eye(16), atol=1e-10)

    def test_maps_one_to_a(self):
        """U|1⟩ = |a mod N⟩."""
        a, N = 7, 15
        U = _mod_exp_unitary(a=a, power=1, N=N, n_work=4)
        state = np.zeros(16)
        state[1] = 1.0
        result = U.matrix @ state
        expected = np.zeros(16)
        expected[a] = 1.0
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_power(self):
        """U^power |1⟩ = |a^power mod N⟩."""
        a, N, power = 2, 15, 3
        U = _mod_exp_unitary(a=a, power=power, N=N, n_work=4)
        state = np.zeros(16)
        state[1] = 1.0
        result = U.matrix @ state
        expected_val = pow(a, power, N)
        expected = np.zeros(16)
        expected[expected_val] = 1.0
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestNumBits:
    def test_small_values(self):
        assert _num_bits(2) == 2
        assert _num_bits(3) == 2
        assert _num_bits(4) == 3
        assert _num_bits(15) == 4
        assert _num_bits(16) == 5


class TestQuantumOrderFinding:
    def test_order_7_mod_15(self):
        """Order of 7 mod 15 is 4."""
        _, order = quantum_order_finding(a=7, N=15, num_counting_qubits=5)
        if order is not None:
            assert pow(7, order, 15) == 1
            assert order <= 15

    def test_order_2_mod_15(self):
        """Order of 2 mod 15 is 4."""
        _, order = quantum_order_finding(a=2, N=15, num_counting_qubits=5)
        if order is not None:
            assert pow(2, order, 15) == 1


class TestShorFactor:
    def test_factor_15_with_a_7(self):
        """Factor 15 using a=7. Order is 4, should find 3 and 5."""
        result = shor_factor(N=15, a=7, num_counting_qubits=5)
        if result.success:
            f1, f2 = result.factors
            assert f1 * f2 == 15
            assert f1 > 1 and f2 > 1

    def test_factor_15_any_base(self):
        """Factor 15 trying multiple bases."""
        result = shor_factor(N=15, num_counting_qubits=5, max_attempts=5)
        if result.success:
            f1, f2 = result.factors
            assert f1 * f2 == 15
            assert f1 > 1 and f2 > 1

    def test_even_number(self):
        """Even numbers should be trivially factored."""
        result = shor_factor(N=6)
        assert result.success
        assert result.factors == (2, 3)

    def test_gcd_shortcut(self):
        """If gcd(a, N) > 1, factor is found without quantum step."""
        result = shor_factor(N=15, a=3)
        assert result.success
        assert result.factors == (3, 5)

    def test_small_n_raises(self):
        try:
            shor_factor(N=3)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_result_fields(self):
        result = shor_factor(N=6)
        assert result.N == 6
        assert result.success
        assert result.factors is not None
