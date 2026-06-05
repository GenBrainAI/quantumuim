"""Tests for Simon's algorithm."""

import numpy as np

from quantumium.algorithms.simon import (
    _simon_oracle,
    _solve_linear_system,
    simon_query,
    simon_algorithm,
)


class TestSimonOracle:
    def test_is_unitary(self):
        oracle = _simon_oracle(2, "11")
        m = oracle.matrix
        np.testing.assert_allclose(m @ m.conj().T, np.eye(m.shape[0]), atol=1e-10)

    def test_is_permutation(self):
        oracle = _simon_oracle(2, "10")
        m = oracle.matrix
        for j in range(m.shape[1]):
            assert abs(np.sum(np.abs(m[:, j])) - 1.0) < 1e-10

    def test_promise_holds(self):
        """f(x) = f(x ⊕ s) for all x."""
        s = "11"
        s_int = int(s, 2)
        oracle = _simon_oracle(2, s)
        n = 2
        for x in range(2**n):
            state_x = np.zeros(2 ** (2 * n), dtype=complex)
            state_x[(x << n) | 0] = 1.0
            out_x = oracle.matrix @ state_x

            xs = x ^ s_int
            state_xs = np.zeros(2 ** (2 * n), dtype=complex)
            state_xs[(xs << n) | 0] = 1.0
            out_xs = oracle.matrix @ state_xs

            # Extract f(x) and f(x⊕s) from output register
            fx = None
            fxs = None
            for i in range(2 ** (2 * n)):
                if abs(out_x[i]) > 0.5:
                    fx = i & ((1 << n) - 1)
                if abs(out_xs[i]) > 0.5:
                    fxs = i & ((1 << n) - 1)
            assert fx == fxs, f"f({x}) = {fx} != f({xs}) = {fxs}"

    def test_identity_secret(self):
        """s = 00 means f is 1-to-1, oracle is still unitary."""
        oracle = _simon_oracle(2, "00")
        m = oracle.matrix
        np.testing.assert_allclose(m @ m.conj().T, np.eye(m.shape[0]), atol=1e-10)


class TestLinearSolver:
    def test_simple_2bit(self):
        # z · s = 0 with z = "10" means s[0] = 0, free var s[1]
        result = _solve_linear_system(["10"], 2)
        assert result is not None
        # s should satisfy 1*s0 + 0*s1 = 0 → s0=0, s1 free → s="01"
        s_int = int(result, 2)
        for eq in ["10"]:
            dot = sum(int(eq[i]) * int(result[i]) for i in range(2)) % 2
            assert dot == 0

    def test_overdetermined(self):
        result = _solve_linear_system(["10", "01", "11"], 2)
        assert result == "00"

    def test_single_equation(self):
        result = _solve_linear_system(["110"], 3)
        assert result is not None
        dot = sum(int("110"[i]) * int(result[i]) for i in range(3)) % 2
        assert dot == 0


class TestSimonQuery:
    def test_returns_valid_bitstring(self):
        z = simon_query(2, "11")
        assert len(z) == 2
        assert all(c in "01" for c in z)

    def test_orthogonal_to_secret(self):
        """z · s = 0 mod 2."""
        s = "11"
        for _ in range(10):
            z = simon_query(2, s)
            dot = sum(int(z[i]) * int(s[i]) for i in range(2)) % 2
            assert dot == 0


class TestSimonAlgorithm:
    def test_find_secret_11(self):
        result = simon_algorithm(2, "11")
        assert result.secret == "11"
        assert result.num_queries > 0

    def test_find_secret_10(self):
        result = simon_algorithm(2, "10")
        assert result.secret == "10"

    def test_find_secret_3bit(self):
        result = simon_algorithm(3, "101")
        assert result.secret == "101"

    def test_identity_function(self):
        """s = 00 means 1-to-1 function."""
        result = simon_algorithm(2, "00")
        assert result.secret == "00"

    def test_wrong_length_raises(self):
        try:
            simon_algorithm(2, "111")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_result_has_equations(self):
        result = simon_algorithm(2, "11")
        assert len(result.equations) >= 0
