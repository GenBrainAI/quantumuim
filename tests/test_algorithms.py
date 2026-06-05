"""Tests for quantum algorithm implementations."""

import numpy as np
import pytest

from quantumium.algorithms.grover import grover_search, grover_oracle, grover_diffuser
from quantumium.algorithms.deutsch_jozsa import deutsch_jozsa, classify_function
from quantumium.algorithms.bernstein_vazirani import bernstein_vazirani, find_secret


class TestGrover:
    def test_single_marked_state(self):
        qc = grover_search(3, [5])
        counts = qc.measure_all(shots=1000)
        bitstring = format(5, "03b")
        assert counts.get(bitstring, 0) > 800

    def test_two_marked_states(self):
        qc = grover_search(3, [2, 6])
        counts = qc.measure_all(shots=2000)
        b2 = format(2, "03b")
        b6 = format(6, "03b")
        assert counts.get(b2, 0) + counts.get(b6, 0) > 1500

    def test_custom_iterations(self):
        qc = grover_search(2, [3], num_iterations=1)
        counts = qc.measure_all(shots=1000)
        assert counts.get("11", 0) > 800

    def test_oracle_phase_flip(self):
        oracle = grover_oracle(2, [3])
        m = oracle.matrix
        assert m[3, 3] == pytest.approx(-1)
        assert m[0, 0] == pytest.approx(1)

    def test_diffuser_is_unitary(self):
        d = grover_diffuser(3)
        m = d.matrix
        assert np.allclose(m @ m.conj().T, np.eye(8))

    def test_invalid_marked_state(self):
        with pytest.raises(ValueError):
            grover_oracle(2, [5])

    def test_no_marked_states(self):
        with pytest.raises(ValueError):
            grover_search(2, [])

    def test_all_marked_states(self):
        with pytest.raises(ValueError):
            grover_search(2, [0, 1, 2, 3])


class TestDeutschJozsa:
    def test_constant_zero(self):
        result = classify_function(3, lambda x: 0)
        assert result == "constant"

    def test_constant_one(self):
        result = classify_function(3, lambda x: 1)
        assert result == "constant"

    def test_balanced_parity(self):
        result = classify_function(3, lambda x: bin(x).count("1") % 2)
        assert result == "balanced"

    def test_balanced_msb(self):
        result = classify_function(3, lambda x: (x >> 2) & 1)
        assert result == "balanced"

    def test_single_qubit_constant(self):
        result = classify_function(1, lambda x: 0)
        assert result == "constant"

    def test_single_qubit_balanced(self):
        result = classify_function(1, lambda x: x)
        assert result == "balanced"


class TestBernsteinVazirani:
    def test_secret_zero(self):
        result = find_secret(3, 0)
        assert result == 0

    def test_secret_all_ones(self):
        result = find_secret(3, 7)
        assert result == 7

    def test_secret_101(self):
        result = find_secret(3, 5)
        assert result == 5

    def test_secret_single_bit(self):
        for bit in range(4):
            secret = 1 << bit
            result = find_secret(4, secret)
            assert result == secret

    def test_invalid_secret(self):
        with pytest.raises(ValueError):
            bernstein_vazirani(3, 8)
