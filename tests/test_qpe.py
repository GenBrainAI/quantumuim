"""Tests for Quantum Phase Estimation."""

import numpy as np
import pytest

from quantumium.core.gate import Gate, Z, S, T
from quantumium.algorithms.qpe import (
    controlled_unitary,
    unitary_power,
    estimate_phase,
)


class TestQPEHelpers:
    def test_controlled_unitary_shape(self):
        cu = controlled_unitary(Z)
        assert cu.matrix.shape == (4, 4)

    def test_controlled_unitary_is_unitary(self):
        cu = controlled_unitary(S)
        m = cu.matrix
        assert np.allclose(m @ m.conj().T, np.eye(4))

    def test_unitary_power(self):
        u2 = unitary_power(Z, 2)
        assert np.allclose(u2.matrix, np.eye(2))

    def test_unitary_power_identity(self):
        u0 = unitary_power(S, 0)
        assert np.allclose(u0.matrix, np.eye(2))


class TestQPE:
    def test_z_gate_phase(self):
        result = estimate_phase(3, Z, shots=4096)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_s_gate_phase(self):
        result = estimate_phase(3, S, shots=4096)
        assert result == pytest.approx(0.25, abs=0.01)

    def test_t_gate_phase(self):
        result = estimate_phase(4, T, shots=4096)
        assert result == pytest.approx(0.125, abs=0.01)

    def test_identity_phase(self):
        I = Gate(np.eye(2, dtype=complex), "I")
        result = estimate_phase(3, I, shots=4096)
        assert result == pytest.approx(0.0, abs=0.01)
