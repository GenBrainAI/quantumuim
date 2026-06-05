"""Tests for Quantum Fourier Transform."""

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import X, H
from quantumium.algorithms.qft import qft, inverse_qft


class TestQFT:
    def test_qft_single_qubit_zero(self):
        """QFT on |0⟩ should produce |+⟩ = (|0⟩+|1⟩)/√2."""
        qc = QuantumCircuit(1)
        qft(qc, [0])
        sv = qc.statevector()
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_qft_single_qubit_one(self):
        """QFT on |1⟩ should produce |−⟩ = (|0⟩−|1⟩)/√2."""
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        qft(qc, [0])
        sv = qc.statevector()
        expected = np.array([1, -1], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_qft_two_qubits_zero(self):
        """QFT on |00⟩ should produce uniform superposition."""
        qc = QuantumCircuit(2)
        qft(qc, [0, 1])
        sv = qc.statevector()
        expected = np.ones(4, dtype=complex) / 2.0
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_qft_three_qubits_zero(self):
        """QFT on |000⟩ should produce uniform superposition over 8 states."""
        qc = QuantumCircuit(3)
        qft(qc, [0, 1, 2])
        sv = qc.statevector()
        expected = np.ones(8, dtype=complex) / np.sqrt(8)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_qft_inverse_identity(self):
        """QFT followed by inverse QFT should return to original state."""
        qc = QuantumCircuit(3)
        qc.apply(X, 0)
        qc.apply(H, 1)

        original_state = qc.statevector().copy()

        qft(qc, [0, 1, 2])
        inverse_qft(qc, [0, 1, 2])

        recovered = qc.statevector()
        np.testing.assert_allclose(recovered, original_state, atol=1e-10)

    def test_qft_inverse_identity_two_qubits(self):
        """QFT† ∘ QFT = I on 2 qubits with non-trivial input."""
        qc = QuantumCircuit(2)
        qc.apply(X, 1)

        original = qc.statevector().copy()

        qft(qc, [0, 1])
        inverse_qft(qc, [0, 1])

        np.testing.assert_allclose(qc.statevector(), original, atol=1e-10)

    def test_qft_preserves_norm(self):
        """QFT should preserve the norm of the state vector."""
        qc = QuantumCircuit(3)
        qc.apply(X, 0)
        qc.apply(H, 2)
        qft(qc, [0, 1, 2])
        sv = qc.statevector()
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    def test_qft_partial_qubits(self):
        """QFT applied to subset of qubits in a larger circuit."""
        qc = QuantumCircuit(3)
        qc.apply(X, 2)
        qft(qc, [0, 1])
        sv = qc.statevector()
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10
