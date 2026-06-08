"""Tests for entanglement measures and quantum information utilities."""

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT
from quantumium.utils.entanglement import (
    partial_trace,
    von_neumann_entropy,
    purity,
    concurrence,
    negativity,
    entanglement_entropy,
    schmidt_decomposition,
)


def _bell_state_rho():
    qc = QuantumCircuit(2)
    qc.apply(H, 0)
    qc.apply(CNOT, 0, 1)
    sv = qc.statevector()
    return np.outer(sv, sv.conj())


def _product_state_rho():
    qc = QuantumCircuit(2)
    qc.apply(H, 0)
    sv = qc.statevector()
    return np.outer(sv, sv.conj())


class TestPartialTrace:
    def test_trace_one_qubit_from_product(self):
        rho = _product_state_rho()
        rho_a = partial_trace(rho, [0], 2)
        expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        np.testing.assert_allclose(rho_a, expected, atol=1e-10)

    def test_trace_other_qubit_from_product(self):
        rho = _product_state_rho()
        rho_b = partial_trace(rho, [1], 2)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        np.testing.assert_allclose(rho_b, expected, atol=1e-10)

    def test_bell_state_reduced_is_mixed(self):
        rho = _bell_state_rho()
        rho_a = partial_trace(rho, [0], 2)
        expected = np.eye(2, dtype=complex) / 2
        np.testing.assert_allclose(rho_a, expected, atol=1e-10)

    def test_trace_preserves_trace(self):
        rho = _bell_state_rho()
        rho_a = partial_trace(rho, [0], 2)
        assert abs(np.trace(rho_a) - 1.0) < 1e-10

    def test_three_qubit_partial_trace(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        sv = qc.statevector()
        rho = np.outer(sv, sv.conj())
        rho_01 = partial_trace(rho, [0, 1], 3)
        assert rho_01.shape == (4, 4)
        assert abs(np.trace(rho_01) - 1.0) < 1e-10

    def test_invalid_size(self):
        rho = np.eye(3, dtype=complex) / 3
        with pytest.raises(ValueError, match="doesn't match"):
            partial_trace(rho, [0], 2)


class TestVonNeumannEntropy:
    def test_pure_state_zero_entropy(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        assert abs(von_neumann_entropy(rho)) < 1e-10

    def test_maximally_mixed_single_qubit(self):
        rho = np.eye(2, dtype=complex) / 2
        assert abs(von_neumann_entropy(rho) - 1.0) < 1e-10

    def test_maximally_mixed_two_qubits(self):
        rho = np.eye(4, dtype=complex) / 4
        assert abs(von_neumann_entropy(rho) - 2.0) < 1e-10

    def test_bell_state_reduced(self):
        rho = _bell_state_rho()
        rho_a = partial_trace(rho, [0], 2)
        assert abs(von_neumann_entropy(rho_a) - 1.0) < 1e-10


class TestPurity:
    def test_pure_state(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        assert abs(purity(rho) - 1.0) < 1e-10

    def test_maximally_mixed(self):
        rho = np.eye(2, dtype=complex) / 2
        assert abs(purity(rho) - 0.5) < 1e-10

    def test_bell_state_is_pure(self):
        rho = _bell_state_rho()
        assert abs(purity(rho) - 1.0) < 1e-10

    def test_bell_reduced_is_mixed(self):
        rho = _bell_state_rho()
        rho_a = partial_trace(rho, [0], 2)
        assert abs(purity(rho_a) - 0.5) < 1e-10


class TestConcurrence:
    def test_bell_state_maximally_entangled(self):
        rho = _bell_state_rho()
        assert abs(concurrence(rho) - 1.0) < 1e-10

    def test_product_state_zero_concurrence(self):
        rho = _product_state_rho()
        assert concurrence(rho) < 1e-10

    def test_separable_state(self):
        rho = np.diag([0.5, 0.0, 0.0, 0.5]).astype(complex)
        c = concurrence(rho)
        assert c >= 0

    def test_wrong_size(self):
        with pytest.raises(ValueError, match="two-qubit"):
            concurrence(np.eye(8, dtype=complex) / 8)


class TestNegativity:
    def test_bell_state_positive_negativity(self):
        rho = _bell_state_rho()
        neg = negativity(rho, [0], 2)
        assert abs(neg - 0.5) < 1e-10

    def test_product_state_zero_negativity(self):
        rho = _product_state_rho()
        neg = negativity(rho, [0], 2)
        assert neg < 1e-10


class TestEntanglementEntropy:
    def test_bell_state(self):
        rho = _bell_state_rho()
        ee = entanglement_entropy(rho, [0], 2)
        assert abs(ee - 1.0) < 1e-10

    def test_product_state(self):
        rho = _product_state_rho()
        ee = entanglement_entropy(rho, [0], 2)
        assert ee < 1e-10


class TestSchmidtDecomposition:
    def test_bell_state(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        sv = qc.statevector()
        coeffs, U, V = schmidt_decomposition(sv, [0], 2)
        assert len(coeffs) == 2
        np.testing.assert_allclose(coeffs, [1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-10)

    def test_product_state(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        sv = qc.statevector()
        coeffs, U, V = schmidt_decomposition(sv, [0], 2)
        assert len(coeffs) == 1
        np.testing.assert_allclose(coeffs, [1.0], atol=1e-10)

    def test_reconstruction(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        sv = qc.statevector()
        coeffs, U, V = schmidt_decomposition(sv, [0], 2)
        reconstructed = sum(c * np.kron(U[:, i], V[i, :]) for i, c in enumerate(coeffs))
        np.testing.assert_allclose(reconstructed, sv, atol=1e-10)
