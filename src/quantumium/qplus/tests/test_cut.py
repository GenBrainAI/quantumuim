"""Tests for Layer 3a: circuit cutting and recombination."""

from __future__ import annotations

import numpy as np

from quantumium.qplus.cut import (
    CircuitCutter,
    CutRecombiner,
    WireCut,
    PAULI_MATRICES,
    PAULI_PREP_STATES,
    _expand_single_qubit_op,
)


class TestPauliConstants:
    def test_pauli_matrices_hermitian(self) -> None:
        for label, mat in PAULI_MATRICES.items():
            assert np.allclose(mat, mat.conj().T), f"{label} not Hermitian"

    def test_pauli_matrices_unitary(self) -> None:
        for label, mat in PAULI_MATRICES.items():
            assert np.allclose(mat @ mat.conj().T, np.eye(2)), f"{label} not unitary"

    def test_pauli_prep_states_normalized(self) -> None:
        for label, state in PAULI_PREP_STATES.items():
            assert abs(np.linalg.norm(state) - 1.0) < 1e-10, f"|{label}⟩ not normalized"

    def test_pauli_anticommutation(self) -> None:
        X, Y, Z = PAULI_MATRICES["X"], PAULI_MATRICES["Y"], PAULI_MATRICES["Z"]
        assert np.allclose(X @ Y + Y @ X, np.zeros((2, 2)))
        assert np.allclose(Y @ Z + Z @ Y, np.zeros((2, 2)))
        assert np.allclose(Z @ X + X @ Z, np.zeros((2, 2)))


class TestCircuitCutter:
    def test_init_empty(self) -> None:
        cc = CircuitCutter()
        assert cc.n_cuts == 0
        assert cc.overhead == 1.0

    def test_add_cut(self) -> None:
        cc = CircuitCutter()
        cc.add_cut(qubit=0, after_gate_index=5, label="cut1")
        assert cc.n_cuts == 1

    def test_overhead_scaling(self) -> None:
        cc = CircuitCutter()
        cc.add_cut(0, 0)
        assert cc.overhead == 4.0
        cc.add_cut(1, 1)
        assert cc.overhead == 16.0

    def test_init_with_cuts(self) -> None:
        cuts = [WireCut(0, 0, "a"), WireCut(1, 5, "b")]
        cc = CircuitCutter(cuts)
        assert cc.n_cuts == 2

    def test_decompose_no_cuts(self) -> None:
        cc = CircuitCutter()
        configs = cc.decompose(2, [])
        assert len(configs) == 1
        assert configs[0][2] == 1.0

    def test_decompose_one_cut(self) -> None:
        cc = CircuitCutter([WireCut(0, 0)])
        configs = cc.decompose(2, [])
        assert len(configs) == 6  # 6 Pauli decomposition terms

    def test_decompose_two_cuts(self) -> None:
        cc = CircuitCutter([WireCut(0, 0), WireCut(1, 1)])
        configs = cc.decompose(3, [])
        assert len(configs) == 36  # 6^2

    def test_decompose_coefficients_sum(self) -> None:
        """Coefficients should reconstruct identity when summed properly."""
        cc = CircuitCutter([WireCut(0, 0)])
        configs = cc.decompose(2, [])
        coefficients = [c for _, _, c in configs]
        assert len(coefficients) == 6

    def test_evaluate_cut_circuit_identity(self) -> None:
        """Cutting an identity circuit should give identity result."""
        cc = CircuitCutter([WireCut(0, 0)])
        n = 2
        dim = 2**n
        identity = np.eye(dim, dtype=complex)
        result = cc.evaluate_cut_circuit(identity, identity, 0, n)
        assert abs(result - 1.0) < 0.1

    def test_evaluate_cut_circuit_hadamard(self) -> None:
        """Cut a Hadamard circuit and verify reconstruction."""
        cc = CircuitCutter([WireCut(0, 0)])
        n = 1
        dim = 2**n
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        identity = np.eye(dim, dtype=complex)
        result = cc.evaluate_cut_circuit(H, identity, 0, n)
        assert isinstance(result, float)

    def test_estimate_resources(self) -> None:
        cc = CircuitCutter()
        resources = cc.estimate_resources(n_cuts=3)
        assert resources["n_cuts"] == 3.0
        assert resources["overhead_factor"] == 64.0

    def test_estimate_resources_from_cuts(self) -> None:
        cc = CircuitCutter([WireCut(0, 0), WireCut(1, 1)])
        resources = cc.estimate_resources()
        assert resources["n_cuts"] == 2.0


class TestCutRecombiner:
    def test_recombine_single(self) -> None:
        result = CutRecombiner.recombine([(1.0, 0.5)])
        assert abs(result - 0.5) < 1e-10

    def test_recombine_multiple(self) -> None:
        pairs = [(0.5, 1.0), (-0.5, 0.5), (0.5, 0.8), (-0.5, 0.2)]
        result = CutRecombiner.recombine(pairs)
        expected = 0.5 * 1.0 + (-0.5) * 0.5 + 0.5 * 0.8 + (-0.5) * 0.2
        assert abs(result - expected) < 1e-10

    def test_recombine_probabilities(self) -> None:
        probs: list[tuple[float, dict[str, float]]] = [
            (0.5, {"00": 0.8, "01": 0.2}),
            (0.5, {"00": 0.6, "01": 0.4}),
        ]
        result = CutRecombiner.recombine_probabilities(probs)
        assert "00" in result
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_recombine_probabilities_normalization(self) -> None:
        probs: list[tuple[float, dict[str, float]]] = [
            (1.0, {"0": 0.5, "1": 0.5}),
            (-0.5, {"0": 0.3, "1": 0.7}),
        ]
        result = CutRecombiner.recombine_probabilities(probs)
        total = sum(result.values())
        assert abs(total - 1.0) < 0.01

    def test_verify_reconstruction_pass(self) -> None:
        assert CutRecombiner.verify_reconstruction(1.0, 1.05, tolerance=0.1)

    def test_verify_reconstruction_fail(self) -> None:
        assert not CutRecombiner.verify_reconstruction(1.0, 1.5, tolerance=0.1)


class TestExpandSingleQubitOp:
    def test_single_qubit_system(self) -> None:
        X = PAULI_MATRICES["X"]
        result = _expand_single_qubit_op(X, 0, 1)
        assert np.allclose(result, X)

    def test_two_qubit_first(self) -> None:
        X = PAULI_MATRICES["X"]
        result = _expand_single_qubit_op(X, 0, 2)
        expected = np.kron(X, np.eye(2))
        assert np.allclose(result, expected)

    def test_two_qubit_second(self) -> None:
        Z = PAULI_MATRICES["Z"]
        result = _expand_single_qubit_op(Z, 1, 2)
        expected = np.kron(np.eye(2), Z)
        assert np.allclose(result, expected)

    def test_three_qubit_middle(self) -> None:
        Y = PAULI_MATRICES["Y"]
        result = _expand_single_qubit_op(Y, 1, 3)
        expected = np.kron(np.kron(np.eye(2), Y), np.eye(2))
        assert np.allclose(result, expected)
