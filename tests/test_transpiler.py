"""Tests for circuit transpiler optimization passes."""

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit, Instruction
from quantumium.core.gate import H, X, Y, Z, S, T, CNOT, SWAP, I, Ry, Rz
from quantumium.transpiler.passes import (
    cancel_inverses,
    remove_identity_gates,
    merge_single_qubit_gates,
    cancel_cnots,
    commute_and_cancel,
    decompose_swap,
    count_gates,
    circuit_depth,
    transpile,
)


def _circuit_unitary(qc: QuantumCircuit) -> np.ndarray:
    """Compute the full unitary matrix of a circuit."""
    n = qc.num_qubits
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for inst in qc.instructions:
        full = qc._expand_gate(inst.gate.matrix, inst.target_qubits, n)
        U = full @ U
    return U


class TestCancelInverses:
    def test_xx_cancels(self):
        qc = QuantumCircuit(1)
        qc.apply(X, 0).apply(X, 0)
        result = cancel_inverses(qc.instructions)
        assert len(result) == 0

    def test_hh_cancels(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0).apply(H, 0)
        result = cancel_inverses(qc.instructions)
        assert len(result) == 0

    def test_zz_cancels(self):
        qc = QuantumCircuit(1)
        qc.apply(Z, 0).apply(Z, 0)
        result = cancel_inverses(qc.instructions)
        assert len(result) == 0

    def test_no_cancel_different_qubits(self):
        qc = QuantumCircuit(2)
        qc.apply(X, 0).apply(X, 1)
        result = cancel_inverses(qc.instructions)
        assert len(result) == 2

    def test_partial_cancel(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0).apply(X, 0).apply(X, 0).apply(H, 0)
        result = cancel_inverses(qc.instructions)
        assert len(result) == 2
        result = cancel_inverses(result)
        assert len(result) == 0

    def test_preserves_unitary(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(X, 0).apply(X, 0).apply(CNOT, 0, 1)
        U_orig = _circuit_unitary(qc)
        optimized = cancel_inverses(qc.instructions)
        qc2 = QuantumCircuit(2)
        for inst in optimized:
            qc2.apply(inst.gate, *inst.target_qubits)
        U_opt = _circuit_unitary(qc2)
        np.testing.assert_allclose(U_orig, U_opt, atol=1e-10)


class TestRemoveIdentity:
    def test_removes_identity(self):
        qc = QuantumCircuit(1)
        qc.apply(I, 0).apply(H, 0).apply(I, 0)
        result = remove_identity_gates(qc.instructions)
        assert len(result) == 1
        assert result[0].gate.name == "H"


class TestMergeSingleQubit:
    def test_merges_consecutive(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0).apply(X, 0).apply(Z, 0)
        result = merge_single_qubit_gates(qc.instructions)
        assert len(result) == 1

    def test_preserves_unitary(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0).apply(T, 0).apply(S, 0)
        U_orig = _circuit_unitary(qc)
        merged = merge_single_qubit_gates(qc.instructions)
        qc2 = QuantumCircuit(1)
        for inst in merged:
            qc2.apply(inst.gate, *inst.target_qubits)
        U_opt = _circuit_unitary(qc2)
        np.testing.assert_allclose(U_orig, U_opt, atol=1e-10)

    def test_stops_at_two_qubit_gate(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(X, 0).apply(CNOT, 0, 1).apply(Z, 0)
        result = merge_single_qubit_gates(qc.instructions)
        assert len(result) == 3

    def test_merges_to_identity_removed(self):
        qc = QuantumCircuit(1)
        qc.apply(X, 0).apply(X, 0)
        result = merge_single_qubit_gates(qc.instructions)
        assert len(result) == 0

    def test_different_qubits_not_merged(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(X, 1)
        result = merge_single_qubit_gates(qc.instructions)
        assert len(result) == 2


class TestCancelCNOTs:
    def test_double_cnot_cancels(self):
        qc = QuantumCircuit(2)
        qc.apply(CNOT, 0, 1).apply(CNOT, 0, 1)
        result = cancel_cnots(qc.instructions)
        assert len(result) == 0

    def test_different_direction_no_cancel(self):
        qc = QuantumCircuit(2)
        qc.apply(CNOT, 0, 1).apply(CNOT, 1, 0)
        result = cancel_cnots(qc.instructions)
        assert len(result) == 2


class TestDecomposeSwap:
    def test_swap_to_three_cnots(self):
        qc = QuantumCircuit(2)
        qc.apply(SWAP, 0, 1)
        result = decompose_swap(qc.instructions)
        assert len(result) == 3
        assert all(r.gate.num_qubits == 2 for r in result)

    def test_preserves_unitary(self):
        qc = QuantumCircuit(2)
        qc.apply(SWAP, 0, 1)
        U_orig = _circuit_unitary(qc)
        decomposed = decompose_swap(qc.instructions)
        qc2 = QuantumCircuit(2)
        for inst in decomposed:
            qc2.apply(inst.gate, *inst.target_qubits)
        U_dec = _circuit_unitary(qc2)
        np.testing.assert_allclose(U_orig, U_dec, atol=1e-10)

    def test_non_swap_untouched(self):
        qc = QuantumCircuit(2)
        qc.apply(CNOT, 0, 1)
        result = decompose_swap(qc.instructions)
        assert len(result) == 1


class TestCommuteAndCancel:
    def test_commute_enables_cancel(self):
        qc = QuantumCircuit(2)
        qc.apply(X, 0).apply(H, 1).apply(X, 0)
        result = commute_and_cancel(qc.instructions)
        assert len(result) == 1
        assert result[0].gate.name == "H"


class TestCountGates:
    def test_counts(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(CNOT, 0, 1).apply(H, 1)
        counts = count_gates(qc.instructions)
        assert counts["H"] == 2
        assert counts["CNOT"] == 1


class TestCircuitDepth:
    def test_parallel_gates(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(X, 1)
        assert circuit_depth(qc.instructions, 2) == 1

    def test_serial_gates(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0).apply(X, 0).apply(Z, 0)
        assert circuit_depth(qc.instructions, 1) == 3

    def test_mixed(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(X, 1)
        assert circuit_depth(qc.instructions, 2) == 3


class TestTranspile:
    def test_reduces_gate_count(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(X, 0).apply(X, 0).apply(CNOT, 0, 1).apply(CNOT, 0, 1)
        optimized = transpile(qc)
        assert len(optimized.instructions) < len(qc.instructions)

    def test_preserves_unitary(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(T, 0).apply(CNOT, 0, 1).apply(X, 1).apply(X, 1)
        U_orig = _circuit_unitary(qc)
        optimized = transpile(qc)
        U_opt = _circuit_unitary(optimized)
        np.testing.assert_allclose(U_orig, U_opt, atol=1e-10)

    def test_empty_circuit(self):
        qc = QuantumCircuit(1)
        optimized = transpile(qc)
        assert len(optimized.instructions) == 0

    def test_already_optimal(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(CNOT, 0, 1)
        optimized = transpile(qc)
        assert len(optimized.instructions) == 2

    def test_custom_passes(self):
        qc = QuantumCircuit(2)
        qc.apply(SWAP, 0, 1)
        optimized = transpile(qc, passes=[decompose_swap])
        assert len(optimized.instructions) == 3

    def test_full_pipeline_complex(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(I, 1)
        qc.apply(X, 2).apply(X, 2)
        qc.apply(CNOT, 0, 1)
        qc.apply(CNOT, 0, 1)
        qc.apply(H, 0).apply(T, 0).apply(S, 0)
        U_orig = _circuit_unitary(qc)
        optimized = transpile(qc)
        U_opt = _circuit_unitary(optimized)
        np.testing.assert_allclose(U_orig, U_opt, atol=1e-10)
        assert len(optimized.instructions) < len(qc.instructions)
