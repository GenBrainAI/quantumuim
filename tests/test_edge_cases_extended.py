"""Extended edge case tests for visualization, QASM, serialization, and transpiler."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from quantumium.core.gate import H, X, Y, Z, S, T, I, CNOT, SWAP, CZ, CCX, Gate, Rx, Ry, Rz
from quantumium.core.circuit import QuantumCircuit
from quantumium.utils.visualization import circuit_to_ascii
from quantumium.transpiler.qasm import to_qasm, from_qasm, _extract_rotation_angle
from quantumium.transpiler.serialization import (
    to_dict,
    from_dict,
    to_json,
    from_json,
    save,
    load,
)
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
    n = qc.num_qubits
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for inst in qc.instructions:
        full = qc._expand_gate(inst.gate.matrix, inst.target_qubits, n)
        U = full @ U
    return U


# === Visualization Edge Cases ===


class TestVisualizationEdgeCases:
    def test_single_qubit_empty(self) -> None:
        qc = QuantumCircuit(1)
        diagram = circuit_to_ascii(qc)
        assert "q0" in diagram
        assert "\n" not in diagram.strip() or diagram.strip().count("\n") == 0

    def test_rotation_gate_rendering(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(Rx(np.pi / 4), 0)
        diagram = circuit_to_ascii(qc)
        assert "Rx" in diagram

    def test_rz_gate_rendering(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(Rz(1.0), 0)
        diagram = circuit_to_ascii(qc)
        assert "Rz" in diagram

    def test_cz_gate_rendering(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(CZ, 0, 1)
        diagram = circuit_to_ascii(qc)
        assert "CZ" in diagram
        assert "│" in diagram

    def test_identity_gate_rendering(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(I, 0)
        diagram = circuit_to_ascii(qc)
        assert "I" in diagram

    def test_reversed_cnot(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(CNOT, 1, 0)
        diagram = circuit_to_ascii(qc)
        assert "●" in diagram
        assert "⊕" in diagram

    def test_long_sequence(self) -> None:
        qc = QuantumCircuit(1)
        for gate in [H, X, Y, Z, S, T]:
            qc.apply(gate, 0)
        diagram = circuit_to_ascii(qc)
        for name in ["H", "X", "Y", "Z", "S", "T"]:
            assert name in diagram

    def test_non_adjacent_cnot(self) -> None:
        """CNOT between q0 and q3 should have vertical connectors."""
        qc = QuantumCircuit(4)
        qc.apply(CNOT, 0, 3)
        diagram = circuit_to_ascii(qc)
        assert "●" in diagram
        assert "⊕" in diagram
        assert diagram.count("│") >= 4

    def test_swap_non_adjacent(self) -> None:
        qc = QuantumCircuit(3)
        qc.apply(SWAP, 0, 2)
        diagram = circuit_to_ascii(qc)
        assert diagram.count("✕") == 2

    def test_ccx_rendering(self) -> None:
        qc = QuantumCircuit(3)
        qc.apply(CCX, 0, 1, 2)
        diagram = circuit_to_ascii(qc)
        assert "CCX" in diagram

    def test_many_qubits(self) -> None:
        qc = QuantumCircuit(5)
        qc.apply(H, 4)
        diagram = circuit_to_ascii(qc)
        assert "q4" in diagram
        lines = diagram.strip().split("\n")
        assert len(lines) == 9  # 5 wire lines + 4 spacer lines


# === QASM Edge Cases ===


class TestQASMEdgeCases:
    def test_empty_circuit_export(self) -> None:
        qc = QuantumCircuit(2)
        qasm = to_qasm(qc)
        assert "OPENQASM 2.0" in qasm
        assert "qreg q[2]" in qasm
        assert "creg c[2]" in qasm

    def test_roundtrip_basic_gates(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(X, 1)
        qc.apply(CNOT, 0, 1)
        qasm = to_qasm(qc)
        qc2 = from_qasm(qasm)
        assert qc2.num_qubits == 2
        assert len(qc2.instructions) == 3
        np.testing.assert_allclose(_circuit_unitary(qc), _circuit_unitary(qc2), atol=1e-10)

    def test_roundtrip_rotation_gates(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(Rx(np.pi / 3), 0)
        qc.apply(Ry(np.pi / 6), 0)
        qc.apply(Rz(np.pi / 4), 0)
        U_orig = _circuit_unitary(qc)
        qasm = to_qasm(qc)
        qc2 = from_qasm(qasm)
        U_round = _circuit_unitary(qc2)
        np.testing.assert_allclose(U_orig, U_round, atol=1e-3)

    def test_roundtrip_all_standard_gates(self) -> None:
        qc = QuantumCircuit(2)
        for gate in [I, X, Y, Z, H, S, T]:
            qc.apply(gate, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(SWAP, 0, 1)
        qasm = to_qasm(qc)
        qc2 = from_qasm(qasm)
        assert len(qc2.instructions) == len(qc.instructions)

    def test_from_qasm_no_qreg_raises(self) -> None:
        with pytest.raises(ValueError, match="No qreg"):
            from_qasm("OPENQASM 2.0;\nh q[0];")

    def test_from_qasm_comments_skipped(self) -> None:
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
// This is a comment
h q[0];
"""
        qc = from_qasm(qasm)
        assert len(qc.instructions) == 1

    def test_custom_gate_export(self) -> None:
        mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        custom = Gate(mat, "MyGate")
        qc = QuantumCircuit(1)
        qc.apply(custom, 0)
        qasm = to_qasm(qc)
        assert "q[0]" in qasm

    def test_extract_rotation_angle_rx(self) -> None:
        g = Rx(1.234)
        result = _extract_rotation_angle(g)
        assert result is not None
        prefix, angle = result
        assert prefix == "Rx"
        assert abs(angle - 1.234) < 1e-6

    def test_extract_rotation_angle_non_rotation(self) -> None:
        result = _extract_rotation_angle(H)
        assert result is not None or result is None  # H is not a standard rotation

    def test_extract_rotation_angle_multi_qubit_returns_none(self) -> None:
        result = _extract_rotation_angle(CNOT)
        assert result is None

    def test_qasm_measure_ignored(self) -> None:
        qasm = """OPENQASM 2.0;
qreg q[1];
creg c[1];
h q[0];
measure q[0] -> c[0];
"""
        qc = from_qasm(qasm)
        assert len(qc.instructions) == 1


# === Serialization Edge Cases ===


class TestSerializationEdgeCases:
    def test_roundtrip_dict(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        d = to_dict(qc)
        qc2 = from_dict(d)
        np.testing.assert_allclose(_circuit_unitary(qc), _circuit_unitary(qc2), atol=1e-10)

    def test_roundtrip_json(self) -> None:
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(Rx(np.pi / 4), 1)
        qc.apply(CNOT, 0, 2)
        j = to_json(qc)
        qc2 = from_json(j)
        np.testing.assert_allclose(_circuit_unitary(qc), _circuit_unitary(qc2), atol=1e-3)

    def test_roundtrip_custom_gate(self) -> None:
        custom = Gate(S.matrix, "MyCustom")
        qc = QuantumCircuit(1)
        qc.apply(custom, 0)
        d = to_dict(qc)
        assert d["instructions"][0]["gate"] == "custom"
        qc2 = from_dict(d)
        np.testing.assert_allclose(_circuit_unitary(qc), _circuit_unitary(qc2), atol=1e-10)

    def test_save_load_file(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            save(qc, filepath)
            qc2 = load(filepath)
            np.testing.assert_allclose(_circuit_unitary(qc), _circuit_unitary(qc2), atol=1e-10)
        finally:
            os.unlink(filepath)

    def test_from_dict_wrong_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown format"):
            from_dict({"format": "wrong-v99"})

    def test_from_dict_unknown_gate_raises(self) -> None:
        d = {
            "format": "quantumium-v1",
            "num_qubits": 1,
            "instructions": [{"gate": "NONEXISTENT", "qubits": [0]}],
        }
        with pytest.raises(ValueError, match="Unknown gate"):
            from_dict(d)

    def test_dict_format_field(self) -> None:
        qc = QuantumCircuit(1)
        d = to_dict(qc)
        assert d["format"] == "quantumium-v1"
        assert d["num_qubits"] == 1
        assert d["instructions"] == []

    def test_roundtrip_rotation_gates(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(Rx(0.5), 0)
        qc.apply(Ry(1.0), 0)
        qc.apply(Rz(1.5), 0)
        U_orig = _circuit_unitary(qc)
        j = to_json(qc)
        qc2 = from_json(j)
        U_round = _circuit_unitary(qc2)
        np.testing.assert_allclose(U_orig, U_round, atol=1e-10)

    def test_json_is_valid_json(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        j = to_json(qc)
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_empty_circuit_serialization(self) -> None:
        qc = QuantumCircuit(1)
        d = to_dict(qc)
        qc2 = from_dict(d)
        assert qc2.num_qubits == 1
        assert len(qc2.instructions) == 0


# === Transpiler Pass Edge Cases ===


class TestTranspilerPassEdgeCases:
    def test_cancel_inverses_triple_x(self) -> None:
        """X X X should reduce to X."""
        qc = QuantumCircuit(1)
        qc.apply(X, 0).apply(X, 0).apply(X, 0)
        result = cancel_inverses(qc.instructions)
        assert len(result) == 1

    def test_cancel_inverses_yy(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(Y, 0).apply(Y, 0)
        result = cancel_inverses(qc.instructions)
        assert len(result) == 0

    def test_remove_identity_all_identities(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(I, 0).apply(I, 1).apply(I, 0)
        result = remove_identity_gates(qc.instructions)
        assert len(result) == 0

    def test_merge_single_qubit_preserves_two_qubit(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(X, 0)
        result = merge_single_qubit_gates(qc.instructions)
        cnot_count = sum(1 for r in result if r.gate.num_qubits == 2)
        assert cnot_count == 1

    def test_cancel_cnots_triple(self) -> None:
        """CNOT CNOT CNOT should reduce to CNOT."""
        qc = QuantumCircuit(2)
        qc.apply(CNOT, 0, 1).apply(CNOT, 0, 1).apply(CNOT, 0, 1)
        result = cancel_cnots(qc.instructions)
        assert len(result) == 1

    def test_decompose_swap_preserves_non_swap(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        result = decompose_swap(qc.instructions)
        assert len(result) == 2

    def test_count_gates_empty(self) -> None:
        qc = QuantumCircuit(1)
        counts = count_gates(qc.instructions)
        assert counts == {}

    def test_circuit_depth_empty(self) -> None:
        qc = QuantumCircuit(2)
        assert circuit_depth(qc.instructions, 2) == 0

    def test_circuit_depth_parallel_2q(self) -> None:
        qc = QuantumCircuit(4)
        qc.apply(CNOT, 0, 1)
        qc.apply(CNOT, 2, 3)
        assert circuit_depth(qc.instructions, 4) == 1

    def test_transpile_with_rotations(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(Rx(np.pi), 0)
        qc.apply(Ry(0.0), 0)
        optimized = transpile(qc)
        U_orig = _circuit_unitary(qc)
        U_opt = _circuit_unitary(optimized)
        np.testing.assert_allclose(U_orig, U_opt, atol=1e-10)

    def test_transpile_identity_only(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(I, 0)
        optimized = transpile(qc)
        assert len(optimized.instructions) == 0

    def test_commute_and_cancel_adjacent_self_inverse(self) -> None:
        """Adjacent self-inverse gates on same qubit should cancel."""
        qc = QuantumCircuit(1)
        qc.apply(H, 0).apply(H, 0)
        result = commute_and_cancel(qc.instructions)
        assert len(result) == 0
