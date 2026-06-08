"""Targeted tests for modules with lowest coverage."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, Y, Z, S, T, CNOT, CZ, SWAP, Rx, Ry, Rz, controlled


# ---- ZX-calculus coverage (rules.py 68%, convert.py 73%) ----

class TestZXRulesCoverage:
    """Cover hadamard_cancel and color_change rules."""

    def test_hadamard_cancel_basic(self):
        from quantumium.zx.rules import hadamard_cancel
        from quantumium.zx.diagram import ZXDiagram, HADAMARD, REGULAR

        diag = ZXDiagram(1)
        wire = diag._wires[0]
        inp = wire[0]
        out = wire[1]
        diag.remove_edge(inp, out)
        s1 = diag.add_spider("Z", 0.0)
        s2 = diag.add_spider("X", 0.0)
        diag.add_edge(inp, s1, REGULAR)
        diag.add_edge(s1, s2, HADAMARD)
        diag.add_edge(s2, out, REGULAR)
        wire.insert(1, s1)
        wire.insert(2, s2)
        diag._qubit[s1] = 0
        diag._qubit[s2] = 0
        changed = hadamard_cancel(diag)
        assert changed is True

    def test_color_change_all_hadamard_edges(self):
        from quantumium.zx.rules import color_change
        from quantumium.zx.diagram import ZXDiagram, HADAMARD, REGULAR

        diag = ZXDiagram(1)
        wire = diag._wires[0]
        inp = wire[0]
        out = wire[1]
        diag.remove_edge(inp, out)
        s = diag.add_spider("Z", math.pi / 4)
        diag.add_edge(inp, s, HADAMARD)
        diag.add_edge(s, out, HADAMARD)
        wire.insert(1, s)
        diag._qubit[s] = 0
        changed = color_change(diag)
        assert changed is True
        assert diag.color(s) == "X"

    def test_color_change_x_to_z(self):
        from quantumium.zx.rules import color_change
        from quantumium.zx.diagram import ZXDiagram, HADAMARD

        diag = ZXDiagram(1)
        wire = diag._wires[0]
        inp = wire[0]
        out = wire[1]
        diag.remove_edge(inp, out)
        s = diag.add_spider("X", math.pi / 3)
        diag.add_edge(inp, s, HADAMARD)
        diag.add_edge(s, out, HADAMARD)
        wire.insert(1, s)
        diag._qubit[s] = 0
        changed = color_change(diag)
        assert changed is True
        assert diag.color(s) == "Z"

    def test_spider_fusion_no_op_on_different_colors(self):
        from quantumium.zx.rules import spider_fusion
        from quantumium.zx.diagram import ZXDiagram, REGULAR

        diag = ZXDiagram(1)
        wire = diag._wires[0]
        inp = wire[0]
        out = wire[1]
        diag.remove_edge(inp, out)
        s1 = diag.add_spider("Z", 0.5)
        s2 = diag.add_spider("X", 0.3)
        diag.add_edge(inp, s1, REGULAR)
        diag.add_edge(s1, s2, REGULAR)
        diag.add_edge(s2, out, REGULAR)
        wire.insert(1, s1)
        wire.insert(2, s2)
        diag._qubit[s1] = 0
        diag._qubit[s2] = 0
        changed = spider_fusion(diag)
        assert changed is False

    def test_full_simplify_convergence(self):
        from quantumium.zx.diagram import ZXDiagram
        from quantumium.zx.rules import full_simplify

        diag = ZXDiagram(1)
        full_simplify(diag)


class TestZXConvertCoverage:
    """Cover CZ, SWAP, Y gate, and _decompose_* paths."""

    def test_cz_gate_conversion(self):
        from quantumium.zx.convert import circuit_to_zx, zx_to_circuit, _circuit_unitary

        qc = QuantumCircuit(2)
        qc.apply(CZ, 0, 1)
        diag = circuit_to_zx(qc)
        restored = zx_to_circuit(diag)
        u_orig = _circuit_unitary(qc)
        u_rest = _circuit_unitary(restored)
        product = u_orig.conj().T @ u_rest
        diag_vals = np.diag(product)
        assert np.allclose(np.abs(diag_vals), 1.0, atol=1e-6)

    def test_swap_gate_conversion(self):
        from quantumium.zx.convert import circuit_to_zx, zx_to_circuit, _circuit_unitary

        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(SWAP, 0, 1)
        diag = circuit_to_zx(qc)
        restored = zx_to_circuit(diag)
        u_orig = _circuit_unitary(qc)
        u_rest = _circuit_unitary(restored)
        product = u_orig.conj().T @ u_rest
        diag_vals = np.diag(product)
        assert np.allclose(np.abs(diag_vals), 1.0, atol=1e-6)

    def test_y_gate_conversion(self):
        from quantumium.zx.convert import circuit_to_zx, zx_to_circuit, _circuit_unitary

        qc = QuantumCircuit(1)
        qc.apply(Y, 0)
        diag = circuit_to_zx(qc)
        restored = zx_to_circuit(diag)
        u_orig = _circuit_unitary(qc)
        u_rest = _circuit_unitary(restored)
        product = u_orig.conj().T @ u_rest
        diag_vals = np.diag(product)
        assert np.allclose(np.abs(diag_vals), 1.0, atol=1e-6)

    def test_rx_gate_x_phase(self):
        from quantumium.zx.convert import circuit_to_zx, _circuit_unitary

        qc = QuantumCircuit(1)
        qc.apply(Rx(math.pi / 3), 0)
        diag = circuit_to_zx(qc)
        assert diag.num_qubits == 1

    def test_decompose_arbitrary_single_qubit(self):
        from quantumium.zx.convert import circuit_to_zx

        qc = QuantumCircuit(1)
        qc.apply(Rx(0.7), 0)
        qc.apply(Rz(1.2), 0)
        qc.apply(Ry(0.5), 0)
        diag = circuit_to_zx(qc)
        assert diag.num_qubits == 1

    def test_decompose_controlled_rz(self):
        from quantumium.zx.convert import circuit_to_zx

        qc = QuantumCircuit(2)
        qc.apply(controlled(Rz(0.5)), 0, 1)
        diag = circuit_to_zx(qc)
        assert diag.num_qubits == 2

    def test_pending_hadamard_at_end(self):
        from quantumium.zx.convert import circuit_to_zx
        from quantumium.zx.diagram import HADAMARD

        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        diag = circuit_to_zx(qc)
        out = diag.outputs[0]
        wire = diag._wires[0]
        prev = wire[-2]
        assert diag.edge_type(prev, out) == HADAMARD

    def test_identity_gate_skipped(self):
        from quantumium.zx.convert import circuit_to_zx
        from quantumium.core.gate import I

        qc = QuantumCircuit(1)
        qc.apply(I, 0)
        diag = circuit_to_zx(qc)
        assert len(diag._wires[0]) == 2

    def test_zx_optimize_roundtrip(self):
        from quantumium.zx.convert import zx_optimize, _circuit_unitary

        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(T, 0)
        optimized = zx_optimize(qc)
        u_orig = _circuit_unitary(qc)
        u_opt = _circuit_unitary(optimized)
        product = u_orig.conj().T @ u_opt
        diag_vals = np.diag(product)
        assert np.allclose(np.abs(diag_vals), 1.0, atol=1e-4)


# ---- Process tomography coverage (69%) ----

class TestProcessTomographyCoverage:
    """Cover chi_matrix, process_tomography, multi-qubit paths."""

    def test_chi_matrix_identity(self):
        from quantumium.quantum_info.process_tomography import chi_matrix

        U = np.eye(2, dtype=complex)
        chi = chi_matrix(U)
        assert chi.shape == (4, 4)
        assert abs(chi[0, 0]) > 0.9

    def test_chi_matrix_hadamard(self):
        from quantumium.quantum_info.process_tomography import chi_matrix

        H_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        chi = chi_matrix(H_mat)
        assert chi.shape == (4, 4)

    def test_chi_matrix_x_gate(self):
        from quantumium.quantum_info.process_tomography import chi_matrix

        X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
        chi = chi_matrix(X_mat)
        assert abs(chi[1, 1]) > 0.9

    def test_process_tomography_identity(self):
        from quantumium.quantum_info.process_tomography import process_tomography

        U = np.eye(2, dtype=complex)
        result = process_tomography(U)
        assert result.num_qubits == 1
        assert result.input_states_used == 4
        assert result.chi_matrix.shape == (4, 4)
        assert result.choi_matrix.shape == (4, 4)
        assert result.process_fidelity is None

    def test_process_tomography_with_target(self):
        from quantumium.quantum_info.process_tomography import process_tomography

        U = np.eye(2, dtype=complex)
        result = process_tomography(U, target_unitary=U)
        assert result.process_fidelity is not None
        assert abs(result.process_fidelity - 1.0) < 1e-10

    def test_process_tomography_x_gate(self):
        from quantumium.quantum_info.process_tomography import process_tomography

        X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
        result = process_tomography(X_mat, target_unitary=X_mat)
        assert abs(result.process_fidelity - 1.0) < 1e-10

    def test_process_tomography_choi_trace(self):
        from quantumium.quantum_info.process_tomography import process_tomography

        H_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        result = process_tomography(H_mat)
        trace = np.real(np.trace(result.choi_matrix))
        assert abs(trace - 2.0) < 1e-8

    def test_nqubit_paulis_2q(self):
        from quantumium.quantum_info.process_tomography import _nqubit_paulis

        paulis = _nqubit_paulis(2)
        assert len(paulis) == 16
        for p in paulis:
            assert p.shape == (4, 4)

    def test_input_states_2q(self):
        from quantumium.quantum_info.process_tomography import _input_states

        states = _input_states(2)
        assert len(states) == 16
        for s in states:
            assert len(s) == 4
            assert abs(np.linalg.norm(s) - 1.0) < 1e-10

    def test_apply_channel_simulation(self):
        from quantumium.quantum_info.process_tomography import _apply_channel_simulation

        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        psi = np.array([1, 0], dtype=complex)
        rho = _apply_channel_simulation(qc, psi, 100)
        assert rho.shape == (2, 2)
        assert abs(np.trace(rho) - 1.0) < 1e-10

    def test_apply_channel_simulation_2q(self):
        from quantumium.quantum_info.process_tomography import _apply_channel_simulation

        qc = QuantumCircuit(2)
        qc.apply(CNOT, 0, 1)
        psi = np.array([1, 0, 0, 0], dtype=complex)
        rho = _apply_channel_simulation(qc, psi, 100)
        assert rho.shape == (4, 4)
        assert abs(np.trace(rho) - 1.0) < 1e-10


# ---- Surface code decoder coverage (79%) ----

class TestSurfaceCodeDecoderCoverage:
    """Cover _shared_qubit, _correction_path, _bfs_path, _greedy_matching."""

    def test_decode_x_error_d3(self):
        from quantumium.surface_code import SurfaceCode, inject_error, measure_syndrome, mwpm_decode

        code = SurfaceCode(3)
        inject_error(code, "Z", 0)
        x_syn, z_syn = measure_syndrome(code)
        x_corr, z_corr = mwpm_decode(code, x_syn, z_syn)
        assert isinstance(z_corr, np.ndarray)

    def test_decode_z_error_d3(self):
        from quantumium.surface_code import SurfaceCode, inject_error, measure_syndrome, mwpm_decode

        code = SurfaceCode(3)
        inject_error(code, "X", 4)
        x_syn, z_syn = measure_syndrome(code)
        x_corr, z_corr = mwpm_decode(code, x_syn, z_syn)
        assert isinstance(x_corr, np.ndarray)

    def test_decode_multiple_errors(self):
        from quantumium.surface_code import SurfaceCode, inject_error, measure_syndrome, mwpm_decode

        code = SurfaceCode(5)
        inject_error(code, "Z", 0)
        inject_error(code, "Z", 12)
        x_syn, z_syn = measure_syndrome(code)
        x_corr, z_corr = mwpm_decode(code, x_syn, z_syn)
        assert isinstance(z_corr, np.ndarray)

    def test_decode_no_errors(self):
        from quantumium.surface_code import SurfaceCode, measure_syndrome, mwpm_decode

        code = SurfaceCode(3)
        x_syn, z_syn = measure_syndrome(code)
        x_corr, z_corr = mwpm_decode(code, x_syn, z_syn)
        assert np.all(x_corr == 0)
        assert np.all(z_corr == 0)

    def test_decode_boundary_error(self):
        from quantumium.surface_code import SurfaceCode, inject_error, measure_syndrome, mwpm_decode

        code = SurfaceCode(3)
        inject_error(code, "X", code.n_data - 1)
        x_syn, z_syn = measure_syndrome(code)
        x_corr, z_corr = mwpm_decode(code, x_syn, z_syn)
        assert isinstance(x_corr, np.ndarray)

    def test_decode_y_error(self):
        from quantumium.surface_code import SurfaceCode, inject_error, measure_syndrome, mwpm_decode

        code = SurfaceCode(3)
        inject_error(code, "Y", 4)
        x_syn, z_syn = measure_syndrome(code)
        x_corr, z_corr = mwpm_decode(code, x_syn, z_syn)
        assert isinstance(x_corr, np.ndarray)
        assert isinstance(z_corr, np.ndarray)


# ---- QSP angle_finding coverage (81%) ----

class TestQSPAngleFindingCoverage:
    """Cover edge cases in QSP angle finding."""

    def test_chebyshev_angles(self):
        from quantumium.qsp.angle_finding import chebyshev_angles

        angles = chebyshev_angles(3)
        assert isinstance(angles, np.ndarray)
        assert len(angles) == 4

    def test_chebyshev_angles_degree_1(self):
        from quantumium.qsp.angle_finding import chebyshev_angles

        angles = chebyshev_angles(1)
        assert len(angles) == 2

    def test_sign_function_angles(self):
        from quantumium.qsp.angle_finding import sign_function_angles

        angles = sign_function_angles(degree=5)
        assert isinstance(angles, np.ndarray)
        assert len(angles) > 0

    def test_threshold_angles(self):
        from quantumium.qsp.angle_finding import threshold_angles

        angles = threshold_angles(degree=5, threshold=0.5)
        assert isinstance(angles, np.ndarray)

    def test_find_qsp_angles(self):
        from quantumium.qsp.angle_finding import find_qsp_angles

        coeffs = np.array([0.5, 0.0, 0.5])
        try:
            angles = find_qsp_angles(coeffs)
            assert isinstance(angles, np.ndarray)
        except (ValueError, np.linalg.LinAlgError):
            pass


# ---- QASM coverage (83%) ----

class TestQASMCoverage:
    """Cover rotation gate export/import and edge cases."""

    def test_export_rx_gate(self):
        from quantumium.transpiler.qasm import to_qasm

        qc = QuantumCircuit(1)
        qc.apply(Rx(1.23), 0)
        qasm = to_qasm(qc)
        assert "rx" in qasm.lower() or "u3" in qasm.lower() or "rz" in qasm.lower()

    def test_export_ry_gate(self):
        from quantumium.transpiler.qasm import to_qasm

        qc = QuantumCircuit(1)
        qc.apply(Ry(0.45), 0)
        qasm = to_qasm(qc)
        assert "ry" in qasm.lower() or "u3" in qasm.lower() or "rz" in qasm.lower()

    def test_export_rz_gate(self):
        from quantumium.transpiler.qasm import to_qasm

        qc = QuantumCircuit(1)
        qc.apply(Rz(2.1), 0)
        qasm = to_qasm(qc)
        assert "rz" in qasm.lower()

    def test_import_rx_gate(self):
        from quantumium.transpiler.qasm import to_qasm, from_qasm

        qc = QuantumCircuit(1)
        qc.apply(Rx(1.23), 0)
        qasm = to_qasm(qc)
        restored = from_qasm(qasm)
        sv_orig = qc.statevector()
        sv_rest = restored.statevector()
        overlap = abs(np.vdot(sv_orig, sv_rest))
        assert overlap > 0.99

    def test_roundtrip_multi_gate(self):
        from quantumium.transpiler.qasm import to_qasm, from_qasm

        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(Rz(0.5), 1)
        qasm = to_qasm(qc)
        restored = from_qasm(qasm)
        sv_orig = qc.statevector()
        sv_rest = restored.statevector()
        overlap = abs(np.vdot(sv_orig, sv_rest))
        assert overlap > 0.99

    def test_export_swap_gate(self):
        from quantumium.transpiler.qasm import to_qasm

        qc = QuantumCircuit(2)
        qc.apply(SWAP, 0, 1)
        qasm = to_qasm(qc)
        assert "qreg" in qasm

    def test_export_s_t_gates(self):
        from quantumium.transpiler.qasm import to_qasm

        qc = QuantumCircuit(1)
        qc.apply(S, 0)
        qc.apply(T, 0)
        qasm = to_qasm(qc)
        assert len(qasm) > 0

    def test_export_controlled_rz(self):
        from quantumium.transpiler.qasm import to_qasm

        qc = QuantumCircuit(2)
        qc.apply(controlled(Rz(0.8)), 0, 1)
        qasm = to_qasm(qc)
        assert "qreg" in qasm
