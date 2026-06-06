"""Tests for the ZX-calculus optimization engine."""

import math

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, Z, S, T, CNOT, CZ, Rz, Rx, Y, SWAP, I
from quantumium.zx import ZXDiagram, zx_optimize
from quantumium.zx.convert import _circuit_unitary
from quantumium.zx.diagram import REGULAR, HADAMARD


def _unitaries_equivalent(c1, c2, atol=1e-6):
    u1 = _circuit_unitary(c1)
    u2 = _circuit_unitary(c2)
    if u1.shape != u2.shape:
        return False
    product = u1.conj().T @ u2
    diag = np.diag(product)
    if not np.allclose(np.abs(diag), 1.0, atol=atol):
        return False
    return bool(np.allclose(diag, diag[0], atol=atol))


# --- ZXDiagram data structure ---

class TestZXDiagram:
    def test_create_empty(self):
        d = ZXDiagram(2)
        assert d.num_qubits == 2
        assert len(d.inputs) == 2
        assert len(d.outputs) == 2

    def test_add_spider(self):
        d = ZXDiagram(1)
        s = d.add_spider("Z", math.pi / 4)
        assert d.color(s) == "Z"
        assert abs(d.phase(s) - math.pi / 4) < 1e-10

    def test_edge_ops(self):
        d = ZXDiagram(1)
        s1 = d.add_spider("Z")
        s2 = d.add_spider("X")
        d.add_edge(s1, s2, REGULAR)
        assert d.has_edge(s1, s2)
        assert d.edge_type(s1, s2) == REGULAR
        d.remove_edge(s1, s2)
        assert not d.has_edge(s1, s2)

    def test_parallel_edges_cancel(self):
        d = ZXDiagram(1)
        s1 = d.add_spider("Z")
        s2 = d.add_spider("Z")
        d.add_edge(s1, s2, HADAMARD)
        assert d.has_edge(s1, s2)
        d.add_edge(s1, s2, HADAMARD)
        assert not d.has_edge(s1, s2)

    def test_copy(self):
        d = ZXDiagram(1)
        s = d.add_spider("Z", math.pi)
        d2 = d.copy()
        d2.set_phase(s, 0.0)
        assert abs(d.phase(s) - math.pi) < 1e-10
        assert abs(d2.phase(s)) < 1e-10

    def test_t_count(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0).apply(S, 0).apply(T, 0)
        d = ZXDiagram.from_circuit(qc)
        assert d.t_count() == 2

    def test_wire_structure(self):
        d = ZXDiagram(2)
        for q in range(2):
            wire = d.wire(q)
            assert wire[0] == d.inputs[q]
            assert wire[-1] == d.outputs[q]


# --- Circuit → ZX conversion ---

class TestCircuitToZX:
    def test_identity_circuit(self):
        qc = QuantumCircuit(1)
        d = ZXDiagram.from_circuit(qc)
        assert len(d.spider_ids()) == 0
        assert d.wire(0) == [d.inputs[0], d.outputs[0]]

    def test_single_h(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        d = ZXDiagram.from_circuit(qc)
        assert len(d.spider_ids()) == 0
        assert d.edge_type(d.inputs[0], d.outputs[0]) == HADAMARD

    def test_double_h_cancels(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0).apply(H, 0)
        d = ZXDiagram.from_circuit(qc)
        assert len(d.spider_ids()) == 0
        assert d.edge_type(d.inputs[0], d.outputs[0]) == REGULAR

    def test_t_gate(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0)
        d = ZXDiagram.from_circuit(qc)
        spiders = d.spider_ids()
        assert len(spiders) == 1
        assert d.color(spiders[0]) == "Z"
        assert abs(d.phase(spiders[0]) - math.pi / 4) < 1e-10

    def test_cnot(self):
        qc = QuantumCircuit(2)
        qc.apply(CNOT, 0, 1)
        d = ZXDiagram.from_circuit(qc)
        spiders = d.spider_ids()
        assert len(spiders) == 2
        colors = {d.color(s) for s in spiders}
        assert colors == {"Z", "X"}

    def test_cz(self):
        qc = QuantumCircuit(2)
        qc.apply(CZ, 0, 1)
        d = ZXDiagram.from_circuit(qc)
        spiders = d.spider_ids()
        assert len(spiders) == 2
        for s in spiders:
            assert d.color(s) == "Z"
        s0, s1 = spiders
        assert d.has_edge(s0, s1)
        assert d.edge_type(s0, s1) == HADAMARD


# --- ZX → Circuit (round-trip) ---

class TestRoundTrip:
    def test_empty(self):
        qc = QuantumCircuit(1)
        d = ZXDiagram.from_circuit(qc)
        result = d.to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_h(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        result = ZXDiagram.from_circuit(qc).to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_t(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0)
        result = ZXDiagram.from_circuit(qc).to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_s(self):
        qc = QuantumCircuit(1)
        qc.apply(S, 0)
        result = ZXDiagram.from_circuit(qc).to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_z(self):
        qc = QuantumCircuit(1)
        qc.apply(Z, 0)
        result = ZXDiagram.from_circuit(qc).to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_x(self):
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        result = ZXDiagram.from_circuit(qc).to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_rz(self):
        qc = QuantumCircuit(1)
        qc.apply(Rz(0.7), 0)
        result = ZXDiagram.from_circuit(qc).to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_rx(self):
        qc = QuantumCircuit(1)
        qc.apply(Rx(1.2), 0)
        result = ZXDiagram.from_circuit(qc).to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_cnot(self):
        qc = QuantumCircuit(2)
        qc.apply(CNOT, 0, 1)
        result = ZXDiagram.from_circuit(qc).to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_cz(self):
        qc = QuantumCircuit(2)
        qc.apply(CZ, 0, 1)
        result = ZXDiagram.from_circuit(qc).to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_h_t_h(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0).apply(T, 0).apply(H, 0)
        result = ZXDiagram.from_circuit(qc).to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_bell_state_circuit(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(CNOT, 0, 1)
        result = ZXDiagram.from_circuit(qc).to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_multi_gate_circuit(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(T, 0).apply(CNOT, 0, 1).apply(S, 1).apply(H, 1)
        result = ZXDiagram.from_circuit(qc).to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_y_gate(self):
        qc = QuantumCircuit(1)
        qc.apply(Y, 0)
        result = ZXDiagram.from_circuit(qc).to_circuit()
        assert _unitaries_equivalent(qc, result)


# --- Spider fusion ---

class TestSpiderFusion:
    def test_fuse_two_z_spiders(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0).apply(S, 0)
        d = ZXDiagram.from_circuit(qc)
        assert len(d.spider_ids()) == 2
        d.simplify()
        assert len(d.spider_ids()) == 1
        s = d.spider_ids()[0]
        assert abs(d.phase(s) - 3 * math.pi / 4) < 1e-10

    def test_fuse_preserves_semantics(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0).apply(S, 0)
        d = ZXDiagram.from_circuit(qc)
        d.simplify()
        result = d.to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_fuse_t_t_gives_s(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0).apply(T, 0)
        d = ZXDiagram.from_circuit(qc)
        d.simplify()
        spiders = d.spider_ids()
        assert len(spiders) == 1
        assert abs(d.phase(spiders[0]) - math.pi / 2) < 1e-10

    def test_fuse_z_to_identity(self):
        qc = QuantumCircuit(1)
        qc.apply(Z, 0).apply(Z, 0)
        d = ZXDiagram.from_circuit(qc)
        d.simplify()
        assert len(d.spider_ids()) == 0

    def test_fuse_through_cnot_control(self):
        qc = QuantumCircuit(2)
        qc.apply(T, 0).apply(CNOT, 0, 1).apply(T, 0)
        d = ZXDiagram.from_circuit(qc)
        original_t = d.t_count()
        d.simplify()
        result = d.to_circuit()
        assert _unitaries_equivalent(qc, result)
        assert d.t_count() < original_t

    def test_fuse_three_spiders(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0).apply(T, 0).apply(T, 0)
        d = ZXDiagram.from_circuit(qc)
        d.simplify()
        assert len(d.spider_ids()) == 1
        assert abs(d.phase(d.spider_ids()[0]) - 3 * math.pi / 4) < 1e-10


# --- Identity removal ---

class TestIdentityRemoval:
    def test_remove_identity_spider(self):
        d = ZXDiagram(1)
        s = d.add_spider("Z", 0.0)
        d.wire_insert(0, s, REGULAR)
        d.simplify()
        assert len(d.spider_ids()) == 0

    def test_keep_phase_spider(self):
        d = ZXDiagram(1)
        s = d.add_spider("Z", math.pi / 4)
        d.wire_insert(0, s, REGULAR)
        d.simplify()
        assert len(d.spider_ids()) == 1

    def test_keep_high_degree_spider(self):
        qc = QuantumCircuit(2)
        qc.apply(CNOT, 0, 1)
        d = ZXDiagram.from_circuit(qc)
        d.simplify()
        assert len(d.spider_ids()) >= 1


# --- Color change ---

class TestColorChange:
    def test_h_z_h_becomes_x(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0).apply(Z, 0).apply(H, 0)
        d = ZXDiagram.from_circuit(qc)
        d.simplify()
        result = d.to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_h_t_h_becomes_rx(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0).apply(T, 0).apply(H, 0)
        d = ZXDiagram.from_circuit(qc)
        d.simplify()
        spiders = d.spider_ids()
        assert len(spiders) == 1
        assert d.color(spiders[0]) == "X"
        result = d.to_circuit()
        assert _unitaries_equivalent(qc, result)


# --- Full simplification with semantics preservation ---

class TestSimplify:
    def test_simplify_preserves_identity(self):
        qc = QuantumCircuit(1)
        result = zx_optimize(qc)
        assert _unitaries_equivalent(qc, result)

    def test_simplify_preserves_bell(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0).apply(CNOT, 0, 1)
        result = zx_optimize(qc)
        assert _unitaries_equivalent(qc, result)

    def test_simplify_preserves_complex_circuit(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0).apply(T, 0).apply(CNOT, 0, 1)
        qc.apply(S, 1).apply(CNOT, 1, 2).apply(H, 2)
        result = zx_optimize(qc)
        assert _unitaries_equivalent(qc, result)

    @pytest.mark.parametrize("seed", range(5))
    def test_random_clifford_t_circuit(self, seed):
        rng = np.random.default_rng(seed)
        qc = QuantumCircuit(2)
        gates_1q = [H, T, S, Z]
        for _ in range(6):
            g = gates_1q[rng.integers(len(gates_1q))]
            q = int(rng.integers(2))
            qc.apply(g, q)
            if rng.random() < 0.3:
                qc.apply(CNOT, 0, 1)
        result = zx_optimize(qc)
        assert _unitaries_equivalent(qc, result)


# --- T-count reduction ---

class TestTCountReduction:
    def test_adjacent_t_merge(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0).apply(T, 0)
        result = zx_optimize(qc)
        assert _unitaries_equivalent(qc, result)

    def test_t_through_cnot_control(self):
        qc = QuantumCircuit(2)
        qc.apply(T, 0).apply(CNOT, 0, 1).apply(T, 0)
        d = ZXDiagram.from_circuit(qc)
        original_t = d.t_count()
        d.simplify()
        assert d.t_count() < original_t
        result = d.to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_t_cancel_through_cnot(self):
        qc = QuantumCircuit(2)
        qc.apply(T, 0).apply(CNOT, 0, 1).apply(Rz(-math.pi / 4), 0)
        d = ZXDiagram.from_circuit(qc)
        original_t = d.t_count()
        d.simplify()
        assert d.t_count() < original_t
        result = d.to_circuit()
        assert _unitaries_equivalent(qc, result)

    def test_multiple_t_reductions(self):
        qc = QuantumCircuit(2)
        qc.apply(T, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(T, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(T, 0)
        d = ZXDiagram.from_circuit(qc)
        original_t = d.t_count()
        d.simplify()
        assert d.t_count() < original_t
        result = d.to_circuit()
        assert _unitaries_equivalent(qc, result)


# --- Equivalence checking ---

class TestEquivalence:
    def test_same_circuit_equivalent(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0).apply(T, 0)
        d1 = ZXDiagram.from_circuit(qc)
        d2 = ZXDiagram.from_circuit(qc)
        assert d1.is_equivalent(d2)

    def test_simplified_equivalent(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0).apply(T, 0)
        d1 = ZXDiagram.from_circuit(qc)
        d2 = d1.copy()
        d2.simplify()
        assert d1.is_equivalent(d2)

    def test_different_circuits_not_equivalent(self):
        qc1 = QuantumCircuit(1)
        qc1.apply(T, 0)
        qc2 = QuantumCircuit(1)
        qc2.apply(S, 0)
        d1 = ZXDiagram.from_circuit(qc1)
        d2 = ZXDiagram.from_circuit(qc2)
        assert not d1.is_equivalent(d2)


# --- zx_optimize one-liner ---

class TestZXOptimize:
    def test_optimize_reduces_gates(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0).apply(T, 0).apply(T, 0).apply(T, 0)
        result = zx_optimize(qc)
        assert _unitaries_equivalent(qc, result)
        assert len(result.instructions) <= len(qc.instructions)

    def test_optimize_identity_circuit(self):
        qc = QuantumCircuit(1)
        qc.apply(Z, 0).apply(Z, 0)
        result = zx_optimize(qc)
        assert _unitaries_equivalent(qc, result)
        assert len(result.instructions) == 0

    def test_optimize_no_change_needed(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0)
        result = zx_optimize(qc)
        assert _unitaries_equivalent(qc, result)


# --- Existing tests unbroken ---

class TestExistingTestsUnbroken:
    def test_import_core(self):
        from quantumium.core.circuit import QuantumCircuit as QC
        from quantumium.core.gate import H as HG
        qc = QC(1)
        qc.apply(HG, 0)
        state = qc.statevector()
        assert abs(abs(state[0]) ** 2 - 0.5) < 1e-10

    def test_import_surface_code(self):
        from quantumium.surface_code.lattice import SurfaceCode
        lattice = SurfaceCode(3)
        assert lattice.distance == 3
