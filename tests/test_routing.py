"""Tests for qubit routing and mapping (SABRE algorithm)."""

from __future__ import annotations

import numpy as np

from quantumium.transpiler.routing import (
    CouplingMap,
    Layout,
    SABRERouter,
    RoutingResult,
    route,
)
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT, X, Z, S, Rz, controlled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_coupling_compliance(result: RoutingResult, coupling_map: CouplingMap):
    """Assert every 2Q gate in routed circuit operates on connected qubits."""
    for inst in result.circuit.instructions:
        if inst.gate.num_qubits == 2 and inst.gate.name != "SWAP":
            q0, q1 = inst.target_qubits
            assert coupling_map.is_connected(q0, q1), (
                f"Gate {inst.gate.name} on ({q0},{q1}) violates coupling"
            )
        elif inst.gate.name == "SWAP":
            q0, q1 = inst.target_qubits
            assert coupling_map.is_connected(q0, q1), f"SWAP on ({q0},{q1}) violates coupling"


def _remap_counts(counts: dict[str, int], layout: Layout, num_logical: int) -> dict[str, int]:
    """Map measurement bitstrings from physical to logical qubit order."""
    mapped = {}
    for bitstring, count in counts.items():
        n = len(bitstring)
        logical_bits = ["0"] * num_logical
        for lq in range(num_logical):
            pq = layout.logical_to_physical(lq)
            if pq < n:
                bit_pos = n - 1 - pq
                logical_bits[num_logical - 1 - lq] = bitstring[bit_pos]
        key = "".join(logical_bits)
        mapped[key] = mapped.get(key, 0) + count
    return mapped


# ===========================================================================
# CouplingMap Tests
# ===========================================================================


class TestCouplingMap:
    def test_linear_topology(self):
        cm = CouplingMap.from_linear(5)
        assert cm.num_qubits == 5
        assert len(cm.edges) == 4
        assert cm.is_connected(0, 1)
        assert cm.is_connected(3, 4)
        assert not cm.is_connected(0, 2)

    def test_grid_topology(self):
        cm = CouplingMap.from_grid(2, 3)
        assert cm.num_qubits == 6
        assert cm.is_connected(0, 1)
        assert cm.is_connected(0, 3)
        assert cm.is_connected(1, 4)
        assert not cm.is_connected(0, 4)

    def test_heavy_hex_topology(self):
        cm = CouplingMap.from_heavy_hex(4)
        assert cm.num_qubits > 4
        assert cm.is_connected(0, 1)

    def test_custom_edges(self):
        cm = CouplingMap([(0, 1), (1, 2), (2, 3), (0, 3)])
        assert cm.num_qubits == 4
        assert cm.is_connected(0, 3)
        assert not cm.is_connected(0, 2)

    def test_distance_linear(self):
        cm = CouplingMap.from_linear(5)
        assert cm.distance(0, 0) == 0
        assert cm.distance(0, 1) == 1
        assert cm.distance(0, 4) == 4
        assert cm.distance(2, 4) == 2

    def test_distance_grid(self):
        cm = CouplingMap.from_grid(3, 3)
        assert cm.distance(0, 8) == 4  # corner to corner
        assert cm.distance(0, 4) == 2  # corner to center

    def test_shortest_path(self):
        cm = CouplingMap.from_linear(5)
        path = cm.shortest_path(0, 4)
        assert path == [0, 1, 2, 3, 4]
        assert cm.shortest_path(2, 2) == [2]

    def test_shortest_path_grid(self):
        cm = CouplingMap.from_grid(2, 3)
        path = cm.shortest_path(0, 5)
        assert len(path) == cm.distance(0, 5) + 1
        assert path[0] == 0
        assert path[-1] == 5

    def test_neighbors(self):
        cm = CouplingMap.from_linear(4)
        assert cm.neighbors(0) == {1}
        assert cm.neighbors(1) == {0, 2}
        assert cm.neighbors(3) == {2}

    def test_symmetric_edges(self):
        cm = CouplingMap([(0, 1), (2, 1)])
        assert cm.is_connected(0, 1)
        assert cm.is_connected(1, 0)
        assert cm.is_connected(1, 2)
        assert cm.is_connected(2, 1)

    def test_repr(self):
        cm = CouplingMap.from_linear(3)
        assert "3 qubits" in repr(cm)
        assert "2 edges" in repr(cm)


# ===========================================================================
# Layout Tests
# ===========================================================================


class TestLayout:
    def test_trivial_layout(self):
        layout = Layout.trivial(4)
        for i in range(4):
            assert layout.logical_to_physical(i) == i
            assert layout.physical_to_logical(i) == i

    def test_custom_layout(self):
        layout = Layout({0: 2, 1: 0, 2: 1})
        assert layout.logical_to_physical(0) == 2
        assert layout.logical_to_physical(1) == 0
        assert layout.physical_to_logical(2) == 0
        assert layout.physical_to_logical(0) == 1

    def test_swap(self):
        layout = Layout.trivial(3)
        layout.swap(0, 1)
        assert layout.logical_to_physical(0) == 1
        assert layout.logical_to_physical(1) == 0
        assert layout.physical_to_logical(0) == 1
        assert layout.physical_to_logical(1) == 0

    def test_double_swap(self):
        layout = Layout.trivial(3)
        layout.swap(0, 1)
        layout.swap(0, 1)
        for i in range(3):
            assert layout.logical_to_physical(i) == i

    def test_copy_independence(self):
        layout = Layout.trivial(3)
        copy = layout.copy()
        copy.swap(0, 1)
        assert layout.logical_to_physical(0) == 0
        assert copy.logical_to_physical(0) == 1

    def test_mapping_property(self):
        layout = Layout({0: 2, 1: 0, 2: 1})
        m = layout.mapping
        assert m == {0: 2, 1: 0, 2: 1}


# ===========================================================================
# SABRE Router — Correctness Tests
# ===========================================================================


class TestSABRECorrectness:
    def test_already_compatible(self):
        """Circuit that already satisfies coupling needs no SWAPs."""
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(CNOT, 1, 2)
        cm = CouplingMap.from_linear(3)
        result = route(qc, cm, seed=42)
        assert result.swap_count == 0
        _check_coupling_compliance(result, cm)

    def test_single_nonlocal_cnot(self):
        """Single CNOT between non-adjacent qubits requires routing."""
        qc = QuantumCircuit(4)
        qc.apply(CNOT, 0, 3)
        cm = CouplingMap.from_linear(4)
        result = route(qc, cm, seed=42)
        _check_coupling_compliance(result, cm)
        has_cnot = any(
            inst.gate.num_qubits == 2 and inst.gate.name != "SWAP"
            for inst in result.circuit.instructions
        )
        assert has_cnot

    def test_bell_state_preserved(self):
        """Routed Bell state gives same measurement statistics."""
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        cm = CouplingMap.from_linear(2)
        result = route(qc, cm, seed=42)
        orig = qc.measure_all(shots=5000)
        routed = result.circuit.measure_all(shots=5000)
        mapped = _remap_counts(routed, result.initial_layout, 2)
        for key in ("00", "11"):
            assert mapped.get(key, 0) > 1000

    def test_ghz3_on_linear(self):
        """GHZ-3 routed on linear topology preserves correlations."""
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(CNOT, 0, 2)
        cm = CouplingMap.from_linear(3)
        result = route(qc, cm, seed=42)
        _check_coupling_compliance(result, cm)
        routed = result.circuit.measure_all(shots=5000)
        mapped = _remap_counts(routed, result.initial_layout, 3)
        assert mapped.get("000", 0) > 1000
        assert mapped.get("111", 0) > 1000

    def test_single_qubit_only(self):
        """Circuit with only single-qubit gates needs no SWAPs."""
        qc = QuantumCircuit(4)
        qc.apply(H, 0)
        qc.apply(X, 1)
        qc.apply(Z, 2)
        qc.apply(S, 3)
        cm = CouplingMap.from_linear(4)
        result = route(qc, cm, seed=42)
        assert result.swap_count == 0
        assert result.routed_gate_count == 4

    def test_deterministic_with_seed(self):
        """Same seed produces same routing."""
        qc = QuantumCircuit(4)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 3)
        qc.apply(CNOT, 1, 3)
        cm = CouplingMap.from_linear(4)
        r1 = route(qc, cm, seed=99)
        r2 = route(qc, cm, seed=99)
        assert r1.swap_count == r2.swap_count
        assert len(r1.circuit.instructions) == len(r2.circuit.instructions)

    def test_routing_with_custom_layout(self):
        """Custom initial layout is respected."""
        qc = QuantumCircuit(3)
        qc.apply(CNOT, 0, 2)
        cm = CouplingMap.from_linear(3)
        layout = Layout({0: 0, 1: 2, 2: 1})
        result = route(qc, cm, initial_layout=layout, seed=42)
        _check_coupling_compliance(result, cm)


# ===========================================================================
# Coupling Compliance Tests
# ===========================================================================


class TestCouplingCompliance:
    def test_qft4_linear(self):
        """QFT-4 on linear topology: all 2Q gates on connected qubits."""
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.apply(H, i)
            for j in range(i + 1, 4):
                angle = np.pi / (2 ** (j - i))
                crz = controlled(Rz(angle))
                qc.apply(crz, j, i)
        cm = CouplingMap.from_linear(4)
        result = route(qc, cm, seed=42)
        _check_coupling_compliance(result, cm)
        assert result.swap_count > 0

    def test_qft4_grid(self):
        """QFT-4 on grid: fewer SWAPs than linear due to richer connectivity."""
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.apply(H, i)
            for j in range(i + 1, 4):
                angle = np.pi / (2 ** (j - i))
                crz = controlled(Rz(angle))
                qc.apply(crz, j, i)
        cm_lin = CouplingMap.from_linear(4)
        cm_grid = CouplingMap.from_grid(2, 2)
        r_lin = route(qc, cm_lin, seed=42)
        r_grid = route(qc, cm_grid, seed=42)
        _check_coupling_compliance(r_lin, cm_lin)
        _check_coupling_compliance(r_grid, cm_grid)
        assert r_grid.swap_count <= r_lin.swap_count

    def test_all_to_all_no_swaps(self):
        """Fully connected topology needs no SWAPs."""
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        cm = CouplingMap(edges)
        qc = QuantumCircuit(4)
        qc.apply(CNOT, 0, 3)
        qc.apply(CNOT, 1, 3)
        qc.apply(CNOT, 2, 0)
        result = route(qc, cm, seed=42)
        assert result.swap_count == 0
        _check_coupling_compliance(result, cm)

    def test_chain_of_cnots(self):
        """Long chain of CNOTs all satisfy coupling after routing."""
        qc = QuantumCircuit(5)
        qc.apply(CNOT, 0, 4)
        qc.apply(CNOT, 1, 3)
        qc.apply(CNOT, 2, 4)
        qc.apply(CNOT, 0, 2)
        cm = CouplingMap.from_linear(5)
        result = route(qc, cm, seed=42)
        _check_coupling_compliance(result, cm)

    def test_heavy_hex_compliance(self):
        """Routing on heavy-hex respects all coupling constraints."""
        qc = QuantumCircuit(4)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 3)
        qc.apply(CNOT, 1, 2)
        cm = CouplingMap.from_heavy_hex(5)
        result = route(qc, cm, seed=42)
        _check_coupling_compliance(result, cm)


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    def test_two_qubit_circuit(self):
        """Minimal 2-qubit circuit on 2-qubit coupling."""
        qc = QuantumCircuit(2)
        qc.apply(CNOT, 0, 1)
        cm = CouplingMap([(0, 1)])
        result = route(qc, cm, seed=42)
        assert result.swap_count == 0

    def test_empty_circuit(self):
        """Circuit with no gates routes trivially."""
        qc = QuantumCircuit(3)
        cm = CouplingMap.from_linear(3)
        result = route(qc, cm, seed=42)
        assert result.swap_count == 0
        assert result.routed_gate_count == 0

    def test_single_gate_circuit(self):
        """Single H gate needs no routing."""
        qc = QuantumCircuit(3)
        qc.apply(H, 1)
        cm = CouplingMap.from_linear(3)
        result = route(qc, cm, seed=42)
        assert result.swap_count == 0
        assert result.routed_gate_count == 1

    def test_circuit_larger_than_coupling(self):
        """Circuit uses fewer qubits than coupling map has."""
        qc = QuantumCircuit(2)
        qc.apply(CNOT, 0, 1)
        cm = CouplingMap.from_linear(5)
        result = route(qc, cm, seed=42)
        assert result.swap_count == 0
        _check_coupling_compliance(result, cm)

    def test_mixed_1q_2q_gates(self):
        """Circuit with interleaved single and two-qubit gates."""
        qc = QuantumCircuit(4)
        qc.apply(H, 0)
        qc.apply(H, 1)
        qc.apply(CNOT, 0, 3)
        qc.apply(X, 2)
        qc.apply(CNOT, 1, 2)
        qc.apply(H, 3)
        cm = CouplingMap.from_linear(4)
        result = route(qc, cm, seed=42)
        _check_coupling_compliance(result, cm)


# ===========================================================================
# RoutingResult Tests
# ===========================================================================


class TestRoutingResult:
    def test_result_fields(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 2)
        cm = CouplingMap.from_linear(3)
        result = route(qc, cm, seed=42)
        assert isinstance(result, RoutingResult)
        assert isinstance(result.circuit, QuantumCircuit)
        assert isinstance(result.initial_layout, Layout)
        assert isinstance(result.final_layout, Layout)
        assert result.original_gate_count == 2
        assert result.routed_gate_count >= 2

    def test_swap_count_matches(self):
        """swap_count matches actual SWAP gates in routed circuit."""
        qc = QuantumCircuit(4)
        qc.apply(CNOT, 0, 3)
        qc.apply(CNOT, 1, 3)
        cm = CouplingMap.from_linear(4)
        result = route(qc, cm, seed=42)
        actual_swaps = sum(1 for inst in result.circuit.instructions if inst.gate.name == "SWAP")
        assert actual_swaps == result.swap_count


# ===========================================================================
# SABRE Algorithm Properties
# ===========================================================================


class TestSABREProperties:
    def test_backward_pass_improves(self):
        """Multiple iterations can find better solutions than single pass."""
        qc = QuantumCircuit(5)
        qc.apply(CNOT, 0, 4)
        qc.apply(CNOT, 1, 3)
        qc.apply(CNOT, 2, 4)
        cm = CouplingMap.from_linear(5)
        r1 = SABRERouter(seed=42, max_iterations=1).route(qc, cm)
        r3 = SABRERouter(seed=42, max_iterations=3).route(qc, cm)
        assert r3.swap_count <= r1.swap_count

    def test_richer_topology_fewer_swaps(self):
        """Grid topology produces fewer SWAPs than linear for same circuit."""
        qc = QuantumCircuit(4)
        qc.apply(CNOT, 0, 3)
        qc.apply(CNOT, 1, 3)
        qc.apply(CNOT, 0, 2)
        cm_lin = CouplingMap.from_linear(4)
        cm_grid = CouplingMap.from_grid(2, 2)
        r_lin = route(qc, cm_lin, seed=42)
        r_grid = route(qc, cm_grid, seed=42)
        assert r_grid.swap_count <= r_lin.swap_count

    def test_router_class_direct(self):
        """SABRERouter can be used directly."""
        router = SABRERouter(seed=42, max_iterations=2)
        qc = QuantumCircuit(3)
        qc.apply(CNOT, 0, 2)
        cm = CouplingMap.from_linear(3)
        result = router.route(qc, cm)
        _check_coupling_compliance(result, cm)


# ===========================================================================
# Measurement Preservation Tests
# ===========================================================================


class TestMeasurementPreservation:
    def test_bell_statistics(self):
        """Routed Bell state preserves 50/50 |00> + |11> distribution."""
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        cm = CouplingMap.from_linear(2)
        result = route(qc, cm, seed=42)
        routed_counts = result.circuit.measure_all(shots=5000)
        mapped = _remap_counts(routed_counts, result.initial_layout, 2)
        assert mapped.get("00", 0) > 1500
        assert mapped.get("11", 0) > 1500
        assert mapped.get("01", 0) < 100
        assert mapped.get("10", 0) < 100

    def test_x_gate_statistics(self):
        """X gate on qubit 0 preserves |1> outcome after routing."""
        qc = QuantumCircuit(3)
        qc.apply(X, 0)
        cm = CouplingMap.from_linear(3)
        result = route(qc, cm, seed=42)
        routed_counts = result.circuit.measure_all(shots=1000)
        mapped = _remap_counts(routed_counts, result.initial_layout, 3)
        assert mapped.get("001", 0) == 1000

    def test_ghz4_statistics_on_grid(self):
        """GHZ-4 routed on grid preserves all-zero/all-one correlations."""
        qc = QuantumCircuit(4)
        qc.apply(H, 0)
        for i in range(1, 4):
            qc.apply(CNOT, 0, i)
        cm = CouplingMap.from_grid(2, 2)
        result = route(qc, cm, seed=42)
        _check_coupling_compliance(result, cm)
        routed_counts = result.circuit.measure_all(shots=5000)
        mapped = _remap_counts(routed_counts, result.initial_layout, 4)
        assert mapped.get("0000", 0) > 1500
        assert mapped.get("1111", 0) > 1500
        spurious = sum(v for k, v in mapped.items() if k not in ("0000", "1111"))
        assert spurious < 100


# ===========================================================================
# Integration with Existing Transpiler
# ===========================================================================


class TestTranspilerIntegration:
    def test_route_then_decompose_swap(self):
        """Routed SWAPs can be decomposed to CNOTs via existing pass."""
        from quantumium.transpiler.passes import decompose_swap

        qc = QuantumCircuit(4)
        qc.apply(CNOT, 0, 3)
        cm = CouplingMap.from_linear(4)
        result = route(qc, cm, seed=42)
        decomposed = decompose_swap(result.circuit.instructions)
        for inst in decomposed:
            assert inst.gate.name != "SWAP"
            if inst.gate.num_qubits == 2:
                assert cm.is_connected(*inst.target_qubits)

    def test_existing_tests_unbroken(self):
        """Importing routing doesn't break existing transpiler."""
        from quantumium.transpiler.passes import transpile

        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(H, 0)
        opt = transpile(qc)
        assert len(opt.instructions) <= len(qc.instructions)
