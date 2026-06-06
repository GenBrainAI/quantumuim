"""Tests for the resource estimation module."""

import math

import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, Y, Z, S, T, CNOT, CZ, SWAP, CCX, Rz, Rx, Ry
from quantumium.resource_estimation import (
    CircuitAnalyzer,
    LogicalResources,
    PhysicalEstimate,
    ResourceEstimator,
    MagicStateFactory,
)


# ──────────────────────────────────────────────
# CircuitAnalyzer — T-gate counting
# ──────────────────────────────────────────────

class TestTCount:
    def test_single_t_gate(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0)
        assert CircuitAnalyzer().t_count(qc) == 1

    def test_no_t_gates(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        assert CircuitAnalyzer().t_count(qc) == 0

    def test_multiple_t_gates(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0)
        qc.apply(T, 0)
        qc.apply(T, 0)
        assert CircuitAnalyzer().t_count(qc) == 3

    def test_toffoli_t_count(self):
        qc = QuantumCircuit(3)
        qc.apply(CCX, 0, 1, 2)
        assert CircuitAnalyzer().t_count(qc) == 7

    def test_mixed_t_and_clifford(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(T, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(T, 1)
        assert CircuitAnalyzer().t_count(qc) == 2

    def test_toffoli_plus_t(self):
        qc = QuantumCircuit(3)
        qc.apply(CCX, 0, 1, 2)
        qc.apply(T, 0)
        assert CircuitAnalyzer().t_count(qc) == 8


# ──────────────────────────────────────────────
# CircuitAnalyzer — T-depth
# ──────────────────────────────────────────────

class TestTDepth:
    def test_sequential_t(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0)
        qc.apply(T, 0)
        qc.apply(T, 0)
        assert CircuitAnalyzer().t_depth(qc) == 3

    def test_parallel_t(self):
        qc = QuantumCircuit(3)
        qc.apply(T, 0)
        qc.apply(T, 1)
        qc.apply(T, 2)
        assert CircuitAnalyzer().t_depth(qc) == 1

    def test_no_t_depth_zero(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        assert CircuitAnalyzer().t_depth(qc) == 0

    def test_mixed_depth(self):
        qc = QuantumCircuit(2)
        qc.apply(T, 0)
        qc.apply(T, 1)
        qc.apply(CNOT, 0, 1)
        qc.apply(T, 0)
        assert CircuitAnalyzer().t_depth(qc) == 2

    def test_toffoli_depth(self):
        qc = QuantumCircuit(3)
        qc.apply(CCX, 0, 1, 2)
        assert CircuitAnalyzer().t_depth(qc) == 1


# ──────────────────────────────────────────────
# CircuitAnalyzer — Clifford counting
# ──────────────────────────────────────────────

class TestCliffordCount:
    def test_basic_cliffords(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(X, 1)
        qc.apply(CNOT, 0, 1)
        counts = CircuitAnalyzer().clifford_count(qc)
        assert counts["H"] == 1
        assert counts["X"] == 1
        assert counts["CNOT"] == 1

    def test_t_not_clifford(self):
        qc = QuantumCircuit(1)
        qc.apply(T, 0)
        counts = CircuitAnalyzer().clifford_count(qc)
        assert len(counts) == 0

    def test_swap_is_clifford(self):
        qc = QuantumCircuit(2)
        qc.apply(SWAP, 0, 1)
        counts = CircuitAnalyzer().clifford_count(qc)
        assert counts["SWAP"] == 1


# ──────────────────────────────────────────────
# CircuitAnalyzer — rotation gates
# ──────────────────────────────────────────────

class TestRotationCount:
    def test_rz_detected(self):
        qc = QuantumCircuit(1)
        qc.apply(Rz(0.5), 0)
        assert CircuitAnalyzer().rotation_count(qc) == 1

    def test_rx_ry_detected(self):
        qc = QuantumCircuit(1)
        qc.apply(Rx(0.3), 0)
        qc.apply(Ry(0.7), 0)
        assert CircuitAnalyzer().rotation_count(qc) == 2

    def test_rotation_t_cost(self):
        analyzer = CircuitAnalyzer(rotation_precision=1e-10)
        qc = QuantumCircuit(1)
        qc.apply(Rz(0.5), 0)
        res = analyzer.analyze(qc)
        assert res.n_rotation_gates == 1
        assert res.rotation_t_cost > 0
        assert res.total_t_budget == res.n_t_gates + res.rotation_t_cost


# ──────────────────────────────────────────────
# CircuitAnalyzer — full analysis
# ──────────────────────────────────────────────

class TestAnalyze:
    def test_bell_state_analysis(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        res = CircuitAnalyzer().analyze(qc)
        assert res.n_qubits == 2
        assert res.n_t_gates == 0
        assert res.n_clifford == 2
        assert res.circuit_depth == 2

    def test_t_circuit_analysis(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(T, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(T, 1)
        res = CircuitAnalyzer().analyze(qc)
        assert res.n_t_gates == 2
        assert res.n_clifford == 2

    def test_summary_string(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        qc.apply(T, 0)
        res = CircuitAnalyzer().analyze(qc)
        s = res.summary()
        assert "Logical Resources" in s
        assert "T-gates" in s

    def test_gate_counts_dict(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(H, 1)
        qc.apply(CNOT, 0, 1)
        res = CircuitAnalyzer().analyze(qc)
        assert res.gate_counts["H"] == 2
        assert res.gate_counts["CNOT"] == 1


# ──────────────────────────────────────────────
# LogicalResources
# ──────────────────────────────────────────────

class TestLogicalResources:
    def test_total_t_budget(self):
        lr = LogicalResources(
            n_qubits=5, n_t_gates=10, t_depth=3,
            n_clifford=20, n_measurements=0, circuit_depth=15,
            n_rotation_gates=2, rotation_t_cost=100,
        )
        assert lr.total_t_budget == 110

    def test_summary_contains_fields(self):
        lr = LogicalResources(
            n_qubits=3, n_t_gates=7, t_depth=4,
            n_clifford=10, n_measurements=1, circuit_depth=12,
        )
        s = lr.summary()
        assert "3" in s
        assert "7" in s


# ──────────────────────────────────────────────
# MagicStateFactory
# ──────────────────────────────────────────────

class TestMagicStateFactory:
    def test_output_error_one_round(self):
        f = MagicStateFactory(code_distance=7, physical_error_rate=1e-3)
        assert abs(f.output_error_rate - 35.0 * (1e-3) ** 3) < 1e-15

    def test_output_error_two_rounds(self):
        f = MagicStateFactory(code_distance=7, physical_error_rate=1e-3, n_distillation_rounds=2)
        p1 = 35.0 * (1e-3) ** 3
        p2 = 35.0 * p1 ** 3
        assert abs(f.output_error_rate - p2) < 1e-30

    def test_qubits_per_factory(self):
        f = MagicStateFactory(code_distance=7)
        expected = 15 * (2 * 7 * 7 - 1)
        assert f.qubits_per_factory == expected

    def test_cycles_per_t_state(self):
        f = MagicStateFactory(code_distance=7)
        assert f.cycles_per_t_state == 5 * 7

    def test_throughput(self):
        f = MagicStateFactory(code_distance=7)
        tp = f.throughput_per_us(1.0)
        assert tp == pytest.approx(1.0 / 35.0)

    def test_input_states(self):
        f1 = MagicStateFactory(code_distance=5, n_distillation_rounds=1)
        assert f1.input_states == 15
        f2 = MagicStateFactory(code_distance=5, n_distillation_rounds=2)
        assert f2.input_states == 225

    def test_rounds_needed(self):
        rounds = MagicStateFactory.rounds_needed(1e-3, 1e-15)
        assert rounds >= 2

    def test_summary(self):
        f = MagicStateFactory(code_distance=7)
        s = f.summary()
        assert "MagicStateFactory" in s
        assert "d=7" in s


# ──────────────────────────────────────────────
# ResourceEstimator — code distance
# ──────────────────────────────────────────────

class TestCodeDistance:
    def test_distance_is_odd(self):
        est = ResourceEstimator(physical_error_rate=1e-3)
        d = est.code_distance(1e-10, 1000)
        assert d % 2 == 1

    def test_distance_minimum_3(self):
        est = ResourceEstimator(physical_error_rate=1e-4)
        d = est.code_distance(0.1, 1)
        assert d >= 3

    def test_tighter_target_needs_larger_distance(self):
        est = ResourceEstimator(physical_error_rate=1e-3)
        d1 = est.code_distance(1e-6, 100)
        d2 = est.code_distance(1e-12, 100)
        assert d2 >= d1

    def test_more_ops_needs_larger_distance(self):
        est = ResourceEstimator(physical_error_rate=1e-3)
        d1 = est.code_distance(1e-10, 100)
        d2 = est.code_distance(1e-10, 100000)
        assert d2 >= d1


# ──────────────────────────────────────────────
# ResourceEstimator — factory count
# ──────────────────────────────────────────────

class TestFactoryCount:
    def test_no_t_gates_no_factory(self):
        est = ResourceEstimator()
        assert est.factory_count(0, 0, 7) == 0

    def test_single_factory_minimum(self):
        est = ResourceEstimator()
        assert est.factory_count(10, 10, 7) >= 1

    def test_more_t_more_factories(self):
        est = ResourceEstimator()
        f1 = est.factory_count(100, 10, 7)
        f2 = est.factory_count(1000, 10, 7)
        assert f2 >= f1


# ──────────────────────────────────────────────
# ResourceEstimator — full estimation
# ──────────────────────────────────────────────

class TestFullEstimation:
    def test_bell_state_estimate(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        est = ResourceEstimator()
        result = est.estimate(qc, target_error=1e-6)
        assert result.total_qubits > 0
        assert result.code_distance >= 3
        assert result.n_factories == 0

    def test_t_heavy_circuit(self):
        qc = QuantumCircuit(3)
        for _ in range(10):
            qc.apply(T, 0)
            qc.apply(T, 1)
            qc.apply(T, 2)
        est = ResourceEstimator()
        result = est.estimate(qc)
        assert result.n_factories >= 1
        assert result.factory_qubits > 0
        assert result.total_qubits > result.data_qubits

    def test_toffoli_circuit(self):
        qc = QuantumCircuit(3)
        qc.apply(CCX, 0, 1, 2)
        est = ResourceEstimator()
        result = est.estimate(qc)
        assert result.logical.n_t_gates == 7
        assert result.n_factories >= 1

    def test_rotation_circuit(self):
        qc = QuantumCircuit(1)
        qc.apply(Rz(0.123), 0)
        qc.apply(Rz(0.456), 0)
        est = ResourceEstimator(rotation_precision=1e-10)
        result = est.estimate(qc)
        assert result.logical.n_rotation_gates == 2
        assert result.logical.total_t_budget > 0

    def test_estimate_summary(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(T, 0)
        qc.apply(CNOT, 0, 1)
        est = ResourceEstimator()
        result = est.estimate(qc)
        s = result.summary()
        assert "Physical Resource Estimate" in s
        assert "Code distance" in s
        assert "Total physical qubits" in s

    def test_runtime_positive(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(T, 0)
        qc.apply(CNOT, 0, 1)
        est = ResourceEstimator()
        result = est.estimate(qc)
        assert result.runtime_us > 0
        assert result.runtime_seconds > 0

    def test_physical_error_rate_stored(self):
        est = ResourceEstimator(physical_error_rate=5e-4)
        qc = QuantumCircuit(1)
        qc.apply(T, 0)
        result = est.estimate(qc)
        assert result.physical_error_rate == 5e-4


# ──────────────────────────────────────────────
# Scaling and monotonicity
# ──────────────────────────────────────────────

class TestScaling:
    def test_more_qubits_more_physical(self):
        est = ResourceEstimator()
        qc1 = QuantumCircuit(2)
        qc1.apply(H, 0)
        qc1.apply(CNOT, 0, 1)

        qc2 = QuantumCircuit(10)
        for i in range(9):
            qc2.apply(H, i)
            qc2.apply(CNOT, i, i + 1)

        r1 = est.estimate(qc1)
        r2 = est.estimate(qc2)
        assert r2.total_qubits > r1.total_qubits

    def test_tighter_error_more_qubits(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(T, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(T, 1)
        qc.apply(CNOT, 1, 2)

        est = ResourceEstimator()
        r1 = est.estimate(qc, target_error=1e-3)
        r2 = est.estimate(qc, target_error=1e-12)
        assert r2.code_distance >= r1.code_distance
        assert r2.total_qubits >= r1.total_qubits

    def test_better_physical_fewer_qubits(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(T, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(T, 1)

        est_noisy = ResourceEstimator(physical_error_rate=1e-3)
        est_clean = ResourceEstimator(physical_error_rate=1e-4)
        r_noisy = est_noisy.estimate(qc, target_error=1e-10)
        r_clean = est_clean.estimate(qc, target_error=1e-10)
        assert r_clean.code_distance <= r_noisy.code_distance


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────

class TestEdgeCases:
    def test_clifford_only_no_factory(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(CNOT, 1, 2)
        est = ResourceEstimator()
        result = est.estimate(qc)
        assert result.n_factories == 0
        assert result.factory_qubits == 0

    def test_single_gate_circuit(self):
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        est = ResourceEstimator()
        result = est.estimate(qc)
        assert result.total_qubits > 0

    def test_empty_logical_resources(self):
        lr = LogicalResources(
            n_qubits=1, n_t_gates=0, t_depth=0,
            n_clifford=0, n_measurements=0, circuit_depth=0,
        )
        assert lr.total_t_budget == 0

    def test_routing_overhead_applied(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        est1 = ResourceEstimator(routing_overhead=1.0)
        est2 = ResourceEstimator(routing_overhead=2.0)
        r1 = est1.estimate(qc)
        r2 = est2.estimate(qc)
        assert r2.data_qubits > r1.data_qubits
