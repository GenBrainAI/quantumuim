"""Tests for QAOA implementation."""

import numpy as np

from quantumium.algorithms.qaoa import (
    maxcut_cost_operator,
    qaoa_circuit,
    qaoa_expectation,
    qaoa_maxcut,
)


class TestMaxCutCost:
    def test_triangle_graph(self):
        """Triangle graph: every partition cuts exactly 2 edges."""
        edges = [(0, 1), (1, 2), (0, 2)]
        cost = maxcut_cost_operator(3, edges)
        assert len(cost) == 8
        assert cost[0] == 0.0  # |000⟩ — no cuts
        assert cost[7] == 0.0  # |111⟩ — no cuts
        assert cost[1] == 2.0  # |001⟩ — cuts (0,2) and (1,2)
        assert cost[3] == 2.0  # |011⟩ — cuts (0,1) and (0,2)

    def test_single_edge(self):
        """Single edge (0,1): max cut = 1."""
        cost = maxcut_cost_operator(2, [(0, 1)])
        assert cost[0] == 0.0  # |00⟩
        assert cost[1] == 1.0  # |01⟩
        assert cost[2] == 1.0  # |10⟩
        assert cost[3] == 0.0  # |11⟩

    def test_no_edges(self):
        """Graph with no edges: all costs are 0."""
        cost = maxcut_cost_operator(3, [])
        np.testing.assert_array_equal(cost, np.zeros(8))


class TestQAOACircuit:
    def test_circuit_output_normalized(self):
        """QAOA circuit should produce a normalized state."""
        cost = maxcut_cost_operator(3, [(0, 1), (1, 2)])
        gammas = np.array([0.5])
        betas = np.array([0.3])
        qc = qaoa_circuit(3, cost, gammas, betas)
        sv = qc.statevector()
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    def test_expectation_bounded(self):
        """Expectation value should be between min and max cost."""
        edges = [(0, 1), (1, 2), (0, 2)]
        cost = maxcut_cost_operator(3, edges)
        gammas = np.array([1.0])
        betas = np.array([0.5])
        exp_val = qaoa_expectation(3, cost, gammas, betas)
        assert 0 <= exp_val <= max(cost)

    def test_zero_params_uniform(self):
        """With gamma=0, beta=0, state stays uniform → expectation = mean(cost)."""
        edges = [(0, 1)]
        cost = maxcut_cost_operator(2, edges)
        gammas = np.array([0.0])
        betas = np.array([0.0])
        exp_val = qaoa_expectation(2, cost, gammas, betas)
        expected_mean = np.mean(cost)
        assert abs(exp_val - expected_mean) < 1e-10

    def test_multi_layer(self):
        """p=2 circuit should also produce valid output."""
        cost = maxcut_cost_operator(2, [(0, 1)])
        gammas = np.array([0.5, 1.0])
        betas = np.array([0.3, 0.7])
        qc = qaoa_circuit(2, cost, gammas, betas)
        sv = qc.statevector()
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10


class TestQAOAMaxCut:
    def test_simple_graph_finds_cut(self):
        """QAOA should find a good cut for a simple 3-node path graph."""
        edges = [(0, 1), (1, 2)]
        result = qaoa_maxcut(3, edges, p=1, num_starts=5, max_iter=50)
        assert result.optimal_value > 1.0
        assert len(result.optimal_bitstring) == 3
        assert len(result.history) > 0

    def test_single_edge_optimal(self):
        """For a single edge, QAOA should achieve close to max cut = 1."""
        result = qaoa_maxcut(2, [(0, 1)], p=2, num_starts=5, max_iter=80)
        assert result.optimal_value > 0.8
