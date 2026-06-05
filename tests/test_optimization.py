"""Tests for quantum optimization applications."""

import numpy as np

from quantumium.applications.optimization import (
    solve_maxcut,
    portfolio_optimization,
)


class TestMaxCut:
    def test_triangle(self):
        adj = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        result = solve_maxcut(adj, p=1, num_starts=5, max_iter=50)
        assert result.problem.startswith("MaxCut")
        assert result.objective_value >= 1.0
        assert len(result.solution) == 3

    def test_path_graph(self):
        adj = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        result = solve_maxcut(adj, p=1, num_starts=5, max_iter=50)
        assert result.objective_value >= 1.0


class TestPortfolio:
    def test_simple_portfolio(self):
        returns = np.array([0.1, 0.2, 0.15])
        cov = np.array([
            [0.05, 0.01, 0.02],
            [0.01, 0.08, 0.03],
            [0.02, 0.03, 0.06],
        ])
        result = portfolio_optimization(
            returns, cov, risk_aversion=0.5, budget=2,
            p=1, num_starts=5, max_iter=50,
        )
        assert result.problem.startswith("Portfolio")
        assert len(result.solution) == 3
