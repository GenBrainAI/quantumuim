"""Quantum optimization applications.

Provides problem encodings and solvers for combinatorial optimization
problems using QAOA and variational methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..algorithms.qaoa import qaoa_maxcut, maxcut_cost_operator, QAOAResult


@dataclass
class OptimizationResult:
    """Result from a quantum optimization run."""
    problem: str
    solution: str
    objective_value: float
    qaoa_result: QAOAResult


def solve_maxcut(
    adjacency: list[list[int]] | NDArray,
    p: int = 2,
    num_starts: int = 10,
    max_iter: int = 100,
) -> OptimizationResult:
    """Solve MaxCut using QAOA.

    Args:
        adjacency: Adjacency matrix or list of lists.
        p: QAOA depth.
        num_starts: Random restarts.
        max_iter: Optimization iterations.
    """
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] != 0:
                edges.append((i, j))

    result = qaoa_maxcut(n, edges, p=p, num_starts=num_starts, max_iter=max_iter)

    partition = result.optimal_bitstring
    cut_value = sum(1 for i, j in edges if partition[i] != partition[j])

    return OptimizationResult(
        problem=f"MaxCut ({n} nodes, {len(edges)} edges)",
        solution=partition,
        objective_value=float(cut_value),
        qaoa_result=result,
    )


def solve_tsp_qaoa(
    distances: list[list[float]] | NDArray,
    p: int = 1,
    penalty: float = 10.0,
    num_starts: int = 10,
    max_iter: int = 100,
) -> OptimizationResult:
    """Solve small TSP instances using QAOA with penalty encoding.

    Uses one-hot encoding: n² qubits for n cities.
    Only practical for n ≤ 4 on simulators due to qubit count.

    Args:
        distances: n×n distance matrix.
        p: QAOA depth.
        penalty: Penalty weight for constraint violations.
        num_starts: Random restarts.
        max_iter: Optimization iterations.
    """
    dist = np.asarray(distances, dtype=float)
    n = dist.shape[0]
    num_qubits = n * n

    dim = 2**num_qubits
    cost = np.zeros(dim, dtype=float)

    for basis_state in range(dim):
        bits = [(basis_state >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]
        assignment = np.array(bits).reshape(n, n)

        # Penalty: each row should have exactly one 1
        row_penalty = sum((assignment[i].sum() - 1) ** 2 for i in range(n))
        # Penalty: each column should have exactly one 1
        col_penalty = sum((assignment[:, j].sum() - 1) ** 2 for j in range(n))

        # Route cost
        route_cost = 0.0
        for step in range(n):
            next_step = (step + 1) % n
            for city_a in range(n):
                for city_b in range(n):
                    if assignment[step, city_a] and assignment[next_step, city_b]:
                        route_cost += dist[city_a, city_b]

        cost[basis_state] = -(route_cost + penalty * (row_penalty + col_penalty))

    from ..algorithms.qaoa import qaoa_circuit, qaoa_expectation
    from ..core.gate import H
    from ..core.circuit import QuantumCircuit

    best_value = -np.inf
    best_bitstring = None
    best_params = None

    for _ in range(num_starts):
        gammas = np.random.uniform(0, 2 * np.pi, size=p)
        betas = np.random.uniform(0, np.pi, size=p)
        params = np.concatenate([gammas, betas])

        for _ in range(max_iter):
            g, b = params[:p], params[p:]
            val = qaoa_expectation(num_qubits, cost, g, b)

            grad = np.zeros_like(params)
            eps = 0.01
            for i in range(len(params)):
                params_p = params.copy()
                params_p[i] += eps
                val_p = qaoa_expectation(num_qubits, cost, params_p[:p], params_p[p:])
                grad[i] = (val_p - val) / eps
            params += 0.1 * grad

        g_f, b_f = params[:p], params[p:]
        final_val = qaoa_expectation(num_qubits, cost, g_f, b_f)
        if final_val > best_value:
            best_value = final_val
            best_params = params.copy()

            qc = qaoa_circuit(num_qubits, cost, g_f, b_f)
            counts = qc.measure_all(shots=512)
            best_bitstring = max(counts, key=lambda k: counts[k])

    qaoa_result = QAOAResult(
        optimal_params=best_params,
        optimal_value=best_value,
        optimal_bitstring=best_bitstring,
        history=[],
    )

    return OptimizationResult(
        problem=f"TSP ({n} cities)",
        solution=best_bitstring,
        objective_value=-best_value,
        qaoa_result=qaoa_result,
    )


def portfolio_optimization(
    expected_returns: NDArray[np.floating],
    covariance: NDArray[np.floating],
    risk_aversion: float = 0.5,
    budget: int | None = None,
    p: int = 1,
    num_starts: int = 10,
    max_iter: int = 100,
) -> OptimizationResult:
    """Portfolio optimization using QAOA.

    Maximize: μᵀx - q·xᵀΣx   subject to  Σxᵢ = budget

    Each qubit represents whether to include an asset.

    Args:
        expected_returns: Expected return per asset.
        covariance: Covariance matrix.
        risk_aversion: Risk aversion parameter q.
        budget: Number of assets to select (None = no constraint).
        p: QAOA depth.
        num_starts: Random restarts.
        max_iter: Iterations.
    """
    n_assets = len(expected_returns)
    num_qubits = n_assets
    dim = 2**num_qubits

    cost = np.zeros(dim, dtype=float)
    penalty = 10.0

    for basis_state in range(dim):
        x = np.array([(basis_state >> (num_qubits - 1 - i)) & 1 for i in range(n_assets)], dtype=float)

        objective = np.dot(expected_returns, x) - risk_aversion * x @ covariance @ x

        if budget is not None:
            constraint = penalty * (np.sum(x) - budget) ** 2
            objective -= constraint

        cost[basis_state] = objective

    from ..algorithms.qaoa import qaoa_circuit, qaoa_expectation

    best_value = -np.inf
    best_params = None
    best_bitstring = None

    for _ in range(num_starts):
        gammas = np.random.uniform(0, 2 * np.pi, size=p)
        betas = np.random.uniform(0, np.pi, size=p)
        params = np.concatenate([gammas, betas])

        for _ in range(max_iter):
            g, b = params[:p], params[p:]
            val = qaoa_expectation(num_qubits, cost, g, b)

            grad = np.zeros_like(params)
            eps = 0.01
            for i in range(len(params)):
                params_p = params.copy()
                params_p[i] += eps
                val_p = qaoa_expectation(num_qubits, cost, params_p[:p], params_p[p:])
                grad[i] = (val_p - val) / eps
            params += 0.1 * grad

        g_f, b_f = params[:p], params[p:]
        final_val = qaoa_expectation(num_qubits, cost, g_f, b_f)
        if final_val > best_value:
            best_value = final_val
            best_params = params.copy()
            qc = qaoa_circuit(num_qubits, cost, g_f, b_f)
            counts = qc.measure_all(shots=512)
            best_bitstring = max(counts, key=lambda k: counts[k])

    qaoa_result = QAOAResult(
        optimal_params=best_params,
        optimal_value=best_value,
        optimal_bitstring=best_bitstring,
        history=[],
    )

    selected = [i for i, b in enumerate(best_bitstring) if b == '1']
    return OptimizationResult(
        problem=f"Portfolio ({n_assets} assets, budget={budget})",
        solution=best_bitstring,
        objective_value=best_value,
        qaoa_result=qaoa_result,
    )
