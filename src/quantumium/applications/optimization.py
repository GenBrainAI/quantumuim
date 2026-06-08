"""Quantum optimization applications.

Provides problem encodings and solvers for combinatorial optimization
problems using QAOA and variational methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..algorithms.qaoa import qaoa_maxcut, QAOAResult


@dataclass
class OptimizationResult:
    """Result from a quantum optimization run.

    Attributes:
        problem: Human-readable description of the problem instance
            (e.g., ``"MaxCut (5 nodes, 7 edges)"``).
        solution: Optimal bitstring found by the solver, where each
            character ``'0'``/``'1'`` indicates the assignment of the
            corresponding decision variable.
        objective_value: Value of the objective function at the returned
            solution.
        qaoa_result: Full :class:`QAOAResult` from the underlying QAOA
            run, including optimal parameters and convergence history.
    """

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
    """Solve the MaxCut problem on a graph using QAOA.

    Partitions the graph vertices into two sets to maximize the number
    of edges crossing the partition. The adjacency matrix is converted
    to an edge list internally.

    Args:
        adjacency: Symmetric adjacency matrix of shape ``(n, n)`` as a
            list of lists or NumPy array. Non-zero entries indicate edges.
        p: Number of QAOA layers (circuit depth).
        num_starts: Number of random parameter initializations.
        max_iter: Maximum optimizer iterations per start.

    Returns:
        OptimizationResult: Contains the partition bitstring, the cut
            value (number of edges cut), and the full QAOA result.
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
    """Solve small Travelling Salesman Problem instances using QAOA.

    Encodes the TSP as a QUBO using one-hot encoding with ``n**2`` qubits
    for *n* cities. Row and column constraints (each city visited exactly
    once, each time-step has exactly one city) are enforced via quadratic
    penalty terms. Only practical for ``n <= 4`` on statevector simulators
    due to the exponential qubit count.

    Args:
        distances: Distance matrix of shape ``(n, n)`` as a list of lists
            or NumPy array.
        p: Number of QAOA layers (circuit depth).
        penalty: Weight of the penalty terms for constraint violations.
            Higher values encourage feasible solutions at the cost of
            harder optimization.
        num_starts: Number of random parameter initializations.
        max_iter: Maximum optimizer iterations per start.

    Returns:
        OptimizationResult: Contains the best bitstring encoding the
            route (one-hot, row-major), the total route distance as the
            objective value, and the full QAOA result.
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
    """Solve mean-variance portfolio optimization using QAOA.

    Maximizes the objective ``mu^T x - q * x^T Sigma x`` subject to an
    optional budget constraint ``sum(x) = budget``, where each binary
    variable ``x_i`` indicates whether asset *i* is selected. The budget
    constraint is enforced with a quadratic penalty term.

    Args:
        expected_returns: Expected return for each asset, shape ``(n,)``.
        covariance: Covariance matrix of asset returns, shape ``(n, n)``.
        risk_aversion: Risk aversion parameter *q*. Higher values penalize
            portfolio variance more heavily.
        budget: Number of assets to select. If ``None``, no cardinality
            constraint is imposed.
        p: Number of QAOA layers (circuit depth).
        num_starts: Number of random parameter initializations.
        max_iter: Maximum optimizer iterations per start.

    Returns:
        OptimizationResult: Contains the selected-asset bitstring, the
            objective value, and the full QAOA result.
    """
    n_assets = len(expected_returns)
    num_qubits = n_assets
    dim = 2**num_qubits

    cost = np.zeros(dim, dtype=float)
    penalty = 10.0

    for basis_state in range(dim):
        x = np.array(
            [(basis_state >> (num_qubits - 1 - i)) & 1 for i in range(n_assets)], dtype=float
        )

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

    [i for i, b in enumerate(best_bitstring) if b == "1"]
    return OptimizationResult(
        problem=f"Portfolio ({n_assets} assets, budget={budget})",
        solution=best_bitstring,
        objective_value=best_value,
        qaoa_result=qaoa_result,
    )
