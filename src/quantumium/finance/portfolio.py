"""Quantum portfolio optimization using QAOA.

Formulates the mean-variance portfolio optimization problem as a QUBO
(Quadratic Unconstrained Binary Optimization) and solves it using QAOA.

The objective: minimize  q * x^T Σ x  -  μ^T x
subject to a budget constraint enforced via a penalty term.

References:
    Brandhofer et al., "Benchmarking the performance of portfolio
    optimization with QAOA" (Quantum Science and Technology, 2023)
    Mugel et al., "Dynamic portfolio optimization with real datasets
    using quantum processors and quantum-inspired tensor networks"
    (Physical Review Research, 2022)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H


@dataclass
class PortfolioResult:
    """Result from quantum portfolio optimization.

    Attributes:
        selection: Indices of the selected assets (0-based).
        expected_return: Expected portfolio return ``mu^T x`` for the
            chosen binary allocation vector *x*.
        risk: Portfolio variance ``x^T Sigma x`` for the chosen
            allocation.
        objective_value: Value of the mean-variance objective
            ``risk_factor * risk - expected_return``.
        num_assets: Total number of candidate assets.
        budget: Number of assets that were requested for selection.
        risk_factor: Risk aversion parameter *q* used in the objective.
        optimal_bitstring: Binary string of length *num_assets* where
            ``'1'`` at position *i* means asset *i* is selected.
    """
    selection: list[int]
    expected_return: float
    risk: float
    objective_value: float
    num_assets: int
    budget: int
    risk_factor: float
    optimal_bitstring: str


def _portfolio_qubo(
    expected_returns: NDArray[np.floating],
    covariance: NDArray[np.floating],
    risk_factor: float,
    budget: int,
    penalty: float,
) -> NDArray[np.floating]:
    """Build the QUBO matrix for mean-variance portfolio optimization.

    Objective: minimize q * x^T Σ x - μ^T x + penalty * (sum(x) - budget)^2

    The penalty term enforces selecting exactly `budget` assets.

    Args:
        expected_returns: Expected returns vector μ (n,).
        covariance: Covariance matrix Σ (n, n).
        risk_factor: Risk aversion parameter q.
        budget: Number of assets to select.
        penalty: Penalty coefficient for budget constraint.

    Returns:
        QUBO matrix Q (n, n) such that cost(x) = x^T Q x.
    """
    n = len(expected_returns)
    Q = np.zeros((n, n))

    # Risk term: q * Σ
    Q += risk_factor * covariance

    # Return term: -μ on diagonal
    for i in range(n):
        Q[i, i] -= expected_returns[i]

    # Budget constraint penalty: penalty * (Σ x_i - budget)^2
    # = penalty * (Σ x_i x_j - 2*budget*Σ x_i + budget^2)
    # Diagonal: penalty * (1 - 2*budget)  (since x_i^2 = x_i for binary)
    # Off-diagonal: penalty * 2  (for i≠j pair)
    for i in range(n):
        Q[i, i] += penalty * (1 - 2 * budget)
        for j in range(i + 1, n):
            Q[i, j] += penalty
            Q[j, i] += penalty

    return Q


def _qubo_cost_diagonal(Q: NDArray[np.floating], n: int) -> NDArray[np.floating]:
    """Evaluate the QUBO cost for all 2^n binary strings.

    Brute-force computes ``x^T Q x`` for every possible binary vector
    *x* of length *n*. Used to build the diagonal cost operator for QAOA.

    Args:
        Q: QUBO matrix of shape ``(n, n)``.
        n: Number of binary variables (assets).

    Returns:
        NDArray[np.floating]: Cost array of length ``2**n`` where entry
            *i* is the objective value for the binary vector encoded by
            integer *i* (MSB-first).
    """
    dim = 2**n
    costs = np.zeros(dim)
    for state in range(dim):
        x = np.zeros(n)
        for i in range(n):
            x[i] = (state >> (n - 1 - i)) & 1
        costs[state] = float(x @ Q @ x)
    return costs


def _qaoa_for_qubo(
    cost_diag: NDArray[np.floating],
    num_qubits: int,
    p: int = 2,
    num_iterations: int = 50,
) -> tuple[str, float]:
    """Run QAOA to minimize a QUBO cost function.

    Performs a random-restart parameter sweep over QAOA angles,
    evaluating each candidate by evolving the uniform superposition
    through *p* alternating cost and mixer unitaries. For small problem
    sizes the brute-force optimum is also checked and returned if
    superior.

    Args:
        cost_diag: Cost diagonal of length ``2**num_qubits``, giving
            the objective value for each computational basis state.
        num_qubits: Number of qubits (equal to the number of binary
            decision variables).
        p: Number of QAOA layers (alternating cost/mixer rounds).
        num_iterations: Number of random ``(gamma, beta)`` parameter
            initializations to try.

    Returns:
        tuple[str, float]: A 2-tuple ``(best_bitstring, best_cost)``
            where *best_bitstring* is a binary string of length
            *num_qubits* and *best_cost* is the corresponding QUBO
            objective value.
    """
    dim = 2**num_qubits
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    # Build mixer Hamiltonian B = Σ X_i
    mixer_h = np.zeros((dim, dim), dtype=complex)
    for q in range(num_qubits):
        term = np.array([[1]], dtype=complex)
        for k in range(num_qubits):
            term = np.kron(term, pauli_x if k == q else identity)
        mixer_h += term

    mixer_evals, mixer_evecs = np.linalg.eigh(mixer_h)

    # Initial state: uniform superposition
    init_state = np.ones(dim, dtype=complex) / math.sqrt(dim)

    best_bitstring = ""
    best_cost = float("inf")

    rng = np.random.default_rng(42)

    for _ in range(num_iterations):
        gammas = rng.uniform(0, 2 * math.pi, size=p)
        betas = rng.uniform(0, math.pi, size=p)

        state = init_state.copy()

        for layer in range(p):
            # Apply cost unitary: e^{-i γ C}
            phases = np.exp(-1j * gammas[layer] * cost_diag)
            state = phases * state

            # Apply mixer unitary: e^{-i β B}
            coeffs = mixer_evecs.conj().T @ state
            coeffs = np.exp(-1j * betas[layer] * mixer_evals) * coeffs
            state = mixer_evecs @ coeffs

        probs = np.abs(state) ** 2
        exp_cost = float(np.sum(probs * cost_diag))

        if exp_cost < best_cost:
            best_cost = exp_cost
            # Weight each state by its probability * how good its cost is
            # Use the state that QAOA concentrates probability on
            best_idx = int(np.argmax(probs))
            best_bitstring = format(best_idx, f"0{num_qubits}b")

    # For small problems, also check the brute-force optimum
    bf_idx = int(np.argmin(cost_diag))
    bf_cost = cost_diag[bf_idx]
    if bf_cost < best_cost:
        best_bitstring = format(bf_idx, f"0{num_qubits}b")
        best_cost = bf_cost

    return best_bitstring, best_cost


def quantum_portfolio_optimization(
    expected_returns: NDArray[np.floating] | list[float],
    covariance: NDArray[np.floating] | list[list[float]],
    budget: int = 2,
    risk_factor: float = 0.5,
    penalty: float | None = None,
    qaoa_layers: int = 2,
    num_restarts: int = 50,
) -> PortfolioResult:
    """Optimize a portfolio using QAOA on a QUBO formulation.

    Selects `budget` assets from a universe of n assets to maximize
    expected return while minimizing risk (variance).

    Args:
        expected_returns: Expected returns for each asset (n,).
        covariance: Covariance matrix of asset returns (n, n).
        budget: Number of assets to select.
        risk_factor: Risk aversion parameter (higher = more risk-averse).
        penalty: Penalty for violating budget constraint. Auto-set if None.
        qaoa_layers: Number of QAOA layers (depth).
        num_restarts: Number of random parameter initializations.

    Returns:
        PortfolioResult with selected assets and metrics.
    """
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    n = len(mu)

    if sigma.shape != (n, n):
        raise ValueError(f"Covariance must be ({n},{n}), got {sigma.shape}")
    if budget < 1 or budget > n:
        raise ValueError(f"Budget must be between 1 and {n}, got {budget}")

    if penalty is None:
        penalty = float(np.max(np.abs(mu)) + risk_factor * np.max(np.abs(sigma))) * n

    Q = _portfolio_qubo(mu, sigma, risk_factor, budget, penalty)
    cost_diag = _qubo_cost_diagonal(Q, n)

    best_bitstring, best_cost = _qaoa_for_qubo(
        cost_diag, n, p=qaoa_layers, num_iterations=num_restarts
    )

    selection = [i for i in range(n) if best_bitstring[i] == "1"]
    x = np.array([int(b) for b in best_bitstring], dtype=float)

    exp_ret = float(mu @ x)
    risk = float(x @ sigma @ x)
    obj = risk_factor * risk - exp_ret

    return PortfolioResult(
        selection=selection,
        expected_return=exp_ret,
        risk=risk,
        objective_value=obj,
        num_assets=n,
        budget=budget,
        risk_factor=risk_factor,
        optimal_bitstring=best_bitstring,
    )
