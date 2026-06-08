"""Tests for quantumium.finance — option pricing, risk analysis, portfolio."""

import math
import numpy as np
import pytest

from quantumium.finance.option_pricing import (
    OptionPricingResult,
    black_scholes_price,
    european_option_price,
    _norm_cdf,
    _discretize_lognormal,
    _payoff,
)
from quantumium.finance.risk_analysis import (
    RiskAnalysisResult,
    classical_var,
    quantum_var,
)
from quantumium.finance.portfolio import (
    PortfolioResult,
    quantum_portfolio_optimization,
    _portfolio_qubo,
    _qubo_cost_diagonal,
)


# ============================================================
# Black-Scholes Tests
# ============================================================


class TestBlackScholes:
    def test_call_basic(self):
        price = black_scholes_price(100, 100, 0.05, 0.2, 1.0, "call")
        assert 8.0 < price < 12.0  # BS call ~10.45

    def test_put_basic(self):
        price = black_scholes_price(100, 100, 0.05, 0.2, 1.0, "put")
        assert 4.0 < price < 8.0  # BS put ~5.57

    def test_put_call_parity(self):
        S, K, r, vol, T = 100, 100, 0.05, 0.2, 1.0
        call = black_scholes_price(S, K, r, vol, T, "call")
        put = black_scholes_price(S, K, r, vol, T, "put")
        assert abs((call - put) - (S - K * math.exp(-r * T))) < 1e-10

    def test_deep_itm_call(self):
        price = black_scholes_price(200, 100, 0.05, 0.2, 1.0, "call")
        assert price > 95.0  # deep ITM, mostly intrinsic

    def test_deep_otm_call(self):
        price = black_scholes_price(50, 100, 0.05, 0.2, 1.0, "call")
        assert price < 1.0

    def test_zero_maturity(self):
        assert black_scholes_price(110, 100, 0.05, 0.2, 0, "call") == 10.0
        assert black_scholes_price(90, 100, 0.05, 0.2, 0, "put") == 10.0
        assert black_scholes_price(110, 100, 0.05, 0.2, 0, "put") == 0.0

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="option_type"):
            black_scholes_price(100, 100, 0.05, 0.2, 1.0, "straddle")

    def test_high_volatility(self):
        price = black_scholes_price(100, 100, 0.05, 0.8, 1.0, "call")
        assert price > 25.0  # high vol = expensive option


# ============================================================
# Distribution Helpers
# ============================================================


class TestHelpers:
    def test_norm_cdf(self):
        assert abs(_norm_cdf(0) - 0.5) < 1e-10
        assert _norm_cdf(10) > 0.999
        assert _norm_cdf(-10) < 0.001

    def test_discretize_lognormal_sums_to_one(self):
        _, probs = _discretize_lognormal(100, 0.05, 0.2, 1.0, 16)
        assert abs(np.sum(probs) - 1.0) < 1e-10

    def test_discretize_lognormal_positive(self):
        prices, probs = _discretize_lognormal(100, 0.05, 0.2, 1.0, 16)
        assert np.all(prices > 0)
        assert np.all(probs >= 0)

    def test_payoff_call(self):
        prices = np.array([90, 95, 100, 105, 110], dtype=float)
        payoff = _payoff(prices, 100, "call")
        np.testing.assert_array_almost_equal(payoff, [0, 0, 0, 5, 10])

    def test_payoff_put(self):
        prices = np.array([90, 95, 100, 105, 110], dtype=float)
        payoff = _payoff(prices, 100, "put")
        np.testing.assert_array_almost_equal(payoff, [10, 5, 0, 0, 0])


# ============================================================
# Quantum Option Pricing
# ============================================================


class TestEuropeanOptionPrice:
    def test_call_returns_result(self):
        result = european_option_price(100, 100, num_qubits=3)
        assert isinstance(result, OptionPricingResult)
        assert result.quantum_price > 0
        assert result.classical_price > 0

    def test_call_reasonable_error(self):
        result = european_option_price(100, 100, num_qubits=4)
        assert result.relative_error < 0.5  # within 50% of BS

    def test_put_reasonable(self):
        result = european_option_price(100, 100, option_type="put", num_qubits=4)
        assert result.quantum_price > 0
        assert result.option_type == "put"

    def test_more_qubits_better(self):
        r3 = european_option_price(100, 100, num_qubits=3)
        r5 = european_option_price(100, 100, num_qubits=5)
        # More qubits should generally give better discretization
        assert r5.num_price_bins > r3.num_price_bins

    def test_deep_itm_call(self):
        result = european_option_price(150, 100, num_qubits=4)
        assert result.quantum_price > 30.0

    def test_deep_otm_put(self):
        result = european_option_price(150, 100, option_type="put", num_qubits=4)
        assert result.quantum_price < 5.0

    def test_result_fields(self):
        result = european_option_price(100, 90, rate=0.03, volatility=0.3, maturity=0.5)
        assert result.spot == 100
        assert result.strike == 90
        assert result.risk_free_rate == 0.03
        assert result.volatility == 0.3
        assert result.maturity == 0.5


# ============================================================
# Classical VaR
# ============================================================


class TestClassicalVaR:
    def test_basic(self):
        losses = np.array([1, 2, 3, 4, 5], dtype=float)
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        var, cvar = classical_var(losses, probs, confidence=0.8)
        assert var >= 4.0  # 80th percentile
        assert cvar >= var  # CVaR >= VaR always

    def test_cvar_ge_var(self):
        rng = np.random.default_rng(42)
        losses = np.sort(rng.standard_normal(32))
        probs = np.ones(32) / 32
        var, cvar = classical_var(losses, probs, 0.95)
        assert cvar >= var - 1e-10

    def test_uniform_distribution(self):
        losses = np.arange(100, dtype=float)
        probs = np.ones(100) / 100
        var, cvar = classical_var(losses, probs, 0.95)
        assert var >= 94  # 95th percentile of 0..99

    def test_concentrated_distribution(self):
        losses = np.array([0, 10], dtype=float)
        probs = np.array([0.99, 0.01])
        var, cvar = classical_var(losses, probs, 0.5)
        assert var == 0.0


# ============================================================
# Quantum VaR
# ============================================================


class TestQuantumVaR:
    def test_returns_result(self):
        result = quantum_var(mu=0, sigma=1, num_qubits=3)
        assert isinstance(result, RiskAnalysisResult)

    def test_var_positive_mu(self):
        result = quantum_var(mu=2, sigma=1, confidence=0.95, num_qubits=4)
        assert result.var_classical > 0

    def test_cvar_ge_var(self):
        result = quantum_var(mu=0, sigma=1, confidence=0.95, num_qubits=4)
        assert result.cvar_classical >= result.var_classical - 1e-10
        assert result.cvar_quantum >= result.var_quantum - 1e-10

    def test_higher_sigma_higher_var(self):
        r1 = quantum_var(mu=0, sigma=0.5, num_qubits=3)
        r2 = quantum_var(mu=0, sigma=2.0, num_qubits=3)
        assert r2.var_classical > r1.var_classical

    def test_quantum_matches_classical(self):
        result = quantum_var(mu=0, sigma=1, num_qubits=4)
        assert abs(result.var_quantum - result.var_classical) < 2.0

    def test_distribution_type(self):
        result = quantum_var()
        assert result.distribution == "gaussian"


# ============================================================
# Portfolio QUBO
# ============================================================


class TestPortfolioQUBO:
    def test_qubo_shape(self):
        mu = np.array([0.1, 0.2, 0.15])
        sigma = np.eye(3) * 0.01
        Q = _portfolio_qubo(mu, sigma, risk_factor=0.5, budget=1, penalty=1.0)
        assert Q.shape == (3, 3)

    def test_qubo_symmetric(self):
        mu = np.array([0.1, 0.2, 0.15])
        sigma = np.array([[0.04, 0.01, 0.005], [0.01, 0.09, 0.01], [0.005, 0.01, 0.06]])
        Q = _portfolio_qubo(mu, sigma, 0.5, 2, 1.0)
        np.testing.assert_array_almost_equal(Q, Q.T)

    def test_cost_diagonal_length(self):
        Q = np.eye(3) * 0.1
        costs = _qubo_cost_diagonal(Q, 3)
        assert len(costs) == 8  # 2^3

    def test_cost_zero_for_zero_state(self):
        mu = np.array([0.1, 0.2])
        sigma = np.eye(2) * 0.01
        Q = _portfolio_qubo(mu, sigma, 0.5, 1, 0)
        costs = _qubo_cost_diagonal(Q, 2)
        assert costs[0] == 0.0  # |00> = no assets selected


# ============================================================
# Quantum Portfolio Optimization
# ============================================================


class TestPortfolioOptimization:
    def test_returns_result(self):
        mu = [0.1, 0.2, 0.15, 0.05]
        sigma = np.diag([0.04, 0.09, 0.06, 0.02]).tolist()
        result = quantum_portfolio_optimization(mu, sigma, budget=2)
        assert isinstance(result, PortfolioResult)

    def test_selects_correct_budget(self):
        mu = [0.1, 0.2, 0.15, 0.05]
        sigma = np.diag([0.04, 0.09, 0.06, 0.02]).tolist()
        result = quantum_portfolio_optimization(mu, sigma, budget=2, num_restarts=100)
        assert len(result.selection) == 2

    def test_three_assets(self):
        mu = [0.15, 0.10, 0.05]
        sigma = [[0.04, 0.005, 0.001], [0.005, 0.02, 0.002], [0.001, 0.002, 0.01]]
        result = quantum_portfolio_optimization(mu, sigma, budget=1, num_restarts=100)
        assert len(result.selection) == 1
        assert result.num_assets == 3

    def test_budget_constraint_violated_raises(self):
        with pytest.raises(ValueError, match="Budget"):
            quantum_portfolio_optimization([0.1, 0.2], [[0.04, 0], [0, 0.09]], budget=0)

    def test_covariance_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Covariance"):
            quantum_portfolio_optimization([0.1, 0.2], [[0.04]], budget=1)

    def test_result_fields(self):
        mu = [0.1, 0.2]
        sigma = [[0.04, 0.01], [0.01, 0.09]]
        result = quantum_portfolio_optimization(mu, sigma, budget=1, risk_factor=0.3)
        assert result.risk_factor == 0.3
        assert result.budget == 1
        assert len(result.optimal_bitstring) == 2

    def test_high_return_preferred(self):
        mu = [0.01, 0.50]
        sigma = [[0.01, 0], [0, 0.01]]
        result = quantum_portfolio_optimization(
            mu,
            sigma,
            budget=1,
            risk_factor=0.1,
            penalty=10.0,
            qaoa_layers=3,
            num_restarts=200,
        )
        # With low risk and very different returns, should prefer asset 1
        assert 1 in result.selection
