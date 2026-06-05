"""Quantum Finance Demo — option pricing, risk analysis, portfolio optimization.

Demonstrates:
1. European option pricing via quantum amplitude estimation vs Black-Scholes
2. Value at Risk (VaR) and Conditional VaR estimation
3. Portfolio optimization using QAOA
"""

import numpy as np

from quantumium.finance import (
    european_option_price,
    black_scholes_price,
    quantum_var,
    classical_var,
    quantum_portfolio_optimization,
)


def demo_option_pricing():
    print("=" * 60)
    print("  European Option Pricing")
    print("=" * 60)

    S, K, r, vol, T = 100.0, 100.0, 0.05, 0.2, 1.0

    print(f"\nParameters: S={S}, K={K}, r={r}, σ={vol}, T={T}")
    print()

    for n_qubits in [3, 4, 5]:
        result = european_option_price(S, K, r, vol, T, "call", num_qubits=n_qubits)
        print(f"  {n_qubits} qubits ({result.num_price_bins:3d} bins): "
              f"Quantum={result.quantum_price:8.4f}  "
              f"BS={result.classical_price:8.4f}  "
              f"Error={result.relative_error:6.2%}")

    print("\n  Put option (4 qubits):")
    put = european_option_price(S, K, r, vol, T, "put", num_qubits=4)
    bs_put = black_scholes_price(S, K, r, vol, T, "put")
    print(f"    Quantum={put.quantum_price:.4f}  BS={bs_put:.4f}")

    parity = (put.quantum_price - european_option_price(S, K, r, vol, T, "call", num_qubits=4).quantum_price)
    theory = S - K * np.exp(-r * T)
    print(f"    Put-Call parity check: C-P={-parity:.4f}  S-K*exp(-rT)={theory:.4f}")


def demo_risk_analysis():
    print("\n" + "=" * 60)
    print("  Value at Risk (VaR) Analysis")
    print("=" * 60)

    print("\n  Gaussian loss distribution (μ=0, σ=1)")
    for conf in [0.90, 0.95, 0.99]:
        result = quantum_var(mu=0, sigma=1, confidence=conf, num_qubits=4)
        print(f"\n    {conf:.0%} confidence:")
        print(f"      VaR  — Quantum: {result.var_quantum:7.4f}  Classical: {result.var_classical:7.4f}")
        print(f"      CVaR — Quantum: {result.cvar_quantum:7.4f}  Classical: {result.cvar_classical:7.4f}")

    print("\n  Higher volatility (σ=2):")
    r2 = quantum_var(mu=0, sigma=2, confidence=0.95, num_qubits=4)
    print(f"    95% VaR = {r2.var_classical:.4f}  CVaR = {r2.cvar_classical:.4f}")

    print("\n  Custom discrete distribution:")
    losses = np.array([0, 1, 2, 5, 10, 20], dtype=float)
    probs = np.array([0.5, 0.2, 0.15, 0.08, 0.05, 0.02])
    var, cvar = classical_var(losses, probs, 0.95)
    print(f"    95% VaR = {var:.2f}  CVaR = {cvar:.2f}")


def demo_portfolio():
    print("\n" + "=" * 60)
    print("  Portfolio Optimization (QAOA)")
    print("=" * 60)

    assets = ["TECH", "BOND", "GOLD", "REIT"]
    mu = [0.12, 0.04, 0.06, 0.08]
    sigma = [
        [0.04, 0.005, -0.002, 0.01],
        [0.005, 0.01, 0.001, 0.003],
        [-0.002, 0.001, 0.02, 0.002],
        [0.01, 0.003, 0.002, 0.03],
    ]

    print(f"\n  Assets: {assets}")
    print(f"  Expected returns: {mu}")
    print(f"  Select 2 of 4 assets\n")

    for q in [0.1, 0.5, 1.0]:
        result = quantum_portfolio_optimization(
            mu, sigma, budget=2, risk_factor=q, qaoa_layers=2, num_restarts=100
        )
        selected = [assets[i] for i in result.selection]
        print(f"  Risk factor q={q}:")
        print(f"    Selected: {selected}")
        print(f"    Return: {result.expected_return:.4f}  Risk: {result.risk:.6f}")
        print(f"    Bitstring: {result.optimal_bitstring}")
        print()


if __name__ == "__main__":
    demo_option_pricing()
    demo_risk_analysis()
    demo_portfolio()
    print("Done!")
