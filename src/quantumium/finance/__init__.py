"""Quantum finance applications.

Provides quantum-accelerated financial computations:

- **option_pricing**: European option pricing via quantum amplitude
  estimation, with a classical Black-Scholes benchmark for comparison.
- **risk_analysis**: Value at Risk (VaR) and Conditional VaR (CVaR)
  using quantum amplitude estimation on discretized loss distributions.
- **portfolio**: Mean-variance portfolio optimization formulated as a
  QUBO and solved with QAOA.
"""

from .option_pricing import (
    OptionPricingResult,
    european_option_price,
    black_scholes_price,
)
from .risk_analysis import (
    RiskAnalysisResult,
    quantum_var,
    classical_var,
)
from .portfolio import (
    PortfolioResult,
    quantum_portfolio_optimization,
)

__all__ = [
    "OptionPricingResult",
    "european_option_price",
    "black_scholes_price",
    "RiskAnalysisResult",
    "quantum_var",
    "classical_var",
    "PortfolioResult",
    "quantum_portfolio_optimization",
]
