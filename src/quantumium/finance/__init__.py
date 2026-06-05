"""Quantum finance applications.

Option pricing, risk analysis, and portfolio optimization using
quantum amplitude estimation, QAOA, and variational methods.
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
