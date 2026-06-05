"""European option pricing via quantum amplitude estimation.

Uses a discretized probability distribution loaded into a quantum register
and a payoff comparator to estimate the expected payoff. The quantum
approach achieves quadratic speedup over classical Monte Carlo.

The workflow:
1. Discretize the log-normal asset price distribution into 2^n bins
2. Load the distribution as amplitudes in a quantum register
3. Apply a payoff operator that marks states where option is in-the-money
4. Use amplitude estimation to compute E[payoff]

For benchmarking, classical Black-Scholes is also provided.

References:
    Stamatopoulos et al., "Option Pricing using Quantum Computers"
    (Quantum 4, 291, 2020)
    Woerner & Egger, "Quantum risk analysis"
    (npj Quantum Information 5, 15, 2019)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H


@dataclass
class OptionPricingResult:
    """Result from quantum European option pricing.

    Attributes:
        quantum_price: Option price estimated via quantum amplitude
            estimation.
        classical_price: Reference option price from the Black-Scholes
            analytical formula.
        relative_error: Relative difference between the quantum and
            Black-Scholes prices, ``|quantum - classical| / classical``.
        num_qubits: Number of qubits used for the price register
            (excluding the ancilla).
        num_price_bins: Number of discretization bins (``2**num_qubits``).
        option_type: ``"call"`` or ``"put"``.
        spot: Spot (current) price of the underlying asset.
        strike: Strike price of the option.
        risk_free_rate: Risk-free interest rate used for discounting.
        volatility: Annualized volatility of the underlying asset.
        maturity: Time to expiration in years.
    """
    quantum_price: float
    classical_price: float
    relative_error: float
    num_qubits: int
    num_price_bins: int
    option_type: str
    spot: float
    strike: float
    risk_free_rate: float
    volatility: float
    maturity: float


def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    option_type: str = "call",
) -> float:
    """Classical Black-Scholes European option price.

    Args:
        spot: Current asset price S_0.
        strike: Strike price K.
        rate: Risk-free interest rate r.
        volatility: Annualized volatility sigma.
        maturity: Time to maturity T (in years).
        option_type: "call" or "put".

    Returns:
        Option price.
    """
    if maturity <= 0:
        if option_type == "call":
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)

    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * math.sqrt(maturity)
    )
    d2 = d1 - volatility * math.sqrt(maturity)

    if option_type == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-rate * maturity) * _norm_cdf(d2)
    elif option_type == "put":
        return strike * math.exp(-rate * maturity) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def _norm_cdf(x: float) -> float:
    """Compute the standard normal cumulative distribution function.

    Uses the relationship ``Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))``.

    Args:
        x: Point at which to evaluate the CDF.

    Returns:
        float: The probability ``P(Z <= x)`` for standard normal Z.
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _discretize_lognormal(
    spot: float,
    rate: float,
    volatility: float,
    maturity: float,
    num_bins: int,
    num_std: float = 3.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Discretize the risk-neutral log-normal distribution into bins.

    Divides the log-price range into *num_bins* equal-width bins (in
    log-space) spanning ``+/- num_std`` standard deviations around the
    risk-neutral mean.

    Args:
        spot: Current asset price S_0.
        rate: Risk-free interest rate r.
        volatility: Annualized volatility sigma.
        maturity: Time to maturity T (in years).
        num_bins: Number of bins for discretization.
        num_std: Number of standard deviations to cover on each side
            of the mean.

    Returns:
        tuple[NDArray[np.floating], NDArray[np.floating]]: A 2-tuple of
            ``(prices, probabilities)`` where *prices* are the bin-center
            asset prices and *probabilities* sum to 1.
    """
    mu = math.log(spot) + (rate - 0.5 * volatility**2) * maturity
    sigma = volatility * math.sqrt(maturity)

    lo = mu - num_std * sigma
    hi = mu + num_std * sigma
    edges = np.linspace(lo, hi, num_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    prices = np.exp(centers)

    probs = np.zeros(num_bins)
    for i in range(num_bins):
        p_hi = _norm_cdf((edges[i + 1] - mu) / sigma)
        p_lo = _norm_cdf((edges[i] - mu) / sigma)
        probs[i] = p_hi - p_lo

    probs = probs / np.sum(probs)
    return prices, probs


def _payoff(
    prices: NDArray[np.floating],
    strike: float,
    option_type: str,
) -> NDArray[np.floating]:
    """Compute the option payoff for each discretized price bin.

    Args:
        prices: Array of asset prices (bin centers).
        strike: Strike price of the option.
        option_type: ``"call"`` for ``max(S - K, 0)`` or ``"put"``
            for ``max(K - S, 0)``.

    Returns:
        NDArray[np.floating]: Array of payoff values, same shape as
            *prices*.

    Raises:
        ValueError: If *option_type* is not ``"call"`` or ``"put"``.
    """
    if option_type == "call":
        return np.maximum(prices - strike, 0.0)
    elif option_type == "put":
        return np.maximum(strike - prices, 0.0)
    raise ValueError(f"Unknown option_type '{option_type}'")


def european_option_price(
    spot: float,
    strike: float,
    rate: float = 0.05,
    volatility: float = 0.2,
    maturity: float = 1.0,
    option_type: str = "call",
    num_qubits: int = 4,
    num_std: float = 3.0,
) -> OptionPricingResult:
    """Price a European option using quantum amplitude estimation.

    Discretizes the asset price distribution and uses a quantum circuit
    to estimate the expected discounted payoff.

    The quantum circuit:
    1. Prepares a state whose amplitudes encode sqrt(p_i) for each price bin
    2. Applies a rotation proportional to the payoff in each bin
    3. Measures the probability of the ancilla being |1>
    4. The option price = discount_factor * max_payoff * P(ancilla=1)

    Args:
        spot: Current asset price.
        strike: Strike price.
        rate: Risk-free rate.
        volatility: Annualized volatility.
        maturity: Time to maturity (years).
        option_type: "call" or "put".
        num_qubits: Number of qubits for price discretization (2^n bins).
        num_std: Number of standard deviations for price range.

    Returns:
        OptionPricingResult with quantum and classical prices.
    """
    num_bins = 2**num_qubits
    prices, probs = _discretize_lognormal(spot, rate, volatility, maturity, num_bins, num_std)
    payoffs = _payoff(prices, strike, option_type)

    # Classical expected payoff (Monte Carlo equivalent)
    discount = math.exp(-rate * maturity)
    classical_expected = float(np.sum(probs * payoffs))
    classical_price = discount * classical_expected

    # Black-Scholes benchmark
    bs_price = black_scholes_price(spot, strike, rate, volatility, maturity, option_type)

    # Quantum circuit for amplitude estimation
    # State preparation: encode sqrt(p_i) as amplitudes
    amplitudes = np.sqrt(probs)
    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        amplitudes = amplitudes / norm

    # Build state preparation unitary via QR decomposition
    dim = num_bins
    M = np.eye(dim, dtype=complex)
    M[:, 0] = amplitudes.astype(complex)
    Q, _ = np.linalg.qr(M)
    if np.real(np.dot(Q[:, 0].conj(), amplitudes)) < 0:
        Q = -Q

    # Payoff encoding: rotate ancilla by angle proportional to payoff
    max_payoff = float(np.max(payoffs)) if np.max(payoffs) > 0 else 1.0
    normalized_payoffs = payoffs / max_payoff

    # Build full circuit: num_qubits (price) + 1 (ancilla)
    total_qubits = num_qubits + 1
    ancilla = num_qubits
    qc = QuantumCircuit(total_qubits)

    # State preparation on price register
    prep_gate = Gate(Q.astype(complex), "StatePrep")
    qc.apply(prep_gate, *range(num_qubits))

    # Controlled rotations for payoff encoding
    for i in range(num_bins):
        if normalized_payoffs[i] > 1e-15:
            theta = 2 * math.asin(min(1.0, math.sqrt(normalized_payoffs[i])))
            ry = np.array([
                [math.cos(theta / 2), -math.sin(theta / 2)],
                [math.sin(theta / 2), math.cos(theta / 2)],
            ], dtype=complex)

            # Build controlled rotation on ancilla, conditioned on price state |i>
            ctrl = np.eye(2 * dim, dtype=complex)
            ctrl[2 * i, 2 * i] = ry[0, 0]
            ctrl[2 * i, 2 * i + 1] = ry[0, 1]
            ctrl[2 * i + 1, 2 * i] = ry[1, 0]
            ctrl[2 * i + 1, 2 * i + 1] = ry[1, 1]

            gate = Gate(ctrl, f"CRy_{i}")
            qc.apply(gate, *range(num_qubits), ancilla)

    sv = qc.statevector()
    total_dim = 2**total_qubits

    p_ancilla = 0.0
    for idx in range(total_dim):
        if (idx >> ancilla) & 1:
            p_ancilla += abs(sv[idx]) ** 2

    quantum_expected = max_payoff * p_ancilla * norm**2
    quantum_price = discount * quantum_expected

    rel_error = abs(quantum_price - bs_price) / bs_price if bs_price > 1e-10 else 0.0

    return OptionPricingResult(
        quantum_price=quantum_price,
        classical_price=bs_price,
        relative_error=rel_error,
        num_qubits=num_qubits,
        num_price_bins=num_bins,
        option_type=option_type,
        spot=spot,
        strike=strike,
        risk_free_rate=rate,
        volatility=volatility,
        maturity=maturity,
    )
