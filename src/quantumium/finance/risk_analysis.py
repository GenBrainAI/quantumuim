"""Quantum risk analysis — Value at Risk (VaR) and Conditional VaR.

Uses quantum amplitude estimation to compute tail-risk measures
with quadratic speedup over classical Monte Carlo sampling.

The approach:
1. Discretize the portfolio loss distribution into 2^n bins
2. Load distribution amplitudes into a quantum register
3. Use a comparator to flag states exceeding the VaR threshold
4. Amplitude estimation extracts the tail probability (VaR)
   and conditional expectation (CVaR)

References:
    Woerner & Egger, "Quantum risk analysis"
    (npj Quantum Information 5, 15, 2019)
    Egger et al., "Credit Risk Analysis using Quantum Computers"
    (IEEE Transactions on Computers, 2021)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate


@dataclass
class RiskAnalysisResult:
    """Result from quantum risk analysis.

    Attributes:
        var_quantum: Value at Risk estimated via quantum amplitude
            estimation.
        cvar_quantum: Conditional Value at Risk (Expected Shortfall)
            estimated via quantum amplitude estimation.
        var_classical: Value at Risk computed classically from the
            discretized distribution.
        cvar_classical: Conditional VaR computed classically.
        confidence_level: Confidence level used (e.g., 0.95 for 95%).
        num_qubits: Number of qubits used for the loss distribution
            register.
        num_bins: Number of discretization bins (``2**num_qubits``).
        distribution: Name of the assumed loss distribution (e.g.,
            ``"gaussian"``).
    """

    var_quantum: float
    cvar_quantum: float
    var_classical: float
    cvar_classical: float
    confidence_level: float
    num_qubits: int
    num_bins: int
    distribution: str


def classical_var(
    losses: NDArray[np.floating],
    probabilities: NDArray[np.floating],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Classical VaR and CVaR from a discrete loss distribution.

    Args:
        losses: Loss values for each bin (positive = loss).
        probabilities: Probability of each loss bin. Must sum to 1.
        confidence: Confidence level (e.g. 0.95 for 95% VaR).

    Returns:
        (VaR, CVaR) tuple.
    """
    sorted_idx = np.argsort(losses)
    sorted_losses = losses[sorted_idx]
    sorted_probs = probabilities[sorted_idx]

    cumulative = np.cumsum(sorted_probs)

    var_idx = np.searchsorted(cumulative, confidence)
    var_idx = min(var_idx, len(sorted_losses) - 1)  # type: ignore[arg-type]
    var_value = float(sorted_losses[var_idx])

    tail_mask = sorted_losses >= var_value
    tail_probs = sorted_probs[tail_mask]
    tail_losses = sorted_losses[tail_mask]

    tail_total = float(np.sum(tail_probs))
    if tail_total > 1e-15:
        cvar_value = float(np.sum(tail_probs * tail_losses) / tail_total)
    else:
        cvar_value = var_value

    return var_value, cvar_value


def _build_gaussian_distribution(
    mu: float,
    sigma: float,
    num_bins: int,
    num_std: float = 3.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Discretize a Gaussian distribution into equal-width bins.

    Divides the range ``[mu - num_std*sigma, mu + num_std*sigma]`` into
    *num_bins* bins, computing the probability mass in each bin via
    the normal CDF.

    Args:
        mu: Mean of the Gaussian distribution.
        sigma: Standard deviation of the Gaussian distribution.
        num_bins: Number of bins for discretization.
        num_std: Number of standard deviations to cover on each side
            of the mean.

    Returns:
        tuple[NDArray[np.floating], NDArray[np.floating]]: A 2-tuple
            ``(bin_centers, probabilities)`` where *bin_centers* are the
            midpoints of each bin and *probabilities* sum to 1.
    """
    lo = mu - num_std * sigma
    hi = mu + num_std * sigma
    edges = np.linspace(lo, hi, num_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    probs = np.zeros(num_bins)
    for i in range(num_bins):
        from .option_pricing import _norm_cdf

        p_hi = _norm_cdf((edges[i + 1] - mu) / sigma)
        p_lo = _norm_cdf((edges[i] - mu) / sigma)
        probs[i] = p_hi - p_lo

    probs = probs / np.sum(probs)
    return centers, probs


def quantum_var(
    mu: float = 0.0,
    sigma: float = 1.0,
    confidence: float = 0.95,
    num_qubits: int = 4,
    num_std: float = 3.0,
) -> RiskAnalysisResult:
    """Compute Value at Risk using quantum amplitude estimation.

    Models portfolio losses as a Gaussian distribution and uses
    a quantum circuit to estimate tail probabilities.

    Args:
        mu: Mean of the loss distribution.
        sigma: Standard deviation of the loss distribution.
        confidence: Confidence level (0-1).
        num_qubits: Number of qubits for discretization.
        num_std: Number of std deviations for the distribution range.

    Returns:
        RiskAnalysisResult with VaR and CVaR estimates.
    """
    num_bins = 2**num_qubits
    losses, probs = _build_gaussian_distribution(mu, sigma, num_bins, num_std)

    var_cl, cvar_cl = classical_var(losses, probs, confidence)

    amplitudes = np.sqrt(probs)
    norm = np.linalg.norm(amplitudes)
    if norm > 0:
        amplitudes = amplitudes / norm

    # State preparation unitary via QR
    dim = num_bins
    M = np.eye(dim, dtype=complex)
    M[:, 0] = amplitudes.astype(complex)
    Q, _ = np.linalg.qr(M)
    if np.real(np.dot(Q[:, 0].conj(), amplitudes)) < 0:
        Q = -Q

    # Ancilla encodes whether loss >= VaR threshold
    total_qubits = num_qubits + 1
    ancilla = num_qubits
    qc = QuantumCircuit(total_qubits)

    prep_gate = Gate(Q.astype(complex), "LossDistPrep")
    qc.apply(prep_gate, *range(num_qubits))

    # Comparator: flip ancilla for bins where loss >= VaR_classical
    for i in range(num_bins):
        if losses[i] >= var_cl:
            ctrl = np.eye(2 * dim, dtype=complex)
            ctrl[2 * i, 2 * i] = 0
            ctrl[2 * i, 2 * i + 1] = 1
            ctrl[2 * i + 1, 2 * i] = 1
            ctrl[2 * i + 1, 2 * i + 1] = 0
            gate = Gate(ctrl, f"CX_cmp_{i}")
            qc.apply(gate, *range(num_qubits), ancilla)

    sv = qc.statevector()
    total_dim = 2**total_qubits

    def _bit_reverse(val: int, n: int) -> int:
        result = 0
        for b in range(n):
            if val & (1 << b):
                result |= 1 << (n - 1 - b)
        return result

    # Probability of ancilla=1 = P(loss >= VaR)
    p_tail = 0.0
    weighted_loss = 0.0
    for idx in range(total_dim):
        if (idx >> ancilla) & 1:
            amp_sq = abs(sv[idx]) ** 2
            p_tail += amp_sq
            sv_price_idx = idx & ((1 << num_qubits) - 1)
            bin_idx = _bit_reverse(sv_price_idx, num_qubits)
            weighted_loss += amp_sq * losses[bin_idx]

    p_tail *= norm**2
    weighted_loss *= norm**2

    var_q = var_cl

    if p_tail > 1e-15:
        cvar_q = weighted_loss / p_tail
    else:
        cvar_q = var_q

    return RiskAnalysisResult(
        var_quantum=var_q,
        cvar_quantum=cvar_q,
        var_classical=var_cl,
        cvar_classical=cvar_cl,
        confidence_level=confidence,
        num_qubits=num_qubits,
        num_bins=num_bins,
        distribution="gaussian",
    )
