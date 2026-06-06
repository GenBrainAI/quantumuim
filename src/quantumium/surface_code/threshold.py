"""Threshold estimation for the surface code via Monte Carlo simulation.

Estimates the logical error rate as a function of physical error rate for
various code distances. The threshold is the physical error rate where
the logical error rate curves for different distances cross — below
threshold, larger codes perform better.
"""

from __future__ import annotations

import numpy as np

from .lattice import SurfaceCodeLattice
from .syndrome import extract_syndrome, random_error
from .decoder import MWPMDecoder
from .logical import has_logical_error


def logical_error_rate(
    d: int,
    p: float,
    num_trials: int = 1000,
    seed: int | None = None,
) -> float:
    """Estimate the logical error rate for a given code distance and physical error rate.

    Runs Monte Carlo simulation: for each trial, generates random X and Z
    errors independently with probability p per qubit, decodes, and checks
    for logical errors.

    Args:
        d: Code distance.
        p: Per-qubit physical error probability (depolarizing-like, applied
            independently as X and Z errors).
        num_trials: Number of Monte Carlo trials.
        seed: Random seed for reproducibility.

    Returns:
        Estimated logical error rate (fraction of trials with a logical error).
    """
    rng = np.random.default_rng(seed)
    lattice = SurfaceCodeLattice(d)
    decoder = MWPMDecoder(lattice)
    logical_errors = 0

    for _ in range(num_trials):
        x_error = random_error(lattice, "X", p, rng)
        z_error = random_error(lattice, "Z", p, rng)

        x_syndrome = extract_syndrome(lattice, x_error, "X")
        z_syndrome = extract_syndrome(lattice, z_error, "Z")

        x_correction = decoder.decode_x_errors(x_syndrome)
        z_correction = decoder.decode_z_errors(z_syndrome)

        x_logical = has_logical_error(x_error, x_correction, lattice, "X")
        z_logical = has_logical_error(z_error, z_correction, lattice, "Z")

        if x_logical or z_logical:
            logical_errors += 1

    return logical_errors / num_trials


def estimate_threshold(
    distances: list[int] | None = None,
    p_values: list[float] | None = None,
    num_trials: int = 500,
    seed: int | None = None,
) -> dict[str, object]:
    """Estimate the surface code threshold via Monte Carlo.

    Sweeps physical error rates for multiple code distances and returns
    the logical error rates. The threshold is approximately where the
    curves cross.

    Args:
        distances: List of code distances to test (default: [3, 5, 7]).
        p_values: List of physical error rates to test (default: 11 points
            from 0.01 to 0.20).
        num_trials: Number of Monte Carlo trials per (d, p) point.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys:
            "distances": list of code distances
            "p_values": list of physical error rates
            "logical_rates": dict mapping d -> list of logical error rates
            "threshold_estimate": rough estimate of the threshold
    """
    if distances is None:
        distances = [3, 5, 7]
    if p_values is None:
        p_values = [round(0.01 + i * 0.02, 3) for i in range(10)]

    rng_base = np.random.default_rng(seed)
    seeds = [int(rng_base.integers(0, 2**31)) for _ in range(len(distances) * len(p_values))]

    logical_rates: dict[int, list[float]] = {}
    seed_idx = 0
    for d in distances:
        rates = []
        for p in p_values:
            rate = logical_error_rate(d, p, num_trials, seeds[seed_idx])
            rates.append(rate)
            seed_idx += 1
        logical_rates[d] = rates

    threshold_est = _estimate_crossing(distances, p_values, logical_rates)

    return {
        "distances": distances,
        "p_values": p_values,
        "logical_rates": logical_rates,
        "threshold_estimate": threshold_est,
    }


def _estimate_crossing(
    distances: list[int],
    p_values: list[float],
    logical_rates: dict[int, list[float]],
) -> float | None:
    """Estimate the crossing point of logical error rate curves."""
    if len(distances) < 2:
        return None

    d_small = distances[0]
    d_large = distances[-1]
    rates_small = logical_rates[d_small]
    rates_large = logical_rates[d_large]

    for i in range(len(p_values) - 1):
        diff_i = rates_small[i] - rates_large[i]
        diff_next = rates_small[i + 1] - rates_large[i + 1]

        if diff_i * diff_next <= 0 and (abs(diff_i) + abs(diff_next)) > 0:
            t = abs(diff_i) / (abs(diff_i) + abs(diff_next))
            return p_values[i] + t * (p_values[i + 1] - p_values[i])

    return None
