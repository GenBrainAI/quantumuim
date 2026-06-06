"""Monte Carlo threshold estimation for surface codes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantumium.surface_code.lattice import SurfaceCode
from quantumium.surface_code.syndrome import inject_random_errors, measure_syndrome
from quantumium.surface_code.decoder import mwpm_decode
from quantumium.surface_code.logical import has_logical_error


@dataclass
class ThresholdResult:
    """Results from a threshold estimation run."""
    distances: list[int]
    error_rates: list[float]
    logical_error_rates: dict[int, list[float]] = field(default_factory=dict)
    num_trials: int = 0


def estimate_threshold(
    distances: list[int] | None = None,
    error_rates: list[float] | None = None,
    num_trials: int = 1000,
    seed: int | None = None,
) -> ThresholdResult:
    """Estimate the error correction threshold via Monte Carlo simulation.

    For each (distance, physical_error_rate) pair, runs num_trials rounds of:
    inject random depolarizing errors → measure syndrome → decode → check logical error.
    """
    if distances is None:
        distances = [3, 5, 7]
    if error_rates is None:
        error_rates = [0.001, 0.005, 0.01, 0.015, 0.02]

    rng = np.random.default_rng(seed)
    result = ThresholdResult(
        distances=distances,
        error_rates=error_rates,
        num_trials=num_trials,
    )

    for d in distances:
        logical_rates: list[float] = []
        code = SurfaceCode(d)
        for p in error_rates:
            logical_errors = 0
            for _ in range(num_trials):
                code.reset()
                inject_random_errors(code, p, rng)
                x_syn, z_syn = measure_syndrome(code)
                x_corr, z_corr = mwpm_decode(code, x_syn, z_syn)
                if has_logical_error(code, x_corr, z_corr):
                    logical_errors += 1
            logical_rates.append(logical_errors / num_trials)
        result.logical_error_rates[d] = logical_rates

    return result
