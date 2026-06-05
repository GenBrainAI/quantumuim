"""Observables and diagnostic quantities for open system evolution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class EvolutionResult:
    """Result from Lindblad master equation integration.

    Attributes:
        times: Array of time points.
        states: List of density matrices at each time point.
        expect_values: Dict mapping operator labels to expectation
            value arrays (populated if observables were provided).
    """
    times: NDArray[np.floating]
    states: list[NDArray[np.complexfloating]]
    expect_values: dict[str, NDArray[np.floating]]


def expect(
    rho: NDArray[np.complexfloating],
    operator: NDArray[np.complexfloating],
) -> complex:
    """Compute expectation value Tr(ρ · O)."""
    return complex(np.trace(rho @ operator))


def purity(rho: NDArray[np.complexfloating]) -> float:
    """Compute the purity Tr(ρ²) of a density matrix.

    Returns 1 for pure states, 1/d for maximally mixed states.
    """
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho: NDArray[np.complexfloating]) -> float:
    """Compute the von Neumann entropy S = -Tr(ρ log₂ ρ).

    Returns entropy in bits.
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    entropy = 0.0
    for lam in eigenvalues:
        lam = float(np.real(lam))
        if lam > 1e-15:
            entropy -= lam * math.log2(lam)
    return entropy


def coherence(rho: NDArray[np.complexfloating]) -> float:
    """Compute the l1-norm of coherence (sum of off-diagonal magnitudes)."""
    d = rho.shape[0]
    total = 0.0
    for i in range(d):
        for j in range(d):
            if i != j:
                total += abs(rho[i, j])
    return total
