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
    """Compute the expectation value of an operator in a given state.

    Evaluates Tr(rho * O) for density matrix rho and operator O.

    Args:
        rho (NDArray[np.complexfloating]): Density matrix (d x d).
        operator (NDArray[np.complexfloating]): Observable operator (d x d).

    Returns:
        complex: The expectation value Tr(rho * O).
    """
    return complex(np.trace(rho @ operator))


def purity(rho: NDArray[np.complexfloating]) -> float:
    """Compute the purity of a density matrix.

    Purity is defined as Tr(rho^2). It equals 1 for pure states and
    1/d for the maximally mixed state of dimension d.

    Args:
        rho (NDArray[np.complexfloating]): Density matrix (d x d).

    Returns:
        float: Purity value in [1/d, 1].
    """
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho: NDArray[np.complexfloating]) -> float:
    """Compute the von Neumann entropy of a density matrix.

    Defined as S = -Tr(rho * log2(rho)). Returns 0 for pure states and
    log2(d) for the maximally mixed state of dimension d.

    Args:
        rho (NDArray[np.complexfloating]): Density matrix (d x d).

    Returns:
        float: Entropy in bits (base-2 logarithm).
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    entropy = 0.0
    for lam in eigenvalues:
        lam = float(np.real(lam))
        if lam > 1e-15:
            entropy -= lam * math.log2(lam)
    return entropy


def coherence(rho: NDArray[np.complexfloating]) -> float:
    """Compute the l1-norm of coherence of a density matrix.

    The l1-norm coherence is the sum of the absolute values of all
    off-diagonal elements. It equals 0 for diagonal (incoherent) states
    and is maximal for states like |+>.

    Args:
        rho (NDArray[np.complexfloating]): Density matrix (d x d).

    Returns:
        float: The l1-norm of coherence (non-negative).
    """
    d = rho.shape[0]
    total = 0.0
    for i in range(d):
        for j in range(d):
            if i != j:
                total += abs(rho[i, j])
    return total
