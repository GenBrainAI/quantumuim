"""QSVT application primitives.

Demonstrates the "grand unification" by implementing three fundamental
quantum algorithms as special cases of QSVT:
1. Matrix inversion (→ HHL-like)
2. Hamiltonian simulation (→ product formulas)
3. Amplitude amplification (→ Grover-like)

Each reduces to choosing the right QSP polynomial and angles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .qsvt import QSVTResult, block_encoding, qsvt_transform
from .angle_finding import find_qsp_angles


def qsvt_matrix_inversion(
    A: NDArray[np.complexfloating],
    kappa_cutoff: float = 0.1,
    degree: int = 15,
    alpha: float | None = None,
) -> QSVTResult:
    """Approximate matrix inversion A⁻¹ via QSVT.

    Applies a polynomial approximation of 1/x to the singular values
    of A. Small singular values below kappa_cutoff are regularized
    to avoid divergence.

    The polynomial P(x) ≈ 1/(α·x) for |x| > kappa_cutoff,
    smoothly goes to 0 for |x| < kappa_cutoff/2.

    Args:
        A: Input matrix to invert.
        kappa_cutoff: Regularization threshold for small singular values.
        degree: Polynomial degree (higher = better approximation).
        alpha: Block-encoding norm. Auto-computed if None.

    Returns:
        QSVTResult where transformed_matrix ≈ A⁻¹/α (up to normalization).
    """
    A = np.asarray(A, dtype=complex)

    if alpha is None:
        alpha = float(np.linalg.norm(A, ord=2)) * 1.01

    # Build Chebyshev approximation of regularized 1/x
    grid = np.linspace(-1, 1, 100)

    target = np.zeros_like(grid)
    for i, x in enumerate(grid):
        ax = abs(x)
        if ax > kappa_cutoff:
            target[i] = 1.0 / x
        elif ax > kappa_cutoff / 2:
            # Smooth transition
            t = (ax - kappa_cutoff / 2) / (kappa_cutoff / 2)
            target[i] = math.copysign(t / kappa_cutoff, x)

    # Normalize to fit in [-1, 1]
    max_abs = np.max(np.abs(target))
    if max_abs > 0:
        target = target * 0.95 / max_abs

    # Fit to Chebyshev basis
    from .angle_finding import _chebyshev, _fit_chebyshev
    coeffs = _fit_chebyshev(target, grid, degree)

    angles = find_qsp_angles(coeffs, num_iterations=500, learning_rate=0.01)

    return qsvt_transform(A, angles, alpha)


def qsvt_hamiltonian_sim(
    H: NDArray[np.complexfloating],
    time: float = 1.0,
    degree: int = 15,
) -> QSVTResult:
    """Hamiltonian simulation e^{-iHt} via QSVT.

    Approximates cos(Ht) + i·sin(Ht) applied to the eigenvalues
    of H using a Jacobi-Anger expansion in Chebyshev polynomials.

    For Hermitian H with ‖H‖ ≤ 1 (after normalization by alpha),
    this realizes a polynomial approximation to e^{-iαt·x}.

    Args:
        H: Hermitian Hamiltonian matrix.
        time: Evolution time.
        degree: Polynomial degree (higher = longer time or better accuracy).

    Returns:
        QSVTResult where transformed_matrix ≈ e^{-iHt}.
    """
    H = np.asarray(H, dtype=complex)

    if not np.allclose(H, H.conj().T, atol=1e-10):
        raise ValueError("H must be Hermitian")

    alpha = float(np.linalg.norm(H, ord=2)) * 1.01
    tau = alpha * time

    # Approximate cos(τx) via Chebyshev fitting on a grid
    from .angle_finding import _fit_chebyshev
    grid = np.linspace(-1, 1, 100)
    target = np.cos(tau * grid)

    max_abs = np.max(np.abs(target))
    if max_abs > 0.99:
        target = target * 0.99 / max_abs

    cos_coeffs = _fit_chebyshev(target, grid, degree)

    angles = find_qsp_angles(cos_coeffs, num_iterations=2000, learning_rate=0.005)

    return qsvt_transform(H, angles, alpha)


def qsvt_amplitude_amplification(
    A: NDArray[np.complexfloating],
    num_iterations: int = 3,
) -> QSVTResult:
    """Amplitude amplification as a special case of QSVT.

    Applies a Chebyshev polynomial T_d(x) to the singular values,
    which amplifies values near 0 toward 1 (the core of Grover-like
    speedup).

    For d iterations of Grover, the effective polynomial is
    T_d(x) = cos(d·arccos(x)), which maps small x to values
    oscillating between -1 and 1.

    Args:
        A: Input matrix (block-encoded operator).
        num_iterations: Number of amplification rounds (= Chebyshev degree).

    Returns:
        QSVTResult with amplified singular values.
    """
    A = np.asarray(A, dtype=complex)
    d = 2 * num_iterations + 1

    from .angle_finding import chebyshev_angles
    angles = chebyshev_angles(d)

    return qsvt_transform(A, angles)
