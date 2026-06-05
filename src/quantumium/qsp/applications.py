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
    """Approximate matrix inversion via QSVT (HHL-like algorithm).

    Applies a polynomial approximation of 1/x to the singular values
    of A using Quantum Singular Value Transformation. Small singular
    values below kappa_cutoff are regularized with a smooth transition
    to avoid divergence.

    The polynomial P(x) approximates 1/(alpha*x) for |x| > kappa_cutoff
    and smoothly transitions to 0 for |x| < kappa_cutoff/2.

    Args:
        A (NDArray[np.complexfloating]): Input matrix to invert (n x n).
        kappa_cutoff (float): Regularization threshold below which
            singular values are suppressed. Defaults to 0.1.
        degree (int): Polynomial degree for the approximation. Higher
            values improve accuracy but increase circuit depth.
            Defaults to 15.
        alpha (float | None): Block-encoding normalization factor.
            If None, automatically set to 1.01 * ||A||_op.

    Returns:
        QSVTResult: Result where transformed_matrix approximates
            A^{-1}/alpha (up to normalization from the polynomial fit).
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
    """Simulate Hamiltonian time-evolution e^{-iHt} via QSVT.

    Approximates the cosine component cos(H*t) applied to the
    eigenvalues of H using a Chebyshev polynomial fit. After
    normalization by alpha = 1.01 * ||H||_op, the polynomial
    approximates cos(alpha * t * x) for x in [-1, 1].

    Args:
        H (NDArray[np.complexfloating]): Hermitian Hamiltonian matrix
            (n x n). Must be Hermitian (H = H^dagger).
        time (float): Evolution time. Defaults to 1.0.
        degree (int): Polynomial degree for the Chebyshev approximation.
            Higher values support longer evolution times or better
            accuracy. Defaults to 15.

    Returns:
        QSVTResult: Result where transformed_matrix approximates
            e^{-iHt} (specifically the cos(Ht) component).

    Raises:
        ValueError: If H is not Hermitian.
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
    """Perform amplitude amplification as a special case of QSVT.

    Applies a Chebyshev polynomial T_d(x) to the singular values of
    the block-encoded matrix, which amplifies small singular values
    toward 1. This is the core mechanism underlying the quadratic
    Grover-like speedup.

    The effective polynomial T_d(x) = cos(d * arccos(x)) maps small
    x values to oscillating values between -1 and 1.

    Args:
        A (NDArray[np.complexfloating]): Input matrix to amplify
            (block-encoded operator, n x n).
        num_iterations (int): Number of amplification rounds. The
            Chebyshev polynomial degree is 2 * num_iterations + 1.
            Defaults to 3.

    Returns:
        QSVTResult: Result with amplified singular values in the
            transformed_matrix.
    """
    A = np.asarray(A, dtype=complex)
    d = 2 * num_iterations + 1

    from .angle_finding import chebyshev_angles
    angles = chebyshev_angles(d)

    return qsvt_transform(A, angles)
