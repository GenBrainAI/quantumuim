"""QSP angle-finding algorithms.

Given a target polynomial P(a), find QSP processing angles {φ_k} such
that the (0,0) entry of the QSP sequence equals P(a).

The key constraint: P must satisfy |P(a)|² ≤ 1 for all a ∈ [-1, 1],
and deg(P) = d requires d+1 angles.

For specific important polynomials (Chebyshev, sign, threshold),
we provide direct constructions. For arbitrary polynomials, we use
an optimization-based approach.

References:
    Dong et al., "Efficient phase-factor evaluation in quantum signal
    processing" (Physical Review A 103, 042419, 2021)
    Haah, "Product decomposition of periodic functions in quantum
    signal processing" (Quantum 3, 190, 2019)
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def find_qsp_angles(
    target_poly_coeffs: NDArray[np.floating] | list[float],
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
) -> NDArray[np.floating]:
    """Find QSP angles for a target polynomial via optimization.

    Uses gradient descent to minimize the L2 error between the
    QSP polynomial and the target polynomial evaluated on a grid.

    The target polynomial P(a) = Σ c_k T_k(a) should be expressed
    in the Chebyshev basis, where T_k is the k-th Chebyshev polynomial.

    Args:
        target_poly_coeffs: Chebyshev coefficients [c_0, c_1, ..., c_d].
        num_iterations: Optimization iterations.
        learning_rate: Step size for gradient descent.

    Returns:
        QSP angles array of length d+1.
    """
    coeffs = np.asarray(target_poly_coeffs, dtype=float)
    d = len(coeffs) - 1
    num_angles = d + 1

    grid = np.linspace(-1, 1, 100)
    target_vals = np.zeros(len(grid))
    for k, c in enumerate(coeffs):
        target_vals += c * _chebyshev(k, grid)

    max_val = np.max(np.abs(target_vals))
    if max_val > 1.0:
        target_vals = target_vals / max_val

    rng = np.random.default_rng(42)
    angles = rng.uniform(-0.1, 0.1, size=num_angles)

    for iteration in range(num_iterations):
        qsp_vals = _evaluate_qsp_on_grid(angles, grid)
        error = qsp_vals - target_vals
        loss = float(np.mean(np.abs(error) ** 2))

        grad = np.zeros(num_angles)
        eps = 1e-6
        for j in range(num_angles):
            angles_plus = angles.copy()
            angles_plus[j] += eps
            vals_plus = _evaluate_qsp_on_grid(angles_plus, grid)
            loss_plus = float(np.mean(np.abs(vals_plus - target_vals) ** 2))
            grad[j] = (loss_plus - loss) / eps

        angles -= learning_rate * grad

        if loss < 1e-12:
            break

    return angles


def _evaluate_qsp_on_grid(
    angles: NDArray[np.floating],
    grid: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Evaluate Re(P(a)) on a grid of signal values."""
    from .signal_processing import qsp_sequence

    result = np.zeros(len(grid))
    for i, a in enumerate(grid):
        U = qsp_sequence(angles, a)
        result[i] = float(np.real(U[0, 0]))
    return result


def _chebyshev(n: int, x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Evaluate Chebyshev polynomial T_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    T_prev2 = np.ones_like(x)
    T_prev1 = x.copy()
    for _ in range(2, n + 1):
        T_curr = 2 * x * T_prev1 - T_prev2
        T_prev2 = T_prev1
        T_prev1 = T_curr
    return T_prev1


def chebyshev_angles(degree: int) -> NDArray[np.floating]:
    """QSP angles that realize the Chebyshev polynomial T_d(a).

    For T_d, the QSP angles have a known closed form:
    φ_0 = (1-d)π/4 if d is odd, (2-d)π/4 if d is even,
    φ_k = π/4 for 1 ≤ k ≤ d-1,
    φ_d = π/4.

    These come from the factorization of T_d in the QSP convention.
    """
    if degree == 0:
        return np.array([0.0])

    if degree == 1:
        return np.array([0.0, 0.0])

    angles = np.full(degree + 1, math.pi / 4)

    if degree % 2 == 1:
        angles[0] = (1 - degree) * math.pi / 4
    else:
        angles[0] = (2 - degree) * math.pi / 4

    return angles


def sign_function_angles(degree: int = 15, kappa: float = 1.0) -> NDArray[np.floating]:
    """QSP angles approximating the sign function sgn(a).

    Uses Chebyshev polynomial approximation of erf(κa) as a smooth
    approximation to the sign function.

    Args:
        degree: Polynomial degree (higher = sharper transition).
            Must be odd for sign function (odd polynomial).
        kappa: Sharpness parameter (larger = closer to step function).

    Returns:
        QSP angles of length degree + 1.
    """
    if degree % 2 == 0:
        degree += 1

    grid = np.linspace(-1, 1, 100)
    target = np.tanh(kappa * grid * 3)

    max_abs = np.max(np.abs(target))
    if max_abs > 0.99:
        target = target * 0.99 / max_abs

    coeffs = _fit_chebyshev(target, grid, degree)

    return find_qsp_angles(coeffs, num_iterations=2000, learning_rate=0.005)


def threshold_angles(
    threshold: float = 0.5,
    degree: int = 15,
    width: float = 0.1,
) -> NDArray[np.floating]:
    """QSP angles approximating a threshold (step) function.

    H(a - threshold) ≈ 1 if a > threshold, 0 if a < threshold,
    with smooth transition of given width.

    Args:
        threshold: Location of the step.
        degree: Polynomial degree.
        width: Width of the transition region.

    Returns:
        QSP angles.
    """
    grid = np.linspace(-1, 1, 100)
    target = 0.5 * (1 + np.tanh((grid - threshold) / width))

    target = target * 0.98 + 0.01

    coeffs = _fit_chebyshev(target, grid, degree)

    return find_qsp_angles(coeffs, num_iterations=2000, learning_rate=0.005)


def _fit_chebyshev(
    values: NDArray[np.floating],
    grid: NDArray[np.floating],
    degree: int,
) -> NDArray[np.floating]:
    """Fit a function to Chebyshev polynomial coefficients."""
    basis = np.zeros((len(grid), degree + 1))
    for k in range(degree + 1):
        basis[:, k] = _chebyshev(k, grid)

    coeffs, _, _, _ = np.linalg.lstsq(basis, values, rcond=None)
    return coeffs
