"""Distance measures between quantum states.

Implements trace distance, Bures distance, Hellinger distance, and quantum
relative entropy for density matrices.

References:
    Nielsen & Chuang, "Quantum Computation and Quantum Information" (2010), Ch. 9
    Wilde, "Quantum Information Theory" (2nd ed., 2017), Ch. 9-11
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _ensure_density_matrix(
    rho: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Validate and coerce an input to a density matrix.

    If the input is a 1-D statevector, converts it to a density matrix
    via the outer product |psi><psi|. If already 2-D, validates that
    it is square.

    Args:
        rho: A statevector (1-D array of length d) or density matrix
            (2-D array of shape (d, d)).

    Returns:
        A (d x d) complex density matrix.

    Raises:
        ValueError: If the input is 2-D but not square.
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    if rho.shape[0] != rho.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {rho.shape}")
    return rho


def trace_distance(
    rho: NDArray[np.complexfloating],
    sigma: NDArray[np.complexfloating],
) -> float:
    """Trace distance between two quantum states.

    D(rho, sigma) = (1/2) ||rho - sigma||_1 = (1/2) sum_i |lambda_i|

    where lambda_i are eigenvalues of (rho - sigma).

    The trace distance is a metric on density matrices with range [0, 1].
    D = 0 iff rho = sigma; D = 1 iff rho and sigma have orthogonal support.

    Args:
        rho: First density matrix (or statevector).
        sigma: Second density matrix (or statevector).

    Returns:
        Trace distance in [0, 1].
    """
    rho = _ensure_density_matrix(rho)
    sigma = _ensure_density_matrix(sigma)
    if rho.shape != sigma.shape:
        raise ValueError("States must have the same dimension")

    diff = rho - sigma
    eigvals = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigvals)))


def bures_distance(
    rho: NDArray[np.complexfloating],
    sigma: NDArray[np.complexfloating],
) -> float:
    """Bures distance between two quantum states.

    d_B(rho, sigma) = sqrt(2(1 - sqrt(F(rho, sigma))))

    where F is the Uhlmann fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.

    The Bures distance is a metric with range [0, sqrt(2)].

    Args:
        rho: First density matrix (or statevector).
        sigma: Second density matrix (or statevector).

    Returns:
        Bures distance in [0, sqrt(2)].
    """
    rho = _ensure_density_matrix(rho)
    sigma = _ensure_density_matrix(sigma)
    if rho.shape != sigma.shape:
        raise ValueError("States must have the same dimension")

    fid = _uhlmann_fidelity(rho, sigma)
    return float(np.sqrt(max(0.0, 2 * (1 - np.sqrt(max(0.0, fid))))))


def hellinger_distance(
    rho: NDArray[np.complexfloating],
    sigma: NDArray[np.complexfloating],
) -> float:
    """Quantum Hellinger distance between two states.

    d_H(rho, sigma) = sqrt(2(1 - Tr(sqrt(rho) sqrt(sigma))))

    where sqrt denotes the matrix square root.

    Args:
        rho: First density matrix (or statevector).
        sigma: Second density matrix (or statevector).

    Returns:
        Hellinger distance in [0, sqrt(2)].
    """
    rho = _ensure_density_matrix(rho)
    sigma = _ensure_density_matrix(sigma)
    if rho.shape != sigma.shape:
        raise ValueError("States must have the same dimension")

    sqrt_rho = _matrix_sqrt(rho)
    sqrt_sigma = _matrix_sqrt(sigma)
    overlap = float(np.real(np.trace(sqrt_rho @ sqrt_sigma)))
    return float(np.sqrt(max(0.0, 2 * (1 - overlap))))


def relative_entropy(
    rho: NDArray[np.complexfloating],
    sigma: NDArray[np.complexfloating],
    tol: float = 1e-15,
) -> float:
    """Quantum relative entropy (Kullback-Leibler divergence).

    S(rho || sigma) = Tr(rho (log2 rho - log2 sigma))

    Returns infinity if the support of rho is not contained in the support
    of sigma.

    Args:
        rho: First density matrix.
        sigma: Second density matrix.
        tol: Eigenvalue threshold for numerical stability.

    Returns:
        Relative entropy in [0, inf).
    """
    rho = _ensure_density_matrix(rho)
    sigma = _ensure_density_matrix(sigma)
    if rho.shape != sigma.shape:
        raise ValueError("States must have the same dimension")

    eigvals_rho, eigvecs_rho = np.linalg.eigh(rho)
    eigvals_sigma, eigvecs_sigma = np.linalg.eigh(sigma)

    # Check support condition: if rho has weight on the kernel of sigma, S = inf
    for i, val_r in enumerate(eigvals_rho):
        if val_r > tol:
            vec = eigvecs_rho[:, i]
            sigma_expectation = float(np.real(vec.conj() @ sigma @ vec))
            if sigma_expectation < tol:
                return float("inf")

    log_rho = _matrix_log2(rho, tol)
    log_sigma = _matrix_log2(sigma, tol)
    return float(np.real(np.trace(rho @ (log_rho - log_sigma))))


def _uhlmann_fidelity(
    rho: NDArray[np.complexfloating],
    sigma: NDArray[np.complexfloating],
) -> float:
    """Compute the Uhlmann fidelity between two density matrices.

    F(rho, sigma) = (Tr sqrt(sqrt(rho) * sigma * sqrt(rho)))^2

    Args:
        rho: First density matrix (d x d).
        sigma: Second density matrix (d x d).

    Returns:
        Fidelity value in [0, 1].
    """
    sqrt_rho = _matrix_sqrt(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    eigvals = np.linalg.eigvalsh(product)
    return float(np.sum(np.sqrt(np.maximum(eigvals, 0.0)))) ** 2


def _matrix_sqrt(
    A: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Compute the matrix square root of a positive semidefinite matrix.

    Uses eigendecomposition: sqrt(A) = V * diag(sqrt(eigenvalues)) * V^dag.
    Negative eigenvalues (due to numerical noise) are clipped to zero.

    Args:
        A: A positive semidefinite (d x d) complex matrix.

    Returns:
        The matrix square root, a (d x d) complex matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def _matrix_log2(
    A: NDArray[np.complexfloating],
    tol: float = 1e-15,
) -> NDArray[np.complexfloating]:
    """Matrix log base 2 of a positive semidefinite matrix.

    Eigenvalues below tol are treated as zero (their log is set to 0
    rather than -inf, matching the convention 0 log 0 = 0).
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    log_eigvals = np.zeros_like(eigvals, dtype=float)
    mask = eigvals > tol
    log_eigvals[mask] = np.log2(eigvals[mask])
    return eigvecs @ np.diag(log_eigvals).astype(complex) @ eigvecs.conj().T
