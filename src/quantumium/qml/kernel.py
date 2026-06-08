"""Quantum kernel estimation for machine learning.

Computes kernel values k(x, x') = |<phi(x)|phi(x')>|^2 using
the fidelity between quantum feature-map states. This kernel
can be used with classical SVM or other kernel methods.

Based on: Havlicek et al., "Supervised learning with quantum-enhanced
feature spaces" (Nature, 2019).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .encoding import zz_feature_map, angle_encode


def quantum_kernel(
    x1: NDArray[np.floating],
    x2: NDArray[np.floating],
    feature_map: str = "zz",
    reps: int = 2,
) -> float:
    """Compute the quantum kernel value between two data points.

    k(x1, x2) = |<0|U(x1)^dag U(x2)|0>|^2

    This is the fidelity between the two feature-map states.

    Args:
        x1: First data point.
        x2: Second data point.
        feature_map: Encoding strategy — "zz" or "angle".
        reps: Repetitions for the ZZ feature map.

    Returns:
        Kernel value in [0, 1].
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    if len(x1) != len(x2):
        raise ValueError("Data points must have the same dimension")

    # Get statevectors for both data points
    if feature_map == "zz":
        sv1 = zz_feature_map(x1, reps=reps).statevector()
        sv2 = zz_feature_map(x2, reps=reps).statevector()
    elif feature_map == "angle":
        sv1 = angle_encode(x1).statevector()
        sv2 = angle_encode(x2).statevector()
    else:
        raise ValueError(f"Unknown feature map: {feature_map}")

    # Kernel = fidelity = |<phi(x1)|phi(x2)>|^2
    return float(abs(np.dot(sv1.conj(), sv2)) ** 2)


def quantum_kernel_matrix(
    X: NDArray[np.floating],
    Y: NDArray[np.floating] | None = None,
    feature_map: str = "zz",
    reps: int = 2,
) -> NDArray[np.floating]:
    """Compute the quantum kernel (Gram) matrix.

    Args:
        X: Data matrix, shape (n_samples, n_features).
        Y: Optional second data matrix. If None, computes K(X, X).
        feature_map: Encoding strategy — "zz" or "angle".
        reps: Repetitions for the ZZ feature map.

    Returns:
        Kernel matrix, shape (n_X, n_Y) or (n_X, n_X) if Y is None.
    """
    X = np.asarray(X, dtype=float)
    symmetric = Y is None
    if symmetric:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)

    n_x = len(X)
    n_y = len(Y)
    K = np.zeros((n_x, n_y))

    # Cache statevectors to avoid recomputation
    sv_x = []
    for i in range(n_x):
        if feature_map == "zz":
            sv = zz_feature_map(X[i], reps=reps).statevector()
        else:
            sv = angle_encode(X[i]).statevector()
        sv_x.append(sv)

    if symmetric:
        sv_y = sv_x
    else:
        sv_y = []
        for j in range(n_y):
            if feature_map == "zz":
                sv = zz_feature_map(Y[j], reps=reps).statevector()
            else:
                sv = angle_encode(Y[j]).statevector()
            sv_y.append(sv)

    for i in range(n_x):
        start_j = i if symmetric else 0
        for j in range(start_j, n_y):
            val = float(abs(np.dot(sv_x[i].conj(), sv_y[j])) ** 2)
            K[i, j] = val
            if symmetric and i != j:
                K[j, i] = val

    return K
