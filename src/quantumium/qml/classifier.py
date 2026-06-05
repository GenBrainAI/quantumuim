"""Variational Quantum Classifier (VQC).

A hybrid quantum-classical classifier that uses a parameterized quantum
circuit as the model, trained via classical optimization.

Architecture:
1. Feature encoding: maps classical data x -> quantum state |phi(x)>
2. Variational ansatz: parameterized unitary V(theta)|phi(x)>
3. Measurement: expectation of Z on qubit 0 -> prediction
4. Classical optimizer: updates theta to minimize loss
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H, Ry, Rz, CNOT, Z
from .encoding import angle_encode


@dataclass
class VQCResult:
    weights: NDArray[np.floating]
    bias: float
    loss_history: list[float]
    accuracy: float


def _build_ansatz(
    qc: QuantumCircuit,
    weights: NDArray[np.floating],
    num_qubits: int,
    num_layers: int,
) -> None:
    """Append a hardware-efficient variational ansatz to the circuit."""
    idx = 0
    for _ in range(num_layers):
        for q in range(num_qubits):
            qc.apply(Ry(weights[idx]), q)
            idx += 1
            qc.apply(Rz(weights[idx]), q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.apply(CNOT, q, q + 1)


def _predict_single(
    x: NDArray[np.floating],
    weights: NDArray[np.floating],
    bias: float,
    num_layers: int,
) -> float:
    """Get raw prediction for a single data point."""
    num_qubits = len(x)
    qc = QuantumCircuit(num_qubits)

    # Encode data
    for i, xi in enumerate(x):
        qc.apply(Ry(xi), i)

    # Apply ansatz
    _build_ansatz(qc, weights, num_qubits, num_layers)

    # Measure <Z> on qubit 0
    sv = qc.statevector()
    n = len(sv)
    expectation = 0.0
    for i in range(n):
        # Qubit 0 is LSB: if bit 0 is set, eigenvalue is -1, else +1
        sign = -1.0 if (i & 1) else 1.0
        expectation += sign * abs(sv[i]) ** 2

    return expectation + bias


def _loss_fn(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    weights: NDArray[np.floating],
    bias: float,
    num_layers: int,
) -> float:
    """Mean squared error loss."""
    total = 0.0
    for i in range(len(X)):
        pred = _predict_single(X[i], weights, bias, num_layers)
        total += (pred - y[i]) ** 2
    return total / len(X)


def variational_classifier(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    num_layers: int = 2,
    learning_rate: float = 0.1,
    epochs: int = 50,
    seed: int | None = None,
) -> VQCResult:
    """Train a variational quantum classifier.

    Uses angle encoding + hardware-efficient ansatz + Z measurement.
    Labels should be +1 or -1.

    Args:
        X: Training data, shape (num_samples, num_features).
        y: Labels, shape (num_samples,), values in {-1, +1}.
        num_layers: Depth of the variational ansatz.
        learning_rate: Step size for parameter-shift gradient descent.
        epochs: Number of training iterations.
        seed: Random seed for weight initialization.

    Returns:
        VQCResult with trained weights, bias, loss history, and accuracy.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D (samples x features)")
    if len(y) != len(X):
        raise ValueError("X and y must have same number of samples")

    num_samples, num_qubits = X.shape
    num_weights = num_layers * num_qubits * 2  # Ry + Rz per qubit per layer

    rng = np.random.default_rng(seed)
    weights = rng.uniform(-np.pi, np.pi, num_weights)
    bias = 0.0

    loss_history = []
    shift = np.pi / 2

    for epoch in range(epochs):
        loss = _loss_fn(X, y, weights, bias, num_layers)
        loss_history.append(loss)

        # Parameter-shift gradients for weights
        grad = np.zeros_like(weights)
        for k in range(len(weights)):
            w_plus = weights.copy()
            w_plus[k] += shift
            w_minus = weights.copy()
            w_minus[k] -= shift
            grad[k] = (
                _loss_fn(X, y, w_plus, bias, num_layers)
                - _loss_fn(X, y, w_minus, bias, num_layers)
            ) / (2 * np.sin(shift))

        # Gradient for bias (finite difference)
        eps = 0.01
        grad_bias = (
            _loss_fn(X, y, weights, bias + eps, num_layers)
            - _loss_fn(X, y, weights, bias - eps, num_layers)
        ) / (2 * eps)

        weights -= learning_rate * grad
        bias -= learning_rate * grad_bias

    # Compute final accuracy
    correct = 0
    for i in range(num_samples):
        pred = _predict_single(X[i], weights, bias, num_layers)
        if (pred >= 0 and y[i] > 0) or (pred < 0 and y[i] < 0):
            correct += 1

    return VQCResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        accuracy=correct / num_samples,
    )
