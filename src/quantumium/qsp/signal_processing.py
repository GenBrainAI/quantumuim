"""Quantum Signal Processing (QSP) core routines.

QSP applies a polynomial transformation to the eigenvalues of a
single-qubit signal rotation. Given signal rotation W(a) = exp(i*a*Z)
and a sequence of processing rotations exp(i*φ_k*Z), the interleaved
product realizes a polynomial transformation:

    ⟨0| U_QSP |0⟩ = P(a)

where P is a polynomial of degree d (= number of signal rotations)
determined by the processing angles {φ_k}.

Convention (Wx):
    Signal operator: W(a) = [[a, i√(1-a²)], [i√(1-a²), a]]
    Processing operator: S(φ) = [[e^{iφ}, 0], [0, e^{-iφ}]]
    QSP sequence: S(φ_0) · W(a) · S(φ_1) · W(a) · ... · S(φ_d)

The (0,0) entry of the resulting 2x2 unitary is P(a), a degree-d
polynomial in a that satisfies |P(a)|² + (1-a²)|Q(a)|² = 1.

References:
    Low & Chuang, "Optimal Hamiltonian Simulation by Quantum Signal
    Processing" (PRL 118, 010501, 2017)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate


@dataclass
class QSPResult:
    """Result from a Quantum Signal Processing computation.

    Attributes:
        polynomial_values (NDArray[np.complexfloating]): The QSP
            polynomial P(a) evaluated at each signal value.
        angles (NDArray[np.floating]): The QSP processing angles
            [phi_0, phi_1, ..., phi_d] that define the polynomial.
        degree (int): Degree of the QSP polynomial (number of signal
            rotations).
        signal_values (NDArray[np.floating]): The signal values a
            at which the polynomial was evaluated.
    """
    polynomial_values: NDArray[np.complexfloating]
    angles: NDArray[np.floating]
    degree: int
    signal_values: NDArray[np.floating]


def _signal_operator(a: float) -> NDArray[np.complexfloating]:
    """W(a) signal rotation matrix for signal value a ∈ [-1, 1]."""
    s = math.sqrt(max(0.0, 1 - a * a))
    return np.array([
        [a, 1j * s],
        [1j * s, a],
    ], dtype=complex)


def _processing_operator(phi: float) -> NDArray[np.complexfloating]:
    """S(φ) = diag(e^{iφ}, e^{-iφ}) processing rotation."""
    return np.array([
        [np.exp(1j * phi), 0],
        [0, np.exp(-1j * phi)],
    ], dtype=complex)


def qsp_sequence(
    angles: NDArray[np.floating] | list[float],
    signal_value: float,
) -> NDArray[np.complexfloating]:
    """Evaluate the QSP unitary for a given signal value.

    Computes: S(φ_0) · W(a) · S(φ_1) · W(a) · ... · S(φ_d)

    Args:
        angles: QSP processing angles [φ_0, φ_1, ..., φ_d].
        signal_value: Signal value a ∈ [-1, 1].

    Returns:
        2×2 unitary matrix.
    """
    angles = np.asarray(angles, dtype=float)
    W = _signal_operator(signal_value)

    U = _processing_operator(angles[0])
    for k in range(1, len(angles)):
        U = U @ W @ _processing_operator(angles[k])

    return U


def evaluate_qsp_polynomial(
    angles: NDArray[np.floating] | list[float],
    signal_values: NDArray[np.floating] | list[float],
) -> NDArray[np.complexfloating]:
    """Evaluate the QSP polynomial P(a) = ⟨0|U_QSP|0⟩ at multiple points.

    Args:
        angles: QSP processing angles.
        signal_values: Array of signal values in [-1, 1].

    Returns:
        Array of P(a) values (complex).
    """
    signal_values = np.asarray(signal_values, dtype=float)
    results = np.zeros(len(signal_values), dtype=complex)

    for i, a in enumerate(signal_values):
        U = qsp_sequence(angles, a)
        results[i] = U[0, 0]

    return results


def qsp_circuit(
    angles: NDArray[np.floating] | list[float],
    signal_gate: Gate | None = None,
    signal_value: float = 0.5,
) -> QuantumCircuit:
    """Build a single-qubit QSP circuit.

    If signal_gate is None, constructs W(signal_value) as the signal.

    Args:
        angles: QSP processing angles [φ_0, ..., φ_d].
        signal_gate: Optional explicit signal unitary (single-qubit).
        signal_value: Signal value if no gate provided.

    Returns:
        Single-qubit QuantumCircuit implementing the QSP sequence.
    """
    angles = np.asarray(angles, dtype=float)
    qc = QuantumCircuit(1)

    if signal_gate is None:
        W = _signal_operator(signal_value)
        signal_gate = Gate(W, f"W({signal_value:.3f})")

    S0 = Gate(_processing_operator(angles[0]), f"S({angles[0]:.3f})")
    qc.apply(S0, 0)

    for k in range(1, len(angles)):
        qc.apply(signal_gate, 0)
        Sk = Gate(_processing_operator(angles[k]), f"S({angles[k]:.3f})")
        qc.apply(Sk, 0)

    return qc
