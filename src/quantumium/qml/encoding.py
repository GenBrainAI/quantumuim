"""Data encoding circuits for quantum machine learning.

Maps classical data vectors into quantum states using various strategies:
- Amplitude encoding: log(N) qubits for N features
- Angle encoding: N qubits for N features via Ry rotations
- ZZ feature map: entangling feature map for kernel methods
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H, Ry, Rz, CNOT


def amplitude_encode(data: NDArray[np.floating]) -> QuantumCircuit:
    """Encode a classical vector into quantum amplitudes.

    Maps an N-dimensional vector to log2(N) qubits by preparing
    a state whose amplitudes are proportional to the data values.

    Args:
        data: Real-valued vector. Length must be a power of 2.

    Returns:
        Circuit that prepares the encoded state from |0...0>.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"Data length must be a power of 2, got {n}")

    num_qubits = int(np.log2(n))
    norm = np.linalg.norm(data)
    if norm < 1e-15:
        raise ValueError("Data vector must be non-zero")

    data_norm = data / norm
    dim = 2**num_qubits

    # Build unitary that maps |0> to |data> via QR decomposition
    M = np.eye(dim, dtype=complex)
    M[:, 0] = data_norm
    Q, _ = np.linalg.qr(M)
    if np.real(np.dot(Q[:, 0].conj(), data_norm)) < 0:
        Q *= -1

    qc = QuantumCircuit(num_qubits)
    prep_gate = Gate(Q, "AmpEnc")
    qubits = list(range(num_qubits - 1, -1, -1))
    qc.apply(prep_gate, *qubits)
    return qc


def angle_encode(data: NDArray[np.floating]) -> QuantumCircuit:
    """Encode a classical vector using rotation angles.

    Each feature x_i is encoded as Ry(x_i) on qubit i.
    Uses N qubits for N features.

    Args:
        data: Real-valued feature vector.

    Returns:
        Circuit with one qubit per feature.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 1:
        raise ValueError("Data must have at least 1 feature")

    qc = QuantumCircuit(n)
    for i, x in enumerate(data):
        qc.apply(Ry(x), i)
    return qc


def zz_feature_map(data: NDArray[np.floating], reps: int = 2) -> QuantumCircuit:
    """ZZ feature map — entangling encoding for kernel methods.

    Implements the Havlicek et al. (2019) feature map:
    U(x) = prod_r [exp(i sum_{ij} x_i x_j Z_i Z_j) exp(i sum_i x_i H_i)] H^n

    Each repetition applies:
    1. Hadamard on all qubits
    2. Rz(2*x_i) on each qubit i
    3. CNOT + Rz(2*x_i*x_j) + CNOT for each pair (i, j)

    Args:
        data: Feature vector of length N (N qubits used).
        reps: Number of repetitions of the feature map layer.

    Returns:
        Circuit implementing the feature map.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 1:
        raise ValueError("Data must have at least 1 feature")

    qc = QuantumCircuit(n)

    for _ in range(reps):
        for i in range(n):
            qc.apply(H, i)

        for i in range(n):
            qc.apply(Rz(2.0 * data[i]), i)

        for i in range(n):
            for j in range(i + 1, n):
                qc.apply(CNOT, i, j)
                qc.apply(Rz(2.0 * data[i] * data[j]), j)
                qc.apply(CNOT, i, j)

    return qc
