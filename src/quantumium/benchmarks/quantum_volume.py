"""Quantum Volume (QV) estimation.

Quantum Volume measures the largest random circuit of equal width and depth
that a quantum computer can successfully execute. QV = 2^n where n is the
maximum circuit width for which the heavy output probability exceeds 2/3.

Protocol (simplified):
1. Generate random SU(4) circuits of width n and depth n
2. Simulate ideal heavy outputs (probabilities > median)
3. Run circuits on the device (or noisy simulator)
4. Check if heavy output probability > 2/3

Reference: Cross et al., "Validating quantum computers using randomized
model circuits" (arXiv:1811.12926)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate


def _random_su4(rng: np.random.Generator) -> NDArray[np.complexfloating]:
    """Generate a Haar-random SU(4) matrix (random 2-qubit gate).

    Uses QR decomposition of a random complex Gaussian matrix with
    phase correction to ensure Haar measure, then normalizes the
    determinant to 1.

    Args:
        rng: NumPy random generator instance.

    Returns:
        A (4 x 4) unitary matrix with determinant 1.
    """
    # Use QR decomposition of a random complex matrix
    Z = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    # Ensure Haar measure by fixing phases
    diag = np.diag(R)
    phase = diag / np.abs(diag)
    Q = Q @ np.diag(phase)
    # Ensure det = 1
    det = np.linalg.det(Q)
    Q = Q / (det ** (1 / 4))
    return Q


def _build_qv_circuit(
    num_qubits: int,
    depth: int,
    rng: np.random.Generator,
) -> QuantumCircuit:
    """Build a random model circuit for quantum volume estimation.

    Each layer randomly permutes the qubits and applies independent
    Haar-random SU(4) gates to consecutive pairs.

    Args:
        num_qubits: Number of qubits (circuit width).
        depth: Number of layers (circuit depth).
        rng: NumPy random generator instance.

    Returns:
        A QuantumCircuit containing the random model circuit.
    """
    qc = QuantumCircuit(num_qubits)

    for d in range(depth):
        # Random permutation of qubits
        perm = rng.permutation(num_qubits).tolist()
        # Apply SU(4) to consecutive pairs
        for i in range(0, num_qubits - 1, 2):
            q1 = perm[i]
            q2 = perm[i + 1]
            su4 = _random_su4(rng)
            gate = Gate(su4, f"SU4_d{d}_{i}")
            qc.apply(gate, q1, q2)

    return qc


def _heavy_output_probability(
    ideal_probs: NDArray[np.floating],
    noisy_probs: NDArray[np.floating],
) -> float:
    """Compute the heavy output probability for a single circuit.

    Heavy outputs are bitstrings whose ideal probability exceeds the
    median of the ideal distribution. The heavy output probability is
    the total noisy probability mass on those heavy bitstrings.

    Args:
        ideal_probs: Ideal output probability distribution (1-D array of
            length 2^n).
        noisy_probs: Noisy output probability distribution (same shape).

    Returns:
        Heavy output probability in [0, 1].
    """
    median = float(np.median(ideal_probs))
    heavy_mask = ideal_probs > median
    return float(np.sum(noisy_probs[heavy_mask]))


@dataclass
class QVResult:
    """Container for quantum volume estimation results.

    Attributes:
        num_qubits: Number of qubits used in the QV test.
        quantum_volume: Estimated quantum volume (2^n if passed, 2^(n-1)
            otherwise).
        heavy_output_prob: Average heavy output probability across all
            random circuits.
        passed: Whether the heavy output probability exceeded the threshold.
        num_circuits: Number of random model circuits averaged over.
        threshold: Heavy output probability threshold used (typically 2/3).
    """
    num_qubits: int
    quantum_volume: int
    heavy_output_prob: float
    passed: bool
    num_circuits: int
    threshold: float


def quantum_volume(
    num_qubits: int = 3,
    num_circuits: int = 100,
    error_rate: float = 0.01,
    threshold: float = 2 / 3,
    seed: int | None = None,
) -> QVResult:
    """Estimate quantum volume for a given number of qubits.

    Generates random model circuits and estimates the heavy output
    probability under a simple depolarizing noise model.

    Args:
        num_qubits: Number of qubits (circuit width = depth = num_qubits).
        num_circuits: Number of random circuits to average over.
        error_rate: Per-gate depolarizing error probability.
        threshold: Heavy output probability threshold (default 2/3).
        seed: Random seed.

    Returns:
        QVResult with quantum volume and heavy output probability.
    """
    if num_qubits < 2:
        raise ValueError("Quantum volume requires at least 2 qubits")

    rng = np.random.default_rng(seed)
    depth = num_qubits
    dim = 2**num_qubits

    hop_values = []

    for _ in range(num_circuits):
        qc = _build_qv_circuit(num_qubits, depth, rng)

        # Ideal probabilities
        sv_ideal = qc.statevector()
        ideal_probs = np.abs(sv_ideal) ** 2

        # Noisy probabilities: simple depolarizing model
        # Each gate has error_rate chance of depolarizing, mixing toward uniform
        num_gates = depth * (num_qubits // 2)
        noise_factor = (1 - error_rate) ** num_gates
        noisy_probs = noise_factor * ideal_probs + (1 - noise_factor) / dim

        hop = _heavy_output_probability(ideal_probs, noisy_probs)
        hop_values.append(hop)

    avg_hop = float(np.mean(hop_values))
    passed = avg_hop > threshold
    qv = 2**num_qubits if passed else 2 ** (num_qubits - 1)

    return QVResult(
        num_qubits=num_qubits,
        quantum_volume=qv,
        heavy_output_prob=avg_hop,
        passed=passed,
        num_circuits=num_circuits,
        threshold=threshold,
    )
