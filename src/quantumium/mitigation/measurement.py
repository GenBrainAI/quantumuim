"""Measurement error mitigation via calibration matrix inversion.

On real hardware, readout errors cause |0⟩ to be measured as |1⟩ and
vice versa. By characterizing these errors with calibration circuits,
we can invert the error to recover corrected probability distributions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def calibration_matrix(
    error_rates: dict[int, tuple[float, float]],
) -> NDArray[np.floating]:
    """Build the readout calibration matrix from per-qubit error rates.

    Args:
        error_rates: Maps qubit index to (p0_given_1, p1_given_0) where
            p0_given_1 = P(measure 0 | prepared 1)
            p1_given_0 = P(measure 1 | prepared 0)

    Returns:
        2^n x 2^n calibration matrix A such that noisy = A @ ideal.
    """
    qubits = sorted(error_rates.keys())
    len(qubits)

    single_matrices = []
    for q in qubits:
        p01, p10 = error_rates[q]
        m = np.array(
            [
                [1 - p10, p01],
                [p10, 1 - p01],
            ]
        )
        single_matrices.append(m)

    result = single_matrices[0]
    for m in single_matrices[1:]:
        result = np.kron(result, m)

    return result


def mitigate_probabilities(
    cal_matrix: NDArray[np.floating],
    noisy_probs: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Apply inverse calibration matrix to noisy probabilities.

    Uses pseudoinverse with non-negativity clipping and renormalization.
    """
    inv = np.linalg.pinv(cal_matrix)
    corrected = inv @ noisy_probs
    corrected = np.clip(corrected, 0, None)
    total = corrected.sum()
    if total > 0:
        corrected = corrected / total
    return corrected


def mitigate_counts(
    cal_matrix: NDArray[np.floating],
    counts: dict[str, int],
    num_qubits: int,
) -> dict[str, float]:
    """Mitigate measurement counts using calibration matrix.

    Args:
        cal_matrix: Readout calibration matrix from calibration_matrix().
        counts: Raw measurement counts {bitstring: count}.
        num_qubits: Number of qubits.

    Returns:
        Corrected probability distribution {bitstring: probability}.
    """
    total_shots = sum(counts.values())
    dim = 2**num_qubits

    noisy_probs = np.zeros(dim)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        noisy_probs[idx] = count / total_shots

    corrected = mitigate_probabilities(cal_matrix, noisy_probs)

    result = {}
    for i in range(dim):
        if corrected[i] > 1e-10:
            result[format(i, f"0{num_qubits}b")] = float(corrected[i])
    return result
