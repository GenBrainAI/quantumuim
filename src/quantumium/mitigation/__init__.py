"""Quantum error mitigation techniques for NISQ devices.

This package provides error mitigation strategies that reduce the effect of
noise without the overhead of full quantum error correction. Techniques
include Zero-Noise Extrapolation (ZNE), measurement error mitigation via
calibration matrix inversion, and Pauli twirling for converting coherent
errors to stochastic noise.

Modules:
    zne: Zero-Noise Extrapolation via unitary folding and polynomial extrapolation.
    measurement: Readout error mitigation using calibration matrices.
    twirling: Pauli twirling to convert coherent errors into stochastic channels.
"""

from .zne import (
    noise_scaled_circuits,
    richardson_extrapolate,
    linear_extrapolate,
    zne_expectation,
)
from .measurement import (
    calibration_matrix,
    mitigate_counts,
    mitigate_probabilities,
)
from .twirling import (
    pauli_twirl_cnot,
    twirled_circuit_ensemble,
    twirled_expectation,
)

__all__ = [
    "noise_scaled_circuits",
    "richardson_extrapolate",
    "linear_extrapolate",
    "zne_expectation",
    "calibration_matrix",
    "mitigate_counts",
    "mitigate_probabilities",
    "pauli_twirl_cnot",
    "twirled_circuit_ensemble",
    "twirled_expectation",
]
