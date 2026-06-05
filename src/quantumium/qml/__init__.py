"""Quantum machine learning — data encoding, variational classifiers, and kernels."""

from .encoding import amplitude_encode, angle_encode, zz_feature_map
from .classifier import variational_classifier, VQCResult
from .kernel import quantum_kernel, quantum_kernel_matrix

__all__ = [
    "amplitude_encode",
    "angle_encode",
    "zz_feature_map",
    "variational_classifier",
    "VQCResult",
    "quantum_kernel",
    "quantum_kernel_matrix",
]
