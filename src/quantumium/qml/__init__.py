"""Quantum machine learning -- data encoding, variational classifiers, and kernels.

This package provides tools for quantum-enhanced machine learning:

- **encoding**: Data-encoding circuits (amplitude, angle, ZZ feature map)
  that map classical feature vectors into quantum states.
- **classifier**: A variational quantum classifier (VQC) trained with
  parameter-shift gradient descent.
- **kernel**: Quantum kernel estimation for use with classical kernel
  methods such as SVM, based on quantum state fidelity.
"""

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
