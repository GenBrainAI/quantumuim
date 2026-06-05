"""Quantum Signal Processing (QSP) and Quantum Singular Value Transformation (QSVT).

The unified algorithmic framework that generalizes QPE, Hamiltonian simulation,
amplitude amplification, and matrix arithmetic into polynomial transformations
of block-encoded matrices.

References:
    Gilyén et al., "Quantum singular value transformation and beyond"
    (STOC 2019)
    Martyn et al., "Grand Unification of Quantum Algorithms"
    (PRX Quantum 2, 040203, 2021)
"""

from .signal_processing import (
    QSPResult,
    qsp_sequence,
    qsp_circuit,
    evaluate_qsp_polynomial,
)
from .qsvt import (
    QSVTResult,
    block_encoding,
    qsvt_circuit,
    qsvt_transform,
)
from .angle_finding import (
    find_qsp_angles,
    chebyshev_angles,
    sign_function_angles,
    threshold_angles,
)
from .applications import (
    qsvt_matrix_inversion,
    qsvt_hamiltonian_sim,
    qsvt_amplitude_amplification,
)

__all__ = [
    "QSPResult",
    "qsp_sequence",
    "qsp_circuit",
    "evaluate_qsp_polynomial",
    "QSVTResult",
    "block_encoding",
    "qsvt_circuit",
    "qsvt_transform",
    "find_qsp_angles",
    "chebyshev_angles",
    "sign_function_angles",
    "threshold_angles",
    "qsvt_matrix_inversion",
    "qsvt_hamiltonian_sim",
    "qsvt_amplitude_amplification",
]
