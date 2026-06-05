"""Tensor network simulation for quantum circuits.

Matrix Product State (MPS) representation enables efficient simulation
of low-entanglement quantum circuits with controllable bond dimension.

References:
    Vidal, "Efficient simulation of slightly entangled quantum computations"
    (PRL 91, 147902, 2003)
    Schollwöck, "The density-matrix renormalization group in the age of
    matrix product states" (Annals of Physics 326, 96-192, 2011)
"""

from .mps import (
    MPS,
    mps_from_statevector,
    mps_zero_state,
    mps_product_state,
)
from .operations import (
    apply_single_qubit_gate,
    apply_two_qubit_gate,
    measure_qubit,
    entanglement_entropy,
    inner_product,
)

__all__ = [
    "MPS",
    "mps_from_statevector",
    "mps_zero_state",
    "mps_product_state",
    "apply_single_qubit_gate",
    "apply_two_qubit_gate",
    "measure_qubit",
    "entanglement_entropy",
    "inner_product",
]
