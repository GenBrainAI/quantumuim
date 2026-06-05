"""Quantum information theory toolkit.

This package provides channel representations and conversions (Kraus, Choi,
superoperator, Pauli Transfer Matrix), distance measures between quantum
states (trace distance, Bures distance, Hellinger distance, relative entropy),
and quantum process tomography for characterizing unknown channels.

Modules:
    channels: Quantum channel class with lazy conversion between Kraus, Choi,
        superoperator, and PTM representations.
    distances: Distance and divergence measures between density matrices.
    process_tomography: Quantum process tomography and chi-matrix reconstruction.
"""

from .channels import (
    QuantumChannel,
    kraus_to_choi,
    choi_to_kraus,
    kraus_to_superoperator,
    superoperator_to_kraus,
    kraus_to_ptm,
    compose_channels,
    tensor_channels,
    channel_fidelity,
)
from .distances import (
    trace_distance,
    bures_distance,
    hellinger_distance,
    relative_entropy,
)
from .process_tomography import (
    ProcessTomographyResult,
    process_tomography,
    chi_matrix,
)

__all__ = [
    "QuantumChannel",
    "kraus_to_choi",
    "choi_to_kraus",
    "kraus_to_superoperator",
    "superoperator_to_kraus",
    "kraus_to_ptm",
    "compose_channels",
    "tensor_channels",
    "channel_fidelity",
    "trace_distance",
    "bures_distance",
    "hellinger_distance",
    "relative_entropy",
    "ProcessTomographyResult",
    "process_tomography",
    "chi_matrix",
]
