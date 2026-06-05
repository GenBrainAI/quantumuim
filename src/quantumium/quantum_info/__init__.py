"""Quantum information theory toolkit.

Channel representations, distance measures, and process tomography.
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
