"""Quantum error correction codes for fault-tolerant computation."""

from .bit_flip import (
    bit_flip_encode,
    bit_flip_syndrome,
    bit_flip_correct,
    bit_flip_decode,
)
from .phase_flip import (
    phase_flip_encode,
    phase_flip_syndrome,
    phase_flip_correct,
    phase_flip_decode,
)
from .shor_code import (
    shor_encode,
    shor_syndrome,
    shor_correct,
    shor_decode,
)

__all__ = [
    "bit_flip_encode",
    "bit_flip_syndrome",
    "bit_flip_correct",
    "bit_flip_decode",
    "phase_flip_encode",
    "phase_flip_syndrome",
    "phase_flip_correct",
    "phase_flip_decode",
    "shor_encode",
    "shor_syndrome",
    "shor_correct",
    "shor_decode",
]
