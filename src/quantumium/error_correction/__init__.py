"""Quantum error correction codes for fault-tolerant computation.

This package provides implementations of fundamental quantum error correction
codes, including the 3-qubit bit-flip code, the 3-qubit phase-flip code, and
the 9-qubit Shor code. Each code module exposes encode, syndrome measurement,
correction, and decode functions that operate on NumPy statevector arrays.

Modules:
    bit_flip: 3-qubit bit-flip code [[3,1,1]] for correcting single X errors.
    phase_flip: 3-qubit phase-flip code [[3,1,1]] for correcting single Z errors.
    shor_code: 9-qubit Shor code [[9,1,3]] for correcting arbitrary single-qubit errors.
"""

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
