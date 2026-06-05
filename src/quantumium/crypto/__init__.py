"""Quantum cryptography -- key distribution protocols and primitives.

This package provides quantum cryptographic building blocks:

- **bb84**: Simulation of the Bennett-Brassard 1984 (BB84) quantum key
  distribution protocol, including optional eavesdropper modelling and
  QBER-based eavesdropper detection.
- **qrng**: Quantum random number generation using Hadamard-based
  superposition and measurement, suitable for cryptographic key material.
"""

from .bb84 import bb84_protocol, BB84Result
from .qrng import quantum_random_bits, quantum_random_int

__all__ = [
    "bb84_protocol",
    "BB84Result",
    "quantum_random_bits",
    "quantum_random_int",
]
