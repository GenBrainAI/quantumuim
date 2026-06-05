"""Quantum random number generation.

Uses Hadamard + measurement to generate truly random bits
from quantum superposition — the theoretical gold standard
for randomness generation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import H


def quantum_random_bits(num_bits: int, seed: int | None = None) -> list[int]:
    """Generate random bits using quantum measurement.

    Each bit comes from measuring a qubit in |+> = H|0> state,
    giving a 50/50 outcome by the Born rule.

    Args:
        num_bits: Number of random bits to generate.
        seed: Optional seed for the measurement sampling RNG.
            In a real quantum system, outcomes are intrinsically random.
            The seed is only used here for simulation reproducibility.

    Returns:
        List of random bits (0 or 1).
    """
    if num_bits < 1:
        raise ValueError("Must generate at least 1 bit")

    bits = []
    rng = np.random.default_rng(seed)

    for _ in range(num_bits):
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        sv = qc.statevector()
        prob_0 = float(abs(sv[0]) ** 2)
        bits.append(0 if rng.random() < prob_0 else 1)

    return bits


def quantum_random_int(low: int, high: int, seed: int | None = None) -> int:
    """Generate a random integer in [low, high) using quantum randomness.

    Args:
        low: Inclusive lower bound.
        high: Exclusive upper bound.
        seed: Optional seed for reproducibility.

    Returns:
        Random integer in [low, high).
    """
    if high <= low:
        raise ValueError(f"high ({high}) must be greater than low ({low})")

    range_size = high - low
    num_bits = int(np.ceil(np.log2(range_size))) if range_size > 1 else 1

    rng_seed = seed
    while True:
        bits = quantum_random_bits(num_bits, seed=rng_seed)
        value = sum(b << i for i, b in enumerate(bits))
        if value < range_size:
            return low + value
        # Rejection sampling: if value is out of range, regenerate
        if rng_seed is not None:
            rng_seed += 1
