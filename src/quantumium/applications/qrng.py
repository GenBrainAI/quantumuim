"""Quantum Random Number Generation (QRNG).

Generates cryptographically unpredictable random numbers using quantum
measurement of superposition states. The randomness is fundamentally
guaranteed by quantum mechanics — Born's rule ensures each measurement
outcome is intrinsically probabilistic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import H
from ..core.noise import NoiseModel


def random_bits(
    num_bits: int,
    noise_model: NoiseModel | None = None,
) -> str:
    """Generate a string of random bits using quantum measurement.

    Each bit is produced by measuring a qubit in the |+⟩ state.
    Uses batches of qubits for efficiency.
    """
    batch_size = min(num_bits, 8)
    bits = []

    while len(bits) < num_bits:
        remaining = num_bits - len(bits)
        n = min(batch_size, remaining)
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.apply(H, q)

        if noise_model is not None:
            counts = qc.noisy_measure(noise_model, shots=1)
        else:
            counts = qc.measure_all(shots=1)

        bitstring = next(iter(counts.keys()))
        bits.extend(bitstring)

    return "".join(bits[:num_bits])


def random_int(
    low: int,
    high: int,
    noise_model: NoiseModel | None = None,
) -> int:
    """Generate a random integer in [low, high) using quantum randomness.

    Uses rejection sampling to ensure uniform distribution.
    """
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high})")

    range_size = high - low
    num_bits = int(np.ceil(np.log2(range_size))) if range_size > 1 else 1

    while True:
        bits = random_bits(num_bits, noise_model)
        value = int(bits, 2)
        if value < range_size:
            return low + value


def random_float(noise_model: NoiseModel | None = None, precision: int = 32) -> float:
    """Generate a random float in [0, 1) using quantum randomness.

    Args:
        noise_model: Optional noise model.
        precision: Number of random bits (default 32 gives ~9 decimal digits).
    """
    bits = random_bits(precision, noise_model)
    value = int(bits, 2)
    return value / (2**precision)


def random_bytes(
    num_bytes: int,
    noise_model: NoiseModel | None = None,
) -> bytes:
    """Generate random bytes using quantum measurement."""
    bits = random_bits(num_bytes * 8, noise_model)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def random_choice(
    items: list,
    noise_model: NoiseModel | None = None,
) -> object:
    """Select a random item from a list using quantum randomness."""
    if not items:
        raise ValueError("Cannot choose from empty list")
    idx = random_int(0, len(items), noise_model)
    return items[idx]


def random_shuffle(
    items: list,
    noise_model: NoiseModel | None = None,
) -> list:
    """Return a new list with items shuffled using quantum randomness (Fisher-Yates)."""
    result = list(items)
    for i in range(len(result) - 1, 0, -1):
        j = random_int(0, i + 1, noise_model)
        result[i], result[j] = result[j], result[i]
    return result


def entropy_test(num_bits: int = 1000, noise_model: NoiseModel | None = None) -> dict:
    """Run basic statistical tests on quantum random bits.

    Returns a dict with:
    - bits: the generated bitstring
    - ones_ratio: fraction of 1s (should be ~0.5)
    - runs_count: number of runs (consecutive same-bit sequences)
    - longest_run: longest run of same bits
    - min_entropy_estimate: conservative entropy estimate (bits per bit)
    """
    bits = random_bits(num_bits, noise_model)

    ones = bits.count("1")
    ones_ratio = ones / num_bits

    runs = 1
    longest_run = 1
    current_run = 1
    for i in range(1, num_bits):
        if bits[i] == bits[i - 1]:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            runs += 1
            current_run = 1

    p_max = max(ones_ratio, 1 - ones_ratio)
    min_entropy = -np.log2(p_max) if p_max < 1 else 0.0

    return {
        "num_bits": num_bits,
        "ones_ratio": ones_ratio,
        "runs_count": runs,
        "longest_run": longest_run,
        "min_entropy_estimate": min_entropy,
        "bits_preview": bits[:64] + ("..." if num_bits > 64 else ""),
    }
