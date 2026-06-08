"""Quantum Random Number Generation (QRNG).

Generates cryptographically unpredictable random numbers using quantum
measurement of superposition states. The randomness is fundamentally
guaranteed by quantum mechanics — Born's rule ensures each measurement
outcome is intrinsically probabilistic.
"""

from __future__ import annotations

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.gate import H
from ..core.noise import NoiseModel


def random_bits(
    num_bits: int,
    noise_model: NoiseModel | None = None,
) -> str:
    """Generate a string of random bits using quantum measurement.

    Each bit is produced by preparing a qubit in the |+> state
    (via a Hadamard gate) and measuring in the computational basis.
    Qubits are processed in batches for efficiency.

    Args:
        num_bits: Number of random bits to generate. Must be >= 1.
        noise_model: Optional :class:`NoiseModel` to apply during
            measurement, simulating hardware imperfections.

    Returns:
        str: A string of ``'0'`` and ``'1'`` characters of length
            *num_bits*.
    """
    batch_size = min(num_bits, 8)
    bits = []  # type: ignore[var-annotated]

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
    """Generate a uniformly distributed random integer in [low, high).

    Uses rejection sampling on quantum random bits to guarantee a
    perfectly uniform distribution over the requested range, regardless
    of whether the range is a power of two.

    Args:
        low: Inclusive lower bound of the range.
        high: Exclusive upper bound of the range. Must be > *low*.
        noise_model: Optional :class:`NoiseModel` to apply during
            measurement.

    Returns:
        int: A random integer ``r`` satisfying ``low <= r < high``.

    Raises:
        ValueError: If *low* >= *high*.
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
    """Generate a uniformly distributed random float in [0, 1).

    Produces *precision* quantum random bits and interprets them as the
    binary fraction ``0.b_1 b_2 ... b_n``.

    Args:
        noise_model: Optional :class:`NoiseModel` to apply during
            measurement.
        precision: Number of random bits to use. Default is 32,
            yielding approximately 9 decimal digits of precision.

    Returns:
        float: A random value ``f`` satisfying ``0.0 <= f < 1.0``.
    """
    bits = random_bits(precision, noise_model)
    value = int(bits, 2)
    return value / (2**precision)  # type: ignore[no-any-return]


def random_bytes(
    num_bytes: int,
    noise_model: NoiseModel | None = None,
) -> bytes:
    """Generate random bytes using quantum measurement.

    Each byte is assembled from 8 quantum random bits. The result is
    suitable for cryptographic seeding or key material.

    Args:
        num_bytes: Number of random bytes to generate. Must be >= 1.
        noise_model: Optional :class:`NoiseModel` to apply during
            measurement.

    Returns:
        bytes: A bytes object of length *num_bytes*.
    """
    bits = random_bits(num_bytes * 8, noise_model)
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


def random_choice(
    items: list,  # type: ignore[type-arg]
    noise_model: NoiseModel | None = None,
) -> object:
    """Select a random item from a list using quantum randomness.

    Uses :func:`random_int` internally to pick a uniformly random index.

    Args:
        items: Non-empty list of items to choose from.
        noise_model: Optional :class:`NoiseModel` to apply during
            measurement.

    Returns:
        object: A randomly selected element from *items*.

    Raises:
        ValueError: If *items* is empty.
    """
    if not items:
        raise ValueError("Cannot choose from empty list")
    idx = random_int(0, len(items), noise_model)
    return items[idx]


def random_shuffle(
    items: list,  # type: ignore[type-arg]
    noise_model: NoiseModel | None = None,
) -> list:  # type: ignore[type-arg]
    """Return a new list with items shuffled using quantum randomness.

    Implements the Fisher-Yates (Knuth) shuffle algorithm with swap
    indices drawn from quantum random bits, producing a uniformly
    random permutation.

    Args:
        items: List of items to shuffle. The original list is not
            modified.
        noise_model: Optional :class:`NoiseModel` to apply during
            measurement.

    Returns:
        list: A new list containing the same elements in a randomly
            permuted order.
    """
    result = list(items)
    for i in range(len(result) - 1, 0, -1):
        j = random_int(0, i + 1, noise_model)
        result[i], result[j] = result[j], result[i]
    return result


def entropy_test(num_bits: int = 1000, noise_model: NoiseModel | None = None) -> dict:  # type: ignore[type-arg]
    """Run basic statistical tests on quantum random bits.

    Generates *num_bits* quantum random bits and computes several
    summary statistics useful for validating randomness quality.

    Args:
        num_bits: Number of random bits to generate and test.
            Defaults to 1000.
        noise_model: Optional :class:`NoiseModel` to apply during
            measurement.

    Returns:
        dict: A dictionary with the following keys:

            - ``"num_bits"`` (int): Number of bits generated.
            - ``"ones_ratio"`` (float): Fraction of ``'1'`` bits
              (ideally ~0.5).
            - ``"runs_count"`` (int): Number of runs (consecutive
              sequences of identical bits).
            - ``"longest_run"`` (int): Length of the longest run.
            - ``"min_entropy_estimate"`` (float): Conservative
              min-entropy estimate in bits per bit.
            - ``"bits_preview"`` (str): First 64 bits (with ``"..."``
              appended if truncated).
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
