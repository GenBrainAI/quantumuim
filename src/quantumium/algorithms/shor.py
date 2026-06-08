"""Shor's factoring algorithm — educational implementation.

This implements Shor's algorithm for small numbers (N ≤ 21) using
quantum order-finding via QPE with explicitly constructed modular
exponentiation unitaries. Not intended for large-scale factoring —
the gate matrices are built as full 2^n × 2^n permutations.

Algorithm outline:
1. Pick random a < N, check gcd(a, N) = 1
2. Use QPE to find the order r of a mod N (a^r ≡ 1 mod N)
3. If r is even and a^(r/2) ≢ -1 mod N, then
   gcd(a^(r/2) ± 1, N) gives non-trivial factors
"""

from __future__ import annotations

from math import gcd
from dataclasses import dataclass

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H, X


@dataclass
class ShorResult:
    """Result container for Shor's factoring algorithm.

    Attributes
    ----------
    N : int
        The number that was factored.
    a : int
        The base used for order finding.
    order : int or None
        The order of ``a`` mod ``N`` found by QPE, or None if order-finding
        was not used (e.g., trivial gcd factor found).
    factors : tuple[int, int] or None
        A pair of non-trivial factors ``(p, q)`` such that ``p * q == N``,
        or None if factoring failed.
    phase_estimate : float
        The phase value measured by QPE (0.0 if QPE was not used).
    success : bool
        Whether non-trivial factors were successfully found.
    """

    N: int
    a: int
    order: int | None
    factors: tuple[int, int] | None
    phase_estimate: float
    success: bool


def _mod_exp_unitary(a: int, power: int, N: int, n_work: int) -> Gate:
    """Build the modular exponentiation unitary |x> -> |a^power * x mod N>.

    For computational basis states |x> with x < N, maps to |(a^power * x) mod N>.
    States |x> with x >= N are mapped to themselves (identity on overflow).

    Parameters
    ----------
    a : int
        The base for modular exponentiation.
    power : int
        The exponent (a is raised to this power mod N).
    N : int
        The modulus.
    n_work : int
        Number of work qubits (must satisfy 2^n_work >= N).

    Returns
    -------
    Gate
        A unitary gate on n_work qubits implementing the permutation.
    """
    dim = 2**n_work
    matrix = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        if x < N:
            y = (pow(a, power, N) * x) % N
            matrix[y, x] = 1.0
        else:
            matrix[x, x] = 1.0
    return Gate(matrix, f"ModExp(a={a},p={power},N={N})")


def _controlled_unitary(U: Gate) -> Gate:
    """Build a controlled-U gate.

    The resulting gate acts as identity when the control qubit is |0>
    and applies U to the target register when the control is |1>.

    Parameters
    ----------
    U : Gate
        The unitary gate to control.

    Returns
    -------
    Gate
        A gate on (1 + U.num_qubits) qubits with the control on the first qubit.
    """
    n = U.matrix.shape[0]
    dim = 2 * n
    matrix = np.eye(dim, dtype=complex)
    matrix[n:, n:] = U.matrix
    return Gate(matrix, f"C-{U.name}")


def _num_bits(N: int) -> int:
    """Return the number of bits needed to represent the integer N.

    Parameters
    ----------
    N : int
        A non-negative integer.

    Returns
    -------
    int
        The minimum number of bits required, at least 1.
    """
    return max(N.bit_length(), 1)


def _inverse_qft_inplace(qc: QuantumCircuit, qubits: list[int]) -> None:
    """Apply inverse QFT to specified qubits within an existing circuit.

    Appends SWAP and controlled-phase gates directly to the given circuit,
    modifying it in place.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to modify.
    qubits : list[int]
        Indices of the qubits to which the inverse QFT is applied.
    """
    n = len(qubits)
    for i in range(n // 2):
        j = n - 1 - i
        qi, qj = qubits[i], qubits[j]
        from ..core.gate import SWAP

        qc.apply(SWAP, qi, qj)

    for i in range(n):
        for j in range(i):
            angle = -np.pi / (2 ** (i - j))
            cr = _controlled_phase(angle)
            qc.apply(cr, qubits[j], qubits[i])
        qc.apply(H, qubits[i])


def _controlled_phase(angle: float) -> Gate:
    """Build a controlled phase gate that applies exp(i*angle) to |11>.

    Parameters
    ----------
    angle : float
        The phase angle in radians.

    Returns
    -------
    Gate
        A two-qubit gate that is identity on |00>, |01>, |10> and applies
        exp(i*angle) to |11>.
    """
    matrix = np.eye(4, dtype=complex)
    matrix[3, 3] = np.exp(1j * angle)
    return Gate(matrix, f"CP({angle:.3f})")


def _continued_fraction_order(phase: float, N: int, max_denom: int) -> int | None:
    """Extract a candidate order from a phase estimate using continued fractions.

    Expands the phase as a continued fraction and returns the denominator
    of the best convergent that could be a valid order.

    Parameters
    ----------
    phase : float
        The estimated phase theta in [0, 1), from QPE measurement.
    N : int
        The number being factored (used for range checks).
    max_denom : int
        Maximum allowed denominator for convergents.

    Returns
    -------
    int or None
        A candidate order r, or None if no valid candidate was found.
    """
    if abs(phase) < 1e-10:
        return None

    # Continued fraction expansion
    a_coeffs = []
    x = phase
    for _ in range(20):
        a_coeffs.append(int(x))
        frac = x - int(x)
        if abs(frac) < 1e-10:
            break
        x = 1.0 / frac

    # Build convergents and check which gives valid order
    for k in range(1, len(a_coeffs) + 1):
        # Build convergent p/q from first k coefficients
        p, q = 0, 1
        for i in range(k - 1, -1, -1):
            p, q = q, a_coeffs[i] * q + p
        # Now p/q is the convergent (but swapped)
        p, q = q, p

        if q <= 0 or q > max_denom:
            continue
        if pow(1, 1) and q > 0:
            # Check if q is the order
            if pow(int(round(phase * q)), 1) >= 0:
                pass
        # Verify: a^q ≡ 1 mod N would be checked by caller
        if 0 < q <= max_denom:
            return q

    return None


def quantum_order_finding(
    a: int,
    N: int,
    num_counting_qubits: int | None = None,
) -> tuple[float, int | None]:
    """Find the order of a mod N using quantum phase estimation.

    The order r is the smallest positive integer such that a^r = 1 (mod N).
    This is the quantum subroutine at the heart of Shor's algorithm.

    Parameters
    ----------
    a : int
        The base whose order is to be found. Must satisfy gcd(a, N) = 1.
    N : int
        The modulus.
    num_counting_qubits : int or None, optional
        Number of counting (precision) qubits for QPE. Defaults to
        ``2 * ceil(log2(N)) + 1`` for sufficient precision.

    Returns
    -------
    tuple[float, int | None]
        A pair ``(phase_estimate, candidate_order)`` where ``phase_estimate``
        is the raw phase from QPE and ``candidate_order`` is the extracted
        order r (or None if extraction failed).
    """
    n_work = _num_bits(N)
    if num_counting_qubits is None:
        num_counting_qubits = 2 * n_work + 1

    total = num_counting_qubits + n_work
    qc = QuantumCircuit(total)

    # Initialize work register to |1⟩ (= |00...01⟩)
    qc.apply(X, num_counting_qubits + n_work - 1)

    # Hadamard on counting qubits
    for q in range(num_counting_qubits):
        qc.apply(H, q)

    # Controlled modular exponentiation
    for q in range(num_counting_qubits):
        power = 2 ** (num_counting_qubits - 1 - q)
        mod_exp = _mod_exp_unitary(a, power, N, n_work)
        c_mod_exp = _controlled_unitary(mod_exp)
        work_qubits = list(range(num_counting_qubits, total))
        qc.apply(c_mod_exp, q, *work_qubits)

    # Inverse QFT on counting register
    counting_qubits = list(range(num_counting_qubits))
    _inverse_qft_inplace(qc, counting_qubits)

    # Measure
    counts = qc.measure_all(shots=4096)

    # Extract counting register result
    counting_counts: dict[str, int] = {}
    for bitstring, count in counts.items():
        counting_bits = bitstring[:num_counting_qubits]
        counting_counts[counting_bits] = counting_counts.get(counting_bits, 0) + count

    best = max(counting_counts, key=counting_counts.get)  # type: ignore[arg-type]
    measured_int = int(best, 2)
    phase = measured_int / (2**num_counting_qubits)

    # Find order via continued fractions
    order = _continued_fraction_order(phase, N, N)
    if order is not None and pow(a, order, N) != 1:
        # Try multiples
        for mult in range(2, N):
            candidate = order * mult
            if candidate > N:
                break
            if pow(a, candidate, N) == 1:
                order = candidate
                break
        else:
            order = None

    return phase, order


def shor_factor(
    N: int,
    a: int | None = None,
    num_counting_qubits: int | None = None,
    max_attempts: int = 10,
) -> ShorResult:
    """Factor an integer N using Shor's algorithm.

    Implements the full Shor's factoring pipeline: check for trivial factors
    (even N, gcd), then use quantum order-finding via QPE to extract factors.
    Supports small numbers (N <= 21) due to the explicit unitary construction.

    Parameters
    ----------
    N : int
        The composite number to factor. Must be >= 4 and should not be
        a prime or prime power for meaningful results.
    a : int or None, optional
        Base for order finding. If None, tries values from 2 up to
        ``min(N, max_attempts + 2)``.
    num_counting_qubits : int or None, optional
        Number of precision qubits for QPE. Defaults to
        ``2 * ceil(log2(N)) + 1``.
    max_attempts : int, optional
        Maximum number of bases to try (default 10).

    Returns
    -------
    ShorResult
        Contains the factors if successful (``success=True``), or
        ``factors=None`` if all attempts failed.

    Raises
    ------
    ValueError
        If ``N < 4``.
    """
    if N < 4:
        raise ValueError(f"N must be >= 4, got {N}")
    if N % 2 == 0:
        return ShorResult(
            N=N, a=2, order=None, factors=(2, N // 2), phase_estimate=0.0, success=True
        )

    candidates = [a] if a is not None else list(range(2, min(N, max_attempts + 2)))

    for base in candidates:
        g = gcd(base, N)
        if g > 1:
            return ShorResult(
                N=N, a=base, order=None, factors=(g, N // g), phase_estimate=0.0, success=True
            )

        phase, order = quantum_order_finding(base, N, num_counting_qubits)

        if order is None or order == 0:
            continue

        if order % 2 != 0:
            continue

        half_power = pow(base, order // 2, N)
        if half_power == N - 1:
            continue

        f1 = gcd(half_power + 1, N)
        f2 = gcd(half_power - 1, N)

        if f1 > 1 and f1 < N:
            return ShorResult(
                N=N, a=base, order=order, factors=(f1, N // f1), phase_estimate=phase, success=True
            )
        if f2 > 1 and f2 < N:
            return ShorResult(
                N=N, a=base, order=order, factors=(f2, N // f2), phase_estimate=phase, success=True
            )

    return ShorResult(
        N=N,
        a=candidates[-1] if candidates else 2,
        order=None,
        factors=None,
        phase_estimate=0.0,
        success=False,
    )
