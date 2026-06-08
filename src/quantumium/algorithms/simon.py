"""Simon's algorithm — finding the hidden period of a black-box function.

Given f: {0,1}^n → {0,1}^n with the promise that there exists s ∈ {0,1}^n
such that f(x) = f(y) iff x ⊕ y ∈ {0, s}, find s.

Classical complexity: Ω(2^(n/2)) queries
Quantum complexity: O(n) queries

Simon's algorithm is historically important as the first example of
an exponential quantum speedup, predating Shor's algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H


@dataclass
class SimonResult:
    """Result container for Simon's algorithm.

    Attributes
    ----------
    secret : str
        The recovered hidden string s as a binary string of length n.
        ``"000...0"`` indicates that f is one-to-one (no nontrivial period).
    equations : list[str]
        The collected linear equations z such that z . s = 0 (mod 2),
        each represented as a binary string of length n.
    num_queries : int
        Total number of quantum oracle queries used.
    """

    secret: str
    equations: list[str]
    num_queries: int


def _simon_oracle(n: int, secret: str) -> Gate:
    """Build the oracle U_f for Simon's problem.

    U_f|x⟩|0⟩ = |x⟩|f(x)⟩ where f(x) = f(x ⊕ s).

    Implementation: f(x) = x if x < x⊕s, else x⊕s.
    This ensures f(x) = f(x⊕s) for all x.
    """
    s = int(secret, 2)
    dim = 2 ** (2 * n)
    matrix = np.zeros((dim, dim), dtype=complex)

    for x in range(2**n):
        xs = x ^ s
        fx = min(x, xs)
        in_idx = (x << n) | 0
        out_idx = (x << n) | fx
        matrix[out_idx, in_idx] = 1.0

    for x in range(2**n):
        xs = x ^ s
        fx = min(x, xs)
        for y in range(1, 2**n):
            in_idx = (x << n) | y
            out_idx = (x << n) | (y ^ fx)
            matrix[out_idx, in_idx] = 1.0

    return Gate(matrix, f"Simon(s={secret})")


def simon_query(n: int, secret: str) -> str:
    """Run one iteration of Simon's algorithm to obtain a single linear equation.

    Prepares a uniform superposition on the input register, applies the Simon
    oracle, and measures the input register to obtain a bitstring z satisfying
    z . s = 0 (mod 2).

    Parameters
    ----------
    n : int
        Number of input bits (each register has n qubits; total circuit uses 2n qubits).
    secret : str
        The hidden string s as a binary string of length n.

    Returns
    -------
    str
        A bitstring z of length n such that z . s = 0 (mod 2).
    """
    total = 2 * n
    qc = QuantumCircuit(total)

    for q in range(n):
        qc.apply(H, q)

    oracle = _simon_oracle(n, secret)
    # Reverse qubit order within each register so oracle's bit ordering
    # matches circuit convention (qubit 0 = LSB)
    qubit_order = list(range(n - 1, -1, -1)) + list(range(total - 1, n - 1, -1))
    qc.apply(oracle, *qubit_order)

    for q in range(n):
        qc.apply(H, q)

    counts = qc.measure_all(shots=1)
    bitstring = list(counts.keys())[0]
    return bitstring[n:]


def _solve_linear_system(equations: list[str], n: int) -> str | None:
    """Solve the system z . s = 0 over GF(2) via Gaussian elimination to find s.

    Parameters
    ----------
    equations : list[str]
        List of binary strings, each of length n, representing linear equations.
    n : int
        Number of bits in the secret string s.

    Returns
    -------
    str or None
        The nontrivial solution s as a binary string, or ``None`` if no
        solution could be determined.
    """
    if not equations:
        return None

    matrix = np.array([[int(b) for b in eq] for eq in equations], dtype=int)

    # Gaussian elimination over GF(2)
    rows, cols = matrix.shape
    pivot_cols = []

    for col in range(cols):
        pivot_row = None
        for row in range(len(pivot_cols), rows):
            if matrix[row, col] == 1:
                pivot_row = row
                break
        if pivot_row is None:
            continue

        matrix[[len(pivot_cols), pivot_row]] = matrix[[pivot_row, len(pivot_cols)]]
        for row in range(rows):
            if row != len(pivot_cols) and matrix[row, col] == 1:
                matrix[row] = (matrix[row] + matrix[len(pivot_cols)]) % 2
        pivot_cols.append(col)

    if len(pivot_cols) >= n:
        return "0" * n

    # Find free variable (not a pivot column)
    free_cols = [c for c in range(n) if c not in pivot_cols]
    if not free_cols:
        return "0" * n

    # Set first free variable to 1, solve for pivots
    s = [0] * n
    s[free_cols[0]] = 1

    for i in range(len(pivot_cols) - 1, -1, -1):
        pc = pivot_cols[i]
        val = 0
        for j in range(pc + 1, n):
            val = (val + matrix[i, j] * s[j]) % 2
        s[pc] = val

    return "".join(str(b) for b in s)


def _verify_period(n: int, secret: str, candidate: str) -> bool:
    """Verify a candidate period by checking f(x) = f(x XOR candidate) for all x."""
    s = int(secret, 2)
    c = int(candidate, 2)
    for x in range(2**n):
        fx = min(x, x ^ s)
        fxc = min(x ^ c, (x ^ c) ^ s)
        if fx != fxc:
            return False
    return True


def simon_algorithm(
    n: int,
    secret: str,
    max_queries: int | None = None,
) -> SimonResult:
    """Run Simon's algorithm to find the hidden period string s.

    Repeatedly queries the oracle to collect linear equations z . s = 0 (mod 2),
    then solves the resulting linear system over GF(2) via Gaussian elimination
    to recover the secret string s.

    Parameters
    ----------
    n : int
        Number of input bits. Each register has n qubits; the oracle maps
        {0,1}^n to {0,1}^n.
    secret : str
        The hidden string s as a binary string of length n. If s = "000...0",
        the function f is one-to-one (no period).
    max_queries : int or None, optional
        Maximum number of quantum queries before giving up. Defaults to 5*n,
        which is sufficient with high probability.

    Returns
    -------
    SimonResult
        A result object containing the recovered secret string, the collected
        equations, and the total number of queries performed.

    Raises
    ------
    ValueError
        If ``len(secret) != n``.
    """
    if len(secret) != n:
        raise ValueError(f"Secret must be {n} bits, got {len(secret)}")
    if max_queries is None:
        max_queries = 5 * n

    equations: list[str] = []
    num_queries = 0

    def _rank(eqs: list[str]) -> int:
        if not eqs:
            return 0
        mat = np.array([[int(b) for b in eq] for eq in eqs], dtype=int)
        pivots = 0
        for col in range(n):
            pivot_row = None
            for row in range(pivots, len(mat)):
                if mat[row, col] == 1:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue
            mat[[pivots, pivot_row]] = mat[[pivot_row, pivots]]
            for row in range(len(mat)):
                if row != pivots and mat[row, col] == 1:
                    mat[row] = (mat[row] + mat[pivots]) % 2
            pivots += 1
        return pivots

    queries_at_full_rank = 0

    for _ in range(max_queries):
        z = simon_query(n, secret)
        num_queries += 1

        if z not in equations and z != "0" * n:
            equations.append(z)

        rank = _rank(equations)

        if rank >= n:
            return SimonResult(
                secret="0" * n,
                equations=equations,
                num_queries=num_queries,
            )

        if rank == n - 1:
            queries_at_full_rank += 1
            if queries_at_full_rank >= n + 2:
                candidate = _solve_linear_system(equations, n)
                if candidate is not None and candidate != "0" * n:
                    if _verify_period(n, secret, candidate):
                        return SimonResult(
                            secret=candidate,
                            equations=equations,
                            num_queries=num_queries,
                        )

    candidate = _solve_linear_system(equations, n) or "0" * n
    if candidate != "0" * n and not _verify_period(n, secret, candidate):
        candidate = "0" * n
    return SimonResult(secret=candidate, equations=equations, num_queries=num_queries)
