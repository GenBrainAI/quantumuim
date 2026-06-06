"""Fermionic and qubit operator algebra."""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional


class FermionOperator:
    """Sum of products of fermionic ladder operators with complex coefficients.

    Each term is keyed by a tuple of (mode_index, is_creation) pairs.
    Example: a†_0 a_1 -> key ((0, True), (1, False))
    """

    def __init__(self, terms: Optional[Dict[Tuple, complex]] = None):
        self._terms: Dict[Tuple, complex] = {}
        if terms:
            for key, coeff in terms.items():
                if abs(coeff) > 1e-15:
                    self._terms[key] = complex(coeff)

    @staticmethod
    def creation(mode: int) -> FermionOperator:
        return FermionOperator({((mode, True),): 1.0})

    @staticmethod
    def annihilation(mode: int) -> FermionOperator:
        return FermionOperator({((mode, False),): 1.0})

    @staticmethod
    def identity() -> FermionOperator:
        return FermionOperator({(): 1.0})

    @staticmethod
    def zero() -> FermionOperator:
        return FermionOperator()

    @property
    def terms(self) -> Dict[Tuple, complex]:
        return dict(self._terms)

    @property
    def num_modes(self) -> int:
        modes = set()
        for key in self._terms:
            for mode, _ in key:
                modes.add(mode)
        return max(modes) + 1 if modes else 0

    @property
    def is_hermitian(self) -> bool:
        diff = (self - self.dagger()).normal_order().compress()
        return all(abs(v) < 1e-10 for v in diff._terms.values())

    def dagger(self) -> FermionOperator:
        result = {}
        for key, coeff in self._terms.items():
            new_key = tuple((mode, not is_creation) for mode, is_creation in reversed(key))
            result[new_key] = result.get(new_key, 0) + np.conj(coeff)
        return FermionOperator(result)

    def normal_order(self) -> FermionOperator:
        result = FermionOperator()
        for key, coeff in self._terms.items():
            ordered = _normal_order_term(key, coeff)
            result = result + ordered
        return result.compress()

    def compress(self, tol: float = 1e-15) -> FermionOperator:
        return FermionOperator({k: v for k, v in self._terms.items() if abs(v) > tol})

    def __add__(self, other: FermionOperator) -> FermionOperator:
        result = dict(self._terms)
        for key, coeff in other._terms.items():
            result[key] = result.get(key, 0) + coeff
        return FermionOperator(result)

    def __sub__(self, other: FermionOperator) -> FermionOperator:
        result = dict(self._terms)
        for key, coeff in other._terms.items():
            result[key] = result.get(key, 0) - coeff
        return FermionOperator(result)

    def __mul__(self, other) -> FermionOperator:
        if isinstance(other, (int, float, complex)):
            return FermionOperator({k: v * other for k, v in self._terms.items()})
        if isinstance(other, FermionOperator):
            result = {}
            for k1, c1 in self._terms.items():
                for k2, c2 in other._terms.items():
                    key = k1 + k2
                    result[key] = result.get(key, 0) + c1 * c2
            return FermionOperator(result)
        return NotImplemented

    def __rmul__(self, other) -> FermionOperator:
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self) -> FermionOperator:
        return self * (-1)

    def __eq__(self, other) -> bool:
        if not isinstance(other, FermionOperator):
            return NotImplemented
        s = self.compress()
        o = other.compress()
        if set(s._terms.keys()) != set(o._terms.keys()):
            return False
        return all(abs(s._terms[k] - o._terms[k]) < 1e-10 for k in s._terms)

    def __repr__(self) -> str:
        if not self._terms:
            return "FermionOperator(0)"
        parts = []
        for key, coeff in self._terms.items():
            if not key:
                parts.append(f"{coeff}")
            else:
                ops = " ".join(
                    f"a{'†' if cr else ''}_{m}" for m, cr in key
                )
                parts.append(f"{coeff} [{ops}]")
        return " + ".join(parts)


def _normal_order_term(key: Tuple, coeff: complex) -> FermionOperator:
    """Normal-order a single term using bubble sort with anti-commutation."""
    ops = list(key)
    result = FermionOperator()
    current_coeff = coeff

    changed = True
    while changed:
        changed = False
        for i in range(len(ops) - 1):
            m_left, cr_left = ops[i]
            m_right, cr_right = ops[i + 1]

            if cr_left and cr_right:
                if m_left > m_right:
                    ops[i], ops[i + 1] = ops[i + 1], ops[i]
                    current_coeff = -current_coeff
                    changed = True
                elif m_left == m_right:
                    return result
            elif not cr_left and not cr_right:
                if m_left < m_right:
                    ops[i], ops[i + 1] = ops[i + 1], ops[i]
                    current_coeff = -current_coeff
                    changed = True
                elif m_left == m_right:
                    return result
            elif not cr_left and cr_right:
                # a_i a†_j = delta_ij - a†_j a_i
                if m_left == m_right:
                    # a_i a†_i = 1 - a†_i a_i
                    remaining = ops[:i] + ops[i + 2:]
                    identity_term = _normal_order_term(tuple(remaining), current_coeff)
                    result = result + identity_term
                    ops[i], ops[i + 1] = ops[i + 1], ops[i]
                    current_coeff = -current_coeff
                    changed = True
                else:
                    ops[i], ops[i + 1] = ops[i + 1], ops[i]
                    current_coeff = -current_coeff
                    changed = True

    result = result + FermionOperator({tuple(ops): current_coeff})
    return result


PAULI_MULT = {
    ("I", "I"): ("I", 1),
    ("I", "X"): ("X", 1),
    ("I", "Y"): ("Y", 1),
    ("I", "Z"): ("Z", 1),
    ("X", "I"): ("X", 1),
    ("X", "X"): ("I", 1),
    ("X", "Y"): ("Z", 1j),
    ("X", "Z"): ("Y", -1j),
    ("Y", "I"): ("Y", 1),
    ("Y", "X"): ("Z", -1j),
    ("Y", "Y"): ("I", 1),
    ("Y", "Z"): ("X", 1j),
    ("Z", "I"): ("Z", 1),
    ("Z", "X"): ("Y", 1j),
    ("Z", "Y"): ("X", -1j),
    ("Z", "Z"): ("I", 1),
}


def _multiply_pauli_strings(s1: str, s2: str) -> Tuple[str, complex]:
    """Multiply two Pauli strings, return (result_string, phase)."""
    n = max(len(s1), len(s2))
    s1 = s1.ljust(n, "I")
    s2 = s2.ljust(n, "I")
    result = []
    phase = 1
    for a, b in zip(s1, s2):
        r, p = PAULI_MULT[(a, b)]
        result.append(r)
        phase *= p
    return "".join(result), phase


class QubitOperator:
    """Sum of Pauli string products with complex coefficients.

    Each term is keyed by a Pauli string like "XYZII".
    """

    def __init__(self, terms: Optional[Dict[str, complex]] = None):
        self._terms: Dict[str, complex] = {}
        if terms:
            for key, coeff in terms.items():
                if abs(coeff) > 1e-15:
                    self._terms[key] = complex(coeff)

    @staticmethod
    def identity() -> QubitOperator:
        return QubitOperator({"I": 1.0})

    @staticmethod
    def zero() -> QubitOperator:
        return QubitOperator()

    @property
    def terms(self) -> Dict[str, complex]:
        return dict(self._terms)

    @property
    def num_qubits(self) -> int:
        if not self._terms:
            return 0
        return max(len(k) for k in self._terms)

    def compress(self, tol: float = 1e-15) -> QubitOperator:
        result = {}
        for key, coeff in self._terms.items():
            stripped = key.rstrip("I")
            if not stripped:
                stripped = "I"
            result[stripped] = result.get(stripped, 0) + coeff
        return QubitOperator({k: v for k, v in result.items() if abs(v) > tol})

    def to_matrix(self, n_qubits: Optional[int] = None) -> np.ndarray:
        """Build the full 2^n x 2^n matrix representation."""
        if n_qubits is None:
            n_qubits = self.num_qubits
        if n_qubits == 0:
            n_qubits = 1

        pauli_matrices = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }

        dim = 2**n_qubits
        result = np.zeros((dim, dim), dtype=complex)

        for pauli_str, coeff in self._terms.items():
            padded = pauli_str.ljust(n_qubits, "I")
            mat = np.array([[1.0]], dtype=complex)
            for ch in padded:
                mat = np.kron(mat, pauli_matrices[ch])
            result += coeff * mat

        return result

    def __add__(self, other: QubitOperator) -> QubitOperator:
        result = dict(self._terms)
        for key, coeff in other._terms.items():
            result[key] = result.get(key, 0) + coeff
        return QubitOperator(result)

    def __sub__(self, other: QubitOperator) -> QubitOperator:
        result = dict(self._terms)
        for key, coeff in other._terms.items():
            result[key] = result.get(key, 0) - coeff
        return QubitOperator(result)

    def __mul__(self, other) -> QubitOperator:
        if isinstance(other, (int, float, complex)):
            return QubitOperator({k: v * other for k, v in self._terms.items()})
        if isinstance(other, QubitOperator):
            result = {}
            for k1, c1 in self._terms.items():
                for k2, c2 in other._terms.items():
                    key, phase = _multiply_pauli_strings(k1, k2)
                    result[key] = result.get(key, 0) + c1 * c2 * phase
            return QubitOperator(result)
        return NotImplemented

    def __rmul__(self, other) -> QubitOperator:
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self) -> QubitOperator:
        return self * (-1)

    def __eq__(self, other) -> bool:
        if not isinstance(other, QubitOperator):
            return NotImplemented
        s = self.compress()
        o = other.compress()
        if set(s._terms.keys()) != set(o._terms.keys()):
            return False
        return all(abs(s._terms[k] - o._terms[k]) < 1e-10 for k in s._terms)

    def __repr__(self) -> str:
        if not self._terms:
            return "QubitOperator(0)"
        parts = []
        for key, coeff in self._terms.items():
            parts.append(f"{coeff} [{key}]")
        return " + ".join(parts)
