"""Fermion-to-qubit transforms: Jordan-Wigner and Bravyi-Kitaev."""

from __future__ import annotations

from quantumium.fermion.operators import FermionOperator, QubitOperator


def jordan_wigner(fermion_op: FermionOperator, n_qubits: int | None = None) -> QubitOperator:
    """Transform a FermionOperator to a QubitOperator via Jordan-Wigner."""
    if n_qubits is None:
        n_qubits = fermion_op.num_modes
    if n_qubits == 0:
        n_qubits = 1

    result = QubitOperator.zero()
    for term_key, coeff in fermion_op.terms.items():
        if not term_key:
            result = result + QubitOperator({"I": coeff})
            continue

        term_op = QubitOperator.identity()
        for mode, is_creation in term_key:
            ladder = _jw_ladder(mode, n_qubits, is_creation)
            term_op = term_op * ladder
        result = result + term_op * coeff

    return result.compress()


def _jw_ladder(mode: int, n_qubits: int, is_creation: bool) -> QubitOperator:
    """Build JW qubit operator for a single ladder operator.

    a†_j = (X_j - iY_j)/2  ⊗  Z_{j-1} ⊗ ... ⊗ Z_0
    a_j  = (X_j + iY_j)/2  ⊗  Z_{j-1} ⊗ ... ⊗ Z_0
    """
    x_str = ["I"] * n_qubits
    y_str = ["I"] * n_qubits

    x_str[mode] = "X"
    y_str[mode] = "Y"

    for j in range(mode):
        x_str[j] = "Z"
        y_str[j] = "Z"

    x_term = QubitOperator({"".join(x_str): 0.5})
    if is_creation:
        y_term = QubitOperator({"".join(y_str): -0.5j})
    else:
        y_term = QubitOperator({"".join(y_str): 0.5j})

    return x_term + y_term


def bravyi_kitaev(fermion_op: FermionOperator, n_qubits: int | None = None) -> QubitOperator:
    """Transform a FermionOperator to a QubitOperator via Bravyi-Kitaev."""
    if n_qubits is None:
        n_qubits = fermion_op.num_modes
    if n_qubits == 0:
        n_qubits = 1

    result = QubitOperator.zero()
    for term_key, coeff in fermion_op.terms.items():
        if not term_key:
            result = result + QubitOperator({"I": coeff})
            continue

        term_op = QubitOperator.identity()
        for mode, is_creation in term_key:
            ladder = _bk_ladder(mode, n_qubits, is_creation)
            term_op = term_op * ladder
        result = result + term_op * coeff

    return result.compress()


def _bk_ladder(mode: int, n_qubits: int, is_creation: bool) -> QubitOperator:
    """Build BK qubit operator for a single ladder operator.

    Uses Fenwick tree encoding for O(log n) Pauli weight.
    X-string (A): X on {mode} ∪ U(mode), Z on P(mode)
    Y-string (B): Y on mode, X on U(mode), Z on P(mode) \\ C(mode)
    a†_j = (A - iB) / 2
    a_j  = (A + iB) / 2
    """
    update = _bk_update_set(mode, n_qubits)
    parity = _bk_parity_set(mode)
    children = _bk_children_set(mode)

    # X-string (A): X on {mode} ∪ U, Z on P
    a_str = ["I"] * n_qubits
    a_str[mode] = "X"
    for q in update:
        if q < n_qubits:
            a_str[q] = "X"
    for q in parity:
        if q < n_qubits:
            a_str[q] = "Z"

    # Y-string (B): Y on mode, X on U, Z on P \ C
    b_str = ["I"] * n_qubits
    b_str[mode] = "Y"
    for q in update:
        if q < n_qubits:
            b_str[q] = "X"
    for q in parity:
        if q < n_qubits and q not in children:
            b_str[q] = "Z"

    a_term = QubitOperator({"".join(a_str): 0.5})
    if is_creation:
        b_term = QubitOperator({"".join(b_str): -0.5j})
    else:
        b_term = QubitOperator({"".join(b_str): 0.5j})

    return a_term + b_term


def _bk_update_set(mode: int, n_qubits: int) -> set:
    """Qubits whose parity includes mode — ancestors in Fenwick tree (elements > mode)."""
    result = set()
    j = mode | (mode + 1)
    while j < n_qubits:
        result.add(j)
        j = j | (j + 1)
    return result


def _bk_parity_set(mode: int) -> set:
    """Qubits that store partial parity needed for mode (elements < mode)."""
    result = set()
    if mode == 0:
        return result
    j = mode - 1
    while j >= 0:
        result.add(j)
        j = (j & (j + 1)) - 1
    return result


def _bk_children_set(mode: int) -> set:
    """Qubits covering [g(mode), mode-1] where g(mode) = (mode & (mode+1)).

    These are the children in the Fenwick tree whose ranges are
    subsumed by mode's range.
    """
    result = set()
    g = mode & (mode + 1)
    if g == mode:
        return result
    j = mode - 1
    while j >= g:
        result.add(j)
        j = (j & (j + 1)) - 1
    return result
