"""Stabilizer tableau for efficient Clifford circuit simulation.

Represents an n-qubit stabilizer state using a (2n+1) × (2n+1) binary
tableau following the Aaronson-Gottesman formalism:

    [ x_11 ... x_1n | z_11 ... z_1n | r_1 ]   ← destabilizers
    [ ...           | ...            | ... ]
    [ x_n1 ... x_nn | z_n1 ... z_nn | r_n ]
    [ x_{n+1,1} ... | z_{n+1,1} ... | r_{n+1}]  ← stabilizers
    [ ...           | ...            | ... ]
    [ x_{2n,1} ...  | z_{2n,1} ...  | r_{2n}]

Row i represents the Pauli: (-1)^{r_i} · X^{x_{i,1}} Z^{z_{i,1}} ⊗ ...

References:
    Aaronson & Gottesman, "Improved simulation of stabilizer circuits"
    (Physical Review A, 70(5), 052328, 2004)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class StabilizerTableau:
    """Binary tableau for stabilizer state simulation.

    The tableau has shape (2n, 2n+1) where n = num_qubits.
    Rows 0..n-1 are destabilizers, rows n..2n-1 are stabilizers.
    Columns 0..n-1 are X bits, n..2n-1 are Z bits, 2n is the phase bit.
    """

    def __init__(self, num_qubits: int) -> None:
        self.n = num_qubits
        # Initialize to |0...0>: destabilizers = X_i, stabilizers = Z_i
        self.table = np.zeros((2 * num_qubits, 2 * num_qubits + 1), dtype=np.uint8)
        for i in range(num_qubits):
            self.table[i, i] = 1              # destabilizer i: X_i
            self.table[num_qubits + i, num_qubits + i] = 1  # stabilizer i: Z_i

    @property
    def num_qubits(self) -> int:
        return self.n

    def copy(self) -> StabilizerTableau:
        t = StabilizerTableau.__new__(StabilizerTableau)
        t.n = self.n
        t.table = self.table.copy()
        return t

    def _x(self, row: int, col: int) -> int:
        return int(self.table[row, col])

    def _z(self, row: int, col: int) -> int:
        return int(self.table[row, self.n + col])

    def _r(self, row: int) -> int:
        return int(self.table[row, 2 * self.n])

    def _set_r(self, row: int, val: int) -> None:
        self.table[row, 2 * self.n] = val & 1

    def stabilizer_generators(self) -> list[str]:
        """Return string representations of the stabilizer generators."""
        result = []
        for i in range(self.n, 2 * self.n):
            result.append(self._row_to_pauli(i))
        return result

    def destabilizer_generators(self) -> list[str]:
        """Return string representations of the destabilizer generators."""
        result = []
        for i in range(self.n):
            result.append(self._row_to_pauli(i))
        return result

    def _row_to_pauli(self, row: int) -> str:
        sign = "-" if self._r(row) else "+"
        paulis = []
        for j in range(self.n):
            x, z = self._x(row, j), self._z(row, j)
            if x == 0 and z == 0:
                paulis.append("I")
            elif x == 1 and z == 0:
                paulis.append("X")
            elif x == 0 and z == 1:
                paulis.append("Z")
            else:
                paulis.append("Y")
        return sign + "".join(paulis)

    def _rowmult(self, target: int, source: int) -> None:
        """Multiply row target by row source (left-multiply Pauli)."""
        phase = self._phase_product(target, source)
        self._set_r(target, (self._r(target) + self._r(source) + phase) % 2)
        for j in range(self.n):
            self.table[target, j] ^= self.table[source, j]
            self.table[target, self.n + j] ^= self.table[source, self.n + j]

    def _phase_product(self, i: int, j: int) -> int:
        """Compute phase exponent when multiplying row i by row j.

        Returns the number of times i contributes to the phase (mod 2).
        The phase from multiplying two Paulis P_a P_b is i^g where
        g = Σ_k f(x_a,z_a, x_b,z_b) and f encodes the Pauli commutation.
        """
        g = 0
        for k in range(self.n):
            xa, za = self._x(i, k), self._z(i, k)
            xb, zb = self._x(j, k), self._z(j, k)
            if xa == 1 and za == 1:      # Y
                g += zb - xb             # Y*I=Y(0), Y*X=Z(-1), Y*Y=I(0), Y*Z=X(1)
            elif xa == 1 and za == 0:    # X
                g += zb * (2 * xb - 1)   # X*I=X(0), X*X=I(0), X*Y=Z(1), X*Z=Y(-1)
            elif xa == 0 and za == 1:    # Z
                g += xb * (1 - 2 * zb)   # Z*I=Z(0), Z*X=Y(1), Z*Y=X(-1), Z*Z=I(0)
        # g mod 4: if g%4 == 0 or 1, phase contribution is 0; if 2 or 3, it's 1
        return (g % 4) // 2


def hadamard(tableau: StabilizerTableau, qubit: int) -> None:
    """Apply Hadamard gate to a qubit: H swaps X↔Z, flips phase for Y."""
    for i in range(2 * tableau.n):
        x = tableau._x(i, qubit)
        z = tableau._z(i, qubit)
        # Phase: r ^= x*z (Y → -Y under H)
        tableau._set_r(i, (tableau._r(i) + x * z) % 2)
        # Swap X and Z
        tableau.table[i, qubit] = z
        tableau.table[i, tableau.n + qubit] = x


def phase_gate(tableau: StabilizerTableau, qubit: int) -> None:
    """Apply S gate: X→Y, Z→Z, Y→-X."""
    for i in range(2 * tableau.n):
        x = tableau._x(i, qubit)
        z = tableau._z(i, qubit)
        # Phase: r ^= x*z
        tableau._set_r(i, (tableau._r(i) + x * z) % 2)
        # Z ^= X (X→Y=XZ, Z→Z)
        tableau.table[i, tableau.n + qubit] = (z ^ x) & 1


def cnot(tableau: StabilizerTableau, control: int, target: int) -> None:
    """Apply CNOT with given control and target qubits."""
    for i in range(2 * tableau.n):
        xc = tableau._x(i, control)
        zc = tableau._z(i, control)
        xt = tableau._x(i, target)
        zt = tableau._z(i, target)
        # Phase: r ^= x_c * z_t * (x_t ^ z_c ^ 1)
        tableau._set_r(i, (tableau._r(i) + xc * zt * (xt ^ zc ^ 1)) % 2)
        # X_target ^= X_control
        tableau.table[i, target] = (xt ^ xc) & 1
        # Z_control ^= Z_target
        tableau.table[i, tableau.n + control] = (zc ^ zt) & 1


def cz(tableau: StabilizerTableau, qubit1: int, qubit2: int) -> None:
    """Apply CZ gate (equivalent to H on target, CNOT, H on target)."""
    hadamard(tableau, qubit2)
    cnot(tableau, qubit1, qubit2)
    hadamard(tableau, qubit2)


def pauli_x(tableau: StabilizerTableau, qubit: int) -> None:
    """Apply Pauli X gate."""
    for i in range(2 * tableau.n):
        z = tableau._z(i, qubit)
        tableau._set_r(i, (tableau._r(i) + z) % 2)


def pauli_y(tableau: StabilizerTableau, qubit: int) -> None:
    """Apply Pauli Y gate."""
    for i in range(2 * tableau.n):
        x = tableau._x(i, qubit)
        z = tableau._z(i, qubit)
        tableau._set_r(i, (tableau._r(i) + x + z) % 2)


def pauli_z(tableau: StabilizerTableau, qubit: int) -> None:
    """Apply Pauli Z gate."""
    for i in range(2 * tableau.n):
        x = tableau._x(i, qubit)
        tableau._set_r(i, (tableau._r(i) + x) % 2)


def measure(
    tableau: StabilizerTableau,
    qubit: int,
    rng: np.random.Generator | None = None,
) -> int:
    """Measure a qubit in the computational basis.

    Implements the Aaronson-Gottesman measurement algorithm.
    Modifies the tableau in place to reflect the post-measurement state.

    Returns:
        0 or 1 measurement outcome.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = tableau.n

    # Check if any stabilizer generator anti-commutes with Z on this qubit
    # (i.e., has X=1 for this qubit)
    p = None
    for i in range(n, 2 * n):
        if tableau._x(i, qubit):
            p = i
            break

    if p is not None:
        # Random outcome case: stabilizer anti-commutes with measurement
        # Set all other rows that anti-commute to commute by row multiplication
        for i in range(2 * n):
            if i != p and tableau._x(i, qubit):
                tableau._rowmult(i, p)

        # Move row p to destabilizer, replace with measurement outcome
        tableau.table[p - n] = tableau.table[p].copy()
        tableau.table[p] = 0
        tableau.table[p, tableau.n + qubit] = 1  # Z_qubit

        outcome = int(rng.integers(0, 2))
        tableau._set_r(p, outcome)
        return outcome
    else:
        # Deterministic outcome case: all stabilizers commute with Z_qubit
        # Compute outcome from destabilizers
        scratch = np.zeros(2 * n + 1, dtype=np.uint8)
        for i in range(n):
            if tableau._x(i, qubit):
                # Multiply scratch by stabilizer row (i + n)
                phase = 0
                for k in range(n):
                    xs, zs = int(scratch[k]), int(scratch[n + k])
                    xr, zr = tableau._x(i + n, k), tableau._z(i + n, k)
                    if xs == 1 and zs == 1:
                        phase += zr - xr
                    elif xs == 1 and zs == 0:
                        phase += zr * (2 * xr - 1)
                    elif xs == 0 and zs == 1:
                        phase += xr * (1 - 2 * zr)
                scratch[2 * n] = (scratch[2 * n] + tableau._r(i + n) + (phase % 4) // 2) % 2
                for k in range(n):
                    scratch[k] ^= tableau.table[i + n, k]
                    scratch[n + k] ^= tableau.table[i + n, n + k]

        return int(scratch[2 * n])


def to_statevector(tableau: StabilizerTableau) -> NDArray[np.complexfloating]:
    """Convert a stabilizer state to its statevector representation.

    Only practical for small numbers of qubits (< ~15).
    """
    n = tableau.n
    dim = 2**n

    # Start with equal superposition, project with each stabilizer
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)

    for s in range(n):
        row = n + s
        # Build the Pauli matrix for this stabilizer
        P = _pauli_matrix(tableau, row)
        sign = (-1) ** tableau._r(row)
        # Project: psi = (I + sign*P)/2 @ psi
        psi = 0.5 * (psi + sign * (P @ psi))
        norm = np.linalg.norm(psi)
        if norm > 1e-15:
            psi /= norm

    return psi


def _pauli_matrix(
    tableau: StabilizerTableau, row: int,
) -> NDArray[np.complexfloating]:
    """Build the full Pauli matrix for a tableau row."""
    n = tableau.n
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    result = np.array([[1]], dtype=complex)
    for j in range(n):
        x, z = tableau._x(row, j), tableau._z(row, j)
        if x == 0 and z == 0:
            op = I2
        elif x == 1 and z == 0:
            op = X
        elif x == 0 and z == 1:
            op = Z
        else:
            op = Y
        result = np.kron(result, op)
    return result
