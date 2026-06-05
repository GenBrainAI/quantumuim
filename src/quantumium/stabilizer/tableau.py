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
        """Initialize a stabilizer tableau for the |0...0> state.

        The initial state has destabilizers X_i and stabilizers Z_i for
        each qubit i, representing the computational basis state |0...0>.

        Args:
            num_qubits (int): Number of qubits to simulate.
        """
        self.n = num_qubits
        # Initialize to |0...0>: destabilizers = X_i, stabilizers = Z_i
        self.table = np.zeros((2 * num_qubits, 2 * num_qubits + 1), dtype=np.uint8)
        for i in range(num_qubits):
            self.table[i, i] = 1              # destabilizer i: X_i
            self.table[num_qubits + i, num_qubits + i] = 1  # stabilizer i: Z_i

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits in this stabilizer state.

        Returns:
            int: Number of qubits.
        """
        return self.n

    def copy(self) -> StabilizerTableau:
        """Create a deep copy of this tableau.

        Returns:
            StabilizerTableau: A new tableau with independently copied data.
        """
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
        """Return string representations of the n stabilizer generators.

        Each generator is formatted as a signed Pauli string, e.g. "+ZII"
        or "-XYZ".

        Returns:
            list[str]: List of n stabilizer generator strings.
        """
        result = []
        for i in range(self.n, 2 * self.n):
            result.append(self._row_to_pauli(i))
        return result

    def destabilizer_generators(self) -> list[str]:
        """Return string representations of the n destabilizer generators.

        Destabilizers anti-commute with their corresponding stabilizer
        and commute with all other stabilizers.

        Returns:
            list[str]: List of n destabilizer generator strings.
        """
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
    """Apply the Hadamard gate to a qubit in the stabilizer tableau.

    Transforms Pauli operators as: X -> Z, Z -> X, Y -> -Y.
    Modifies the tableau in place.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to modify.
        qubit (int): Index of the target qubit (0-indexed).
    """
    for i in range(2 * tableau.n):
        x = tableau._x(i, qubit)
        z = tableau._z(i, qubit)
        # Phase: r ^= x*z (Y → -Y under H)
        tableau._set_r(i, (tableau._r(i) + x * z) % 2)
        # Swap X and Z
        tableau.table[i, qubit] = z
        tableau.table[i, tableau.n + qubit] = x


def phase_gate(tableau: StabilizerTableau, qubit: int) -> None:
    """Apply the S (phase) gate to a qubit in the stabilizer tableau.

    Transforms Pauli operators as: X -> Y, Z -> Z, Y -> -X.
    Modifies the tableau in place.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to modify.
        qubit (int): Index of the target qubit (0-indexed).
    """
    for i in range(2 * tableau.n):
        x = tableau._x(i, qubit)
        z = tableau._z(i, qubit)
        # Phase: r ^= x*z
        tableau._set_r(i, (tableau._r(i) + x * z) % 2)
        # Z ^= X (X→Y=XZ, Z→Z)
        tableau.table[i, tableau.n + qubit] = (z ^ x) & 1


def cnot(tableau: StabilizerTableau, control: int, target: int) -> None:
    """Apply the CNOT (controlled-X) gate to the stabilizer tableau.

    Transforms Pauli operators as:
    X_c -> X_c X_t, Z_t -> Z_c Z_t, and adjusts phases accordingly.
    Modifies the tableau in place.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to modify.
        control (int): Index of the control qubit (0-indexed).
        target (int): Index of the target qubit (0-indexed).
    """
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
    """Apply the controlled-Z (CZ) gate to two qubits in the tableau.

    Implemented as H on qubit2, CNOT(qubit1, qubit2), H on qubit2.
    Modifies the tableau in place.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to modify.
        qubit1 (int): Index of the first qubit (0-indexed).
        qubit2 (int): Index of the second qubit (0-indexed).
    """
    hadamard(tableau, qubit2)
    cnot(tableau, qubit1, qubit2)
    hadamard(tableau, qubit2)


def pauli_x(tableau: StabilizerTableau, qubit: int) -> None:
    """Apply the Pauli X gate to a qubit in the stabilizer tableau.

    Flips the phase of any stabilizer/destabilizer that contains Z on
    this qubit. Modifies the tableau in place.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to modify.
        qubit (int): Index of the target qubit (0-indexed).
    """
    for i in range(2 * tableau.n):
        z = tableau._z(i, qubit)
        tableau._set_r(i, (tableau._r(i) + z) % 2)


def pauli_y(tableau: StabilizerTableau, qubit: int) -> None:
    """Apply the Pauli Y gate to a qubit in the stabilizer tableau.

    Flips the phase of any stabilizer/destabilizer that contains X or Z
    (but not both) on this qubit. Modifies the tableau in place.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to modify.
        qubit (int): Index of the target qubit (0-indexed).
    """
    for i in range(2 * tableau.n):
        x = tableau._x(i, qubit)
        z = tableau._z(i, qubit)
        tableau._set_r(i, (tableau._r(i) + x + z) % 2)


def pauli_z(tableau: StabilizerTableau, qubit: int) -> None:
    """Apply the Pauli Z gate to a qubit in the stabilizer tableau.

    Flips the phase of any stabilizer/destabilizer that contains X on
    this qubit. Modifies the tableau in place.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to modify.
        qubit (int): Index of the target qubit (0-indexed).
    """
    for i in range(2 * tableau.n):
        x = tableau._x(i, qubit)
        tableau._set_r(i, (tableau._r(i) + x) % 2)


def measure(
    tableau: StabilizerTableau,
    qubit: int,
    rng: np.random.Generator | None = None,
) -> int:
    """Measure a qubit in the computational basis.

    Implements the Aaronson-Gottesman measurement algorithm. If the
    outcome is random (some stabilizer anti-commutes with Z on this
    qubit), a random bit is chosen. If deterministic, the outcome is
    computed from the destabilizer rows. Modifies the tableau in place
    to reflect the post-measurement state.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to modify.
        qubit (int): Index of the qubit to measure (0-indexed).
        rng (np.random.Generator | None): Random number generator for
            stochastic outcomes. If None, a default RNG is created.

    Returns:
        int: Measurement outcome, either 0 or 1.
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
    """Convert a stabilizer state to its full statevector representation.

    Projects the uniform superposition through each stabilizer generator
    to recover the unique stabilizer state. Only practical for small
    numbers of qubits (fewer than about 15) due to exponential memory.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to convert.

    Returns:
        NDArray[np.complexfloating]: Normalized statevector of length 2^n.
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
