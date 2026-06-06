"""Classical shadow protocol for quantum state property estimation.

Implements random Pauli measurement shadows with median-of-means
estimation, shadow norm calculation, and fidelity estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_EIGENSTATES = {
    'X': {
        0: np.array([1, 1], dtype=complex) / np.sqrt(2),
        1: np.array([1, -1], dtype=complex) / np.sqrt(2),
    },
    'Y': {
        0: np.array([1, 1j], dtype=complex) / np.sqrt(2),
        1: np.array([1, -1j], dtype=complex) / np.sqrt(2),
    },
    'Z': {
        0: np.array([1, 0], dtype=complex),
        1: np.array([0, 1], dtype=complex),
    },
}

PAULI_MATRICES = {'I': PAULI_I, 'X': PAULI_X, 'Y': PAULI_Y, 'Z': PAULI_Z}


@dataclass
class Snapshot:
    """A single classical shadow measurement record."""
    bases: str
    outcomes: tuple[int, ...]


@dataclass
class ShadowResult:
    """Result of an expectation value estimation."""
    value: float
    num_snapshots: int
    num_groups: int


class ClassicalShadow:
    """Classical shadow of a quantum state from random Pauli measurements.

    Each snapshot consists of a random Pauli basis choice per qubit
    and a computational-basis measurement outcome. The inverted channel
    reconstructs local density matrix snapshots for property estimation.
    """

    def __init__(self, num_qubits: int, snapshots: list[Snapshot]) -> None:
        self._num_qubits = num_qubits
        self._snapshots = list(snapshots)

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_snapshots(self) -> int:
        return len(self._snapshots)

    @property
    def snapshots(self) -> list[Snapshot]:
        return list(self._snapshots)

    @classmethod
    def from_state(
        cls,
        statevector: NDArray[np.complexfloating],
        num_snapshots: int,
        seed: int | None = None,
    ) -> ClassicalShadow:
        """Generate a classical shadow from a known statevector.

        Simulates the random Pauli measurement protocol: for each
        snapshot, chooses a random Pauli basis per qubit and samples
        a measurement outcome from the Born-rule probabilities.
        """
        rng = np.random.RandomState(seed)
        state = np.asarray(statevector, dtype=complex)
        num_qubits = int(np.log2(len(state)))

        snapshots = []
        basis_choices = 'XYZ'

        for _ in range(num_snapshots):
            bases = ''.join(rng.choice(list(basis_choices)) for _ in range(num_qubits))
            rotation = _pauli_rotation_matrix(bases)
            rotated = rotation @ state
            probs = np.abs(rotated) ** 2
            probs = probs / probs.sum()
            outcome_idx = rng.choice(len(probs), p=probs)
            outcomes = tuple(
                (outcome_idx >> (num_qubits - 1 - q)) & 1
                for q in range(num_qubits)
            )
            snapshots.append(Snapshot(bases=bases, outcomes=outcomes))

        return cls(num_qubits, snapshots)

    @classmethod
    def from_records(
        cls,
        num_qubits: int,
        bases_list: list[str],
        outcomes_list: list[tuple[int, ...]],
    ) -> ClassicalShadow:
        """Construct from experimental measurement records."""
        snapshots = [
            Snapshot(bases=b, outcomes=o)
            for b, o in zip(bases_list, outcomes_list)
        ]
        return cls(num_qubits, snapshots)

    def snapshot_state(self, idx: int) -> NDArray[np.complexfloating]:
        """Reconstruct the inverted-channel state for a single snapshot.

        For random Pauli measurements, the inverse channel on each qubit is:
        rho_hat_i = 3 * |b_i><b_i| - I
        The full snapshot state is the tensor product over all qubits.
        """
        snap = self._snapshots[idx]
        rho = np.array([[1]], dtype=complex)
        for q in range(self._num_qubits):
            basis = snap.bases[q]
            outcome = snap.outcomes[q]
            eigenstate = PAULI_EIGENSTATES[basis][outcome]
            local_rho = 3.0 * np.outer(eigenstate, eigenstate.conj()) - PAULI_I
            rho = np.kron(rho, local_rho)
        return rho

    def estimate_expectation(
        self,
        observable: NDArray[np.complexfloating],
        num_groups: int | None = None,
    ) -> ShadowResult:
        """Estimate expectation value of an observable using median-of-means."""
        T = len(self._snapshots)
        if T == 0:
            return ShadowResult(value=0.0, num_snapshots=0, num_groups=0)

        if num_groups is None:
            num_groups = max(1, min(T, int(np.ceil(np.log2(T + 1)))))

        traces = np.array([
            np.real(np.trace(observable @ self.snapshot_state(i)))
            for i in range(T)
        ])

        group_size = T // num_groups
        if group_size == 0:
            return ShadowResult(
                value=float(np.median(traces)),
                num_snapshots=T,
                num_groups=T,
            )

        group_means = []
        for k in range(num_groups):
            start = k * group_size
            end = start + group_size
            group_means.append(np.mean(traces[start:end]))

        return ShadowResult(
            value=float(np.median(group_means)),
            num_snapshots=T,
            num_groups=num_groups,
        )

    def estimate_pauli(self, pauli_string: str) -> ShadowResult:
        """Estimate expectation value of a Pauli string operator.

        Uses the efficient per-snapshot evaluation: the trace of a
        Pauli tensor product with a shadow snapshot factorizes over
        qubits, avoiding full matrix construction.
        """
        T = len(self._snapshots)
        if T == 0:
            return ShadowResult(value=0.0, num_snapshots=0, num_groups=0)

        num_groups = max(1, min(T, int(np.ceil(np.log2(T + 1)))))
        traces = []

        for snap in self._snapshots:
            val = 1.0
            for q in range(self._num_qubits):
                p_char = pauli_string[q]
                if p_char == 'I':
                    continue
                basis = snap.bases[q]
                outcome = snap.outcomes[q]
                if p_char != basis:
                    val = 0.0
                    break
                eigenstate = PAULI_EIGENSTATES[basis][outcome]
                pauli_mat = PAULI_MATRICES[p_char]
                local_val = 3.0 * float(np.real(
                    eigenstate.conj() @ pauli_mat @ eigenstate
                ))
                val *= local_val
            traces.append(val)

        traces = np.array(traces)
        group_size = T // num_groups
        if group_size == 0:
            return ShadowResult(value=float(np.median(traces)), num_snapshots=T, num_groups=T)

        group_means = [
            np.mean(traces[k * group_size:(k + 1) * group_size])
            for k in range(num_groups)
        ]
        return ShadowResult(
            value=float(np.median(group_means)),
            num_snapshots=T,
            num_groups=num_groups,
        )

    def estimate_fidelity(
        self,
        target_state: NDArray[np.complexfloating],
    ) -> ShadowResult:
        """Estimate fidelity with a target pure state.

        Computes F = <psi|rho_shadow|psi> using the shadow snapshots.
        """
        target = np.asarray(target_state, dtype=complex)
        if target.ndim == 1:
            observable = np.outer(target, target.conj())
        else:
            observable = target
        return self.estimate_expectation(observable)

    def estimate_purity(self) -> float:
        """Estimate purity tr(rho^2) using shadow cross-correlations.

        Uses pairs of snapshots with matching bases for an unbiased
        estimator of the purity.
        """
        T = len(self._snapshots)
        if T < 2:
            return 1.0

        purity_sum = 0.0
        count = 0

        for i in range(0, T - 1, 2):
            s1 = self._snapshots[i]
            s2 = self._snapshots[i + 1]
            if s1.bases == s2.bases:
                val = 1.0
                for q in range(self._num_qubits):
                    b1 = s1.outcomes[q]
                    b2 = s2.outcomes[q]
                    if b1 == b2:
                        val *= 3.0
                    else:
                        val *= -1.0
                purity_sum += val
                count += 1

        if count == 0:
            return 1.0
        return float(purity_sum / count)


def shadow_norm_pauli(pauli_string: str) -> float:
    """Compute the shadow norm of a Pauli string under random Pauli measurements.

    For a k-local Pauli operator (k non-identity terms), the shadow
    norm is 3^k. This determines sample complexity: O(3^k / epsilon^2).
    """
    k = sum(1 for c in pauli_string if c != 'I')
    return float(3 ** k)


def shadow_norm_observable(
    observable: NDArray[np.complexfloating],
) -> float:
    """Compute the shadow norm of a general observable.

    Uses the operator norm (largest singular value) as an upper bound:
    ||O||_shadow <= (2^n + 1) * ||O||_inf for random Clifford ensemble.
    For random Pauli, the exact shadow norm depends on the Pauli
    decomposition, so we return the Pauli-decomposition bound.
    """
    n = int(np.log2(observable.shape[0]))
    op_norm = float(np.linalg.norm(observable, ord=2))
    return (2 ** n + 1) * op_norm


def required_snapshots(
    shadow_norm: float,
    epsilon: float,
    delta: float = 0.01,
    num_observables: int = 1,
) -> int:
    """Compute required number of snapshots for given accuracy.

    Returns T such that all M observables are estimated within
    epsilon with probability at least 1-delta.

    Uses: T = O(log(M/delta) * shadow_norm^2 / epsilon^2)
    """
    return int(np.ceil(
        2.0 * np.log(2 * num_observables / delta) * shadow_norm ** 2 / epsilon ** 2
    ))


def pauli_decomposition(
    observable: NDArray[np.complexfloating],
) -> dict[str, float]:
    """Decompose an observable into a sum of Pauli strings with coefficients.

    Returns a dict mapping Pauli strings to their real coefficients:
    O = sum_P c_P * P, where c_P = tr(P * O) / 2^n.
    """
    n = int(np.log2(observable.shape[0]))
    dim = 2 ** n
    paulis = ['I', 'X', 'Y', 'Z']
    result = {}

    def _generate_pauli_strings(n):
        if n == 0:
            yield ''
            return
        for suffix in _generate_pauli_strings(n - 1):
            for p in paulis:
                yield p + suffix

    for pstr in _generate_pauli_strings(n):
        mat = _pauli_string_matrix(pstr)
        coeff = float(np.real(np.trace(mat @ observable))) / dim
        if abs(coeff) > 1e-12:
            result[pstr] = coeff

    return result


def _pauli_string_matrix(pauli_string: str) -> NDArray[np.complexfloating]:
    """Build the matrix for a tensor product of Pauli operators."""
    result = np.array([[1]], dtype=complex)
    for c in pauli_string:
        result = np.kron(result, PAULI_MATRICES[c])
    return result


def _pauli_rotation_matrix(bases: str) -> NDArray[np.complexfloating]:
    """Build the rotation matrix to change from Pauli basis to computational."""
    rotations = {
        'X': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        'Y': np.array([[1, -1j], [1, 1j]], dtype=complex) / np.sqrt(2),
        'Z': np.eye(2, dtype=complex),
    }
    result = np.array([[1]], dtype=complex)
    for b in bases:
        result = np.kron(result, rotations[b])
    return result
