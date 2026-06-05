"""Quantum channel representations and conversions.

Supports four equivalent representations of completely positive trace-preserving
(CPTP) maps and conversions between them:

- **Kraus**: E(rho) = sum_k K_k rho K_k^dag
- **Choi**: J(E) = sum_{i,j} |i><j| ⊗ E(|i><j|)  (Choi-Jamiołkowski isomorphism)
- **Superoperator**: vec(E(rho)) = S vec(rho)  (column-stacking vectorization)
- **PTM**: Pauli Transfer Matrix, real representation in the Pauli basis

References:
    Watrous, "The Theory of Quantum Information" (2018)
    Wood et al., "Tensor networks and graphical calculus for open quantum
    systems" (arXiv:1111.6950)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


def _pauli_basis(num_qubits: int) -> list[NDArray[np.complexfloating]]:
    """Build normalized n-qubit Pauli basis {I, X, Y, Z}^⊗n / sqrt(d)."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    single = [I, X, Y, Z]

    if num_qubits == 1:
        return [p / np.sqrt(2) for p in single]

    basis = [np.array([[1]], dtype=complex)]
    for _ in range(num_qubits):
        basis = [np.kron(b, p) for b in basis for p in single]

    d = 2**num_qubits
    return [b / np.sqrt(d) for b in basis]


def kraus_to_choi(
    kraus_ops: list[NDArray[np.complexfloating]],
) -> NDArray[np.complexfloating]:
    """Convert Kraus operators to the Choi matrix.

    J(E) = sum_k (I ⊗ K_k) |Ω><Ω| (I ⊗ K_k)^dag
    where |Ω> = sum_i |i,i> / sqrt(d) is the maximally entangled state.

    Args:
        kraus_ops: List of Kraus operator matrices (d x d).

    Returns:
        Choi matrix (d^2 x d^2).
    """
    d = kraus_ops[0].shape[0]
    choi = np.zeros((d * d, d * d), dtype=complex)

    for i in range(d):
        for j in range(d):
            eij = np.zeros((d, d), dtype=complex)
            eij[i, j] = 1.0
            mapped = sum(K @ eij @ K.conj().T for K in kraus_ops)
            choi += np.kron(eij, mapped)

    return choi


def choi_to_kraus(
    choi: NDArray[np.complexfloating],
    tol: float = 1e-10,
) -> list[NDArray[np.complexfloating]]:
    """Convert a Choi matrix to Kraus operators via eigendecomposition.

    Args:
        choi: Choi matrix (d^2 x d^2).
        tol: Eigenvalue threshold for discarding zero operators.

    Returns:
        List of Kraus operator matrices.
    """
    d2 = choi.shape[0]
    d = int(np.sqrt(d2))

    eigvals, eigvecs = np.linalg.eigh(choi)
    kraus_ops = []
    for val, vec in zip(eigvals, eigvecs.T):
        if val > tol:
            K = np.sqrt(val) * vec.reshape(d, d).T
            kraus_ops.append(K)

    return kraus_ops


def kraus_to_superoperator(
    kraus_ops: list[NDArray[np.complexfloating]],
) -> NDArray[np.complexfloating]:
    """Convert Kraus operators to the superoperator (Liouville) representation.

    S = sum_k K_k ⊗ K_k*

    Using column-stacking vectorization: vec(E(rho)) = S vec(rho).

    Args:
        kraus_ops: List of Kraus operator matrices.

    Returns:
        Superoperator matrix (d^2 x d^2).
    """
    d = kraus_ops[0].shape[0]
    S = np.zeros((d * d, d * d), dtype=complex)
    for K in kraus_ops:
        S += np.kron(K, K.conj())
    return S


def superoperator_to_kraus(
    S: NDArray[np.complexfloating],
    tol: float = 1e-10,
) -> list[NDArray[np.complexfloating]]:
    """Convert superoperator to Kraus operators via Choi intermediate.

    Args:
        S: Superoperator matrix (d^2 x d^2).
        tol: Eigenvalue threshold.

    Returns:
        List of Kraus operator matrices.
    """
    d2 = S.shape[0]
    d = int(np.sqrt(d2))

    # S -> Choi via reshuffle: J_{(ij),(kl)} = S_{(ik),(jl)}
    choi = np.zeros((d2, d2), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    choi[i * d + j, k * d + l] = S[i * d + k, j * d + l]

    return choi_to_kraus(choi, tol)


def kraus_to_ptm(
    kraus_ops: list[NDArray[np.complexfloating]],
) -> NDArray[np.floating]:
    """Convert Kraus operators to the Pauli Transfer Matrix (PTM).

    R_{ij} = Tr(P_i E(P_j)) where P_i are normalized Pauli operators.

    The PTM is always real for CPTP maps.

    Args:
        kraus_ops: List of Kraus operator matrices.

    Returns:
        Real-valued PTM (d^2 x d^2).
    """
    d = kraus_ops[0].shape[0]
    num_qubits = int(np.log2(d))
    paulis = _pauli_basis(num_qubits)
    n = len(paulis)

    ptm = np.zeros((n, n), dtype=float)
    for i, Pi in enumerate(paulis):
        for j, Pj in enumerate(paulis):
            val = sum(
                np.trace(Pi @ K @ Pj @ K.conj().T) for K in kraus_ops
            )
            ptm[i, j] = float(np.real(val))

    return ptm


def compose_channels(
    kraus_a: list[NDArray[np.complexfloating]],
    kraus_b: list[NDArray[np.complexfloating]],
) -> list[NDArray[np.complexfloating]]:
    """Compose two quantum channels: (B ∘ A)(rho) = B(A(rho)).

    The composed Kraus operators are {L_j K_i} for all i, j.

    Args:
        kraus_a: Kraus operators for channel A (applied first).
        kraus_b: Kraus operators for channel B (applied second).

    Returns:
        Kraus operators for the composed channel.
    """
    return [Lb @ Ka for Ka in kraus_a for Lb in kraus_b]


def tensor_channels(
    kraus_a: list[NDArray[np.complexfloating]],
    kraus_b: list[NDArray[np.complexfloating]],
) -> list[NDArray[np.complexfloating]]:
    """Tensor product of two channels: (A ⊗ B)(rho_AB).

    Args:
        kraus_a: Kraus operators for channel A.
        kraus_b: Kraus operators for channel B.

    Returns:
        Kraus operators for the tensor product channel.
    """
    return [np.kron(Ka, Kb) for Ka in kraus_a for Kb in kraus_b]


def channel_fidelity(
    kraus_ops: list[NDArray[np.complexfloating]],
) -> float:
    """Compute the average fidelity of a channel with the identity.

    F_avg = (d * F_ent + 1) / (d + 1)
    where F_ent = <Ω| J(E) |Ω> / d is the entanglement fidelity
    and |Ω> is the maximally entangled state.

    Args:
        kraus_ops: Kraus operators for the channel.

    Returns:
        Average fidelity in [0, 1].
    """
    d = kraus_ops[0].shape[0]
    f_ent = sum(abs(np.trace(K)) ** 2 for K in kraus_ops) / d**2
    return float((d * f_ent + 1) / (d + 1))


@dataclass
class QuantumChannel:
    """A quantum channel with lazy conversion between representations.

    Create from any representation; others are computed on demand.
    """

    _kraus: list[NDArray[np.complexfloating]] | None = field(default=None, repr=False)
    _choi: NDArray[np.complexfloating] | None = field(default=None, repr=False)
    _superop: NDArray[np.complexfloating] | None = field(default=None, repr=False)
    _ptm: NDArray[np.floating] | None = field(default=None, repr=False)
    dim: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        if self._kraus is not None:
            self.dim = self._kraus[0].shape[0]
        elif self._choi is not None:
            self.dim = int(np.sqrt(self._choi.shape[0]))
        elif self._superop is not None:
            self.dim = int(np.sqrt(self._superop.shape[0]))

    @classmethod
    def from_kraus(
        cls, kraus_ops: list[NDArray[np.complexfloating]], name: str = ""
    ) -> QuantumChannel:
        return cls(_kraus=kraus_ops, name=name)

    @classmethod
    def from_choi(
        cls, choi: NDArray[np.complexfloating], name: str = ""
    ) -> QuantumChannel:
        return cls(_choi=choi, name=name)

    @classmethod
    def from_superoperator(
        cls, superop: NDArray[np.complexfloating], name: str = ""
    ) -> QuantumChannel:
        return cls(_superop=superop, name=name)

    @classmethod
    def from_unitary(
        cls, U: NDArray[np.complexfloating], name: str = ""
    ) -> QuantumChannel:
        return cls(_kraus=[np.asarray(U, dtype=complex)], name=name)

    @property
    def kraus(self) -> list[NDArray[np.complexfloating]]:
        if self._kraus is None:
            if self._choi is not None:
                self._kraus = choi_to_kraus(self._choi)
            elif self._superop is not None:
                self._kraus = superoperator_to_kraus(self._superop)
            else:
                raise ValueError("No representation available")
        return self._kraus

    @property
    def choi(self) -> NDArray[np.complexfloating]:
        if self._choi is None:
            self._choi = kraus_to_choi(self.kraus)
        return self._choi

    @property
    def superoperator(self) -> NDArray[np.complexfloating]:
        if self._superop is None:
            self._superop = kraus_to_superoperator(self.kraus)
        return self._superop

    @property
    def ptm(self) -> NDArray[np.floating]:
        if self._ptm is None:
            self._ptm = kraus_to_ptm(self.kraus)
        return self._ptm

    def apply(self, rho: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        """Apply this channel to a density matrix."""
        return sum(K @ rho @ K.conj().T for K in self.kraus)

    def compose(self, other: QuantumChannel) -> QuantumChannel:
        """Return self ∘ other (other applied first)."""
        return QuantumChannel.from_kraus(
            compose_channels(other.kraus, self.kraus),
            name=f"{self.name}∘{other.name}" if self.name and other.name else "",
        )

    def tensor(self, other: QuantumChannel) -> QuantumChannel:
        """Return self ⊗ other."""
        return QuantumChannel.from_kraus(
            tensor_channels(self.kraus, other.kraus),
            name=f"{self.name}⊗{other.name}" if self.name and other.name else "",
        )

    def average_fidelity(self) -> float:
        """Average fidelity with the identity channel."""
        return channel_fidelity(self.kraus)

    def is_unital(self, tol: float = 1e-10) -> bool:
        """Check if E(I) = I (channel preserves the identity)."""
        d = self.dim
        I = np.eye(d, dtype=complex)
        result = self.apply(I)
        return bool(np.allclose(result, I, atol=tol))

    def __repr__(self) -> str:
        label = f"'{self.name}'" if self.name else "unnamed"
        return f"QuantumChannel({label}, dim={self.dim})"
