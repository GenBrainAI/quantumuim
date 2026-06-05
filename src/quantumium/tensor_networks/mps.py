"""Matrix Product State (MPS) representation.

An n-qubit state |ψ⟩ is represented as a chain of tensors:

    |ψ⟩ = Σ A[0]^{s_0} · A[1]^{s_1} · ... · A[n-1]^{s_{n-1}} |s_0 s_1...s_{n-1}⟩

where each A[k]^{s_k} is a (χ_{k-1} × χ_k) matrix for physical index s_k ∈ {0,1}.
The bond dimension χ controls the trade-off between accuracy and efficiency.

For product states χ=1; for maximally entangled states χ can grow up to 2^(n/2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray


@dataclass
class MPS:
    """Matrix Product State for n qubits.

    Attributes:
        tensors: List of n tensors, each of shape (χ_left, 2, χ_right).
            The middle index is the physical (qubit) dimension.
        num_qubits: Number of qubits.
        max_bond_dim: Maximum allowed bond dimension for truncation.
    """
    tensors: list[NDArray[np.complexfloating]]
    num_qubits: int
    max_bond_dim: int | None = None

    def copy(self) -> MPS:
        return MPS(
            tensors=[t.copy() for t in self.tensors],
            num_qubits=self.num_qubits,
            max_bond_dim=self.max_bond_dim,
        )

    def bond_dimensions(self) -> list[int]:
        """Return the bond dimensions χ_0, χ_1, ..., χ_{n-1}."""
        return [self.tensors[i].shape[2] for i in range(self.num_qubits - 1)]

    def max_bond(self) -> int:
        dims = self.bond_dimensions()
        return max(dims) if dims else 1

    def to_statevector(self) -> NDArray[np.complexfloating]:
        """Contract the MPS into a full statevector (2^n amplitudes).

        Only feasible for small n (≤ ~20).
        """
        n = self.num_qubits
        if n > 25:
            raise ValueError(f"Cannot convert {n}-qubit MPS to statevector (too large)")

        # Start with the first tensor: shape (1, 2, χ_1)
        # Reshape to (2, χ_1)
        result = self.tensors[0][0, :, :]  # (2, χ_1)

        for k in range(1, n):
            # result shape: (2^k, χ_k)
            # tensor[k] shape: (χ_k, 2, χ_{k+1})
            T = self.tensors[k]  # (χ_k, 2, χ_{k+1})
            dim_left = result.shape[0]
            chi_k = result.shape[1]

            # Contract: result @ T reshaped
            # result[i, α] * T[α, s, β] → new[i, s, β] → new[i*2+s, β]
            new_result = np.einsum("ia,asb->isb", result, T)
            result = new_result.reshape(dim_left * 2, T.shape[2])

        # Final result shape: (2^n, 1) or (2^n, χ_n)
        return result.flatten()

    def norm(self) -> float:
        """Compute the norm of the MPS state."""
        sv = self.to_statevector()
        return float(np.linalg.norm(sv))

    def normalize(self) -> None:
        """Normalize the MPS in-place."""
        n = self.norm()
        if n > 1e-15:
            self.tensors[0] = self.tensors[0] / n


def mps_zero_state(num_qubits: int, max_bond_dim: int | None = None) -> MPS:
    """Create an MPS representing |00...0⟩.

    Each tensor is shape (1, 2, 1) with A^0 = [[1]], A^1 = [[0]].
    """
    tensors = []
    for _ in range(num_qubits):
        T = np.zeros((1, 2, 1), dtype=complex)
        T[0, 0, 0] = 1.0  # |0⟩
        tensors.append(T)

    return MPS(tensors=tensors, num_qubits=num_qubits, max_bond_dim=max_bond_dim)


def mps_product_state(
    states: list[NDArray[np.complexfloating]],
    max_bond_dim: int | None = None,
) -> MPS:
    """Create an MPS from a list of single-qubit states.

    Args:
        states: List of n two-element arrays [α, β] for each qubit.
        max_bond_dim: Maximum bond dimension for truncation.
    """
    tensors = []
    for s in states:
        s = np.asarray(s, dtype=complex)
        if s.shape != (2,):
            raise ValueError(f"Each state must be a 2-element array, got shape {s.shape}")
        T = np.zeros((1, 2, 1), dtype=complex)
        T[0, 0, 0] = s[0]
        T[0, 1, 0] = s[1]
        tensors.append(T)

    return MPS(tensors=tensors, num_qubits=len(states), max_bond_dim=max_bond_dim)


def mps_from_statevector(
    sv: NDArray[np.complexfloating],
    max_bond_dim: int | None = None,
) -> MPS:
    """Convert a statevector to MPS form via sequential SVD.

    The statevector is reshaped and decomposed site-by-site using SVD,
    optionally truncating to max_bond_dim.

    Args:
        sv: Statevector of length 2^n.
        max_bond_dim: Maximum bond dimension (None = exact).
    """
    sv = np.asarray(sv, dtype=complex)
    n = int(np.log2(len(sv)))
    if 2**n != len(sv):
        raise ValueError(f"Statevector length {len(sv)} is not a power of 2")

    tensors = []
    remainder = sv.reshape(1, -1)  # (1, 2^n)

    for k in range(n - 1):
        # Reshape to (chi_k * 2, 2^(n-k-1))
        chi_left = remainder.shape[0]
        remainder = remainder.reshape(chi_left * 2, -1)

        # SVD
        U, S, Vh = np.linalg.svd(remainder, full_matrices=False)

        # Truncate
        chi = len(S)
        if max_bond_dim is not None and chi > max_bond_dim:
            chi = max_bond_dim
            U = U[:, :chi]
            S = S[:chi]
            Vh = Vh[:chi, :]

        # Tensor for site k: reshape U to (chi_left, 2, chi)
        T = U.reshape(chi_left, 2, chi)
        tensors.append(T)

        # Remainder for next site
        remainder = np.diag(S) @ Vh

    # Last site: shape (chi_{n-1}, 2, 1)
    chi_left = remainder.shape[0]
    T = remainder.reshape(chi_left, 2, 1)
    tensors.append(T)

    return MPS(tensors=tensors, num_qubits=n, max_bond_dim=max_bond_dim)
