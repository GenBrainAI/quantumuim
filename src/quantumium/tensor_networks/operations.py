"""Operations on Matrix Product States.

Gate application, measurement, entanglement entropy, and inner products.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .mps import MPS


def apply_single_qubit_gate(
    mps: MPS,
    gate: NDArray[np.complexfloating],
    qubit: int,
) -> MPS:
    """Apply a single-qubit gate to a qubit in the MPS.

    This is exact and does not change the bond dimension.
    The gate matrix (2×2) contracts with the physical index.

    Args:
        mps: Input MPS (modified in-place).
        gate: 2×2 unitary matrix.
        qubit: Target qubit index.

    Returns:
        The modified MPS (same object).
    """
    gate = np.asarray(gate, dtype=complex)
    if gate.shape != (2, 2):
        raise ValueError(f"Single-qubit gate must be 2×2, got {gate.shape}")

    T = mps.tensors[qubit]  # (χ_left, 2, χ_right)
    # Contract: new_T[α, s', β] = Σ_s gate[s', s] * T[α, s, β]
    new_T = np.einsum("ij,ajb->aib", gate, T)
    mps.tensors[qubit] = new_T

    return mps


def apply_two_qubit_gate(
    mps: MPS,
    gate: NDArray[np.complexfloating],
    qubit1: int,
    qubit2: int,
) -> MPS:
    """Apply a two-qubit gate to adjacent qubits in the MPS.

    Uses SVD to split the result back into two tensors, truncating
    to max_bond_dim if set.

    The qubits must be adjacent (|qubit1 - qubit2| = 1).
    For non-adjacent qubits, swap gates would be needed.

    Args:
        mps: Input MPS (modified in-place).
        gate: 4×4 unitary matrix.
        qubit1: First qubit index.
        qubit2: Second qubit index.

    Returns:
        The modified MPS.
    """
    gate = np.asarray(gate, dtype=complex)
    if gate.shape != (4, 4):
        raise ValueError(f"Two-qubit gate must be 4×4, got {gate.shape}")

    if abs(qubit1 - qubit2) != 1:
        raise ValueError(f"Qubits must be adjacent, got {qubit1} and {qubit2}")

    left = min(qubit1, qubit2)
    right = left + 1

    # If gate indices are reversed from physical layout, transpose
    if qubit1 > qubit2:
        # Swap gate indices: |s1 s2⟩ → |s2 s1⟩
        swap = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=complex,
        )
        gate = gate @ swap  # Apply swap before gate
        gate = swap @ gate  # Apply swap after gate too? No — just conjugate.
        # Actually: if gate acts on (q1, q2) but q1 > q2, we need to
        # apply it in the tensor order (left, right). So:
        gate_reshaped = gate.reshape(2, 2, 2, 2)
        gate_reshaped = gate_reshaped.transpose(1, 0, 3, 2)
        gate = gate_reshaped.reshape(4, 4)

    T_left = mps.tensors[left]  # (χ_L, 2, χ_mid)
    T_right = mps.tensors[right]  # (χ_mid, 2, χ_R)

    chi_L = T_left.shape[0]
    chi_mid = T_left.shape[2]
    chi_R = T_right.shape[2]

    # Contract the two tensors: Θ[α, s1, s2, β] = Σ_γ T_left[α,s1,γ] * T_right[γ,s2,β]
    theta = np.einsum(
        "asc,csb->asb", T_left.reshape(chi_L, 2, chi_mid), T_right.reshape(chi_mid, 2, chi_R)
    )
    # theta shape: (χ_L, 2, 2, χ_R) — wait, that's wrong.
    # Let me redo: T_left is (χ_L, 2, χ_mid), T_right is (χ_mid, 2, χ_R)
    theta = np.einsum("aig,gjb->aijb", T_left, T_right)  # (χ_L, 2, 2, χ_R)

    # Apply gate: gate[s1',s2', s1, s2] * theta[α, s1, s2, β]
    gate_4d = gate.reshape(2, 2, 2, 2)
    theta_new = np.einsum("ijkl,aklb->aijb", gate_4d, theta)  # (χ_L, 2, 2, χ_R)

    # SVD to split back into two tensors
    theta_mat = theta_new.reshape(chi_L * 2, 2 * chi_R)

    U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)

    # Truncate to max_bond_dim
    chi_new = len(S)
    if mps.max_bond_dim is not None and chi_new > mps.max_bond_dim:
        chi_new = mps.max_bond_dim
        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]

    # Absorb singular values into the right tensor (right-canonical)
    S_diag = np.diag(S)
    new_left = U.reshape(chi_L, 2, chi_new)
    new_right = (S_diag @ Vh).reshape(chi_new, 2, chi_R)

    mps.tensors[left] = new_left
    mps.tensors[right] = new_right

    return mps


def measure_qubit(mps: MPS, qubit: int, rng: np.random.Generator | None = None) -> int:
    """Measure a single qubit, collapsing the MPS.

    Computes the probability of outcome 0 and 1, samples, and
    projects the MPS tensor accordingly.

    Args:
        mps: Input MPS (modified in-place by projection).
        qubit: Qubit to measure.
        rng: Random number generator.

    Returns:
        Measurement outcome (0 or 1).
    """
    if rng is None:
        rng = np.random.default_rng()

    T = mps.tensors[qubit]  # (χ_L, 2, χ_R)

    # Compute reduced density matrix for this qubit
    # ρ_s,s' = Σ_α,β T[α,s,β] * conj(T[α,s',β])  (contracted over all other sites)
    # For a general MPS, this requires contracting environment tensors.
    # For simplicity, we compute probabilities from the full statevector.
    # This is exact but O(2^n) — a proper implementation would use
    # transfer matrices for O(n χ²) scaling.

    sv = mps.to_statevector()
    n = mps.num_qubits
    dim = 2**n

    # MPS convention: qubit 0 is MSB in statevector index
    bit_pos = n - 1 - qubit
    p0 = 0.0
    for idx in range(dim):
        if not ((idx >> bit_pos) & 1):
            p0 += abs(sv[idx]) ** 2

    total = np.linalg.norm(sv) ** 2
    if total > 1e-15:
        p0 /= total

    outcome = 0 if rng.random() < p0 else 1

    # Project: zero out the other outcome in the MPS tensor
    projected = T.copy()
    other = 1 - outcome
    projected[:, other, :] = 0

    # Renormalize
    norm_sq = np.sum(np.abs(projected) ** 2)
    if norm_sq > 1e-15:
        projected = projected / math.sqrt(norm_sq)
    # But we need to account for the rest of the MPS norm too
    # Simplest correct approach: set the tensor and renormalize the whole MPS
    mps.tensors[qubit] = projected
    mps.normalize()

    return outcome


def entanglement_entropy(mps: MPS, cut: int) -> float:
    """Compute the von Neumann entanglement entropy across a bipartition.

    The cut is between qubit `cut-1` and qubit `cut`, splitting the
    system into [0, ..., cut-1] and [cut, ..., n-1].

    Uses the Schmidt decomposition (SVD of the bipartition) to compute
    S = -Σ λ²_i log₂(λ²_i).

    Args:
        mps: Input MPS.
        cut: Position of the cut (1 ≤ cut ≤ n-1).

    Returns:
        Entanglement entropy in bits.
    """
    n = mps.num_qubits
    if cut < 1 or cut >= n:
        raise ValueError(f"Cut must be between 1 and {n - 1}, got {cut}")

    sv = mps.to_statevector()
    2**n
    dim_left = 2**cut
    dim_right = 2 ** (n - cut)

    psi_mat = sv.reshape(dim_left, dim_right)
    _, S, _ = np.linalg.svd(psi_mat, full_matrices=False)

    S = S[S > 1e-15]
    probs = S**2
    probs = probs / np.sum(probs)

    entropy = 0.0
    for p in probs:
        if p > 1e-15:
            entropy -= p * math.log2(p)

    return entropy


def inner_product(mps1: MPS, mps2: MPS) -> complex:
    """Compute the inner product ⟨ψ1|ψ2⟩ between two MPS states.

    Uses sequential contraction from left to right with transfer matrices.

    Args:
        mps1: Bra state.
        mps2: Ket state.

    Returns:
        Complex inner product.
    """
    if mps1.num_qubits != mps2.num_qubits:
        raise ValueError("MPS must have the same number of qubits")

    n = mps1.num_qubits

    # Transfer matrix contraction: left to right
    # E[α,α'] = Σ_s conj(T1[α_old, s, α]) * T2[α'_old, s, α']
    # Start with (1, 1) identity
    E = np.ones((1, 1), dtype=complex)

    for k in range(n):
        T1 = mps1.tensors[k]  # (χ1_L, 2, χ1_R)
        T2 = mps2.tensors[k]  # (χ2_L, 2, χ2_R)

        # E_new[β, β'] = Σ_{α,α',s} E[α,α'] * conj(T1[α,s,β]) * T2[α',s,β']
        E_new = np.einsum("ab,asc,bsd->cd", E, T1.conj(), T2)
        E = E_new

    return complex(E[0, 0])
