"""Quantum Singular Value Transformation (QSVT).

QSVT lifts single-qubit QSP to block-encoded matrices: given a unitary U
that block-encodes a matrix A (i.e. (⟨0|⊗I) U (|0⟩⊗I) = A/α), QSVT
applies a polynomial transformation to the singular values of A.

If A = Σ σ_k |w_k⟩⟨v_k| is the SVD, then QSVT realizes:
    P^(SV)(A) = Σ P(σ_k) |w_k⟩⟨v_k|

where P is the polynomial determined by the QSP phase angles.

This is the "grand unification" — QPE, Hamiltonian simulation, amplitude
amplification, and matrix inversion are all special cases.

References:
    Gilyén et al., "Quantum singular value transformation and beyond:
    exponential improvements for quantum matrix arithmetics" (STOC 2019)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate


@dataclass
class QSVTResult:
    """Result from a Quantum Singular Value Transformation computation.

    Attributes:
        transformed_matrix (NDArray[np.complexfloating]): The matrix
            after polynomial transformation of its singular values,
            extracted from the upper-left block of the QSVT unitary.
        original_matrix (NDArray[np.complexfloating]): The input matrix
            before transformation.
        block_encoding_norm (float): The normalization factor alpha
            used in the block encoding (A_encoded = A / alpha).
        angles (NDArray[np.floating]): The QSP/QSVT phase angles
            that define the polynomial transformation.
        degree (int): Degree of the polynomial transformation
            (len(angles) - 1).
    """
    transformed_matrix: NDArray[np.complexfloating]
    original_matrix: NDArray[np.complexfloating]
    block_encoding_norm: float
    angles: NDArray[np.floating]
    degree: int


def block_encoding(
    A: NDArray[np.complexfloating],
    alpha: float | None = None,
) -> tuple[Gate, float]:
    """Construct a block encoding of matrix A.

    Builds a unitary U such that (⟨0|⊗I) U (|0⟩⊗I) = A/α,
    where α ≥ ‖A‖ is the normalization factor.

    The construction embeds A/α in the upper-left block of a larger
    unitary, completing it via SVD.

    Args:
        A: Matrix to block-encode (n×n).
        alpha: Normalization factor. If None, uses ‖A‖_op.

    Returns:
        (gate, alpha) where gate is the block-encoding unitary and
        alpha is the normalization factor.
    """
    A = np.asarray(A, dtype=complex)
    n = A.shape[0]

    if alpha is None:
        alpha = float(np.linalg.norm(A, ord=2)) * 1.01

    A_normalized = A / alpha

    s_vals = np.linalg.svd(A_normalized, compute_uv=False)
    if np.max(s_vals) > 1.0 + 1e-10:
        raise ValueError(
            f"Cannot block-encode: ‖A/α‖ = {np.max(s_vals):.4f} > 1. "
            f"Increase alpha (currently {alpha})."
        )

    # Build 2n × 2n unitary with A_normalized in upper-left block
    # U = [[A_norm, B], [C, D]] where U is unitary
    # Using the construction: U = [[A_norm, sqrt(I-A_norm A_norm†)],
    #                               [sqrt(I-A_norm† A_norm), -A_norm†]]
    # (valid when ‖A_norm‖ ≤ 1)

    AA_dag = A_normalized @ A_normalized.conj().T
    I_n = np.eye(n, dtype=complex)

    # sqrt(I - A A†)
    evals_l, evecs_l = np.linalg.eigh(I_n - AA_dag)
    evals_l = np.maximum(evals_l, 0)
    B = evecs_l @ np.diag(np.sqrt(evals_l)) @ evecs_l.conj().T

    # sqrt(I - A† A)
    A_dag_A = A_normalized.conj().T @ A_normalized
    evals_r, evecs_r = np.linalg.eigh(I_n - A_dag_A)
    evals_r = np.maximum(evals_r, 0)
    C = evecs_r @ np.diag(np.sqrt(evals_r)) @ evecs_r.conj().T

    U = np.block([
        [A_normalized, B],
        [C, -A_normalized.conj().T],
    ])

    # Verify unitarity — if not exact due to numerics, project to unitary
    if not np.allclose(U @ U.conj().T, np.eye(2 * n), atol=1e-8):
        u_svd, _, vh_svd = np.linalg.svd(U)
        U = u_svd @ vh_svd

    gate = Gate(U, f"BE(A, α={alpha:.3f})")
    return gate, alpha


def _projector_controlled_phase(
    phi: float,
    n: int,
    ancilla_idx: int = 0,
) -> NDArray[np.complexfloating]:
    """Build Π_φ = e^{iφ(2|0⟩⟨0|-I)} acting on the ancilla qubit.

    This is a phase of e^{iφ} on the |0⟩ subspace of the ancilla
    and e^{-iφ} on the |1⟩ subspace, tensored with identity on
    the remaining qubits.

    For a 2n-dimensional space with ancilla as the first n-dim block:
    Π_φ = diag(e^{iφ}, ..., e^{iφ}, e^{-iφ}, ..., e^{-iφ})
          |--- n entries ---|--- n entries ---|
    """
    dim = 2 * n
    phases = np.zeros(dim, dtype=complex)
    phases[:n] = np.exp(1j * phi)
    phases[n:] = np.exp(-1j * phi)
    return np.diag(phases)


def qsvt_circuit(
    block_enc_gate: Gate,
    angles: NDArray[np.floating] | list[float],
) -> QuantumCircuit:
    """Build a QSVT circuit from a block-encoding and QSP angles.

    The QSVT sequence is:
    Π_{φ_0} · U · Π_{φ_1} · U† · Π_{φ_2} · U · ... · Π_{φ_d}

    where U and U† alternate, and Π_{φ_k} are projector-controlled
    phase gates.

    Args:
        block_enc_gate: Block-encoding unitary gate.
        angles: QSP/QSVT phase angles [φ_0, φ_1, ..., φ_d].

    Returns:
        QuantumCircuit implementing the QSVT sequence.
    """
    angles = np.asarray(angles, dtype=float)
    dim = block_enc_gate.matrix.shape[0]
    n = dim // 2
    total_qubits = int(math.log2(dim))

    qc = QuantumCircuit(total_qubits)
    qubit_indices = tuple(range(total_qubits))

    U = block_enc_gate.matrix
    U_dag = U.conj().T

    for k in range(len(angles)):
        # Projector-controlled phase
        Pi_k = _projector_controlled_phase(angles[k], n)
        pi_gate = Gate(Pi_k, f"Π({angles[k]:.3f})")
        qc.apply(pi_gate, *qubit_indices)

        if k < len(angles) - 1:
            if k % 2 == 0:
                u_gate = Gate(U.copy(), f"U")
                qc.apply(u_gate, *qubit_indices)
            else:
                u_dag_gate = Gate(U_dag.copy(), f"U†")
                qc.apply(u_dag_gate, *qubit_indices)

    return qc


def qsvt_transform(
    A: NDArray[np.complexfloating],
    angles: NDArray[np.floating] | list[float],
    alpha: float | None = None,
) -> QSVTResult:
    """Apply a QSVT polynomial transformation to a matrix.

    Given a matrix A and QSP angles, computes P^(SV)(A/α) where P
    is the polynomial determined by the angles.

    The result is extracted from the upper-left block of the
    QSVT circuit's unitary.

    Args:
        A: Input matrix (n×n).
        angles: QSP/QSVT phase angles.
        alpha: Block-encoding normalization. Auto-computed if None.

    Returns:
        QSVTResult with the transformed matrix and metadata.
    """
    A = np.asarray(A, dtype=complex)
    angles = np.asarray(angles, dtype=float)
    n = A.shape[0]

    be_gate, alpha_used = block_encoding(A, alpha)

    U = be_gate.matrix
    U_dag = U.conj().T
    d = len(angles)

    # Build the full QSVT unitary by matrix multiplication
    result = _projector_controlled_phase(angles[0], n)

    for k in range(1, d):
        if (k - 1) % 2 == 0:
            result = result @ U
        else:
            result = result @ U_dag
        result = result @ _projector_controlled_phase(angles[k], n)

    # Extract upper-left n×n block
    transformed = result[:n, :n]

    return QSVTResult(
        transformed_matrix=transformed,
        original_matrix=A,
        block_encoding_norm=alpha_used,
        angles=angles,
        degree=d - 1,
    )
