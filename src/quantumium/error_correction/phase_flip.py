"""3-qubit phase-flip code [[3,1,1]].

Encodes one logical qubit into three physical qubits to correct a
single Z (phase-flip) error on any one qubit.

Logical states:
  |0_L⟩ = |+++⟩
  |1_L⟩ = |---⟩

Stabilizer generators: X₀X₁, X₁X₂

This is the bit-flip code conjugated by Hadamard on every qubit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def phase_flip_encode(state: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Encode a single logical qubit into 3 physical qubits (Hadamard basis).

    Args:
        state: 2-element array [α, β] representing α|0⟩ + β|1⟩.

    Returns:
        8-element statevector: α|+++⟩ + β|---⟩.
    """
    state = np.asarray(state, dtype=complex)
    if state.shape != (2,):
        raise ValueError(f"Input must be a 2-element array, got shape {state.shape}")

    # |+⟩ = (|0⟩ + |1⟩)/√2, |−⟩ = (|0⟩ − |1⟩)/√2
    # |+++⟩ = (1/2√2) Σ|xyz⟩ for all xyz
    # |---⟩ = (1/2√2) Σ(-1)^(x+y+z)|xyz⟩
    encoded = np.zeros(8, dtype=complex)
    for i in range(8):
        parity = bin(i).count("1")
        plus_amp = 1.0 / (2 * np.sqrt(2))
        minus_amp = ((-1) ** parity) / (2 * np.sqrt(2))
        encoded[i] = state[0] * plus_amp + state[1] * minus_amp

    return encoded


def phase_flip_syndrome(state: NDArray[np.complexfloating]) -> tuple[int, int]:
    """Measure the phase-flip syndrome without collapsing the logical state.

    Computes the expectation values of the X_0 X_1 and X_1 X_2 stabilizer
    generators and returns syndrome bits indicating which qubit (if any)
    suffered a Z (phase-flip) error.

    Syndrome table:
      (0, 0) -> no error
      (1, 0) -> Z error on qubit 0
      (1, 1) -> Z error on qubit 1
      (0, 1) -> Z error on qubit 2

    Args:
        state: 8-element statevector in the 3-qubit Hilbert space.

    Returns:
        A tuple (s1, s2) of syndrome bits, each 0 or 1.
    """
    state = np.asarray(state, dtype=complex)

    # X₀X₁ expectation
    # X₀X₁ flips qubits 0 and 1: |abc⟩ → |ā b̄ c⟩ (bits 2 and 1 in our encoding)
    exp_x0x1 = 0.0
    for i in range(8):
        j = i ^ 0b110  # flip bits 2 and 1
        exp_x0x1 += (state[j].conj() * state[i]).real

    # X₁X₂ expectation
    exp_x1x2 = 0.0
    for i in range(8):
        j = i ^ 0b011  # flip bits 1 and 0
        exp_x1x2 += (state[j].conj() * state[i]).real

    s1 = 0 if exp_x0x1 >= 0 else 1
    s2 = 0 if exp_x1x2 >= 0 else 1

    return (s1, s2)


def phase_flip_correct(
    state: NDArray[np.complexfloating],
    syndrome: tuple[int, int] | None = None,
) -> NDArray[np.complexfloating]:
    """Apply a Z correction to the qubit identified by the syndrome.

    Looks up the syndrome in the error map and applies a phase-flip (Z gate)
    to the erroneous qubit by negating the amplitudes where that qubit is |1>.
    If no syndrome is provided, it is measured from the state first.

    Args:
        state: 8-element statevector in the 3-qubit Hilbert space.
        syndrome: Optional precomputed syndrome tuple (s1, s2). If None,
            the syndrome is measured via ``phase_flip_syndrome``.

    Returns:
        Corrected 8-element statevector with the identified phase-flip
        error removed.
    """
    state = np.asarray(state, dtype=complex).copy()
    if syndrome is None:
        syndrome = phase_flip_syndrome(state)

    s1, s2 = syndrome

    error_map = {
        (0, 0): None,
        (1, 0): 0,  # Z on qubit 0
        (1, 1): 1,  # Z on qubit 1
        (0, 1): 2,  # Z on qubit 2
    }

    error_qubit = error_map[(s1, s2)]
    if error_qubit is not None:
        bit = 2 - error_qubit
        for i in range(8):
            if (i >> bit) & 1:
                state[i] *= -1

    return state


def phase_flip_decode(state: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Decode a 3-qubit phase-flip code state back to a single logical qubit.

    Projects the state onto the code space spanned by |+++> and |--->
    by computing inner products with the logical basis states, then
    normalizes the result.

    Args:
        state: 8-element statevector in the 3-qubit Hilbert space,
            expected to be in the code space (no residual errors).

    Returns:
        2-element array [alpha, beta] representing the decoded logical
        state alpha|0> + beta|1>, normalized to unit length.
    """
    state = np.asarray(state, dtype=complex)

    # |+++⟩ = (1/2√2) Σ|i⟩
    plus3 = np.ones(8, dtype=complex) / (2 * np.sqrt(2))

    # |---⟩ = (1/2√2) Σ(-1)^parity(i)|i⟩
    minus3 = np.array([(-1) ** bin(i).count("1") for i in range(8)], dtype=complex) / (
        2 * np.sqrt(2)
    )

    alpha = np.dot(plus3.conj(), state)
    beta = np.dot(minus3.conj(), state)

    decoded = np.array([alpha, beta], dtype=complex)
    norm = np.linalg.norm(decoded)
    if norm > 0:
        decoded = decoded / norm
    return decoded
