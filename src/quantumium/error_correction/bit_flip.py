"""3-qubit bit-flip code [[3,1,1]].

Encodes one logical qubit into three physical qubits to correct a
single X (bit-flip) error on any one qubit.

Logical states:
  |0_L⟩ = |000⟩
  |1_L⟩ = |111⟩

Stabilizer generators: Z₀Z₁, Z₁Z₂
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bit_flip_encode(state: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Encode a single logical qubit into 3 physical qubits.

    Args:
        state: 2-element array [α, β] representing α|0⟩ + β|1⟩.

    Returns:
        8-element statevector: α|000⟩ + β|111⟩.
    """
    state = np.asarray(state, dtype=complex)
    if state.shape != (2,):
        raise ValueError(f"Input must be a 2-element array, got shape {state.shape}")

    encoded = np.zeros(8, dtype=complex)
    encoded[0b000] = state[0]  # α|000⟩
    encoded[0b111] = state[1]  # β|111⟩
    return encoded


def bit_flip_syndrome(state: NDArray[np.complexfloating]) -> tuple[int, int]:
    """Measure the bit-flip syndrome without collapsing the logical state.

    Computes the eigenvalues of the stabilizer generators Z_0 Z_1 and Z_1 Z_2
    from the probability distribution of the statevector and returns the
    corresponding syndrome bits.

    Syndrome table:
      (0, 0) -> no error
      (1, 0) -> error on qubit 0
      (1, 1) -> error on qubit 1
      (0, 1) -> error on qubit 2

    Args:
        state: 8-element statevector in the 3-qubit Hilbert space.

    Returns:
        A tuple (s1, s2) of syndrome bits, each 0 or 1.
    """
    state = np.asarray(state, dtype=complex)
    probs = np.abs(state) ** 2

    # Z₀Z₁ eigenvalue: +1 if bits 0,1 agree, -1 if they differ
    s1_plus = sum(probs[i] for i in range(8) if ((i >> 2) & 1) == ((i >> 1) & 1))
    s1_minus = 1.0 - s1_plus

    # Z₁Z₂ eigenvalue: +1 if bits 1,2 agree, -1 if they differ
    s2_plus = sum(probs[i] for i in range(8) if ((i >> 1) & 1) == (i & 1))
    s2_minus = 1.0 - s2_plus

    s1 = 0 if s1_plus > s1_minus else 1
    s2 = 0 if s2_plus > s2_minus else 1

    return (s1, s2)


def bit_flip_correct(
    state: NDArray[np.complexfloating],
    syndrome: tuple[int, int] | None = None,
) -> NDArray[np.complexfloating]:
    """Apply an X correction to the qubit identified by the syndrome.

    Looks up the syndrome in the error map and applies a bit-flip (X gate)
    to the erroneous qubit by swapping amplitudes in the statevector.
    If no syndrome is provided, it is measured from the state first.

    Args:
        state: 8-element statevector in the 3-qubit Hilbert space.
        syndrome: Optional precomputed syndrome tuple (s1, s2). If None,
            the syndrome is measured via ``bit_flip_syndrome``.

    Returns:
        Corrected 8-element statevector with the identified bit-flip
        error removed.
    """
    state = np.asarray(state, dtype=complex).copy()
    if syndrome is None:
        syndrome = bit_flip_syndrome(state)

    s1, s2 = syndrome

    error_map = {
        (0, 0): None,
        (1, 0): 0,  # X on qubit 0
        (1, 1): 1,  # X on qubit 1
        (0, 1): 2,  # X on qubit 2
    }

    error_qubit = error_map[(s1, s2)]
    if error_qubit is not None:
        bit = 2 - error_qubit  # qubit 0 = bit 2 in 3-bit string
        for i in range(8):
            j = i ^ (1 << bit)
            if i < j:
                state[i], state[j] = state[j], state[i]

    return state


def bit_flip_decode(state: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Decode a 3-qubit bit-flip code state back to a single logical qubit.

    Projects onto the code space spanned by |000> and |111> and returns the
    normalized logical qubit amplitudes. Assumes the state has already been
    corrected (no residual errors).

    Args:
        state: 8-element statevector in the 3-qubit Hilbert space,
            expected to be in the code space.

    Returns:
        2-element array [alpha, beta] representing the decoded logical
        state alpha|0> + beta|1>, normalized to unit length.
    """
    state = np.asarray(state, dtype=complex)
    decoded = np.zeros(2, dtype=complex)
    decoded[0] = state[0b000]
    decoded[1] = state[0b111]
    norm = np.linalg.norm(decoded)
    if norm > 0:
        decoded = decoded / norm
    return decoded
