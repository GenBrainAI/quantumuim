"""Shor 9-qubit code [[9,1,3]].

The first quantum error-correcting code, capable of correcting any
single-qubit error (X, Y, or Z) on any one of the 9 physical qubits.

Construction: phase-flip code on top of bit-flip code.
Each logical qubit is first encoded with the phase-flip code (3 qubits),
then each of those 3 qubits is encoded with the bit-flip code (3 qubits each),
giving 9 physical qubits total.

Logical states:
  |0_L⟩ = (|000⟩ + |111⟩)(|000⟩ + |111⟩)(|000⟩ + |111⟩) / 2√2
  |1_L⟩ = (|000⟩ - |111⟩)(|000⟩ - |111⟩)(|000⟩ - |111⟩) / 2√2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _logical_zero() -> NDArray[np.complexfloating]:
    """Build the logical |0_L> state for the 9-qubit Shor code.

    Constructs the 512-element statevector corresponding to:
    |0_L> = (|000> + |111>)(|000> + |111>)(|000> + |111>) / 2*sqrt(2)

    Returns:
        Normalized 512-element statevector for the logical zero state.
    """
    # (|000⟩ + |111⟩)/√2 for each block
    block_plus = np.zeros(8, dtype=complex)
    block_plus[0b000] = 1 / np.sqrt(2)
    block_plus[0b111] = 1 / np.sqrt(2)

    state = block_plus
    state = np.kron(state, block_plus)
    state = np.kron(state, block_plus)  # wrong order — fix below

    # Actually: tensor product order is block0 ⊗ block1 ⊗ block2
    # We need to rebuild correctly for 9 qubits (512-dim)
    state = np.zeros(512, dtype=complex)
    for b0 in [0b000, 0b111]:
        for b1 in [0b000, 0b111]:
            for b2 in [0b000, 0b111]:
                idx = (b0 << 6) | (b1 << 3) | b2
                state[idx] = 1.0
    state /= np.linalg.norm(state)
    return state


def _logical_one() -> NDArray[np.complexfloating]:
    """Build the logical |1_L> state for the 9-qubit Shor code.

    Constructs the 512-element statevector corresponding to:
    |1_L> = (|000> - |111>)(|000> - |111>)(|000> - |111>) / 2*sqrt(2)

    Returns:
        Normalized 512-element statevector for the logical one state.
    """
    state = np.zeros(512, dtype=complex)
    for b0 in [0b000, 0b111]:
        for b1 in [0b000, 0b111]:
            for b2 in [0b000, 0b111]:
                idx = (b0 << 6) | (b1 << 3) | b2
                parity = (1 if b0 == 0b111 else 0) + \
                         (1 if b1 == 0b111 else 0) + \
                         (1 if b2 == 0b111 else 0)
                sign = (-1) ** parity
                state[idx] = sign
    state /= np.linalg.norm(state)
    return state


def shor_encode(state: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Encode a single logical qubit into 9 physical qubits.

    Args:
        state: 2-element array [α, β] representing α|0⟩ + β|1⟩.

    Returns:
        512-element statevector for the 9-qubit encoded state.
    """
    state = np.asarray(state, dtype=complex)
    if state.shape != (2,):
        raise ValueError(f"Input must be a 2-element array, got shape {state.shape}")

    return state[0] * _logical_zero() + state[1] * _logical_one()


def _bit_flip_syndrome_block(state: NDArray[np.complexfloating], block: int) -> tuple[int, int]:
    """Measure bit-flip syndrome for one of the three 3-qubit blocks.

    Block 0: qubits 0,1,2 (bits 8,7,6)
    Block 1: qubits 3,4,5 (bits 5,4,3)
    Block 2: qubits 6,7,8 (bits 2,1,0)

    Args:
        state: 512-element statevector in the 9-qubit Hilbert space.
        block: Block index (0, 1, or 2).

    Returns:
        A tuple (s1, s2) of syndrome bits for the specified block.
    """
    probs = np.abs(state) ** 2

    if block == 0:
        bit_a, bit_b, bit_c = 8, 7, 6
    elif block == 1:
        bit_a, bit_b, bit_c = 5, 4, 3
    else:
        bit_a, bit_b, bit_c = 2, 1, 0

    # Z_a Z_b: +1 if bits a,b agree
    s1_plus = sum(probs[i] for i in range(512) if ((i >> bit_a) & 1) == ((i >> bit_b) & 1))
    s1 = 0 if s1_plus > 0.5 else 1

    # Z_b Z_c: +1 if bits b,c agree
    s2_plus = sum(probs[i] for i in range(512) if ((i >> bit_b) & 1) == ((i >> bit_c) & 1))
    s2 = 0 if s2_plus > 0.5 else 1

    return (s1, s2)


def _phase_flip_syndrome_blocks(state: NDArray[np.complexfloating]) -> tuple[int, int]:
    """Measure phase-flip syndrome across the three 3-qubit blocks.

    Evaluates the expectation values of the inter-block stabilizer generators
    X^{otimes 3}_{block0} X^{otimes 3}_{block1} and
    X^{otimes 3}_{block1} X^{otimes 3}_{block2}.

    Args:
        state: 512-element statevector in the 9-qubit Hilbert space.

    Returns:
        A tuple (s1, s2) of phase-flip syndrome bits across blocks.
    """
    # X^⊗3_block0 ⊗ X^⊗3_block1: flips all 6 qubits in blocks 0 and 1
    exp1 = 0.0
    for i in range(512):
        j = i ^ 0b111111000  # flip bits 8-3
        exp1 += (state[j].conj() * state[i]).real

    # X^⊗3_block1 ⊗ X^⊗3_block2: flips all 6 qubits in blocks 1 and 2
    exp2 = 0.0
    for i in range(512):
        j = i ^ 0b000111111  # flip bits 5-0
        exp2 += (state[j].conj() * state[i]).real

    s1 = 0 if exp1 >= 0 else 1
    s2 = 0 if exp2 >= 0 else 1

    return (s1, s2)


def shor_syndrome(
    state: NDArray[np.complexfloating],
) -> dict:
    """Measure all syndromes for the 9-qubit Shor code.

    Evaluates the bit-flip syndromes within each of the three 3-qubit
    blocks and the phase-flip syndrome across blocks.

    Args:
        state: 512-element statevector in the 9-qubit Hilbert space.

    Returns:
        A dict with two keys:
            ``"bit_flip_blocks"``: list of three (s1, s2) tuples, one per block.
            ``"phase_flip"``: a single (s1, s2) tuple for inter-block phase errors.
    """
    bf = [_bit_flip_syndrome_block(state, b) for b in range(3)]
    pf = _phase_flip_syndrome_blocks(state)
    return {"bit_flip_blocks": bf, "phase_flip": pf}


def shor_correct(
    state: NDArray[np.complexfloating],
    syndrome: dict | None = None,
) -> NDArray[np.complexfloating]:
    """Apply corrections based on the Shor code syndrome measurement.

    First corrects single X (bit-flip) errors within each 3-qubit block
    using the per-block bit-flip syndromes, then corrects a single Z
    (phase-flip) error across blocks using the inter-block phase syndrome.

    Args:
        state: 512-element statevector in the 9-qubit Hilbert space.
        syndrome: Optional precomputed syndrome dict as returned by
            ``shor_syndrome``. If None, the syndrome is measured first.

    Returns:
        Corrected 512-element statevector with the identified errors
        removed.
    """
    state = np.asarray(state, dtype=complex).copy()
    if syndrome is None:
        syndrome = shor_syndrome(state)

    bf_syndromes = syndrome["bit_flip_blocks"]
    pf_syndrome = syndrome["phase_flip"]

    # Correct bit-flip errors in each block
    error_map = {
        (0, 0): None,
        (1, 0): 0,
        (1, 1): 1,
        (0, 1): 2,
    }

    for block_idx, (s1, s2) in enumerate(bf_syndromes):
        error_pos = error_map[(s1, s2)]
        if error_pos is not None:
            if block_idx == 0:
                bit = 8 - error_pos
            elif block_idx == 1:
                bit = 5 - error_pos
            else:
                bit = 2 - error_pos
            for i in range(512):
                j = i ^ (1 << bit)
                if i < j:
                    state[i], state[j] = state[j], state[i]

    # Correct phase-flip errors across blocks
    pf_error_map = {
        (0, 0): None,
        (1, 0): 0,  # Z on block 0 (any qubit)
        (1, 1): 1,  # Z on block 1
        (0, 1): 2,  # Z on block 2
    }

    error_block = pf_error_map[pf_syndrome]
    if error_block is not None:
        if error_block == 0:
            target_bit = 8  # Z on qubit 0 of block 0
        elif error_block == 1:
            target_bit = 5
        else:
            target_bit = 2
        for i in range(512):
            if (i >> target_bit) & 1:
                state[i] *= -1

    return state


def shor_decode(state: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Decode a 9-qubit Shor code state back to a single logical qubit.

    Projects the state onto the logical code space spanned by |0_L> and
    |1_L> and returns the normalized logical qubit amplitudes.

    Args:
        state: 512-element statevector in the 9-qubit Hilbert space,
            expected to be in the code space (no residual errors).

    Returns:
        2-element array [alpha, beta] representing the decoded logical
        state alpha|0> + beta|1>, normalized to unit length.
    """
    state = np.asarray(state, dtype=complex)

    zero_L = _logical_zero()
    one_L = _logical_one()

    alpha = np.dot(zero_L.conj(), state)
    beta = np.dot(one_L.conj(), state)

    decoded = np.array([alpha, beta], dtype=complex)
    norm = np.linalg.norm(decoded)
    if norm > 0:
        decoded = decoded / norm
    return decoded
