"""Tests for quantum error correction codes."""

import numpy as np

from quantumium.error_correction.bit_flip import (
    bit_flip_encode,
    bit_flip_syndrome,
    bit_flip_correct,
    bit_flip_decode,
)
from quantumium.error_correction.phase_flip import (
    phase_flip_encode,
    phase_flip_syndrome,
    phase_flip_correct,
    phase_flip_decode,
)
from quantumium.error_correction.shor_code import (
    shor_encode,
    shor_syndrome,
    shor_correct,
    shor_decode,
)


# --- Bit-flip code ---

class TestBitFlipEncode:
    def test_encode_zero(self):
        encoded = bit_flip_encode(np.array([1, 0]))
        expected = np.zeros(8, dtype=complex)
        expected[0b000] = 1.0
        np.testing.assert_allclose(encoded, expected, atol=1e-10)

    def test_encode_one(self):
        encoded = bit_flip_encode(np.array([0, 1]))
        expected = np.zeros(8, dtype=complex)
        expected[0b111] = 1.0
        np.testing.assert_allclose(encoded, expected, atol=1e-10)

    def test_encode_plus(self):
        encoded = bit_flip_encode(np.array([1, 1]) / np.sqrt(2))
        assert abs(np.linalg.norm(encoded) - 1.0) < 1e-10
        assert abs(abs(encoded[0b000]) - 1 / np.sqrt(2)) < 1e-10
        assert abs(abs(encoded[0b111]) - 1 / np.sqrt(2)) < 1e-10


class TestBitFlipSyndrome:
    def test_no_error(self):
        state = bit_flip_encode(np.array([1, 0]))
        assert bit_flip_syndrome(state) == (0, 0)

    def test_error_qubit_0(self):
        state = bit_flip_encode(np.array([1, 0]))
        state[0b000], state[0b100] = state[0b100], state[0b000]  # X on qubit 0
        assert bit_flip_syndrome(state) == (1, 0)

    def test_error_qubit_1(self):
        state = bit_flip_encode(np.array([1, 0]))
        state[0b000], state[0b010] = state[0b010], state[0b000]  # X on qubit 1
        assert bit_flip_syndrome(state) == (1, 1)

    def test_error_qubit_2(self):
        state = bit_flip_encode(np.array([1, 0]))
        state[0b000], state[0b001] = state[0b001], state[0b000]  # X on qubit 2
        assert bit_flip_syndrome(state) == (0, 1)


class TestBitFlipCorrect:
    def test_correct_qubit_0(self):
        original = np.array([0.6, 0.8], dtype=complex)
        state = bit_flip_encode(original)
        # Apply X error on qubit 0 (bit 2)
        errored = state.copy()
        for i in range(8):
            j = i ^ (1 << 2)
            if i < j:
                errored[i], errored[j] = errored[j], errored[i]
        corrected = bit_flip_correct(errored)
        decoded = bit_flip_decode(corrected)
        np.testing.assert_allclose(np.abs(decoded), np.abs(original), atol=1e-10)

    def test_correct_qubit_2(self):
        original = np.array([1, 0], dtype=complex)
        state = bit_flip_encode(original)
        errored = state.copy()
        for i in range(8):
            j = i ^ (1 << 0)  # qubit 2 = bit 0
            if i < j:
                errored[i], errored[j] = errored[j], errored[i]
        corrected = bit_flip_correct(errored)
        decoded = bit_flip_decode(corrected)
        np.testing.assert_allclose(np.abs(decoded), np.abs(original), atol=1e-10)

    def test_no_error_unchanged(self):
        original = np.array([1, 0], dtype=complex)
        state = bit_flip_encode(original)
        corrected = bit_flip_correct(state)
        np.testing.assert_allclose(corrected, state, atol=1e-10)


class TestBitFlipDecode:
    def test_roundtrip(self):
        original = np.array([0.6, 0.8], dtype=complex)
        encoded = bit_flip_encode(original)
        decoded = bit_flip_decode(encoded)
        np.testing.assert_allclose(np.abs(decoded), np.abs(original), atol=1e-10)


# --- Phase-flip code ---

class TestPhaseFlipEncode:
    def test_encode_zero_normalized(self):
        encoded = phase_flip_encode(np.array([1, 0]))
        assert abs(np.linalg.norm(encoded) - 1.0) < 1e-10

    def test_encode_one_normalized(self):
        encoded = phase_flip_encode(np.array([0, 1]))
        assert abs(np.linalg.norm(encoded) - 1.0) < 1e-10

    def test_encode_superposition(self):
        encoded = phase_flip_encode(np.array([1, 1]) / np.sqrt(2))
        assert abs(np.linalg.norm(encoded) - 1.0) < 1e-10


class TestPhaseFlipSyndrome:
    def test_no_error(self):
        state = phase_flip_encode(np.array([1, 0]))
        assert phase_flip_syndrome(state) == (0, 0)

    def test_z_error_qubit_0(self):
        state = phase_flip_encode(np.array([1, 0]))
        # Z on qubit 0 (bit 2): |abc⟩ → (-1)^a |abc⟩
        errored = state.copy()
        for i in range(8):
            if (i >> 2) & 1:
                errored[i] *= -1
        assert phase_flip_syndrome(errored) == (1, 0)

    def test_z_error_qubit_1(self):
        state = phase_flip_encode(np.array([1, 0]))
        errored = state.copy()
        for i in range(8):
            if (i >> 1) & 1:
                errored[i] *= -1
        assert phase_flip_syndrome(errored) == (1, 1)

    def test_z_error_qubit_2(self):
        state = phase_flip_encode(np.array([1, 0]))
        errored = state.copy()
        for i in range(8):
            if i & 1:
                errored[i] *= -1
        assert phase_flip_syndrome(errored) == (0, 1)


class TestPhaseFlipCorrect:
    def test_correct_z_error(self):
        original = np.array([0.6, 0.8], dtype=complex)
        state = phase_flip_encode(original)
        # Z error on qubit 1
        errored = state.copy()
        for i in range(8):
            if (i >> 1) & 1:
                errored[i] *= -1
        corrected = phase_flip_correct(errored)
        decoded = phase_flip_decode(corrected)
        np.testing.assert_allclose(np.abs(decoded), np.abs(original), atol=1e-10)

    def test_no_error_unchanged(self):
        state = phase_flip_encode(np.array([1, 0]))
        corrected = phase_flip_correct(state)
        np.testing.assert_allclose(corrected, state, atol=1e-10)


class TestPhaseFlipDecode:
    def test_roundtrip(self):
        original = np.array([0.6, 0.8], dtype=complex)
        encoded = phase_flip_encode(original)
        decoded = phase_flip_decode(encoded)
        np.testing.assert_allclose(np.abs(decoded), np.abs(original), atol=1e-10)


# --- Shor 9-qubit code ---

class TestShorEncode:
    def test_encode_zero_normalized(self):
        encoded = shor_encode(np.array([1, 0]))
        assert abs(np.linalg.norm(encoded) - 1.0) < 1e-10

    def test_encode_one_normalized(self):
        encoded = shor_encode(np.array([0, 1]))
        assert abs(np.linalg.norm(encoded) - 1.0) < 1e-10

    def test_dimension(self):
        encoded = shor_encode(np.array([1, 0]))
        assert len(encoded) == 512

    def test_orthogonal_codewords(self):
        zero_L = shor_encode(np.array([1, 0]))
        one_L = shor_encode(np.array([0, 1]))
        overlap = abs(np.dot(zero_L.conj(), one_L))
        assert overlap < 1e-10


class TestShorSyndrome:
    def test_no_error(self):
        state = shor_encode(np.array([1, 0]))
        syn = shor_syndrome(state)
        assert all(s == (0, 0) for s in syn["bit_flip_blocks"])
        assert syn["phase_flip"] == (0, 0)

    def test_x_error_detected(self):
        state = shor_encode(np.array([1, 0]))
        # X error on qubit 0 (bit 8 in 9-bit string)
        errored = state.copy()
        for i in range(512):
            j = i ^ (1 << 8)
            if i < j:
                errored[i], errored[j] = errored[j], errored[i]
        syn = shor_syndrome(errored)
        assert syn["bit_flip_blocks"][0] != (0, 0)

    def test_z_error_detected(self):
        state = shor_encode(np.array([1, 0]))
        # Z error on qubit in block 0 (bit 8)
        errored = state.copy()
        for i in range(512):
            if (i >> 8) & 1:
                errored[i] *= -1
        syn = shor_syndrome(errored)
        assert syn["phase_flip"] != (0, 0)


class TestShorCorrect:
    def test_correct_x_error(self):
        original = np.array([0.6, 0.8], dtype=complex)
        state = shor_encode(original)
        # X error on qubit 4 (bit 4 in block 1)
        errored = state.copy()
        for i in range(512):
            j = i ^ (1 << 4)
            if i < j:
                errored[i], errored[j] = errored[j], errored[i]
        corrected = shor_correct(errored)
        decoded = shor_decode(corrected)
        np.testing.assert_allclose(np.abs(decoded), np.abs(original), atol=1e-10)

    def test_correct_z_error(self):
        original = np.array([0.6, 0.8], dtype=complex)
        state = shor_encode(original)
        # Z error on qubit in block 2 (bit 2)
        errored = state.copy()
        for i in range(512):
            if (i >> 2) & 1:
                errored[i] *= -1
        corrected = shor_correct(errored)
        decoded = shor_decode(corrected)
        np.testing.assert_allclose(np.abs(decoded), np.abs(original), atol=1e-10)

    def test_no_error_roundtrip(self):
        original = np.array([1, 0], dtype=complex)
        state = shor_encode(original)
        corrected = shor_correct(state)
        decoded = shor_decode(corrected)
        np.testing.assert_allclose(np.abs(decoded), np.abs(original), atol=1e-10)


class TestShorDecode:
    def test_roundtrip_zero(self):
        original = np.array([1, 0], dtype=complex)
        decoded = shor_decode(shor_encode(original))
        np.testing.assert_allclose(np.abs(decoded), np.abs(original), atol=1e-10)

    def test_roundtrip_one(self):
        original = np.array([0, 1], dtype=complex)
        decoded = shor_decode(shor_encode(original))
        np.testing.assert_allclose(np.abs(decoded), np.abs(original), atol=1e-10)

    def test_roundtrip_superposition(self):
        original = np.array([0.6, 0.8], dtype=complex)
        decoded = shor_decode(shor_encode(original))
        np.testing.assert_allclose(np.abs(decoded), np.abs(original), atol=1e-10)
