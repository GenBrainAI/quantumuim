"""Example: Quantum error correction codes.

Demonstrates encoding, error injection, syndrome detection,
correction, and decoding for all three codes in the SDK.
"""

import numpy as np

from quantumium.error_correction.bit_flip import (
    bit_flip_encode, bit_flip_syndrome, bit_flip_correct, bit_flip_decode,
)
from quantumium.error_correction.phase_flip import (
    phase_flip_encode, phase_flip_syndrome, phase_flip_correct, phase_flip_decode,
)
from quantumium.error_correction.shor_code import (
    shor_encode, shor_syndrome, shor_correct, shor_decode,
)


def fidelity(a, b):
    return abs(np.dot(a.conj(), b)) ** 2


original = np.array([0.6, 0.8], dtype=complex)
print("=" * 55)
print("Quantum Error Correction Demo")
print("=" * 55)
print(f"\nOriginal state: {original[0]:.2f}|0⟩ + {original[1]:.2f}|1⟩")

# --- Bit-flip code ---
print("\n--- 3-Qubit Bit-Flip Code [[3,1,1]] ---")
encoded = bit_flip_encode(original)
print(f"Encoded into {len(encoded)} amplitudes")

errored = encoded.copy()
errored[0b000], errored[0b100] = errored[0b100], errored[0b000]
errored[0b111], errored[0b011] = errored[0b011], errored[0b111]
print(f"Injected X error on qubit 0")

syn = bit_flip_syndrome(errored)
print(f"Syndrome: {syn} → error on qubit {({(0,0): 'none', (1,0): 0, (1,1): 1, (0,1): 2}[syn])}")

corrected = bit_flip_correct(errored)
decoded = bit_flip_decode(corrected)
print(f"Decoded:  {decoded[0]:.2f}|0⟩ + {decoded[1]:.2f}|1⟩")
print(f"Fidelity: {fidelity(decoded, original):.6f}")

# --- Phase-flip code ---
print("\n--- 3-Qubit Phase-Flip Code [[3,1,1]] ---")
encoded = phase_flip_encode(original)
print(f"Encoded into {len(encoded)} amplitudes")

errored = encoded.copy()
for i in range(8):
    if (i >> 1) & 1:
        errored[i] *= -1
print("Injected Z error on qubit 1")

syn = phase_flip_syndrome(errored)
print(f"Syndrome: {syn} → error on qubit {({(0,0): 'none', (1,0): 0, (1,1): 1, (0,1): 2}[syn])}")

corrected = phase_flip_correct(errored)
decoded = phase_flip_decode(corrected)
print(f"Decoded:  {abs(decoded[0]):.2f}|0⟩ + {abs(decoded[1]):.2f}|1⟩")
print(f"Fidelity: {fidelity(decoded, original):.6f}")

# --- Shor 9-qubit code ---
print("\n--- Shor 9-Qubit Code [[9,1,3]] ---")
encoded = shor_encode(original)
print(f"Encoded into {len(encoded)} amplitudes (9 qubits)")

errored = encoded.copy()
for i in range(512):
    j = i ^ (1 << 4)
    if i < j:
        errored[i], errored[j] = errored[j], errored[i]
print("Injected X error on qubit 4 (block 1)")

syn = shor_syndrome(errored)
print(f"Bit-flip syndromes: {syn['bit_flip_blocks']}")
print(f"Phase-flip syndrome: {syn['phase_flip']}")

corrected = shor_correct(errored)
decoded = shor_decode(corrected)
print(f"Decoded:  {abs(decoded[0]):.2f}|0⟩ + {abs(decoded[1]):.2f}|1⟩")
print(f"Fidelity: {fidelity(decoded, original):.6f}")

# --- Shor code with Z error ---
print("\n--- Shor Code: Z Error ---")
encoded = shor_encode(original)
errored = encoded.copy()
for i in range(512):
    if (i >> 2) & 1:
        errored[i] *= -1
print("Injected Z error on qubit 6 (block 2)")

corrected = shor_correct(errored)
decoded = shor_decode(corrected)
print(f"Decoded:  {abs(decoded[0]):.2f}|0⟩ + {abs(decoded[1]):.2f}|1⟩")
print(f"Fidelity: {fidelity(decoded, original):.6f}")

print("\n" + "=" * 55)
