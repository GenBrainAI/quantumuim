"""Example: Error mitigation techniques on a simple circuit.

Demonstrates Zero-Noise Extrapolation (ZNE), measurement error
mitigation, and Pauli twirling on a Bell state circuit.
"""

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT, Z

from quantumium.mitigation.zne import (
    noise_scaled_circuits,
    richardson_extrapolate,
    zne_expectation,
)
from quantumium.mitigation.measurement import (
    calibration_matrix,
    mitigate_counts,
)
from quantumium.mitigation.twirling import twirled_expectation


def z0_expectation(sv):
    """Compute ⟨Z₀⟩ from a statevector."""
    n = int(np.log2(len(sv)))
    probs = np.abs(sv) ** 2
    exp = 0.0
    for i, p in enumerate(probs):
        bit = (i >> (n - 1)) & 1
        exp += p * (1 - 2 * bit)
    return exp


# Build a Bell state circuit
qc = QuantumCircuit(2)
qc.apply(H, 0)
qc.apply(CNOT, 0, 1)

print("=" * 55)
print("Error Mitigation Demo — Bell State |Φ+⟩")
print("=" * 55)

# --- 1. Direct evaluation ---
direct = z0_expectation(qc.statevector())
print(f"\n1. Direct ⟨Z₀⟩ = {direct:.4f}")
print("   (Expected: 0.0 for Bell state)")

# --- 2. Zero-Noise Extrapolation ---
print("\n2. Zero-Noise Extrapolation (ZNE)")
scale_factors = [1, 3, 5]
scaled = noise_scaled_circuits(qc, scale_factors)
for k, sc in zip(scale_factors, scaled):
    exp = z0_expectation(sc.statevector())
    print(f"   Scale {k}: {len(sc.instructions)} gates, ⟨Z₀⟩ = {exp:.4f}")

zne_val = zne_expectation(qc, z0_expectation, scale_factors)
print(f"   ZNE extrapolated ⟨Z₀⟩ = {zne_val:.4f}")

# --- 3. Measurement Error Mitigation ---
print("\n3. Measurement Error Mitigation")
error_rates = {0: (0.05, 0.03), 1: (0.08, 0.02)}
cal = calibration_matrix(error_rates)
print(f"   Readout errors: q0=(p0|1={error_rates[0][0]}, p1|0={error_rates[0][1]})")
print(f"                   q1=(p0|1={error_rates[1][0]}, p1|0={error_rates[1][1]})")

raw_counts = qc.measure_all(shots=10000)
print(f"   Raw counts: {raw_counts}")

corrected = mitigate_counts(cal, raw_counts, 2)
print(f"   Corrected:  ", end="")
for k, v in sorted(corrected.items()):
    print(f"{k}: {v:.3f}  ", end="")
print()

# --- 4. Pauli Twirling ---
print("\n4. Pauli Twirling")
twirled = twirled_expectation(qc, z0_expectation, num_samples=50, seed=42)
print(f"   Twirled ⟨Z₀⟩ (50 samples) = {twirled:.4f}")
print(f"   Direct ⟨Z₀⟩              = {direct:.4f}")
print(f"   Difference               = {abs(twirled - direct):.4f}")

print("\n" + "=" * 55)
print("On an ideal simulator, all methods give the same result.")
print("On noisy hardware, mitigation techniques recover the")
print("ideal expectation value from noisy measurements.")
print("=" * 55)
