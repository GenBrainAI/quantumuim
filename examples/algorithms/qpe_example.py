"""Quantum Phase Estimation demonstration.

Estimates eigenvalue phases of various unitary gates.
"""

import numpy as np

from quantumium.core.gate import Gate, T, S, Z
from quantumium.algorithms.qpe import estimate_phase


def main():
    print("=" * 60)
    print("Quantum Phase Estimation — Quantumium")
    print("=" * 60)

    # T gate: eigenvalue e^(iπ/4) → phase = 1/8
    print("\n1. T gate phase estimation")
    print("   T|1⟩ = e^(iπ/4)|1⟩ → expected phase = 1/8 = 0.125")
    phase_t = estimate_phase(3, T, shots=4096)
    print(f"   Estimated phase: {phase_t:.4f}")
    print(f"   Error: {abs(phase_t - 0.125):.4f}")

    # S gate: eigenvalue e^(iπ/2) → phase = 1/4
    print("\n2. S gate phase estimation")
    print("   S|1⟩ = i|1⟩ = e^(iπ/2)|1⟩ → expected phase = 1/4 = 0.25")
    phase_s = estimate_phase(3, S, shots=4096)
    print(f"   Estimated phase: {phase_s:.4f}")
    print(f"   Error: {abs(phase_s - 0.25):.4f}")

    # Z gate: eigenvalue e^(iπ) → phase = 1/2
    print("\n3. Z gate phase estimation")
    print("   Z|1⟩ = -|1⟩ = e^(iπ)|1⟩ → expected phase = 1/2 = 0.5")
    phase_z = estimate_phase(3, Z, shots=4096)
    print(f"   Estimated phase: {phase_z:.4f}")
    print(f"   Error: {abs(phase_z - 0.5):.4f}")

    # Custom rotation: Rz(π/3) → phase = 1/12 ≈ 0.0833
    print("\n4. Custom Rz(π/3) phase estimation (4 counting qubits)")
    theta = np.pi / 3
    rz = Gate(np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)],
    ], dtype=complex), "Rz(π/3)")
    expected = (theta / 2) / (2 * np.pi)  # phase = θ/(4π) for |1⟩ eigenvalue
    phase_rz = estimate_phase(4, rz, shots=8192)
    print(f"   Expected phase: {expected:.4f}")
    print(f"   Estimated phase: {phase_rz:.4f}")
    print(f"   Error: {abs(phase_rz - expected):.4f}")

    print("\n" + "=" * 60)
    print("Phase estimation complete.")
    print("Precision improves with more counting qubits (t bits → 2^t resolution)")


if __name__ == "__main__":
    main()
