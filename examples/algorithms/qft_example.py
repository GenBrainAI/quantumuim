"""Quantum Fourier Transform demonstration.

Shows QFT on various input states and verifies QFT†∘QFT = I.
"""

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import X, H
from quantumium.algorithms.qft import qft, inverse_qft


def main():
    print("=" * 60)
    print("Quantum Fourier Transform — Quantumium")
    print("=" * 60)

    # QFT on |000⟩ → uniform superposition
    print("\n1. QFT on |000⟩ (3 qubits)")
    qc = QuantumCircuit(3)
    qft(qc, [0, 1, 2])
    sv = qc.statevector()
    print(f"   Statevector: {np.round(sv, 4)}")
    print(f"   Expected: uniform 1/√8 ≈ {1/np.sqrt(8):.4f} for each amplitude")
    counts = qc.measure_all(1000)
    print(f"   Measurements (1000 shots): {counts}")

    # QFT on |101⟩ → Fourier basis state
    print("\n2. QFT on |101⟩ (binary 5)")
    qc2 = QuantumCircuit(3)
    qc2.apply(X, 0)
    qc2.apply(X, 2)
    qft(qc2, [0, 1, 2])
    sv2 = qc2.statevector()
    print(f"   Statevector magnitudes: {np.round(np.abs(sv2), 4)}")
    print(f"   Phases (degrees): {np.round(np.angle(sv2, deg=True), 1)}")

    # Round-trip: QFT then inverse QFT
    print("\n3. Round-trip test: QFT† ∘ QFT on |011⟩")
    qc3 = QuantumCircuit(3)
    qc3.apply(X, 0)
    qc3.apply(X, 1)
    original = qc3.statevector()
    qft(qc3, [0, 1, 2])
    print(f"   After QFT:  {np.round(qc3.statevector(), 4)}")
    inverse_qft(qc3, [0, 1, 2])
    recovered = qc3.statevector()
    print(f"   After QFT†: {np.round(recovered, 4)}")
    fidelity = abs(np.dot(original.conj(), recovered)) ** 2
    print(f"   Fidelity with original: {fidelity:.10f}")

    # QFT on superposition
    print("\n4. QFT on superposition state H|0⟩⊗|0⟩⊗|0⟩")
    qc4 = QuantumCircuit(3)
    qc4.apply(H, 0)
    qft(qc4, [0, 1, 2])
    sv4 = qc4.statevector()
    print(f"   Statevector magnitudes: {np.round(np.abs(sv4), 4)}")
    counts4 = qc4.measure_all(1000)
    print(f"   Measurements (1000 shots): {counts4}")

    print("\n" + "=" * 60)
    print("All QFT demonstrations complete.")


if __name__ == "__main__":
    main()
