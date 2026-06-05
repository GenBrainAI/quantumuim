"""State tomography demo — reconstructing quantum states from measurements.

Demonstrates:
  1. Creating quantum states (Bell state, random state)
  2. Collecting tomography measurements in Pauli bases
  3. Reconstructing the density matrix via linear inversion
  4. Comparing with the ideal state
"""
from __future__ import annotations

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT, Ry, X
from quantumium.core.noise import NoiseModel, depolarizing
from quantumium.tomography.state_tomography import (
    collect_tomography_data,
    linear_inversion,
)
from quantumium.benchmarks import state_fidelity


def single_qubit_tomography():
    print("=" * 60)
    print("  State Tomography — Single Qubit |+> State")
    print("=" * 60)

    qc = QuantumCircuit(1)
    qc.apply(H, 0)

    ideal_sv = qc.statevector()
    ideal_rho = np.outer(ideal_sv, ideal_sv.conj())
    print(f"\n  Ideal state: |+> = (|0> + |1>)/√2")
    print(f"  Ideal density matrix:")
    print(f"    {np.round(ideal_rho, 4)}")

    data = collect_tomography_data(qc, shots_per_basis=4096)
    rho_recon = linear_inversion(data, num_qubits=1)

    print(f"\n  Reconstructed density matrix (linear inversion, 4096 shots/basis):")
    print(f"    {np.round(rho_recon, 4)}")

    fid = state_fidelity(ideal_sv, np.diag(rho_recon).real)
    print(f"\n  Diagonal comparison:")
    print(f"    Ideal:   [{ideal_rho[0,0].real:.4f}, {ideal_rho[1,1].real:.4f}]")
    print(f"    Recon:   [{rho_recon[0,0].real:.4f}, {rho_recon[1,1].real:.4f}]")


def bell_state_tomography():
    print("\n" + "=" * 60)
    print("  State Tomography — Bell State (|00> + |11>)/√2")
    print("=" * 60)

    qc = QuantumCircuit(2)
    qc.apply(H, 0)
    qc.apply(CNOT, 0, 1)

    ideal_sv = qc.statevector()
    ideal_rho = np.outer(ideal_sv, ideal_sv.conj())

    print(f"\n  Ideal statevector: {np.round(ideal_sv, 4)}")
    print(f"  Ideal density matrix (4x4):")
    for row in ideal_rho:
        print(f"    [{', '.join(f'{x.real:+.3f}' for x in row)}]")

    data = collect_tomography_data(qc, shots_per_basis=8192)
    rho_recon = linear_inversion(data, num_qubits=2)

    print(f"\n  Reconstructed density matrix:")
    for row in rho_recon:
        print(f"    [{', '.join(f'{x.real:+.3f}' for x in row)}]")

    # Check key properties
    trace = np.trace(rho_recon).real
    purity = np.trace(rho_recon @ rho_recon).real
    print(f"\n  Trace:  {trace:.4f}  (should be 1)")
    print(f"  Purity: {purity:.4f}  (should be 1 for pure state)")


def noisy_tomography():
    print("\n" + "=" * 60)
    print("  State Tomography — Noisy Measurement")
    print("=" * 60)

    qc = QuantumCircuit(1)
    qc.apply(H, 0)

    noise = NoiseModel(default_single_qubit_noise=depolarizing(0.05))

    print(f"\n  State: |+> with 5% depolarizing noise")
    data_noisy = collect_tomography_data(qc, shots_per_basis=4096, noise_model=noise)
    rho_noisy = linear_inversion(data_noisy, num_qubits=1)

    data_ideal = collect_tomography_data(qc, shots_per_basis=4096)
    rho_ideal = linear_inversion(data_ideal, num_qubits=1)

    print(f"\n  Ideal reconstruction:")
    print(f"    {np.round(rho_ideal, 4)}")
    print(f"    Purity: {np.trace(rho_ideal @ rho_ideal).real:.4f}")

    print(f"\n  Noisy reconstruction:")
    print(f"    {np.round(rho_noisy, 4)}")
    print(f"    Purity: {np.trace(rho_noisy @ rho_noisy).real:.4f}")
    print(f"    (Noise reduces purity — mixed state)")


if __name__ == "__main__":
    single_qubit_tomography()
    bell_state_tomography()
    noisy_tomography()
    print("\nDone!")
