"""Quantum information channels and process tomography demo.

Demonstrates:
1. Channel representations (Kraus, Choi, superoperator, PTM)
2. Distance measures between quantum states
3. Process tomography of quantum gates
"""

import numpy as np
from quantumium.quantum_info import (
    QuantumChannel,
    kraus_to_choi,
    kraus_to_ptm,
    trace_distance,
    bures_distance,
    relative_entropy,
    process_tomography,
    chi_matrix,
)
from quantumium.core.noise import depolarizing, amplitude_damping

print("=" * 60)
print("Quantum Information Channels & Process Tomography")
print("=" * 60)

# --- Channel Representations ---
print("\n1. Channel Representations")
print("-" * 40)

dep_channel = QuantumChannel.from_kraus(
    depolarizing(0.1).kraus_operators, name="Depolarizing(0.1)"
)
print(f"Channel: {dep_channel}")
print(f"  Choi matrix shape: {dep_channel.choi.shape}")
print(f"  Superoperator shape: {dep_channel.superoperator.shape}")
print(f"  PTM (Pauli Transfer Matrix):\n{np.round(dep_channel.ptm, 4)}")
print(f"  Unital: {dep_channel.is_unital()}")
print(f"  Average fidelity: {dep_channel.average_fidelity():.4f}")

ad_channel = QuantumChannel.from_kraus(
    amplitude_damping(0.3).kraus_operators, name="AmplDamp(0.3)"
)
print(f"\nChannel: {ad_channel}")
print(f"  Unital: {ad_channel.is_unital()}")
print(f"  Average fidelity: {ad_channel.average_fidelity():.4f}")

# --- Channel Composition ---
print("\n2. Channel Composition")
print("-" * 40)
composed = dep_channel.compose(ad_channel)
print(f"  Depolarizing ∘ AmplDamp fidelity: {composed.average_fidelity():.4f}")
print(f"  (vs individual: dep={dep_channel.average_fidelity():.4f}, "
      f"ad={ad_channel.average_fidelity():.4f})")

# --- Distance Measures ---
print("\n3. Distance Measures")
print("-" * 40)
rho_pure = np.array([[1, 0], [0, 0]], dtype=complex)
rho_mixed = np.eye(2, dtype=complex) / 2
rho_partial = np.array([[0.7, 0.2], [0.2, 0.3]], dtype=complex)

print(f"  Trace distance (|0><0|, I/2):     {trace_distance(rho_pure, rho_mixed):.4f}")
print(f"  Bures distance (|0><0|, I/2):     {bures_distance(rho_pure, rho_mixed):.4f}")
print(f"  Relative entropy (|0><0| || I/2): {relative_entropy(rho_pure, rho_mixed):.4f} bits")
print(f"  Trace distance (|0><0|, partial):  {trace_distance(rho_pure, rho_partial):.4f}")

# --- Process Tomography ---
print("\n4. Process Tomography")
print("-" * 40)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)

for name, gate in [("Identity", np.eye(2)), ("X", X), ("Hadamard", H)]:
    result = process_tomography(gate, target_unitary=gate)
    print(f"  {name} gate:")
    print(f"    Process fidelity (self-comparison): {result.process_fidelity:.6f}")
    print(f"    Input states used: {result.input_states_used}")

# Cross-comparison
result = process_tomography(X, target_unitary=np.eye(2, dtype=complex))
print(f"\n  X vs Identity fidelity: {result.process_fidelity:.6f}")

theta = 0.05
Rz = np.array([[np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)]], dtype=complex)
result = process_tomography(Rz, target_unitary=np.eye(2, dtype=complex))
print(f"  Rz({theta:.2f}) vs Identity fidelity: {result.process_fidelity:.6f}")

# --- Chi Matrix ---
print("\n5. Chi (Process) Matrix")
print("-" * 40)
chi_I = chi_matrix(np.eye(2, dtype=complex))
chi_X = chi_matrix(X)
print(f"  Identity chi matrix diagonal: {np.round(np.real(np.diag(chi_I)), 4)}")
print(f"  X gate chi matrix diagonal:   {np.round(np.real(np.diag(chi_X)), 4)}")

print("\n" + "=" * 60)
print("Done!")
