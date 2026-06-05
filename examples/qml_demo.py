"""Example: Quantum Machine Learning.

Demonstrates data encoding, quantum kernel estimation,
and variational quantum classification.
"""

import numpy as np

from quantumium.qml.encoding import amplitude_encode, angle_encode, zz_feature_map
from quantumium.qml.kernel import quantum_kernel, quantum_kernel_matrix
from quantumium.qml.classifier import variational_classifier

print("=" * 55)
print("Quantum Machine Learning Demo")
print("=" * 55)

# --- Data Encoding ---
print("\n--- 1. Data Encoding ---")

data = np.array([0.6, 0.8])
qc = amplitude_encode(data)
sv = qc.statevector()
print(f"Amplitude encoding of {data}:")
print(f"  {qc.num_qubits} qubit(s), amplitudes: {np.abs(sv).round(4)}")

data3 = np.array([0.5, 1.0, 1.5])
qc2 = angle_encode(data3)
print(f"\nAngle encoding of {data3}:")
print(f"  {qc2.num_qubits} qubits")

data_zz = np.array([0.3, 0.7])
qc3 = zz_feature_map(data_zz, reps=2)
sv3 = qc3.statevector()
print(f"\nZZ feature map of {data_zz}:")
print(f"  {qc3.num_qubits} qubits, |state| = {np.abs(sv3).round(4)}")

# --- Quantum Kernel ---
print("\n--- 2. Quantum Kernel ---")

x1 = np.array([0.0, 0.0])
x2 = np.array([1.0, 0.5])
x3 = np.array([2.0, 1.5])

print(f"k(x1, x1) = {quantum_kernel(x1, x1):.4f}  (self-similarity)")
print(f"k(x1, x2) = {quantum_kernel(x1, x2):.4f}")
print(f"k(x1, x3) = {quantum_kernel(x1, x3):.4f}")
print(f"k(x2, x3) = {quantum_kernel(x2, x3):.4f}")

X = np.array([x1, x2, x3])
K = quantum_kernel_matrix(X, feature_map="zz", reps=1)
print(f"\nKernel matrix (ZZ, reps=1):")
for row in K:
    print(f"  [{' '.join(f'{v:.4f}' for v in row)}]")

# --- Variational Classifier ---
print("\n--- 3. Variational Quantum Classifier ---")

X_train = np.array([
    [0.2, 0.1],
    [0.3, 0.2],
    [2.8, 2.9],
    [3.0, 2.8],
])
y_train = np.array([1.0, 1.0, -1.0, -1.0])

print(f"Training data: {len(X_train)} samples, 2 features")
print(f"Class +1: small angles, Class -1: large angles")

result = variational_classifier(
    X_train, y_train,
    num_layers=2, epochs=40, learning_rate=0.15, seed=42
)

print(f"Training accuracy: {result.accuracy:.0%}")
print(f"Final loss: {result.loss_history[-1]:.4f}")
print(f"Loss reduction: {result.loss_history[0]:.4f} -> {result.loss_history[-1]:.4f}")

print("\n" + "=" * 55)
