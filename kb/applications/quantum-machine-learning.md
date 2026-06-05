# Quantum Machine Learning

## 1. Overview

Quantum Machine Learning (QML) sits at the intersection of quantum computing and machine learning, exploring whether quantum processors can learn from data faster or more expressively than classical computers.

**Key question:** Can quantum computers provide a genuine advantage for ML tasks?

The answer is nuanced — provable exponential speedups exist for specific problems (e.g., HHL for linear systems), while for others the advantage is polynomial or remains an open question.

## 2. Data Encoding (Feature Maps)

Classical data must be encoded into quantum states before processing. The encoding choice significantly impacts algorithm performance.

### Amplitude Encoding
Maps N-dimensional data into log₂(N) qubits via state amplitudes:
- **x** = (x₁, x₂, ..., xₙ) → |ψ⟩ = Σᵢ xᵢ|i⟩ / ‖x‖
- Exponential compression: 2ⁿ features in n qubits
- **Cost:** State preparation may require O(N) gates, negating the compression advantage

### Angle Encoding
Maps each feature to a rotation angle on a dedicated qubit:
- xᵢ → Rᵧ(xᵢ)|0⟩ = cos(xᵢ/2)|0⟩ + sin(xᵢ/2)|1⟩
- Linear scaling: N features require N qubits
- Simple to implement, naturally compatible with variational circuits

### ZZ Feature Map (Pauli Feature Map)
Creates entanglement between features for richer Hilbert space embedding:
- Single-qubit rotations: Rz(xᵢ) on each qubit
- Entangling layers: CNOT + Rz(xᵢxⱼ) for feature interactions
- Maps data to a space where linear separation may capture nonlinear classical boundaries
- **Depth:** scales as O(n²) for all-pairs interaction

### IQP (Instantaneous Quantum Polynomial) Encoding
- Diagonal unitals with computational basis measurements
- Classically hard to simulate (under plausible assumptions)
- Provides theoretical separation between quantum and classical kernels

## 3. Variational Quantum Classifier (VQC)

A hybrid quantum-classical approach where a parameterized quantum circuit acts as the model.

### Architecture
1. **Encode:** Apply feature map U(x) to |0⟩ⁿ
2. **Variational layers:** Apply parameterized ansatz V(θ) — rotations + entanglement
3. **Measure:** Read out expectation value ⟨Z⟩ on designated qubits
4. **Classical optimizer:** Update θ to minimize loss

### Parameter-Shift Rule
Exact gradients on quantum hardware without finite differences:
- ∂f/∂θᵢ = [f(θᵢ + π/2) - f(θᵢ - π/2)] / 2
- Requires only 2 circuit evaluations per parameter
- Works for any gate of the form e^{-iθG/2} where G² = I

### Barren Plateaus
A central challenge — gradients vanish exponentially with circuit depth/width:
- For random circuits on n qubits: Var[∂L/∂θ] ∝ 2⁻ⁿ
- **Mitigations:** local cost functions, shallow circuits, structured ansatze, layer-wise training, parameter initialization strategies

### Expressibility vs Trainability Trade-off
- Highly expressive circuits → more barren plateaus
- Structured/problem-aware ansatze → better trainability but risk underfitting
- Optimal balance depends on the problem and available qubits

## 4. Quantum Kernel Methods

Instead of parameterized circuits, use the quantum computer to compute a kernel:
- K(xᵢ, xⱼ) = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|²
- Build a Gram matrix, feed to classical SVM or kernel regression
- **Advantage:** No barren plateau problem, convex optimization on the classical side

### Fidelity Quantum Kernel
- Prepare |ψ(xᵢ)⟩, apply U†(xⱼ), measure probability of |0⟩ⁿ
- Each kernel entry requires O(1/ε²) shots for precision ε
- Can exhibit quantum advantage when the feature map is classically hard to simulate

### Projected Quantum Kernel
- Measure local observables instead of global fidelity
- More robust to noise, avoids exponential concentration
- K(x,y) = exp(-γ Σₖ ‖⟨ψ(x)|Oₖ|ψ(x)⟩ - ⟨ψ(y)|Oₖ|ψ(y)⟩‖²)

## 5. Quantum Neural Networks

### Quantum Convolutional Neural Networks (QCNN)
- Inspired by classical CNNs: alternating convolutional and pooling layers
- Pooling via measurement: measure subset of qubits, condition subsequent operations
- Logarithmic depth in number of qubits → avoids barren plateaus
- Applications: quantum phase recognition, error correction decoding

### Quantum Reservoir Computing
- Fixed random quantum circuit as reservoir (no training of quantum parameters)
- Only classical readout layer is trained
- Leverages quantum dynamics for temporal pattern recognition
- **Advantage:** avoids barren plateaus entirely since quantum circuit isn't optimized

## 6. Quantum Advantage in ML — Current Understanding

### Provable Advantages
- **Linear systems (HHL):** Exponential speedup for sparse systems, but output is quantum state (readout bottleneck)
- **Recommendation systems (Kerenidis-Prakash):** Exponential speedup, though "dequantized" by Tang's classical algorithm for low-rank case
- **Quantum data:** Learning properties of quantum states/processes has provable exponential advantage

### Heuristic/Empirical Advantages
- Small datasets with specific structure may benefit from quantum kernels
- Quantum generative models (QBMs) may capture correlations classical models struggle with
- Advantage depends heavily on data structure matching the quantum feature map

### No-Go Results
- Random quantum circuits offer no advantage over random classical features
- For generic classical data, encoding overhead can negate any quantum speedup
- Classical surrogates (tensor networks, shadow tomography) close the gap in many cases

## 7. Near-Term QML Landscape

| Approach | Qubits Needed | Noise Tolerance | Classical Competition |
|----------|--------------|-----------------|----------------------|
| VQC | 5-50 | Moderate | Strong — classical NNs |
| Quantum Kernels | 5-100 | Good | Moderate — classical kernels |
| QCNN | 10-100 | Good | Moderate |
| Reservoir | 5-50 | Excellent | Strong |

**Practical advice:** Focus QML on problems with quantum structure (molecular properties, quantum control, materials) where the data is inherently quantum. For classical data, the burden of proof for quantum advantage remains high.

## 8. Relation to Quantumium SDK

Our SDK implements:
- **Data encoding:** amplitude, angle, ZZ feature map
- **VQC:** parameterized circuits with parameter-shift gradient computation
- **Quantum kernels:** fidelity-based kernel with Gram matrix construction
- See `quantumium.qml` package for implementations
