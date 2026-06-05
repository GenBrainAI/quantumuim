# Quantum Mechanics Primer for Quantum Computing

## 1. The Qubit

A classical bit is either 0 or 1. A **qubit** exists in a superposition of both states simultaneously:

|ψ⟩ = α|0⟩ + β|1⟩

where α, β ∈ ℂ and |α|² + |β|² = 1.

- |α|² = probability of measuring 0
- |β|² = probability of measuring 1
- The state |ψ⟩ lives in a 2-dimensional complex Hilbert space ℂ²

### Bloch Sphere Representation

Any single-qubit pure state can be written as:

|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩

where θ ∈ [0, π] and φ ∈ [0, 2π) define a point on the Bloch sphere.

## 2. Multiple Qubits and Tensor Products

An n-qubit system lives in the tensor product space (ℂ²)^⊗n, which has dimension 2^n.

Two-qubit basis states: |00⟩, |01⟩, |10⟩, |11⟩

General two-qubit state:
|ψ⟩ = α₀₀|00⟩ + α₀₁|01⟩ + α₁₀|10⟩ + α₁₁|11⟩

### Entanglement

A state is **entangled** if it cannot be written as a tensor product of individual qubit states.

Bell states (maximally entangled):
- |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
- |Φ⁻⟩ = (|00⟩ - |11⟩) / √2
- |Ψ⁺⟩ = (|01⟩ + |10⟩) / √2
- |Ψ⁻⟩ = (|01⟩ - |10⟩) / √2

## 3. Quantum Gates (Unitary Operators)

Quantum computation proceeds by applying **unitary matrices** to qubit states. A matrix U is unitary iff U†U = UU† = I.

### Single-Qubit Gates

**Pauli Gates:**
```
X = |0 1|    Y = |0 -i|    Z = |1  0|
    |1 0|        |i  0|        |0 -1|
```

- X (NOT gate): flips |0⟩ ↔ |1⟩
- Z: phase flip, |1⟩ → -|1⟩
- Y = iXZ

**Hadamard Gate:**
```
H = (1/√2) |1  1|
            |1 -1|
```
H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
H|1⟩ = |-⟩ = (|0⟩ - |1⟩)/√2

**Phase Gates:**
```
S = |1 0|    T = |1    0  |
    |0 i|        |0 e^iπ/4|
```

**Rotation Gates:**
- Rx(θ) = e^(-iθX/2) — rotation about X-axis
- Ry(θ) = e^(-iθY/2) — rotation about Y-axis
- Rz(θ) = e^(-iθZ/2) — rotation about Z-axis

### Multi-Qubit Gates

**CNOT (Controlled-NOT):**
```
CNOT = |1 0 0 0|
       |0 1 0 0|
       |0 0 0 1|
       |0 0 1 0|
```
Flips target qubit iff control qubit is |1⟩.

**Toffoli (CCNOT):** 3-qubit gate, flips target iff both controls are |1⟩.

**SWAP:** Exchanges two qubits. SWAP = CNOT₁₂ · CNOT₂₁ · CNOT₁₂

**CZ (Controlled-Z):** Applies Z to target iff control is |1⟩.

### Universal Gate Sets

Any unitary operation can be approximated to arbitrary precision using:
- {H, T, CNOT} — standard universal set
- {Rx, Ry, CNOT} — continuous universal set

## 4. Measurement

Measurement in the computational basis {|0⟩, |1⟩}:
- State |ψ⟩ = α|0⟩ + β|1⟩ collapses to |0⟩ with probability |α|² or |1⟩ with probability |β|²
- Measurement is **irreversible** — it destroys superposition
- Projective measurement operators: P₀ = |0⟩⟨0|, P₁ = |1⟩⟨1|

### Measurement in Other Bases

To measure in the {|+⟩, |-⟩} basis, apply H before measuring in computational basis.

## 5. Quantum Circuit Model

Computation as a sequence of:
1. **Initialize** qubits to |0⟩^⊗n
2. **Apply** sequence of quantum gates (unitary transformations)
3. **Measure** some or all qubits

Time flows left to right. Each horizontal line is a qubit wire.

```
|0⟩ ─── H ───●─── M
              │
|0⟩ ─────────X─── M
```
This creates a Bell state |Φ⁺⟩ then measures both qubits.

## 6. Density Matrices

For mixed states (statistical ensembles of pure states):

ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|

Properties:
- ρ is Hermitian (ρ† = ρ)
- Tr(ρ) = 1
- ρ is positive semidefinite
- Pure state: Tr(ρ²) = 1; Mixed state: Tr(ρ²) < 1

## 7. No-Cloning Theorem

It is impossible to create an identical copy of an arbitrary unknown quantum state. This is a fundamental constraint on quantum information processing and is key to quantum cryptography.

## 8. Decoherence and Noise

Real quantum systems interact with their environment, causing:
- **T₁ (relaxation time):** excited state decay |1⟩ → |0⟩
- **T₂ (dephasing time):** loss of phase coherence
- **Gate errors:** imperfect gate implementations
- **Readout errors:** measurement inaccuracies

This is why current quantum computers are called **NISQ** (Noisy Intermediate-Scale Quantum) devices.
