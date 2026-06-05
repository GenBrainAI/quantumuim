# Quantum Information Theory

## 1. Classical vs Quantum Information

Classical information is measured in **bits** (0 or 1). Quantum information is measured in **qubits** — which can exist in superposition and be entangled.

Key differences:
- Qubits cannot be copied (no-cloning theorem)
- Measuring a qubit disturbs it
- Entangled qubits share correlations impossible classically
- Quantum information can be teleported but not transmitted faster than light

## 2. Entropy Measures

### Von Neumann Entropy

The quantum analog of Shannon entropy. For density matrix ρ:

S(ρ) = -Tr(ρ log₂ ρ) = -Σᵢ λᵢ log₂ λᵢ

where λᵢ are eigenvalues of ρ.

Properties:
- S(ρ) = 0 iff ρ is a pure state
- S(ρ) ≤ log₂ d for a d-dimensional system (maximized by maximally mixed state)
- S(ρ) is concave: S(Σᵢ pᵢρᵢ) ≥ Σᵢ pᵢS(ρᵢ)
- Subadditivity: S(ρ_AB) ≤ S(ρ_A) + S(ρ_B)
- Strong subadditivity: S(ρ_ABC) + S(ρ_B) ≤ S(ρ_AB) + S(ρ_BC)

### Quantum Relative Entropy

S(ρ ‖ σ) = Tr(ρ log₂ ρ) - Tr(ρ log₂ σ)

- Always non-negative (Klein's inequality)
- Zero iff ρ = σ
- Not symmetric: S(ρ ‖ σ) ≠ S(σ ‖ ρ) in general
- Monotone under quantum channels: S(Φ(ρ) ‖ Φ(σ)) ≤ S(ρ ‖ σ)

### Mutual Information

I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)

Measures total correlations (classical + quantum) between subsystems A and B.

### Conditional Entropy

S(A|B) = S(ρ_AB) - S(ρ_B)

Unlike classical conditional entropy, quantum conditional entropy can be **negative** — this happens for entangled states and is related to quantum advantage in communication.

## 3. Quantum Channels

A quantum channel Φ is a completely positive, trace-preserving (CPTP) map.

### Kraus Representation
Φ(ρ) = Σₖ Eₖ ρ Eₖ†, where Σₖ Eₖ†Eₖ = I

### Stinespring Dilation
Every channel can be represented as: Φ(ρ) = Tr_E[U(ρ ⊗ |0⟩⟨0|)U†]

where U is a unitary on the system + environment.

### Common Channels

| Channel | Effect | Kraus Operators |
|---------|--------|----------------|
| Depolarizing | ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ) | √(1-3p/4)I, √(p/4)X, √(p/4)Y, √(p/4)Z |
| Amplitude Damping | Energy relaxation T₁ | E₀ = [[1,0],[0,√(1-γ)]], E₁ = [[0,√γ],[0,0]] |
| Phase Damping | Dephasing T₂ | E₀ = √(1-p)I, E₁ = √p|0⟩⟨0|, E₂ = √p|1⟩⟨1| |
| Erasure | Qubit lost with prob p | √(1-p)ρ, √p|e⟩⟨e| (erasure flag) |

### Channel Capacity

Quantum channel capacity C(Φ) = max_ρ [S(Φ(ρ)) - Σₖ pₖ S(Φ(ρₖ))]

The Holevo bound limits classical information extractable from quantum states:
χ = S(ρ) - Σᵢ pᵢ S(ρᵢ) (Holevo quantity)

## 4. Fidelity

### State Fidelity
F(ρ, σ) = (Tr√(√ρ σ √ρ))²

For pure states: F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|²

Properties:
- 0 ≤ F ≤ 1, with F = 1 iff ρ = σ
- Symmetric: F(ρ, σ) = F(σ, ρ)
- Monotone under channels: F(Φ(ρ), Φ(σ)) ≥ F(ρ, σ)

### Process Fidelity
F_proc(Φ, Ψ) = F(J(Φ), J(Ψ)) where J is the Choi-Jamiołkowski isomorphism

### Average Gate Fidelity
F_avg = ∫ ⟨ψ|Φ†∘Ψ(|ψ⟩⟨ψ|)|ψ⟩ dψ

Used to benchmark quantum gates against ideal operations.

## 5. Entanglement Measures

### Entanglement Entropy
For bipartite pure state |ψ⟩_AB: E(ψ) = S(ρ_A) = S(ρ_B)

### Concurrence (2-qubit systems)
C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)

where λᵢ are eigenvalues (decreasing) of √(√ρ ρ̃ √ρ), and ρ̃ = (Y⊗Y)ρ*(Y⊗Y)

### Entanglement of Formation
E_F(ρ) = h((1 + √(1-C²))/2) where h is binary entropy

### Negativity
N(ρ) = (‖ρ^(T_B)‖₁ - 1) / 2

where ρ^(T_B) is the partial transpose. N > 0 implies entanglement.

## 6. Quantum Error Correction

### The Error Correction Conditions (Knill-Laflamme)
A code with projector P corrects errors {Eₐ} iff:
P Eₐ† Eᵦ P = αₐᵦ P for some Hermitian matrix α

### Stabilizer Formalism
- n-qubit stabilizer group S ⊂ {I, X, Y, Z}^⊗n
- Code space = simultaneous +1 eigenspace of all S generators
- [[n, k, d]] code: n physical qubits, k logical qubits, distance d
- Can correct ⌊(d-1)/2⌋ errors

### Key Codes
| Code | Parameters | Notes |
|------|-----------|-------|
| Bit-flip | [[3,1,1]] | Corrects single X errors |
| Phase-flip | [[3,1,1]] | Corrects single Z errors |
| Shor code | [[9,1,3]] | Corrects arbitrary single-qubit errors |
| Steane code | [[7,1,3]] | CSS code, transversal gates |
| Surface code | [[O(d²),1,d]] | 2D layout, ~1% threshold |

### Threshold Theorem
If physical error rate p < p_threshold, fault-tolerant quantum computation is possible with polylogarithmic overhead. Surface code threshold ≈ 1%.

## 7. Quantum Communication Protocols

### Superdense Coding
Send 2 classical bits using 1 qubit + 1 shared Bell pair.

### Quantum Teleportation
Transfer an unknown quantum state using 2 classical bits + 1 shared Bell pair.

### Quantum Key Distribution (BB84)
- Alice sends qubits in random bases (Z or X)
- Bob measures in random bases
- They publicly compare bases, keep matching results
- Eavesdropping introduces detectable errors

## 8. Resource Theory Framework

Quantum information can be viewed through resource theories:
- **Entanglement:** free operations = LOCC, resource = entangled states
- **Coherence:** free operations = incoherent operations, resource = superposition
- **Magic:** free operations = Clifford gates, resource = non-stabilizer states (T gates)

The magic resource is directly related to classical simulability — states with low magic can be efficiently simulated classically (Gottesman-Knill theorem).
