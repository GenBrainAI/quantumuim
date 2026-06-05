# Quantum Error Correction — Deep Dive

## 1. Why QEC Is Necessary

Quantum states are fragile — interaction with the environment causes
decoherence (T₁ energy relaxation, T₂ dephasing). Gate errors compound
exponentially with circuit depth. Without error correction, useful
quantum computation is impossible beyond ~100 gates.

QEC encodes logical qubits into many physical qubits so that errors
can be detected and corrected without measuring (and destroying) the
encoded quantum information.

## 2. The Knill-Laflamme Conditions

A quantum code with codespace projector P corrects error set {Eₐ} iff:

P Eₐ† Eᵦ P = αₐᵦ P

for some Hermitian matrix α. This means:
- Errors are detectable (off-diagonal condition)
- Errors don't corrupt the encoded information (diagonal condition)

## 3. Stabilizer Codes

### Framework
An [[n, k, d]] stabilizer code:
- Uses n physical qubits to encode k logical qubits
- Has distance d (corrects ⌊(d-1)/2⌋ arbitrary errors)
- Defined by a stabilizer group S = ⟨g₁, ..., g_{n-k}⟩ where
  each gᵢ is a Pauli operator and all generators commute

The code space is the simultaneous +1 eigenspace of all stabilizers.

### Error Detection
When error E occurs on the code state, measure each stabilizer gᵢ:
- If gᵢ E |ψ⟩ = +|ψ⟩, syndrome bit = 0 (E commutes with gᵢ)
- If gᵢ E |ψ⟩ = -|ψ⟩, syndrome bit = 1 (E anticommutes with gᵢ)

The syndrome uniquely identifies correctable errors.

### Implemented Codes

#### 3-Qubit Bit-Flip Code [[3,1,1]]
- Stabilizers: Z₀Z₁, Z₁Z₂
- Logical states: |0_L⟩ = |000⟩, |1_L⟩ = |111⟩
- Corrects: single X error on any qubit
- Cannot correct: Z or Y errors

| Syndrome | Error |
|----------|-------|
| (0, 0) | None |
| (1, 0) | X₀ |
| (1, 1) | X₁ |
| (0, 1) | X₂ |

#### 3-Qubit Phase-Flip Code [[3,1,1]]
- Stabilizers: X₀X₁, X₁X₂
- Logical states: |0_L⟩ = |+++⟩, |1_L⟩ = |---⟩
- Corrects: single Z error on any qubit
- Hadamard conjugate of the bit-flip code

#### Shor Code [[9,1,3]]
- Concatenation: phase-flip code ∘ bit-flip code
- Each logical qubit → 3 phase-flip qubits → 9 physical qubits
- Corrects: any single-qubit error (X, Y, or Z)
- 8 stabilizer generators (6 for bit-flip + 2 for phase-flip)

Logical states:
```
|0_L⟩ = (|000⟩+|111⟩)(|000⟩+|111⟩)(|000⟩+|111⟩) / 2√2
|1_L⟩ = (|000⟩-|111⟩)(|000⟩-|111⟩)(|000⟩-|111⟩) / 2√2
```

## 4. CSS Codes

Calderbank-Shor-Steane (CSS) codes separate X and Z error correction:
- Built from two classical codes C₁ ⊇ C₂
- X errors corrected by C₁
- Z errors corrected by C₂⊥

### Steane Code [[7,1,3]]
- Based on the classical [7,4,3] Hamming code
- 6 stabilizer generators (3 X-type, 3 Z-type)
- Supports transversal CNOT, H, and S gates
- Lower overhead than Shor code (7 vs 9 qubits)

## 5. Surface Codes

The leading candidate for near-term fault-tolerant QC:
- [[d², 1, d]] topological code on a 2D grid
- Only nearest-neighbor interactions required
- High threshold: ~1% physical error rate
- Stabilizers: X-type (plaquettes) and Z-type (vertices)

### Resource Estimates
| Distance d | Physical qubits | Error rate floor |
|-----------|----------------|-----------------|
| 3 | 17 | ~10⁻³ |
| 5 | 49 | ~10⁻⁵ |
| 7 | 97 | ~10⁻⁷ |
| 9 | 161 | ~10⁻⁹ |
| 17 | 577 | ~10⁻¹⁵ |

### Logical Operations
- X_L and Z_L: chains of physical X/Z across the lattice
- CNOT_L: lattice surgery (merge and split)
- T_L: magic state distillation (dominant overhead)

## 6. Fault Tolerance

A computation is fault-tolerant if a single physical error causes
at most one error per code block (no error propagation).

### Key Principles
- **Transversal gates**: apply gate qubit-by-qubit across code blocks.
  Errors can't spread within a block. Not all gates can be transversal
  (Eastin-Knill theorem).
- **Flag qubits**: detect when syndrome measurement spreads errors
- **Magic state distillation**: produce high-fidelity T states from
  noisy ones using Clifford circuits

### Threshold Theorem
With concatenated codes, if p < p_th:
- Effective error rate after L levels: p_eff ~ (p/p_th)^(2^L)
- Overhead per logical qubit: O(polylog(1/ε))

## 7. Beyond Stabilizer Codes

### Bosonic Codes
Encode qubits in oscillator modes (cat codes, GKP codes, binomial codes).
Hardware-efficient — one oscillator replaces many physical qubits.

### Quantum LDPC Codes
Low-density parity-check codes with constant rate (k/n → const):
- Fewer physical qubits per logical qubit than surface codes
- Require long-range connectivity
- Active research area (2024-2025 breakthroughs)

### Color Codes
Topological codes on trivalent lattices:
- Support transversal Clifford gates
- Threshold ~0.1-0.3% (lower than surface codes)
- Useful for magic state distillation factories

## 8. Error Mitigation vs Error Correction

| Aspect | Mitigation | Correction |
|--------|-----------|------------|
| Qubits | No overhead | O(d²) per logical qubit |
| Gates | No extra gates | Syndrome circuits |
| Threshold | None | Required (p < p_th) |
| Scalability | Exponential sampling | Polynomial overhead |
| Use case | NISQ (near-term) | Fault-tolerant (future) |

Error mitigation (ZNE, measurement correction, twirling) is a
stopgap for current NISQ devices. True fault-tolerant QC requires
error correction, which needs hardware with error rates below
threshold and enough physical qubits for encoding.
