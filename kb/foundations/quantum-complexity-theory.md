# Quantum Complexity Theory

## 1. Complexity Classes

### BQP — Bounded-Error Quantum Polynomial Time
The class of decision problems solvable by a quantum computer in
polynomial time with error probability ≤ 1/3.

Known relationships:
- P ⊆ BQP ⊆ PSPACE
- BPP ⊆ BQP (classical randomized is a subset)
- Whether NP ⊆ BQP is unknown (believed unlikely)

BQP is the quantum analog of BPP. The error bound 1/3 is arbitrary —
it can be amplified to 2^(-n) by O(n) repetitions.

### QMA — Quantum Merlin-Arthur
The quantum analog of NP (or more precisely, MA). A verifier (Arthur)
receives a polynomial-size quantum witness from the prover (Merlin)
and decides in polynomial time with bounded error.

- NP ⊆ QMA ⊆ PSPACE
- QMA-complete problems: local Hamiltonian problem (k-local, k ≥ 2)

### Other Quantum Classes
| Class | Description |
|-------|-------------|
| QIP | Quantum Interactive Proofs = PSPACE |
| QIP(2) | 2-message quantum interactive proofs |
| QCMA | QMA with classical witness |
| QAC⁰ | Quantum analog of AC⁰ |
| PostBQP | BQP + postselection = PP |

## 2. Quantum Speedups

### Provable Speedups (query/oracle model)

| Problem | Classical | Quantum | Speedup |
|---------|-----------|---------|---------|
| Unstructured search | O(N) | O(√N) Grover | Quadratic |
| Parity | Ω(N) | Ω(N) | None |
| OR | Ω(N) | O(√N) | Quadratic |
| Element distinctness | Θ(N) | O(N^(2/3)) | Polynomial |
| Triangle finding | O(N²) | Õ(N^(5/4)) | Polynomial |
| Formula evaluation | O(N) | O(√N) | Quadratic |

### Believed Speedups (no proof of classical hardness)

| Problem | Best Classical | Quantum | Type |
|---------|---------------|---------|------|
| Integer factoring | exp(n^(1/3)) | O(n³) Shor | Superpolynomial |
| Discrete logarithm | exp(n^(1/3)) | O(n³) Shor | Superpolynomial |
| Simulation of quantum systems | Exponential | Polynomial | Exponential |
| Boson sampling | #P-hard | Native | Exponential |

## 3. Query Complexity Framework

### Bounded-Error Query Complexity Q_ε(f)
Minimum queries to compute f with error ≤ ε on all inputs.

### Polynomial Method (Beals et al.)
deg(f) ≤ Q_ε(f) ≤ O(deg(f)²)

where deg(f) is the minimum degree of a polynomial that approximates f.

### Adversary Method
Q_ε(f) ≥ Adv(f) = max over adversary matrices Γ of:
‖Γ‖ / max_i ‖Γ ∘ D_i‖

The negative-weight adversary bound is tight: Q_ε(f) = Θ(Adv±(f)).

## 4. Quantum Algorithms by Technique

### Quantum Fourier Transform Family
- Shor's algorithm (factoring, discrete log)
- QPE (phase estimation)
- Hidden subgroup problem (abelian groups)
- Simon's algorithm (period finding in black-box functions)

### Amplitude Amplification Family
- Grover's search
- Quantum counting
- Quantum walk search
- Amplitude estimation

### Variational / Hybrid Family
- VQE (chemistry)
- QAOA (optimization)
- Quantum machine learning (QML)
- Quantum approximate counting

### Hamiltonian Simulation Family
- Product formulas (Trotter-Suzuki)
- Linear combination of unitaries (LCU)
- Quantum signal processing (QSP)
- Qubitization

## 5. Quantum Error Correction Complexity

### Threshold Theorem
Fault-tolerant QC is possible with polylog overhead if physical
error rate p < p_th (threshold).

| Code | Threshold | Overhead |
|------|-----------|----------|
| Concatenated [[7,1,3]] | ~10⁻⁵ | O(polylog(1/ε)) per gate |
| Surface code | ~10⁻² | O(d² log(1/ε)) per logical qubit |
| Color code | ~10⁻³ | Similar to surface code |

### Magic State Distillation
T gates cannot be transversal in any code that also has transversal
Cliffords (Eastin-Knill theorem). T gates are implemented via magic
state distillation, which dominates fault-tolerant overhead.

Resource estimates for cryptographically relevant factoring (2048-bit):
- ~20 million physical qubits (surface code)
- ~8 hours runtime
- Physical error rate ~10⁻³

## 6. Quantum Supremacy / Advantage

### Random Circuit Sampling
Google Sycamore (2019): 53 qubits, depth ~20, claimed 200 seconds
vs estimated 10,000 years classically.

Subsequently improved classical simulations reduced the gap, but
the scaling argument holds for larger circuits.

### Computational Hardness Assumptions
- Random circuit sampling is hard to classically simulate
  (under plausible complexity assumptions)
- Boson sampling hardness relies on permanent-of-Gaussians conjecture
- IQP hardness relates to polynomial hierarchy collapse

## 7. Limits of Quantum Computing

- Cannot solve NP-complete problems efficiently (believed)
- Grover's quadratic speedup is optimal for unstructured search
  (BBBV theorem)
- No-cloning theorem prevents copying quantum states
- Holevo bound: n qubits can communicate at most n classical bits
- Quantum error rates must stay below threshold for fault tolerance
