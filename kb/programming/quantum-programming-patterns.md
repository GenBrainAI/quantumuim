# Quantum Programming Patterns

Practical patterns for writing quantum programs, organized by problem type.
These patterns apply across SDKs but examples use Quantumium syntax.

## 1. The Prepare-Entangle-Measure Pattern

The most fundamental quantum programming pattern. Nearly every quantum algorithm follows this three-phase structure:

```
Phase 1: Prepare    — Initialize qubits into useful superpositions
Phase 2: Entangle   — Create correlations via multi-qubit gates
Phase 3: Measure    — Extract classical information
```

**Example: Bell State**
```python
qc = QuantumCircuit(2)
qc.apply(H, 0)        # Prepare: superposition
qc.apply(CNOT, 0, 1)  # Entangle: correlate qubits
qc.measure_all()       # Measure: extract result
```

**When to use:** Every quantum program. The art is choosing what to prepare, how to entangle, and what measurement basis to use.

## 2. The Oracle Pattern

Encode a classical function f(x) as a reversible quantum operation U_f, then use quantum parallelism to evaluate it on all inputs simultaneously.

```
|x>|y> → |x>|y ⊕ f(x)>    (standard oracle)
|x>    → (-1)^f(x)|x>       (phase oracle)
```

**Key design decisions:**
- **Standard oracle** when you need the function value in a register
- **Phase oracle** when you need to mark states for amplitude amplification

**Algorithms using this:** Grover's search, Deutsch-Jozsa, Bernstein-Vazirani, Simon's

**Implementation tip:** Converting a classical boolean function to a phase oracle:
1. Put the target qubit in |-> state (X then H)
2. Apply the standard oracle
3. The target qubit acts as a phase kickback mechanism
4. Result: the phase oracle acts on the input register alone

## 3. The Amplitude Amplification Pattern

Given a quantum state with small probability of being "good", boost that probability to near-certainty using O(sqrt(N/M)) iterations (N states, M good).

```
repeat O(sqrt(N/M)) times:
    1. Apply oracle (flip phase of good states)
    2. Apply diffusion (reflect about mean amplitude)
```

**Key parameters:**
- Too few iterations: probability still low
- Too many iterations: probability decreases again (overshooting)
- Optimal iterations: floor(pi/4 * sqrt(N/M))

**Beyond Grover's:** Amplitude amplification works with any state preparation unitary A, not just Hadamard. If A|0> has overlap sin(theta) with the good subspace, then O(1/sin(theta)) iterations suffice.

## 4. The Phase Estimation Pattern

Extract the eigenvalue of a unitary operator by mapping it to a register of qubits, then applying inverse QFT.

```
1. Prepare eigenstate |ψ> in target register
2. Apply controlled-U^(2^k) for k = 0, 1, ..., t-1
3. Apply inverse QFT to the control register
4. Measure control register → phase estimate
```

**Precision:** t qubits give phase to t bits of precision (error < 2^(-t))

**Algorithms using this:** Shor's factoring, HHL linear solver, quantum counting, energy estimation in chemistry

**Common pitfall:** If the input is not an eigenstate, you get a superposition of eigenvalues — which is sometimes desirable (e.g., ground state energy estimation via VQE/QPE hybrid).

## 5. The Variational Pattern (VQE/QAOA)

Use a classical optimizer to tune parameters of a quantum circuit, minimizing a cost function evaluated on the quantum computer.

```
while not converged:
    1. Prepare parameterized state |ψ(θ)>
    2. Measure expectation value <ψ(θ)|H|ψ(θ)>
    3. Classical optimizer updates θ
```

**Design choices:**

| Choice | Options | Trade-off |
|--------|---------|-----------|
| Ansatz | Hardware-efficient, UCCSD, problem-inspired | Expressibility vs trainability |
| Optimizer | COBYLA, L-BFGS-B, Adam, SPSA | Gradient-free vs gradient-based |
| Gradient | Parameter-shift, finite differences | Exact vs approximate |
| Grouping | Qubit-wise commuting, general commuting | Fewer measurements vs complexity |

**Barren plateau warning:** Random parameterized circuits with many qubits have exponentially vanishing gradients. Mitigations:
- Use shallow circuits
- Use problem-inspired ansatze
- Initialize parameters near identity
- Use local cost functions instead of global ones

## 6. The Tomography Pattern

Reconstruct a quantum state or process by measuring in multiple bases.

```
For each Pauli basis (X, Y, Z per qubit):
    1. Prepare the state
    2. Rotate to measurement basis
    3. Measure many shots
    4. Record statistics

Reconstruct: ρ = (1/d) Σ_P <P> · P
```

**Scaling:** Full state tomography requires 3^n measurement settings for n qubits. Alternatives for large systems:
- **Classical shadow tomography** — O(log n) measurements for specific properties
- **Matrix product state tomography** — efficient for low-entanglement states
- **Randomized measurements** — estimate specific observables without full reconstruction

## 7. The Error Mitigation Pattern

Reduce the impact of hardware noise without full error correction.

**Zero-Noise Extrapolation (ZNE):**
```
For each noise scale factor c ∈ {1, 2, 3}:
    1. Stretch circuit noise by factor c (gate folding)
    2. Measure expectation value E(c)
Extrapolate E(c) → E(0) using polynomial or exponential fit
```

**Measurement Error Mitigation:**
```
1. Prepare each computational basis state
2. Measure to build calibration matrix M
3. For actual experiment results p_noisy:
   Solve: p_ideal = M^(-1) · p_noisy
```

**Probabilistic Error Cancellation:**
```
1. Characterize noise channel as Pauli decomposition
2. Sample correction operations probabilistically
3. Post-process with sign corrections
```

## 8. The Encoding Pattern

Map classical data into quantum states for quantum machine learning or quantum simulation.

| Encoding | Map | Qubits needed | Depth |
|----------|-----|---------------|-------|
| Basis encoding | x → \|x> | n = log₂(N) | O(n) |
| Amplitude encoding | x_i → α_i\|i> | n = log₂(N) | O(N) |
| Angle encoding | x_i → Ry(x_i)\|0> | n = N | O(1) |
| IQP encoding | x → exp(i Σ x_i x_j Z_i Z_j) | n = N | O(N²) |

**Trade-off:** Fewer qubits (amplitude) requires deeper circuits. More qubits (angle) allows shallower circuits but limits data size.

## 9. The Simulation Pattern

Simulate quantum systems by decomposing time evolution into gate sequences.

**Trotter-Suzuki decomposition:**
```
e^(-iHt) ≈ (e^(-iH₁t/n) · e^(-iH₂t/n) · ... · e^(-iHₖt/n))^n
```

**Key considerations:**
- **First-order Trotter** error: O(t²/n)
- **Second-order Trotter** error: O(t³/n²) — use symmetric decomposition
- **Higher order** available but deeper circuits
- **Qubitization/LCU** for near-optimal scaling but more complex circuits

**Fermion-to-qubit mapping:**
- Jordan-Wigner: preserves locality, O(n) gates per term
- Bravyi-Kitaev: O(log n) gates per term but more complex

## 10. The Hybrid Classical-Quantum Pattern

Use quantum computers for the parts classical computers can't do efficiently, keep everything else classical.

**Good quantum subroutines:**
- Sampling from distributions that are hard classically
- Estimating expectation values of quantum Hamiltonians
- Searching unstructured spaces (Grover)
- Simulating quantum systems (chemistry, materials)

**Keep classical:**
- Data preprocessing and feature engineering
- Optimization loop control
- Post-processing and analysis
- Error mitigation and result validation

**Architecture:**
```
Classical preprocessing
  → Quantum circuit construction
  → Quantum execution (shots)
  → Classical post-processing
  → Decision / next iteration
```

## Anti-Patterns to Avoid

1. **Measuring too early** — Measurement collapses superposition. Extract all needed information before measuring.

2. **Ignoring decoherence** — Real hardware has limited coherence time. Keep circuits shallow.

3. **Over-entangling** — Not every problem benefits from maximal entanglement. Match entanglement to problem structure.

4. **Forgetting reversibility** — Quantum gates are unitary. Don't try to implement irreversible classical logic without ancilla qubits.

5. **Classical simulation neglect** — Always simulate small instances classically first to verify correctness before running on hardware.

6. **Ignoring measurement statistics** — A single quantum measurement gives one sample. You need O(1/ε²) shots for precision ε.
