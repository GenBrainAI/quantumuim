# Quantum Optimization

## 1. Overview

Many important real-world problems are combinatorial optimization problems — finding the best solution from a finite but exponentially large set. Quantum computers offer several approaches to tackle these, from variational heuristics on near-term devices to provable speedups on fault-tolerant machines.

## 2. Problem Formulation: QUBO and Ising Models

Most quantum optimization algorithms require problems in one of two equivalent forms.

### Quadratic Unconstrained Binary Optimization (QUBO)
Minimize f(x) = xᵀQx where x ∈ {0,1}ⁿ and Q is an upper-triangular matrix.

### Ising Model
Minimize H = Σᵢⱼ Jᵢⱼ ZᵢZⱼ + Σᵢ hᵢZᵢ where Zᵢ ∈ {-1,+1}

**Conversion:** x = (1 - Z)/2 maps between the two. Any QUBO is an Ising Hamiltonian and vice versa.

### Common Problems as QUBO/Ising
- **MaxCut:** Jᵢⱼ = -1 for edges, hᵢ = 0 → ground state = max cut
- **Travelling Salesman:** Binary variables xᵢₜ (city i at position t), penalty terms for constraints
- **Portfolio Optimization:** min xᵀΣx - μᵀx subject to budget and cardinality constraints
- **Graph Coloring, SAT, Number Partitioning** — all reducible to QUBO with polynomial overhead

## 3. QAOA (Quantum Approximate Optimization Algorithm)

Farhi, Goldstone, Gutmann (2014). The leading near-term quantum optimization algorithm.

### Circuit Structure
For p rounds:
1. Start with |+⟩ⁿ (uniform superposition)
2. Apply cost unitary: e^{-iγₖ C} (diagonal in computational basis)
3. Apply mixer unitary: e^{-iβₖ B} where B = ΣᵢXᵢ (standard mixer)
4. Repeat for k = 1, ..., p
5. Measure in computational basis

### Parameters
- 2p variational parameters (γ₁,...,γₚ, β₁,...,βₚ)
- Optimized classically to maximize ⟨ψ(γ,β)|C|ψ(γ,β)⟩
- p → ∞ recovers adiabatic algorithm (provably reaches optimum)
- Low p (1-5) already gives good approximation ratios for many problems

### Performance
- **MaxCut on 3-regular graphs:** QAOA at p=1 achieves approximation ratio ≥ 0.6924
- **Goemans-Williamson (classical):** achieves 0.878 — still ahead for most instances
- **Advantage regime:** open question. QAOA may outperform on specific instance families

### Mixer Variants
- **Grover mixer:** preserves feasible subspace for constrained problems
- **XY mixer:** conserves Hamming weight (for cardinality constraints)
- **Custom mixers:** designed per problem to respect constraints natively

## 4. Variational Quantum Eigensolver (VQE)

Originally designed for chemistry, VQE is a general variational optimizer.

### Structure
1. Prepare parameterized state |ψ(θ)⟩
2. Measure ⟨ψ(θ)|H|ψ(θ)⟩ via Pauli decomposition of H
3. Classically optimize θ

### Ansatz Choices
- **Hardware-efficient:** arbitrary rotations + entanglement layers (generic, may have barren plateaus)
- **UCCSD:** chemistry-inspired, unitary coupled cluster (deep but physically motivated)
- **Problem-specific:** structure ansatz around known symmetries

### Optimization
- **COBYLA, SPSA, L-BFGS-B** for noisy or noiseless settings
- **Parameter-shift rule** for exact gradients
- **Adaptive VQE (ADAPT-VQE):** grows ansatz iteratively, adding operators that most reduce energy

## 5. Quantum Annealing

A fundamentally different paradigm — continuous-time evolution, not gate-based.

### Principle
1. Start in ground state of simple Hamiltonian H_init (transverse field)
2. Slowly interpolate: H(t) = (1 - t/T) H_init + (t/T) H_final
3. If slow enough (T ∝ 1/Δ²_min), adiabatic theorem guarantees reaching ground state of H_final

### Hardware
- **D-Wave:** 5000+ qubit quantum annealer
- Native Ising model formulation — no gate decomposition needed
- Limited connectivity (Pegasus topology) requires minor embedding

### Comparison to Gate-Based
| Aspect | Quantum Annealing | QAOA (Gate-Based) |
|--------|------------------|-------------------|
| Qubits | 5000+ | 50-100 |
| Connectivity | Sparse (Pegasus) | Programmable |
| Error correction | None (analog) | Possible |
| Problem types | QUBO/Ising only | General |
| Control | Anneal schedule only | Full gate control |

## 6. Grover Adaptive Search (GAS)

Uses Grover's algorithm for optimization with provable quadratic speedup.

### Method
1. Set threshold t
2. Use Grover search to find x with f(x) < t
3. Update t = f(x), repeat
4. Expected O(√N) total iterations to find global minimum

### Limitations
- Requires fault-tolerant quantum computer (deep circuits)
- O(√N) is provably optimal for unstructured search
- For structured problems, QAOA or problem-specific algorithms may do better

## 7. Quantum Walk Optimization

### Spatial Search
- Continuous-time quantum walk on problem graph
- Walk Hamiltonian: H = -γA + |w⟩⟨w| (adjacency + marked vertex)
- Finds marked vertex in O(√N) for many graph families

### QWOA (Quantum Walk Optimization Algorithm)
- Replace QAOA mixer with continuous-time quantum walk
- Exploits problem graph structure for better mixing
- Promising for problems with natural graph structure (network optimization)

## 8. Applications in Quantumium SDK

Our SDK provides:
- **QAOA:** generic QAOA with configurable depth, custom Hamiltonians
- **MaxCut:** specialized QAOA formulation for graph cuts
- **TSP:** QUBO formulation with route constraint penalties
- **Portfolio Optimization:** mean-variance objective as QUBO
- **VQE:** for molecular ground state energy (chemistry crossover)
- **Adiabatic evolution:** with spectral gap analysis
- See `quantumium.algorithms`, `quantumium.applications`, `quantumium.finance`, `quantumium.adiabatic`

## 9. Open Questions

1. **Quantum advantage for optimization:** No unconditional proof that quantum computers solve NP-hard optimization problems faster. Best known: quadratic speedup via Grover for unstructured problems.
2. **QAOA vs classical:** At what depth p does QAOA outperform the best classical algorithms for specific problems?
3. **Noise impact:** How do realistic noise levels affect optimization quality? Error mitigation helps but adds overhead.
4. **Warm-starting:** Can classical pre-processing (relaxations, heuristics) initialize quantum algorithms for better performance?
