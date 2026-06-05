# Variational Quantum Algorithms — Practical Guide

A deep-dive into designing, implementing, and debugging variational quantum algorithms (VQAs). Covers VQE, QAOA, quantum classifiers, and general variational workflows.

## The Variational Framework

All VQAs share the same structure:

```
Cost function:  C(θ) = <ψ(θ)|H|ψ(θ)>
Goal:           min_θ C(θ)
Method:         Classical optimizer updates θ based on quantum measurements
```

The quantum computer evaluates the cost function; the classical computer finds the minimum.

## Ansatz Design

The ansatz (parameterized circuit) is the most critical design choice. It determines what states can be reached and how efficiently.

### Hardware-Efficient Ansatz (HEA)

```
Layer structure:  Ry(θ) Rz(θ) on each qubit → CNOT entangling layer → repeat
Pros:            Matches hardware connectivity, low depth
Cons:            Barren plateaus at scale, no physical motivation
Best for:        Small systems (< 10 qubits), initial exploration
```

### Problem-Inspired Ansatz

**Chemistry (UCCSD):**
- Unitary Coupled Cluster with Singles and Doubles
- Physically motivated excitation operators
- Avoids barren plateaus for chemistry problems
- Deeper circuits but better convergence

**Optimization (QAOA):**
- Alternating problem and mixer unitaries
- Depth p controls approximation quality
- At p → ∞, recovers optimal solution
- In practice, p = 1-5 often sufficient for small instances

**Symmetry-preserving:**
- Restrict ansatz to respect problem symmetries
- Particle number, spin, spatial symmetry
- Reduces parameter space and avoids unphysical states

### Adaptive Ansatz (ADAPT-VQE)

```
1. Start with empty ansatz
2. Compute gradient of each candidate operator
3. Add operator with largest gradient
4. Re-optimize all parameters
5. Repeat until gradient norm < threshold
```

Advantage: Systematically builds the most efficient ansatz for the specific problem.

## Optimizer Selection

### Gradient-Free Optimizers

| Optimizer | Best for | Key property |
|-----------|----------|--------------|
| COBYLA | Noisy cost functions | Constraint handling |
| Nelder-Mead | Low dimensions (< 20) | No gradients needed |
| SPSA | Many parameters, noisy | Only 2 evaluations per step |
| Powell | Smooth landscapes | Direction-set method |

### Gradient-Based Optimizers

| Optimizer | Best for | Key property |
|-----------|----------|--------------|
| L-BFGS-B | Smooth, many parameters | Quasi-Newton, bounded |
| Adam | Training loops | Momentum, adaptive LR |
| Natural gradient | Variational manifold | Metric-aware updates |

### Gradient Computation

**Parameter-shift rule:**
```
∂C/∂θ_i = [C(θ_i + π/2) - C(θ_i - π/2)] / 2
```
- Exact for single-parameter gates (Rx, Ry, Rz)
- Requires 2 circuit evaluations per parameter
- Total: 2p evaluations for p parameters per optimization step

**Finite differences:**
```
∂C/∂θ_i ≈ [C(θ_i + ε) - C(θ_i - ε)] / (2ε)
```
- Approximate, sensitive to ε choice
- Same cost as parameter-shift
- Use when parameter-shift rule doesn't apply

**SPSA (Simultaneous Perturbation):**
```
g(θ) ≈ [C(θ + εΔ) - C(θ - εΔ)] / (2ε) · Δ^(-1)
```
- Only 2 evaluations regardless of parameter count
- Stochastic gradient estimate
- Best for high-dimensional parameter spaces on noisy hardware

## Measurement Strategies

### Hamiltonian Grouping

A Hamiltonian H = Σ_i c_i P_i has many Pauli terms. Measuring each independently is expensive.

**Qubit-wise commuting (QWC) groups:**
- Terms P_i and P_j are QWC if they commute on every qubit
- Can be measured in the same basis simultaneously
- Grouping reduces measurement circuits from O(4^n) to O(3^n) or fewer

**General commuting groups:**
- Larger groups possible with basis rotations
- Fewer groups but deeper measurement circuits

**Shot allocation:**
- Allocate more shots to terms with larger coefficients
- Optimal allocation: shots_i ∝ |c_i| · Var(P_i)

### Statistical Considerations

**Precision:** To estimate <H> to precision ε:
- Need O(Σ|c_i|² / ε²) total shots
- Or O(||c||₁² / ε²) with optimal allocation

**Confidence intervals:**
- Bootstrap resampling for non-parametric CI
- Normal approximation valid for > 1000 shots per term

## Debugging VQA Runs

### Common Failure Modes

**1. Stuck at local minimum**
- Symptom: Energy plateaus well above known ground state
- Fix: Multiple random restarts, simulated annealing, or better ansatz

**2. Barren plateau**
- Symptom: Gradients exponentially small, optimizer makes no progress
- Diagnosis: Compute gradient variance across random initializations
- Fix: Shallower circuit, local cost function, structured initialization

**3. Noise-induced bias**
- Symptom: Converged energy is systematically too high
- Fix: Apply error mitigation (ZNE, measurement error mitigation)

**4. Optimizer divergence**
- Symptom: Cost function oscillates or increases
- Fix: Reduce learning rate, switch to more robust optimizer (COBYLA)

**5. Insufficient shots**
- Symptom: Cost function appears noisy, optimizer zigzags
- Fix: Increase shots, use SPSA or other noise-tolerant optimizer

### Diagnostic Checklist

1. **Verify ansatz expressibility:** Can the ansatz represent the target state? Check by optimizing with exact statevector simulation.

2. **Check gradient landscape:** Plot 1D slices through parameter space. Look for smooth landscape with clear minimum.

3. **Monitor convergence:** Track energy, gradient norm, and parameter changes per iteration.

4. **Compare with classical:** For small instances, compute exact ground state. The VQE result should approach it.

5. **Test noise sensitivity:** Run with and without noise model. Large gap indicates need for error mitigation.

## QAOA-Specific Guide

### Problem Formulation

1. Express combinatorial problem as QUBO: minimize x^T Q x
2. Convert to Ising: H = Σ_{ij} J_{ij} Z_i Z_j + Σ_i h_i Z_i
3. QAOA ansatz: |γ,β> = Π_p [e^{-iβ_p H_M} · e^{-iγ_p H_C}] |+>^n

### Choosing p (Depth)

| p | Approximation ratio | Practical use |
|---|---------------------|---------------|
| 1 | ≥ 0.6924 for MaxCut | Quick feasibility check |
| 2-3 | Significantly better | Good balance of quality vs cost |
| 5+ | Near-optimal for small instances | Research, benchmarking |

### Parameter Strategies

**Fixed-angle conjecture:** For regular graphs, optimal QAOA parameters are independent of the specific graph instance. Precomputed tables exist for common graph families.

**Layer-by-layer training:**
1. Optimize p=1 parameters (γ₁, β₁)
2. Fix them, add p=2 layer, optimize (γ₂, β₂)
3. Repeat for higher p
4. Finally, re-optimize all parameters jointly

**Parameter transfer:** Optimal parameters from small instances often transfer well to larger instances of the same problem class.

## VQE for Chemistry — Practical Workflow

```
1. Define molecule (geometry, basis set)
2. Compute molecular integrals (classical, e.g., PySCF)
3. Map fermionic Hamiltonian to qubit Hamiltonian
   - Jordan-Wigner or Bravyi-Kitaev
4. Group Hamiltonian terms for efficient measurement
5. Choose ansatz (UCCSD or hardware-efficient)
6. Initialize parameters (e.g., from Hartree-Fock)
7. Run VQE optimization loop
8. Apply error mitigation to final result
9. Compare with classical methods (FCI, CCSD(T))
```

### Chemistry-Specific Tips

- **Active space selection:** Only include orbitals near the Fermi level
- **Symmetry reduction:** Exploit molecular symmetries to reduce qubit count
- **Tapering:** Remove qubits corresponding to conserved quantities
- **Initial state:** Start from Hartree-Fock state, not random
