# Quantum Algorithm Catalog

## Overview

Quantum algorithms that provide speedup over classical counterparts, organized by type.

## 1. Oracle-Based / Query Algorithms

### Deutsch-Jozsa Algorithm
- **Problem:** Determine if f:{0,1}ⁿ→{0,1} is constant or balanced
- **Classical:** 2^(n-1)+1 queries (worst case)
- **Quantum:** 1 query (exponential speedup)
- **Key insight:** Quantum parallelism + interference

### Bernstein-Vazirani Algorithm
- **Problem:** Find secret string s given oracle f(x) = s·x (mod 2)
- **Classical:** n queries
- **Quantum:** 1 query
- **Key insight:** Hadamard transform decodes the secret in superposition

### Simon's Algorithm
- **Problem:** Find period s where f(x) = f(y) iff x⊕y ∈ {0,s}
- **Classical:** Ω(2^(n/2)) queries
- **Quantum:** O(n) queries (exponential speedup)
- **Key insight:** Precursor to Shor's algorithm; exploits periodicity structure

### Grover's Search Algorithm
- **Problem:** Find x where f(x) = 1 in unstructured database of N items
- **Classical:** O(N) queries
- **Quantum:** O(√N) queries (quadratic speedup)
- **Key insight:** Amplitude amplification via reflection operators
- **Provably optimal** — no quantum algorithm can do better

## 2. Algebraic / Number-Theoretic

### Shor's Factoring Algorithm
- **Problem:** Factor integer N into primes
- **Classical:** Best known: exp(O(n^(1/3) (log n)^(2/3))) (number field sieve)
- **Quantum:** O(n³) (exponential speedup)
- **Key components:** Quantum Fourier Transform + order finding
- **Impact:** Breaks RSA cryptography

### Quantum Phase Estimation (QPE)
- **Problem:** Estimate eigenvalue e^(2πiφ) of unitary U given eigenvector |u⟩
- **Precision:** t qubits → φ to t bits of precision
- **Key building block** for Shor's, HHL, quantum chemistry
- **Uses:** Inverse QFT to extract phase information

### Quantum Fourier Transform (QFT)
- **Operation:** Maps computational basis to Fourier basis
- **Classical FFT:** O(N log N) = O(n 2ⁿ)
- **Quantum QFT:** O(n²) gates on n qubits
- **Caveat:** Cannot directly read out amplitudes; useful as subroutine

## 3. Linear Algebra

### HHL Algorithm (Harrow-Hassidim-Lloyd)
- **Problem:** Solve Ax = b for x (linear system)
- **Classical:** O(N poly(log(1/ε))) with best iterative methods
- **Quantum:** O(poly(log N, 1/ε)) (exponential speedup, with caveats)
- **Caveats:** Requires efficient state preparation, sparse A, output is quantum state
- **Applications:** Machine learning, optimization, differential equations

### Quantum Singular Value Transformation (QSVT)
- **Unifying framework** for many quantum algorithms
- Implements polynomial transformations of singular values
- Subsumes: QPE, HHL, Grover's, quantum walks, Hamiltonian simulation

## 4. Simulation

### Hamiltonian Simulation
- **Problem:** Simulate quantum system evolution e^(-iHt)
- **Methods:**
  - Product formulas (Trotter-Suzuki): O(t²/ε) gates
  - Linear combination of unitaries (LCU)
  - Qubitization / quantum signal processing
- **BQP-complete** — the original motivation for quantum computing (Feynman)

### Variational Quantum Eigensolver (VQE)
- **Problem:** Find ground state energy of Hamiltonian H
- **Approach:** Hybrid classical-quantum; parameterized quantum circuit
- **NISQ-friendly:** Short circuits, noise-tolerant
- **Applications:** Quantum chemistry, materials science

### Quantum Approximate Optimization Algorithm (QAOA)
- **Problem:** Combinatorial optimization (MaxCut, etc.)
- **Approach:** Alternating layers of problem and mixer Hamiltonians
- **Parameterized by depth p** — higher p → better approximation ratio
- **NISQ-friendly**

## 5. Amplitude Estimation & Counting

### Quantum Counting
- **Problem:** Count the number of solutions to f(x) = 1 among N items
- **Method:** QPE on the Grover operator G — eigenvalues encode the solution count M
- **Classical:** O(N) evaluations
- **Quantum:** O(√N) evaluations, estimates M to multiplicative error
- **Applications:** Counting satisfying assignments, estimating set sizes

### Canonical Amplitude Estimation
- **Problem:** Estimate amplitude a = ⟨ψ|good⟩ to precision ε
- **Method:** QPE on Grover operator — eigenphase encodes sin⁻¹(√a)
- **Speedup:** O(1/ε) vs O(1/ε²) classical (quadratic)
- **Applications:** Monte Carlo integration, option pricing, risk analysis
- **Limitation:** Requires deep circuits (QPE + Grover iterates)

### Iterative Amplitude Estimation (QPE-Free)
- **Problem:** Same as canonical AE but without QPE overhead
- **Method:** Suzuki et al. (2020) — sequence of Grover iterates with classical maximum likelihood
- **Advantage:** Shallower circuits, more NISQ-friendly, same O(1/ε) scaling
- **Trade-off:** Requires more classical post-processing

## 6. Quantum Walks

### Discrete-Time Quantum Walk
- **Coin operator** (Hadamard, Grover, DFT) drives superposition of directions
- **Shift operator** moves walker based on coin state
- **Spreading:** Ballistic O(t) vs classical random walk O(√t) — quadratic speedup
- **Graphs:** Line, cycle, hypercube, 2D grid, arbitrary adjacency

### Continuous-Time Quantum Walk
- **Hamiltonian:** H = γA (adjacency matrix) or H = γL (Laplacian)
- **Evolution:** |ψ(t)⟩ = e^{-iHt}|ψ(0)⟩
- **Applications:** Spatial search (O(√N) on many graphs), graph isomorphism heuristics, transport in biological systems

### Quantum Walk Search (Szegedy)
- Quantization of classical Markov chains
- Quadratic speedup for finding marked elements: O(1/√δ) hitting time vs O(1/δ) classical
- General framework encompassing Grover as special case

## 7. Adiabatic & Annealing

### Adiabatic Quantum Computing
- **Principle:** Slowly evolve from easy ground state to problem ground state
- **H(s) = (1-s)H_init + s·H_final** with s: 0→1
- **Adiabatic theorem:** If evolution is slow enough (T ∝ 1/Δ²_min), system stays in ground state
- **Spectral gap Δ_min** determines required evolution time
- **Equivalent to gate model** in computational power (polynomially)

### Quantum Annealing
- Practical implementation of adiabatic QC on specialized hardware (D-Wave)
- Thermal fluctuations + quantum tunneling for optimization
- No error correction, finite temperature — heuristic, not provably optimal
- **Advantage:** Thousands of qubits, native Ising model support

## 8. Quantum Signal Processing & QSVT

### Quantum Signal Processing (QSP)
- Implements polynomial transformations P(x) of a signal embedded in a unitary
- Signal operator W(x) + processing operator S(φ) → polynomial evaluation
- **Angle finding:** Given target polynomial, find processing angles
- **Universality:** Any bounded polynomial achievable with correct angles

### Quantum Singular Value Transformation (QSVT)
- Generalization of QSP to block-encoded matrices
- Applies polynomial to singular values of block-encoded A
- **Unifying framework** — subsumes as special cases:
  - Hamiltonian simulation (odd polynomial approximating e^{-ix})
  - Amplitude amplification (step function polynomial)
  - Matrix inversion / HHL (1/x polynomial)
  - QPE (Chebyshev-based phase discrimination)
- **Optimal query complexity** for many problems
- Considered the "grand unified theory" of quantum algorithms

## 9. Machine Learning

### Quantum Support Vector Machine (QSVM)
- Uses quantum kernel estimation for classification
- Potential exponential speedup for specific kernel functions

### Quantum Principal Component Analysis
- Exponential speedup for finding principal components of density matrix
- Based on QPE applied to density matrix exponentiation

### Quantum Boltzmann Machine
- Quantum-enhanced sampling for generative models

## 10. Cryptography

### Quantum Key Distribution (BB84)
- Not an algorithm per se, but a protocol
- Information-theoretically secure key exchange
- Uses no-cloning theorem for security

### Quantum Random Number Generation
- True randomness from quantum measurement
- Certifiable randomness via Bell inequality violations

## Complexity Classes

| Class | Description |
|-------|-------------|
| BQP | Bounded-error Quantum Polynomial time — problems efficiently solvable by quantum computers |
| QMA | Quantum Merlin-Arthur — quantum analog of NP |
| PostBQP = PP | BQP with postselection |

Known relations: P ⊆ BPP ⊆ BQP ⊆ PP ⊆ PSPACE
