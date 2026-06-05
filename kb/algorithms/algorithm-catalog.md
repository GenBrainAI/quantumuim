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

## 5. Machine Learning

### Quantum Support Vector Machine (QSVM)
- Uses quantum kernel estimation for classification
- Potential exponential speedup for specific kernel functions

### Quantum Principal Component Analysis
- Exponential speedup for finding principal components of density matrix
- Based on QPE applied to density matrix exponentiation

### Quantum Boltzmann Machine
- Quantum-enhanced sampling for generative models

## 6. Cryptography

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
