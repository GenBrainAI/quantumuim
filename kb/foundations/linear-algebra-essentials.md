# Linear Algebra Essentials for Quantum Computing

## 1. Complex Vector Spaces

Quantum states live in **complex Hilbert spaces** — complete inner product spaces over ℂ.

### Dirac Notation (Bra-Ket)
- **Ket** |ψ⟩ : column vector (state)
- **Bra** ⟨ψ| : row vector (conjugate transpose of ket)
- **Inner product** ⟨φ|ψ⟩ : scalar in ℂ
- **Outer product** |ψ⟩⟨φ| : matrix (operator)

### Computational Basis for n Qubits
Basis vectors: |0⟩, |1⟩, ..., |2ⁿ-1⟩ in ℂ^(2ⁿ)

Standard basis for 1 qubit:
```
|0⟩ = (1, 0)ᵀ
|1⟩ = (0, 1)ᵀ
```

## 2. Inner Products and Norms

Inner product: ⟨φ|ψ⟩ = Σᵢ φᵢ* ψᵢ

Properties:
- ⟨φ|ψ⟩ = ⟨ψ|φ⟩* (conjugate symmetry)
- ⟨ψ|ψ⟩ ≥ 0, equality iff |ψ⟩ = 0 (positive definiteness)
- Linearity in the second argument

Norm: ‖|ψ⟩‖ = √⟨ψ|ψ⟩

Quantum states are normalized: ⟨ψ|ψ⟩ = 1

## 3. Operators and Matrices

### Key Matrix Types

**Hermitian (self-adjoint):** A† = A
- Real eigenvalues
- Orthogonal eigenvectors
- Observables in quantum mechanics are Hermitian

**Unitary:** U†U = UU† = I
- Preserves inner products: ⟨Uφ|Uψ⟩ = ⟨φ|ψ⟩
- Quantum gates are unitary operators
- |det(U)| = 1

**Normal:** AA† = A†A
- Both Hermitian and unitary matrices are normal
- Spectral theorem applies

**Positive semidefinite:** ⟨ψ|A|ψ⟩ ≥ 0 for all |ψ⟩
- Density matrices are positive semidefinite

### Conjugate Transpose (Dagger)
(AB)† = B†A†
(|ψ⟩)† = ⟨ψ|

## 4. Eigenvalues and Eigenvectors

A|λ⟩ = λ|λ⟩

### Spectral Decomposition
For normal operator A with eigenvalues λᵢ and orthonormal eigenvectors |λᵢ⟩:

A = Σᵢ λᵢ |λᵢ⟩⟨λᵢ|

For unitary U: eigenvalues lie on the unit circle (|λ| = 1)
For Hermitian H: eigenvalues are real

## 5. Tensor Products

The tensor product ⊗ combines vector spaces:

If |ψ⟩ ∈ ℂᵐ and |φ⟩ ∈ ℂⁿ, then |ψ⟩ ⊗ |φ⟩ ∈ ℂᵐⁿ

### Properties
- (|ψ₁⟩ + |ψ₂⟩) ⊗ |φ⟩ = |ψ₁⟩⊗|φ⟩ + |ψ₂⟩⊗|φ⟩ (distributive)
- c(|ψ⟩ ⊗ |φ⟩) = (c|ψ⟩) ⊗ |φ⟩ = |ψ⟩ ⊗ (c|φ⟩)
- (A ⊗ B)(|ψ⟩ ⊗ |φ⟩) = (A|ψ⟩) ⊗ (B|φ⟩)

### Shorthand
|ψ⟩ ⊗ |φ⟩ = |ψ⟩|φ⟩ = |ψφ⟩

### Kronecker Product (Matrix Form)
```
A ⊗ B = |a₁₁B  a₁₂B  ...|
         |a₂₁B  a₂₂B  ...|
         |...    ...    ...|
```

## 6. Trace and Partial Trace

**Trace:** Tr(A) = Σᵢ aᵢᵢ

Properties:
- Tr(AB) = Tr(BA) (cyclic)
- Tr(A ⊗ B) = Tr(A) · Tr(B)
- Tr(|ψ⟩⟨φ|) = ⟨φ|ψ⟩

**Partial Trace** (tracing out subsystem B):
TrB(|ψ₁⟩⟨ψ₂| ⊗ |φ₁⟩⟨φ₂|) = |ψ₁⟩⟨ψ₂| · ⟨φ₂|φ₁⟩

Used to obtain the reduced density matrix of a subsystem.

## 7. Key Decompositions

### Singular Value Decomposition (SVD)
Any matrix M = UΣV† where U, V are unitary and Σ is diagonal with non-negative reals.

### Schmidt Decomposition
Any bipartite pure state |ψ⟩_AB can be written as:
|ψ⟩ = Σᵢ λᵢ |aᵢ⟩|bᵢ⟩

where λᵢ > 0, Σᵢ λᵢ² = 1, and {|aᵢ⟩}, {|bᵢ⟩} are orthonormal.

Schmidt rank = number of non-zero λᵢ. State is entangled iff Schmidt rank > 1.

## 8. Important Identities

- e^(iθn̂·σ) = cos(θ)I + i sin(θ)(n̂·σ) — for Pauli vector σ = (X,Y,Z)
- HXH = Z, HZH = X — Hadamard conjugation
- XYX = -Y — Pauli anticommutation
- {σᵢ, σⱼ} = 2δᵢⱼI — Pauli anticommutation relations
- [σᵢ, σⱼ] = 2iεᵢⱼₖσₖ — Pauli commutation relations
