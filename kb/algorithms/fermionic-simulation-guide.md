# Fermionic Simulation Guide

## Overview

Fermionic simulation maps fermionic Hamiltonians (from chemistry, condensed matter) onto qubit operators that quantum computers can execute. The key challenge: fermions obey anti-commutation relations {a_i†, a_j} = δ_ij, which qubits don't naturally respect. Fermion-to-qubit mappings enforce these relations through non-local qubit operations.

## Fermion-to-Qubit Mappings

### Jordan-Wigner Transform

The standard mapping. Each fermionic mode maps to one qubit:

- a_j† → (1/2)(X_j - iY_j) ⊗ Z_{j-1} ⊗ ... ⊗ Z_0
- a_j  → (1/2)(X_j + iY_j) ⊗ Z_{j-1} ⊗ ... ⊗ Z_0

The Z-string enforces anti-commutation but creates O(n) Pauli weight for n modes. Number operator: n_j = (I - Z_j)/2.

Properties:
- Preserves locality of 1D systems
- O(n) Pauli weight for general operators
- Number conservation maps to Z-parity conservation
- Occupation basis: |1⟩ = occupied, |0⟩ = empty

### Bravyi-Kitaev Transform

Encodes both occupation AND parity information, reducing Pauli weight to O(log n):

- Qubit j stores parity of a subset of modes (determined by binary tree)
- Update set U(j): modes whose parity qubit j tracks
- Parity set P(j): qubits needed to compute parity of modes 0..j-1
- Flip set F(j): subset for the creation/annihilation operator

Properties:
- O(log n) Pauli weight — significant improvement for large systems
- More complex implementation than JW
- Same number of qubits as JW (1:1 mode-to-qubit)
- Better suited for systems with many modes

### Parity Transform

Alternative encoding where qubit j stores the parity of modes 0..j:

- Simpler than BK but still O(n) for creation operators
- Number operator is local: n_j = (Z_j - Z_{j-1}*Z_j)/2
- Two-qubit reduction possible when total particle number is conserved

## Fermionic Hamiltonians

### Second-Quantized Form

General electronic Hamiltonian:
H = Σ_{pq} h_pq a_p† a_q + (1/2) Σ_{pqrs} h_pqrs a_p† a_q† a_r a_s

- h_pq: one-electron integrals (kinetic + nuclear attraction)
- h_pqrs: two-electron integrals (electron-electron repulsion)

### Molecular Hamiltonians

For chemistry applications:
1. Choose a basis set (STO-3G, 6-31G, etc.)
2. Compute integrals h_pq, h_pqrs (classical step)
3. Map to qubit Hamiltonian via JW or BK
4. Result: sum of Pauli strings with real coefficients

### Model Hamiltonians

Hubbard model: H = -t Σ_{⟨ij⟩,σ} a_{iσ}† a_{jσ} + U Σ_i n_{i↑} n_{i↓}
- t: hopping parameter, U: on-site interaction
- Fundamental model for strongly correlated electrons

Fermi-Hubbard on a lattice maps naturally to JW with 2*n_sites qubits.

## Implementation Architecture

### FermionOperator

Core data structure representing sums of fermionic ladder operator products:

```
FermionOperator:
  terms: dict[tuple[tuple[int, bool], ...], complex]
    - Key: tuple of (mode_index, is_creation) pairs
    - Value: coefficient
  
  Methods:
  - __add__, __mul__: operator algebra
  - dagger(): Hermitian conjugate
  - normal_order(): reorder to creation-before-annihilation
  - is_hermitian(): check self-adjointness
```

Example: a_0† a_1 + h.c. → {((0,True),(1,False)): 1.0, ((1,True),(0,False)): 1.0}

### QubitOperator

Sum of Pauli strings with coefficients (already exists in our codebase as Hamiltonian terms):

```
QubitOperator:
  terms: dict[tuple[tuple[int, str], ...], complex]
    - Key: tuple of (qubit_index, pauli_label) pairs
    - Value: coefficient
```

### Transforms Module

```python
jordan_wigner(fermion_op: FermionOperator) -> QubitOperator
bravyi_kitaev(fermion_op: FermionOperator) -> QubitOperator
parity_transform(fermion_op: FermionOperator) -> QubitOperator
```

### Hamiltonian Builders

```python
def fermi_hubbard(n_sites, t, U, periodic=False) -> FermionOperator
def molecular_hamiltonian(one_body, two_body) -> FermionOperator
def number_operator(n_modes, mode=None) -> FermionOperator
def hopping_operator(i, j, coefficient=-1.0) -> FermionOperator
```

## Testing Strategy

1. Anti-commutation: verify {a_i†, a_j} = δ_ij after transform
2. Hermiticity: H† = H for physical Hamiltonians
3. Eigenspectrum: compare JW and BK eigenvalues (must match)
4. Known solutions: H2 in minimal basis, 2-site Hubbard
5. Number conservation: [H, N] = 0 for particle-conserving Hamiltonians
6. Operator algebra: normal ordering, addition, multiplication

## References

- Jordan & Wigner (1928): original fermion-to-spin mapping
- Bravyi & Kitaev (2002): logarithmic-weight transform
- Seeley, Richard, Love (2012): Bravyi-Kitaev construction details
- Tranter et al. (2018): comparison of fermion-to-qubit mappings
