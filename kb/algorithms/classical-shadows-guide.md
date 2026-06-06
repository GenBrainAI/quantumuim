# Classical Shadows for Efficient Quantum State Learning

A guide to the classical shadow protocol — predicting many properties of a quantum state from few measurements.

## Motivation

Full quantum state tomography requires exponentially many measurements (3^n Pauli bases for n qubits). Classical shadows achieve exponential savings: predict M observables using only O(log M) measurements, independent of system size for certain observable classes.

Key paper: Huang, Kueng, Preskill — "Predicting Many Properties of a Quantum State from Very Few Measurements" (Nature Physics, 2020).

## The Protocol

### 1. Random Measurement
For each of T rounds:
1. Apply a random unitary U_i drawn from an ensemble (e.g., random Pauli basis or random Clifford)
2. Measure in the computational basis, obtaining outcome b_i
3. Store the "snapshot" (U_i, b_i)

### 2. Classical Shadow Construction
Each snapshot produces an inverted channel estimate:

```
rho_hat_i = M^{-1}(U_i^dag |b_i><b_i| U_i)
```

where M^{-1} is the inverse of the measurement channel. For:
- **Random Pauli**: M^{-1}(rho) = (2^n + 1) * rho - I (tensor product structure)
- **Random Clifford**: M^{-1}(rho) = (2^n + 1) * rho - I (global)

### 3. Median-of-Means Estimation
To predict <O> = tr(O * rho):
1. Split T snapshots into K groups of size T/K
2. Compute mean within each group: mu_k = (K/T) * sum(tr(O * rho_hat_i))
3. Take median across groups

This gives epsilon-accurate estimates with probability 1-delta using:
- T = O(log(M/delta) * max_i(||O_i||_shadow^2) / epsilon^2) measurements

## Shadow Norm

The **shadow norm** ||O||_shadow determines sample complexity:

| Ensemble | Observable Type | Shadow Norm |
|----------|----------------|-------------|
| Random Pauli | k-local Pauli | 3^k (independent of n!) |
| Random Pauli | Fidelity with pure state | 2^n (exponential) |
| Random Clifford | Any trace-bounded | (2^n + 1) * ||O||_inf |
| Random Clifford | Fidelity with stabilizer | O(1) |

**Key insight**: Random Pauli measurements are efficient for local observables (energy, correlation functions). Random Clifford measurements are efficient for global properties (fidelity, entanglement witnesses).

## Derandomized Shadows

Huang et al. (2021) showed you can **derandomize** the protocol:
- Given a fixed set of target observables, choose measurement bases deterministically
- Greedy algorithm: pick the basis that maximally reduces variance for all targets
- Often 10-100x fewer measurements than random shadows

## Implementation Architecture for Quantumium

### Core Classes

```python
class ClassicalShadow:
    """Classical shadow of a quantum state."""
    
    def __init__(self, snapshots: list[tuple[str, int]]):
        """Store measurement records (basis, outcome)."""
    
    @classmethod
    def from_state(cls, state, num_snapshots, ensemble='pauli'):
        """Generate shadow from a known state (simulation)."""
    
    @classmethod  
    def from_counts(cls, bases, outcomes):
        """Construct from experimental measurement records."""
    
    def predict(self, observable, num_groups=None) -> float:
        """Predict expectation value via median-of-means."""
    
    def predict_many(self, observables) -> list[float]:
        """Batch prediction for multiple observables."""
    
    def fidelity_with(self, target_state) -> float:
        """Estimate fidelity with a target state."""
    
    def entanglement_witness(self, partition) -> float:
        """Estimate bipartite entanglement witness."""
```

### Measurement Ensembles

```python
class PauliShadow(ClassicalShadow):
    """Random Pauli basis measurements — efficient for local observables."""
    
class CliffordShadow(ClassicalShadow):
    """Random Clifford measurements — efficient for global properties."""

class DerandomizedShadow(ClassicalShadow):
    """Derandomized protocol for a fixed set of target observables."""
```

### Integration with Existing Modules

- **tomography**: Classical shadows as a fast alternative to MLE tomography
- **benchmarks**: Use shadows for efficient fidelity estimation
- **qml**: Shadow-based kernel methods for quantum machine learning

## Sample Complexity Comparison

| Task | Full Tomography | Random Pauli Shadow | 
|------|----------------|-------------------|
| Single Pauli expectation | 4^n | O(1) |
| All k-local Paulis | 4^n | O(3^k * n^k) |
| State fidelity | 4^n | O(2^n) or O(1) for stabilizers |
| Entanglement detection | 4^n | O(4^k) for k-partite |

## Testing Strategy

1. **Accuracy**: Shadow predictions within epsilon of exact values for known states
2. **Sample efficiency**: Verify O(log M) scaling for M observables
3. **Ensemble correctness**: Pauli shadows match for local observables
4. **Median-of-means**: Verify concentration (predictions within bounds with high probability)
5. **Integration**: Works with existing QuantumCircuit states

## Applications

- **Variational algorithms**: Efficiently estimate many Hamiltonian terms in VQE
- **Quantum device characterization**: Predict gate fidelities from few measurements
- **Entanglement detection**: Certify entanglement without full tomography
- **Quantum machine learning**: Shadow-based feature maps and kernels

## References

- Huang, Kueng, Preskill, "Predicting Many Properties of a Quantum State" (2020)
- Huang et al., "Efficient Estimation of Pauli Observables by Derandomization" (2021)
- Elben et al., "The Randomized Measurement Toolbox" (2022) — comprehensive review
