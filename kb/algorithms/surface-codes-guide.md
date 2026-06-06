# Surface Codes — Practical Guide

The surface code is the leading candidate for fault-tolerant quantum computing. It has the highest known error threshold (~1%) among topological codes and requires only nearest-neighbor interactions on a 2D grid.

## Why Surface Codes Matter

All current quantum hardware is noisy. Error correction is the path from NISQ to fault-tolerant quantum computing. Surface codes are favored because:

1. **High threshold** — tolerates physical error rates up to ~1% (depolarizing noise)
2. **2D nearest-neighbor** — only requires connections between adjacent qubits on a grid
3. **Local stabilizers** — each check involves only 4 (or fewer at boundaries) qubits
4. **Efficient decoding** — minimum-weight perfect matching runs in polynomial time

## The Rotated Planar Code

The standard implementation is the rotated surface code, which uses fewer qubits than the unrotated version.

### Lattice Structure

For code distance d:
- **Data qubits:** d² (arranged on a tilted square lattice)
- **X stabilizers:** (d²-1)/2 (detect Z errors)
- **Z stabilizers:** (d²-1)/2 (detect X errors)
- **Total qubits:** 2d² - 1 (data + ancilla)

```
Distance 3 (d=3): 9 data qubits, 4 X stabilizers, 4 Z stabilizers = 17 total qubits
Distance 5 (d=5): 25 data qubits, 12 X stabilizers, 12 Z stabilizers = 49 total qubits
Distance 7 (d=7): 49 data qubits, 24 X stabilizers, 24 Z stabilizers = 97 total qubits
```

### Stabilizer Generators

Each X-type stabilizer is a product of X operators on the data qubits surrounding a face:
```
S_X = X₁ ⊗ X₂ ⊗ X₃ ⊗ X₄   (interior face)
S_X = X₁ ⊗ X₂               (boundary face, 2 qubits)
```

Each Z-type stabilizer is a product of Z operators on the data qubits surrounding a vertex:
```
S_Z = Z₁ ⊗ Z₂ ⊗ Z₃ ⊗ Z₄   (interior vertex)
S_Z = Z₁ ⊗ Z₂               (boundary vertex, 2 qubits)
```

### Logical Operators

The surface code encodes 1 logical qubit per patch:
- **Logical X:** chain of X operators connecting the left and right rough boundaries
- **Logical Z:** chain of Z operators connecting the top and bottom smooth boundaries

The code distance d equals the minimum weight of any logical operator — it takes at least d single-qubit errors to cause a logical error.

## Syndrome Extraction

### Measurement Protocol

1. Prepare each ancilla qubit in |0⟩ (for Z stabilizers) or |+⟩ (for X stabilizers)
2. Apply CNOT gates between each ancilla and its neighboring data qubits
3. Measure each ancilla in the Z basis (for Z stabilizers) or X basis (for X stabilizers)
4. A measurement outcome of -1 indicates an error has occurred nearby

### Syndrome Vector

The syndrome is a binary vector with one bit per stabilizer:
- 0 = stabilizer eigenvalue +1 (no error detected)
- 1 = stabilizer eigenvalue -1 (error detected)

For a single X error on data qubit i, the Z stabilizers adjacent to qubit i will flag. Similarly for Z errors and X stabilizers.

## Decoding: Minimum-Weight Perfect Matching (MWPM)

### The Decoding Problem

Given a syndrome (set of flagged stabilizers), find the most likely error pattern that produced it.

### MWPM Approach

1. **Build a graph:** nodes are the flagged (defect) stabilizers, edges connect every pair with weight = Manhattan distance on the lattice
2. **Add boundary nodes:** each defect can also be matched to the nearest boundary (representing an error chain that exits the lattice)
3. **Find minimum-weight perfect matching:** pair up all defects such that total edge weight is minimized
4. **Infer correction:** for each matched pair, apply corrections along the shortest path between them

### Why MWPM Works

The most likely error pattern is the one with the fewest single-qubit errors. MWPM finds the minimum-weight set of error chains connecting the observed defects. For independent, identically distributed errors, this is optimal.

### Decoding Complexity

- Naive: O(n³) for n defects (Edmonds' blossom algorithm)
- In practice: very fast because the number of defects is typically small
- For production: Union-Find decoder offers almost-linear time

## Error Threshold

### Monte Carlo Estimation

To estimate the threshold:
1. Fix code distance d
2. For each physical error rate p:
   a. Generate random errors (each qubit independently with probability p)
   b. Compute syndrome
   c. Decode and apply correction
   d. Check if a logical error occurred (correction + error = nontrivial logical operator?)
3. Repeat N times, compute logical error rate P_L(p, d)
4. Plot P_L vs p for different d values
5. Threshold p_th is where curves for different d cross

### Expected Results

| Physical error rate | d=3 logical error rate | d=5 logical error rate | d=7 logical error rate |
|--------------------|-----------------------|-----------------------|-----------------------|
| 0.1% (below threshold) | ~0.3% | ~0.03% | ~0.003% |
| 1.0% (near threshold) | ~3% | ~3% | ~3% |
| 2.0% (above threshold) | ~6% | ~8% | ~10% |

Below threshold: increasing d exponentially suppresses logical errors.
Above threshold: increasing d makes things worse.

## Practical Considerations

### Qubit Overhead

To achieve a logical error rate of 10⁻¹⁵ (required for useful computation):
- Need d ≈ 25-30 for physical error rates of 0.1%
- That's ~1500 physical qubits per logical qubit
- A 100-logical-qubit computer needs ~150,000 physical qubits

### Measurement Errors

In real hardware, syndrome measurements are also noisy. The standard approach:
- Repeat syndrome extraction d times
- Use a 3D matching decoder (space + time dimensions)
- This doesn't change the threshold much but increases qubit-time overhead

### Beyond Depolarizing Noise

Real hardware has biased noise (Z errors more common than X errors):
- XZZX surface code variant exploits Z-biased noise
- Threshold can exceed 10% for highly biased noise
- Rectangular codes (different X and Z distances) save qubits

## Surface Codes vs Other QEC Codes

| Code | Threshold | Connectivity | Qubits per logical | Decoder |
|------|-----------|-------------|-------------------|---------|
| Surface code | ~1.0% | 2D nearest-neighbor | d² data + d²-1 ancilla | MWPM |
| Color code | ~0.8% | 2D, degree-3 | ~d² | Restriction decoder |
| Steane [[7,1,3]] | N/A (no threshold) | All-to-all | 7 | Lookup table |
| Shor [[9,1,3]] | N/A (no threshold) | All-to-all | 9 | Lookup table |

Surface codes win for scalable quantum computing due to their combination of high threshold, local connectivity, and efficient decoding.

## Using Surface Codes in Quantumium

```python
from quantumium.surface_code import (
    SurfaceCode,
    mwpm_decode,
    estimate_threshold,
)

# Create a distance-5 surface code
code = SurfaceCode(distance=5)

# Inject a random X error
code.inject_error('X', qubit=12)

# Extract syndrome
syndrome = code.measure_syndrome()

# Decode and correct
correction = mwpm_decode(code, syndrome)
code.apply_correction(correction)

# Verify logical state preserved
assert code.logical_error() == False

# Estimate threshold via Monte Carlo
results = estimate_threshold(
    distances=[3, 5, 7],
    error_rates=[0.001, 0.005, 0.01, 0.015, 0.02],
    num_trials=1000,
)
```

## Further Reading

- Fowler et al., "Surface codes: Towards practical large-scale quantum computation" (2012)
- Dennis et al., "Topological quantum memory" (2002)
- Google Quantum AI, "Suppressing quantum errors by scaling a surface code logical qubit" (Nature, 2023)
