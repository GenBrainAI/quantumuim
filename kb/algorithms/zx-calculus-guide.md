# ZX-Calculus — Practical Guide

The ZX-calculus is a graphical language for reasoning about quantum circuits. It represents quantum operations as diagrams (ZX-diagrams) that can be simplified using rewrite rules, often producing circuits with significantly fewer gates than traditional optimization passes.

## Why ZX-Calculus Matters

Traditional circuit optimization works gate-by-gate: cancel inverses, merge rotations, commute through controls. ZX-calculus lifts the entire circuit into a graphical representation where global simplifications become local graph rewrites. This catches optimizations that gate-level passes miss.

1. **T-gate reduction** — T gates are the bottleneck for fault-tolerant computation (each requires magic state distillation). ZX-calculus routinely reduces T-count by 30-50%
2. **Cross-gate optimization** — sees through layers of CNOTs and single-qubit gates to find cancellations
3. **Verification** — two circuits are equivalent iff their ZX-diagrams simplify to the same normal form
4. **Hardware compilation** — extract circuits optimized for specific connectivity

## ZX-Diagrams

### Spiders

The basic building blocks are Z-spiders (green) and X-spiders (red):

```
Z-spider(α):  represents  |0⟩⟨0| + e^{iα} |1⟩⟨1|  (phase in Z basis)
X-spider(α):  represents  |+⟩⟨+| + e^{iα} |−⟩⟨−|  (phase in X basis)
```

Each spider has:
- A **phase** α ∈ [0, 2π)
- Any number of **input** and **output** wires
- A **color**: green (Z) or red (X)

### Hadamard Edge

A Hadamard box (yellow square or dashed edge) converts between Z and X bases. An H-edge between two spiders is equivalent to inserting a Hadamard gate on that wire.

### Common Gates as ZX-Diagrams

| Gate | ZX Representation |
|------|-------------------|
| Z(α) | Z-spider with phase α, 1 input, 1 output |
| X(α) | X-spider with phase α, 1 input, 1 output |
| H | Z-spider(0) connected via H-edge to another Z-spider(0), or equivalently a standalone H-box |
| CNOT | Z-spider (control, 1-in 2-out) connected to X-spider (target, 2-in 1-out) |
| CZ | Z-spider (1-in 2-out) connected via H-edge to Z-spider (2-in 1-out) |
| T | Z-spider with phase π/4 |
| S | Z-spider with phase π/2 |

## Rewrite Rules

The power of ZX-calculus comes from a small set of rewrite rules that preserve the semantics of the diagram.

### Spider Fusion

Two connected spiders of the same color merge into one spider with the sum of their phases:

```
Z(α) — Z(β)  →  Z(α + β)     (same for X spiders)
```

This is the most important rule — it enables cascading simplifications.

### Identity Removal

A spider with phase 0 and exactly 2 wires (1 in, 1 out) is the identity:

```
Z(0) with 1 input, 1 output  →  plain wire
```

### Color Change (Hadamard)

An X-spider surrounded by Hadamard edges on all wires equals a Z-spider (and vice versa):

```
H — X(α) — H  =  Z(α)
```

### Bialgebra Rule

A Z-spider connected to an X-spider (each with 2 inputs and 2 outputs) can be "pulled apart":

```
Z(0)   X(0)       X(0) — Z(0)
  \   /    \       /         \
   \ /      \     /           \
    X        →   |    =    wires cross
   / \      /     \           /
  /   \    /       \         /
X(0)   Z(0)       Z(0) — X(0)
```

This "bialgebra" or "Hopf" rule lets spiders of different colors pass through each other.

### π-Copy Rule

A Z-spider with phase π copies through X-spiders:

```
Z(π) — X(α)  =  X(-α) — Z(π)    (with appropriate wire reconnections)
```

### Supplementarity

Two Z-spiders with phases α and α+π connected in a specific pattern can be simplified.

## The Simplification Pipeline

### 1. Circuit → ZX-Diagram

Convert each gate to its ZX representation and compose them by connecting wires.

### 2. Graph-Like Form

Transform the diagram to graph-like form:
- All spiders are Z-spiders (use color change to convert X-spiders)
- All connections between spiders are Hadamard edges (regular edges get fused via spider fusion)
- No parallel edges or self-loops (simplify using the rules)

### 3. Simplification Passes

Apply rewrite rules until no more apply:
1. **Spider fusion** — merge adjacent same-color spiders
2. **Identity removal** — remove phase-0 degree-2 spiders
3. **Local complementation** — for Clifford spiders (phases that are multiples of π/2)
4. **Pivoting** — for pairs of Clifford spiders connected by Hadamard edges
5. **Gadgetization** — isolate non-Clifford phases (T gates) as "phase gadgets" so Clifford simplification can proceed around them

### 4. Circuit Extraction

Convert the simplified ZX-diagram back to a quantum circuit. This is the most complex step:
1. Process Z-spiders in topological order
2. Each spider becomes a combination of CNOTs (for connectivity) and phase gates (for the spider's phase)
3. Use Gaussian elimination to minimize CNOT count

## T-Count Optimization

The primary application is reducing T-count for fault-tolerant circuits.

### Phase Gadgets

A T gate (phase π/4) that appears in the ZX-diagram becomes a "phase gadget" — a Z-spider with phase π/4 connected to the main diagram via Hadamard edges. Phase gadgets can be:

1. **Merged** — two gadgets connected to the same set of qubits combine phases
2. **Cancelled** — gadgets with phases that sum to 0 (mod 2π) are removed
3. **Simplified** — Clifford simplification around gadgets reveals new merging opportunities

### Typical Reductions

| Circuit | Original T-count | After ZX optimization | Reduction |
|---------|-----------------|----------------------|-----------|
| Toffoli (CCX) | 7 | 7 | 0% (already optimal) |
| Mod 5₄ | 28 | 8 | 71% |
| GF(2⁴) mult | 112 | 68 | 39% |
| Adder₈ | 399 | 215 | 46% |

## Integration with Circuit Transpiler

ZX-calculus fits into the optimization pipeline as a "heavy" pass:

```
Input circuit
  → Gate-level passes (fast): inverse cancellation, merge rotations
  → ZX optimization (slower): convert → simplify → extract
  → Gate-level passes (cleanup): decompose to hardware gate set
  → Routing/mapping: satisfy hardware connectivity
```

ZX optimization is most valuable for:
- Circuits with many T/Rz gates (variational, arithmetic)
- Pre-compilation for fault-tolerant execution
- Equivalence checking of circuit transformations

## Practical Considerations

### When to Use ZX-Calculus

- **Yes:** T-count reduction for fault-tolerant circuits, circuit equivalence checking, optimizing Clifford+T circuits
- **Maybe:** General circuit optimization (benefits vary)
- **No:** Circuits with only Clifford gates (already efficiently simulable), very small circuits (overhead not worth it)

### Computational Cost

- Converting circuit to ZX-diagram: O(n) where n = gate count
- Simplification: typically O(n²) to O(n³) for Clifford simplification
- Circuit extraction: O(n³) due to Gaussian elimination
- For large circuits (>10,000 gates), consider only applying to subcircuits

### Limitations

- Circuit extraction can increase 2-qubit gate count even when T-count decreases
- Non-Clifford simplification is limited compared to Clifford simplification
- The simplification process is not confluent — different rule application orders may give different results

## Using ZX-Calculus in Quantumium

```python
from quantumium.zx import ZXDiagram, zx_optimize

# Convert circuit to ZX-diagram
diagram = ZXDiagram.from_circuit(circuit)

# Simplify
diagram.simplify()  # full Clifford + phase gadget optimization

# Extract optimized circuit
optimized = diagram.to_circuit()

# Or use the one-liner
optimized = zx_optimize(circuit)

# Compare T-counts
print(f"Original: {circuit.t_count()} T gates")
print(f"Optimized: {optimized.t_count()} T gates")

# Equivalence check
assert ZXDiagram.from_circuit(circuit).is_equivalent(
    ZXDiagram.from_circuit(optimized)
)
```

## Further Reading

- van de Wetering, "ZX-calculus for the working quantum computer scientist" (2020)
- Kissinger & van de Wetering, "Reducing the number of non-Clifford gates in quantum circuits" (2020)
- Duncan et al., "Graph-theoretic Simplification of Quantum Circuits with the ZX-calculus" (2020)
- PyZX library — reference implementation for benchmarks
