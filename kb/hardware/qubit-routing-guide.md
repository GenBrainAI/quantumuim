# Qubit Routing & Mapping for Hardware Execution

A practical guide to mapping logical quantum circuits onto physical hardware with constrained connectivity.

## The Problem

Quantum hardware has limited qubit connectivity. A superconducting chip may have a heavy-hex lattice where each qubit connects to 2-3 neighbors, but a quantum circuit may require two-qubit gates between any pair. **Qubit routing** bridges this gap by:

1. **Initial placement**: Mapping logical qubits to physical qubits
2. **SWAP insertion**: Adding SWAP gates to move qubit states along the connectivity graph when a required interaction isn't directly available
3. **Optimization**: Minimizing the number of inserted SWAPs (each adds ~3 CNOTs of overhead)

## Hardware Coupling Maps

### IBM Heavy-Hex (127 qubits)
```
Degree-3 connectivity, each qubit connects to 2-3 neighbors.
Linear chains with cross-links every ~4 qubits.
```

### Google Sycamore (72 qubits)
```
2D grid with nearest-neighbor connectivity.
Each qubit connects to 4 neighbors (interior).
```

### IonQ (trapped ions)
```
All-to-all connectivity — no routing needed!
But gate times are slower, so circuit depth still matters.
```

## Algorithms

### Trivial Mapping
Assign logical qubit i to physical qubit i. Only works if the circuit respects the coupling map, which is rare.

### Greedy Heuristic
For each two-qubit gate, find the shortest path between the involved physical qubits and insert SWAPs along that path. Simple but produces many unnecessary SWAPs.

### SABRE (SWAP-based Bidirectional heuristic)
The state-of-the-art routing algorithm (Li et al., 2019):

1. **Forward pass**: Process gates front-to-back. For each non-executable gate, score all candidate SWAPs by how much they reduce the total distance for upcoming gates.
2. **Backward pass**: Process gates back-to-front with the final mapping from the forward pass.
3. **Iterate**: Repeat forward/backward passes, keeping the best result.

Key insight: SABRE looks ahead at future gates (not just the immediate one) to make globally better SWAP choices.

### Noise-Aware Routing
Extend SABRE with hardware calibration data:
- Prefer SWAPs through high-fidelity links
- Avoid qubits with high error rates
- Weight SWAP cost by measured CNOT error rates

## Implementation Architecture for Quantumium

### CouplingMap
```python
class CouplingMap:
    """Physical qubit connectivity graph."""
    def __init__(self, edges):
        self.graph = build_adjacency(edges)
    
    def shortest_path(self, q1, q2) -> list[int]:
        """BFS shortest path between physical qubits."""
    
    def distance(self, q1, q2) -> int:
        """Hop count between physical qubits."""
    
    @classmethod
    def from_heavy_hex(cls, n) -> 'CouplingMap': ...
    
    @classmethod
    def from_grid(cls, rows, cols) -> 'CouplingMap': ...
    
    @classmethod
    def from_linear(cls, n) -> 'CouplingMap': ...
```

### Layout (Initial Placement)
```python
class Layout:
    """Mapping from logical to physical qubits."""
    logical_to_physical: dict[int, int]
    physical_to_logical: dict[int, int]
    
    def swap(self, p1, p2):
        """Update mapping after a SWAP on physical qubits p1, p2."""
```

### Router
```python
class SABRERouter:
    """SABRE routing algorithm."""
    def route(self, circuit, coupling_map, initial_layout=None):
        """Return routed circuit with SWAPs inserted."""
    
    def _score_swap(self, swap, front_layer, extended_set):
        """Score a candidate SWAP by distance reduction."""
```

### Transpiler Integration
```python
from quantumium.transpiler import transpile

routed = transpile(circuit, 
    coupling_map=CouplingMap.from_heavy_hex(127),
    routing_method='sabre',
    optimization_level=2)
```

## Metrics

- **SWAP count**: Number of SWAPs inserted (each = 3 CNOTs)
- **Circuit depth overhead**: Increase in circuit depth after routing
- **Gate count overhead**: Total additional gates
- **Routing time**: Wall-clock time for the routing algorithm

## Benchmarks to Include

| Circuit | Qubits | CNOT count | Expected SWAP overhead |
|---------|--------|------------|----------------------|
| QFT-8 | 8 | 28 | ~10-15 SWAPs (linear) |
| QAOA MaxCut | 10 | ~20 | ~5-10 SWAPs (grid) |
| Random | 20 | 100 | ~30-50 SWAPs (heavy-hex) |

## Testing Strategy

1. **Correctness**: Routed circuit produces same unitary as original (up to qubit permutation)
2. **Coupling compliance**: Every two-qubit gate in routed circuit operates on connected physical qubits
3. **Optimality**: SWAP count within 2x of known optimal for small circuits
4. **Scalability**: Route 50+ qubit circuits in < 1 second
5. **Round-trip**: route(circuit) preserves measurement statistics

## References

- Li et al., "Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices" (2019) — SABRE
- Zulehner et al., "An Efficient Methodology for Mapping Quantum Circuits" (2019) — optimal A* for small circuits
- Cowtan et al., "On the Qubit Routing Problem" (2019) — tket approach
