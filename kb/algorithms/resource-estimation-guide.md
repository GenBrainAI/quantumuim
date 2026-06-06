# Resource Estimation Guide

## Overview

Resource estimation answers the critical question: "How many physical qubits and how long will it take to run this quantum algorithm on fault-tolerant hardware?" It bridges the gap between abstract quantum algorithms and concrete hardware requirements, helping researchers and companies plan quantum computing investments.

## Key Resources to Estimate

### Logical Resources

- **Logical qubits**: number of error-corrected qubits the algorithm needs
- **T-gate count**: number of non-Clifford T gates (dominant cost in fault-tolerant computation)
- **T-depth**: sequential depth of T gates (determines runtime)
- **Clifford gate count**: H, S, CNOT operations (relatively cheap in fault-tolerant schemes)
- **Circuit depth**: total sequential gate count (determines wall-clock time)
- **Measurement count**: mid-circuit and final measurements

### Physical Resources

- **Physical qubits**: logical_qubits × qubits_per_logical_qubit (depends on code distance)
- **Code distance**: surface code distance d needed for target logical error rate
- **Magic state factories**: dedicated qubit regions for T-gate distillation
- **Runtime**: circuit_depth × code_cycle_time

## T-Gate Counting

T gates dominate fault-tolerant cost because:
1. Clifford gates are transversal (free in surface codes)
2. T gates require magic state distillation — expensive factories producing |T⟩ = T|+⟩ states
3. Each T gate consumes one distilled magic state

### Common T-Gate Counts

| Algorithm | T-gates | T-depth |
|-----------|---------|---------|
| Toffoli (CCX) | 7 | 4 |
| Controlled-S | 3 | 1 |
| Rotation Rz(θ) | O(log(1/ε)) | O(log(1/ε)) |
| n-bit adder | O(n) | O(n) |
| n-bit multiplier | O(n²) | O(n) |

### Rotation Synthesis

Arbitrary rotations Rz(θ) must be decomposed into Clifford+T:
- Solovay-Kitaev: O(log³·⁷⁵(1/ε)) T-gates for precision ε
- Gridsynth/Ross-Selinger: O(log(1/ε)) T-gates (optimal)
- Repeat-until-success: probabilistic, average O(log(1/ε))

## Surface Code Model

### Code Distance Selection

For a target logical error rate p_L and physical error rate p_phys:

```
p_L ≈ 0.1 × (p_phys / p_threshold)^((d+1)/2)
```

Where p_threshold ≈ 1% for surface codes. Solving for d:

```
d = 2 × ceil(log(10 × p_L) / log(p_phys / p_threshold)) - 1
```

### Physical Qubit Count

Each logical qubit in a rotated surface code uses 2d² - 1 physical qubits (data + syndrome).

With routing overhead (typically 1.5-2× for nearest-neighbor connectivity):

```
total_physical = n_logical × (2d² - 1) × routing_factor + factory_qubits
```

### Magic State Distillation

15-to-1 distillation factory:
- Input: 15 noisy |T⟩ states at error rate p
- Output: 1 clean |T⟩ state at error rate ~35p³
- Cost: ~15 × (2d² - 1) physical qubits per factory
- Throughput: 1 T-state per ~5d code cycles

Multiple factories needed for T-gate-heavy algorithms.

## Implementation Architecture

### CircuitAnalyzer

Extract logical resource counts from a circuit:

```python
class CircuitAnalyzer:
    def analyze(self, circuit: QuantumCircuit) -> LogicalResources:
        """Count gates, qubits, depth, and T-gates in a circuit."""
    
    def t_count(self, circuit: QuantumCircuit) -> int:
        """Count T and T† gates."""
    
    def t_depth(self, circuit: QuantumCircuit) -> int:
        """Compute sequential T-gate depth."""
    
    def clifford_count(self, circuit: QuantumCircuit) -> dict[str, int]:
        """Count Clifford gates by type."""
```

### LogicalResources

```python
@dataclass
class LogicalResources:
    n_qubits: int
    n_t_gates: int
    t_depth: int
    n_clifford: int
    n_measurements: int
    circuit_depth: int
    
    def summary(self) -> str:
        """Human-readable summary."""
```

### PhysicalEstimate

```python
@dataclass
class PhysicalEstimate:
    logical: LogicalResources
    code_distance: int
    physical_qubits: int
    n_factories: int
    factory_qubits: int
    total_qubits: int
    runtime_seconds: float
    logical_error_rate: float
```

### ResourceEstimator

```python
class ResourceEstimator:
    def __init__(self, 
                 physical_error_rate: float = 1e-3,
                 code_cycle_time_us: float = 1.0,
                 routing_overhead: float = 1.5):
        ...
    
    def estimate(self, circuit: QuantumCircuit,
                 target_error: float = 1e-6) -> PhysicalEstimate:
        """Full resource estimation from circuit to physical requirements."""
    
    def code_distance(self, target_error: float, 
                      n_logical_ops: int) -> int:
        """Compute required code distance for target logical error rate."""
    
    def factory_count(self, t_count: int, t_depth: int,
                      code_distance: int) -> int:
        """How many magic state factories are needed."""
```

## Estimation Workflow

1. **Circuit analysis**: count gates, qubits, depth, T-gates
2. **Code distance**: from target error rate and physical error rate
3. **Data qubits**: n_logical × (2d² - 1) × routing_overhead
4. **Factories**: ceil(t_count / t_depth) factories for throughput matching
5. **Factory qubits**: n_factories × factory_qubit_cost(d)
6. **Total**: data + factory qubits
7. **Runtime**: max(circuit_depth, t_count/n_factories) × code_cycle_time × d

## Testing Strategy

1. **Gate counting**: verify T-count for known circuits (Toffoli = 7T)
2. **T-depth**: verify for parallel vs sequential T gates
3. **Code distance**: verify formula against known values (p_phys=1e-3, p_L=1e-10 → d=13)
4. **Physical qubits**: verify 2d²-1 formula
5. **Factory sizing**: verify throughput matching
6. **Known algorithms**: cross-check against published estimates (Shor's 2048-bit, chemistry)
7. **Scaling**: more qubits or gates → more resources (monotonic)
8. **Edge cases**: Clifford-only circuits (T=0), single-gate circuits

## References

- Fowler et al. (2012): surface code thresholds and overhead
- Litinski (2019): game of surface codes — magic state distillation layouts
- Beverland et al. (2022): Azure Quantum resource estimation
- Ross, Selinger (2016): optimal ancilla-free Clifford+T synthesis
- Gidney, Ekerå (2021): factor 2048-bit RSA in 8 hours using 20M qubits
