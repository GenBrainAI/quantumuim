# Tutorial: Q+ Algorithm — Factor Integers with Fewer Qubits

Q+ is Quantumium's signature algorithm for solving problems larger than
available qubit hardware. It combines three techniques to achieve
54-60% qubit reduction for Shor's factoring algorithm.

## The Three Layers

1. **Compress** — semiclassical QFT and modular arithmetic optimization
   reduce circuit width without changing the algorithm's correctness
2. **Recycle** — measure-and-reset allows qubits to be reused via
   classical feedforward, further reducing the qubit count
3. **Cut & Distribute** — split large circuits across multiple QPUs,
   recombine results classically

## Step 1 — Factor a Semiprime

```python
from quantumium.qplus.shor import ShorQPlus

sqp = ShorQPlus()
result = sqp.factor(15)

print(f"N = {result.N}")
print(f"Factors: {result.factors}")
print(f"Qubits used: {result.qubits_used}")
print(f"Standard Shor qubits: {result.standard_qubits}")
print(f"Qubit savings: {result.savings_pct:.0f}%")
# N = 15
# Factors: (3, 5)
# Qubits used: 6
# Standard Shor qubits: 13
# Qubit savings: 54%
```

## Step 2 — Try Larger Numbers

Q+ handles all semiprimes in the test suite:

```python
for N in [15, 21, 35, 77, 143]:
    result = sqp.factor(N, max_attempts=10)
    print(f"N={N:>3}: factors={result.factors}, "
          f"qubits={result.qubits_used} (saved {result.savings_pct:.0f}%)")
```

Output:

```
N= 15: factors=(3, 5),   qubits=6  (saved 54%)
N= 21: factors=(3, 7),   qubits=7  (saved 53%)
N= 35: factors=(5, 7),   qubits=8  (saved 53%)
N= 77: factors=(7, 11),  qubits=9  (saved 57%)
N=143: factors=(11, 13), qubits=10 (saved 60%)
```

## Step 3 — Explore the Layers Individually

### Compression

```python
from quantumium.qplus.compress import SemiclassicalQFT, ModularArithmeticOptimizer

scqft = SemiclassicalQFT()
optimizer = ModularArithmeticOptimizer()
```

### Qubit Recycling

```python
from quantumium.qplus.recycle import QubitRecycler, FeedForwardController

recycler = QubitRecycler()
controller = FeedForwardController()
```

### Circuit Cutting & Distribution

```python
from quantumium.qplus.cut import CircuitCutter, CutRecombiner
from quantumium.qplus.distribute import QPUOrchestrator

cutter = CircuitCutter()
recombiner = CutRecombiner()
orchestrator = QPUOrchestrator()
```

## Performance

From our [benchmark suite](../benchmarks.md), Q+ factoring times on
commodity hardware (Python 3.11, single core):

| N | Qubits (Q+) | Qubits (standard) | Savings | Time |
|---|-------------|-------------------|---------|------|
| 15 | 6 | 13 | 54% | 0.45 ms |
| 21 | 7 | 15 | 53% | 0.49 ms |
| 35 | 8 | 17 | 53% | 0.89 ms |
| 77 | 9 | 21 | 57% | 1.81 ms |
| 143 | 10 | 25 | 60% | 17.2 ms |

## Next Steps

- [Q+ API Reference](../api/qplus.md) — full class documentation
- [Benchmarks](../benchmarks.md) — comprehensive performance data
- [Bell State Tutorial](bell-state.md) — quantum computing basics
