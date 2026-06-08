# Q+ Algorithm — Beyond-Qubit-Count Quantum Computing

Q+ is a three-layer strategy framework for solving quantum problems larger than available qubit hardware. It reduces Shor's algorithm from 3n+1 qubits to n+2 qubits and supports multi-QPU distribution.

## Quick Start

```python
from quantumium.qplus import ShorQPlus

shor = ShorQPlus(strategy="auto")
result = shor.factor(15)
print(f"Factors: {result.factors}")
print(f"Qubits used: {result.qubits_used} vs standard {result.qubits_standard}")
```

## Architecture

### Layer 1: Algorithmic Compression (`compress.py`)

Replaces the n-qubit QFT counting register with sequential single-qubit measurements and classical phase corrections.

- **SemiclassicalQFT** — Griffiths-Niu (1996) semiclassical QFT: reduces counting register from n qubits to 1 reusable qubit
- **ModularArithmeticOptimizer** — Beauregard (2002) and Haner (2017) optimized modular exponentiation

```python
from quantumium.qplus import SemiclassicalQFT

sqft = SemiclassicalQFT(n_bits=8)
result = sqft.estimate_phase(controlled_unitary_fn, n_work=4)
print(f"Phase: {result.phase}, Qubits: {result.n_qubits_used}")
```

### Layer 2: Qubit Recycling (`recycle.py`)

Measures qubits mid-circuit, resets to |0>, and reuses them for subsequent operations.

- **QubitRecycler** — Manages physical qubit pool with measure-reset-reuse lifecycle
- **FeedForwardController** — Tracks measurement outcomes and applies classically-controlled corrections

```python
from quantumium.qplus import QubitRecycler

recycler = QubitRecycler(n_physical=5)
phase, bits = recycler.simulate_recycled_phase_estimation(ctrl_u, n_work=4, n_counting_bits=9)
```

### Layer 3a: Circuit Cutting (`cut.py`)

Decomposes circuits at wire cuts using Pauli-basis decomposition (Peng et al. 2020).

- **CircuitCutter** — Identifies cut points, generates 6^K subcircuit configurations per K cuts
- **CutRecombiner** — Classical postprocessing to reconstruct full expectation values

```python
from quantumium.qplus import CircuitCutter
from quantumium.qplus.cut import WireCut

cutter = CircuitCutter([WireCut(qubit=0, after_gate_index=5)])
configs = cutter.decompose(n_qubits=4, gates=gate_list)
print(f"Overhead: {cutter.overhead}x")
```

### Layer 3b: Multi-QPU Distribution (`distribute.py`)

Orchestrates circuit execution across multiple quantum processing units.

- **QPUOrchestrator** — Partitions circuits, manages QPU fleet
- Three strategies: `classical_only` (wire cutting), `locc` (mid-circuit measurement), `teleport` (Bell pairs)

```python
from quantumium.qplus import QPUOrchestrator

orch = QPUOrchestrator(n_qpus=2, strategy="classical_only")
plan = orch.partition_circuit(n_qubits=8)
result = orch.execute_distributed(unitary, n_qubits=8)
```

### Full Stack: ShorQPlus (`shor.py`)

Combines all layers to factor integers with minimal qubits.

```python
from quantumium.qplus import ShorQPlus

shor = ShorQPlus(strategy="recycled")
result = shor.factor(143, max_attempts=20)
print(f"143 = {result.factors[0]} x {result.factors[1]}")
print(f"Savings: {result.qubit_savings} qubits ({result.strategy})")
```

## Benchmark Results

| N | n_bits | Standard Shor | Q+ Qubits | Reduction |
|-----|--------|---------------|-----------|-----------|
| 15 | 4 | 13 | 6 | 54% |
| 21 | 5 | 16 | 7 | 56% |
| 35 | 6 | 19 | 8 | 58% |
| 77 | 7 | 22 | 9 | 59% |
| 143 | 8 | 25 | 10 | 60% |

All semiprimes factored successfully with 54-60% qubit reduction.

## Running

```bash
# Tests (113 tests)
python -m pytest src/quantumium/qplus/tests/ -v

# Layer-by-layer demo
python src/quantumium/qplus/examples/qplus_demo.py

# Factoring benchmark
python src/quantumium/qplus/examples/shor_benchmark.py
```

## Key References

- Griffiths & Niu, PRL 76, 3228 (1996) — Semiclassical QFT
- Beauregard, QIC 3(2), 175-185 (2002) — Modular exponentiation in 2n+3 qubits
- Haner et al., QST 2(2), 025009 (2017) — Further qubit reduction to 2n+2
- Peng et al., PRL 125, 150504 (2020) — Wire cutting with Pauli decomposition
