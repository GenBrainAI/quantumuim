# Benchmarks

Quantumium includes a built-in benchmarking module for measuring circuit construction and simulation performance.

## Quick Benchmark

```python
from quantumium.benchmarks.bench_suite import BenchmarkSuite

suite = BenchmarkSuite()
results = suite.run_all()
suite.print_report(results)
```

```bash
python -m quantumium.benchmarks.bench_suite
```

## What's Measured

### Circuit Construction

Time to build circuits of varying sizes:

| Circuit | Qubits | What It Tests |
|---------|--------|---------------|
| Bell state | 2 | Minimal overhead |
| GHZ(n) | 4, 8, 16 | Linear gate chains |
| QFT(n) | 4, 8, 16 | Controlled rotation chains |
| Random Clifford | 4, 8, 16 | Generic construction |

### Simulation Performance

Statevector simulation speed (shots/second):

| Circuit | Qubits | What It Tests |
|---------|--------|---------------|
| Bell state | 2 | Baseline |
| GHZ(n) | 4, 8, 16 | Scaling with qubit count |
| QFT(n) | 4, 8, 16 | Gate-heavy circuits |

### Metrics

- **Construction time** — median wall-clock time over 100 runs (ms)
- **Simulation time** — median statevector + sampling time (ms)
- **Gate count** — number of gates in the circuit
- **Shots/sec** — measurement throughput

## Honest Positioning

Quantumium is a pure-Python/NumPy educational SDK. For raw simulation speed, Qiskit Aer (C++ backend) will be faster. Quantumium prioritizes:

- **Breadth** — 26 modules covering algorithms, applications, and advanced simulation
- **Education** — every implementation documented and tested
- **Portability** — single circuit runs on IBM, Google, and Amazon hardware
- **Optimization** — ZX-calculus T-count reduction unavailable in competing SDKs

## Running the Existing Benchmarking Demo

```bash
python examples/benchmarking_demo.py
```

This runs randomized benchmarking and quantum volume estimation using the `quantumium.benchmarks` module.
