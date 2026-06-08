# Quantumium SDK — Performance Benchmarks

Standardized benchmark suite for measuring simulator performance across
circuit types, simulation backends, and algorithmic workloads.

## Quick Start

```bash
# Run all benchmarks (3 iterations each)
python benchmarks/bench_simulator.py

# Custom iterations and output
python benchmarks/bench_simulator.py --iterations 5 --output my_results.json
```

## Benchmark Categories

### 1. Statevector Simulation (2-8 qubits)

Full statevector simulation using O(4^n) matrix multiply:

| Benchmark | Qubits | Description |
|-----------|--------|-------------|
| Bell | 2 | H + CNOT Bell pair |
| GHZ-3/5/7/8 | 3-8 | GHZ state preparation |
| QFT-3/5/7 | 3-7 | Quantum Fourier Transform |
| Random-NxD | 5-8 | Random 1Q gates + CNOT layers |

### 2. Density Matrix Simulation (2-7 qubits)

Noisy simulation with depolarizing error model (1% single-qubit, 2% two-qubit):

| Benchmark | Qubits | Description |
|-----------|--------|-------------|
| Bell-noisy | 2 | Bell pair with noise |
| GHZ-3/5/7-noisy | 3-7 | GHZ with depolarizing noise |
| Random-3/5/7-noisy | 3-7 | Random circuits with noise |

### 3. MPS Simulation (10-30 qubits)

Matrix Product State simulation — scales beyond statevector limits with bounded entanglement:

| Benchmark | Qubits | Bond dim | Description |
|-----------|--------|----------|-------------|
| MPS-GHZ-10/15/20 | 10-20 | 8 | GHZ via MPS |
| MPS-GHZ-20 | 20 | 16 | Higher bond dimension |
| MPS-GHZ-30 | 30 | 8 | Large-scale MPS |

### 4. VQE / QAOA Optimization

Variational algorithm timing (circuit construction + optimization loop):

| Benchmark | Qubits | Description |
|-----------|--------|-------------|
| VQE-H2 | 2 | H2 ground state, depth=1, 30 iterations |
| QAOA-MaxCut-4 | 4 | 4-node ring graph, p=1 |
| QAOA-MaxCut-6 | 6 | 6-node graph with diagonal, p=1 |

### 5. Q+ Shor Factoring

Q+ enhanced Shor's algorithm — factors semiprimes using 54-60% fewer qubits:

| N | Qubits (Q+) | Qubits (standard) | Savings |
|---|-------------|-------------------|---------|
| 15 | 6 | 13 | 54% |
| 21 | 7 | 15 | 53% |
| 35 | 8 | 17 | 53% |
| 77 | 9 | 21 | 57% |
| 143 | 10 | 25 | 60% |

## Output Format

Results are saved as JSON:

```json
{
  "metadata": {
    "python_version": "3.11.x",
    "platform": "Linux-...",
    "numpy_version": "2.x.x",
    "iterations": 3
  },
  "results": [
    {
      "name": "QFT-5",
      "category": "statevector",
      "num_qubits": 5,
      "circuit_depth": 15,
      "gate_count": 15,
      "mean_ms": 12.5,
      "std_ms": 1.2,
      "peak_memory_delta_kb": 512,
      "times_ms": [13.1, 12.8, 11.6]
    }
  ]
}
```

## Regression Detection

Compare against `baseline_results.json`:

```python
import json

with open("benchmarks/baseline_results.json") as f:
    baseline = {r["name"]: r for r in json.load(f)["results"]}

with open("my_results.json") as f:
    current = {r["name"]: r for r in json.load(f)["results"]}

for name, base in baseline.items():
    curr = current.get(name)
    if curr:
        ratio = curr["mean_ms"] / base["mean_ms"]
        flag = "REGRESSION" if ratio > 1.5 else "OK"
        print(f"{name}: {base['mean_ms']:.1f} -> {curr['mean_ms']:.1f} ms ({ratio:.2f}x) [{flag}]")
```

A >1.5x slowdown on any benchmark warrants investigation.

## Scaling Notes

The SDK's statevector simulator uses full 2^n x 2^n matrix multiplication for each gate,
giving O(4^n) per-gate cost. This limits practical statevector benchmarks to ~8-10 qubits.
For larger circuits, MPS simulation scales polynomially in qubit count (linear in bond dimension),
making it suitable for circuits with bounded entanglement.
