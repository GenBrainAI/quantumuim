# Performance Benchmarks

Measured performance of the Quantumium SDK across simulation backends,
variational algorithms, and Q+ factoring. All results from the standardized
benchmark suite (`benchmarks/bench_simulator.py`).

!!! info "Reference Environment"
    Python 3.11.15, Linux x86_64, NumPy 2.4.6, single core.
    3 iterations per benchmark.

## Statevector Simulation

Full statevector simulation using O(4^n) matrix multiply per gate.
Practical limit: ~8 qubits.

| Benchmark | Qubits | Gates | Depth | Mean (ms) | Std (ms) |
|-----------|--------|-------|-------|-----------|----------|
| Bell (2q) | 2 | 2 | 2 | 0.07 | 0.03 |
| GHZ-3 | 3 | 3 | 3 | 0.24 | 0.03 |
| GHZ-5 | 5 | 5 | 5 | 5.98 | 0.40 |
| GHZ-7 | 7 | 7 | 7 | 443.8 | 29.5 |
| GHZ-8 | 8 | 8 | 8 | 1,452.9 | 29.9 |
| QFT-3 | 3 | 6 | 5 | 0.66 | 0.03 |
| QFT-5 | 5 | 15 | 9 | 42.3 | 10.1 |
| QFT-7 | 7 | 28 | 13 | 1,895.8 | 133.9 |
| Random-5x5 | 5 | 45 | 15 | 82.9 | 32.2 |
| Random-7x3 | 7 | 39 | 9 | 2,616.0 | 52.3 |
| Random-8x3 | 8 | 45 | 9 | 8,059.1 | 22.3 |

The exponential scaling is visible: 7 qubits takes ~444 ms per GHZ gate,
8 qubits takes ~1.45 s. This is the SDK's full-matrix simulation — each
gate applies a 2^n x 2^n unitary.

## Density Matrix Simulation (Noisy)

Noisy simulation with depolarizing error (1% single-qubit, 2% two-qubit):

| Benchmark | Qubits | Gates | Mean (ms) | Std (ms) |
|-----------|--------|-------|-----------|----------|
| Bell-noisy (2q) | 2 | 2 | 1.05 | 0.24 |
| GHZ-3-noisy | 3 | 3 | 2.15 | 0.11 |
| GHZ-5-noisy | 5 | 5 | 22.7 | 13.9 |
| GHZ-7-noisy | 7 | 7 | 711.4 | 26.2 |
| Random-3x5-noisy | 3 | 25 | 19.1 | 13.2 |
| Random-5x3-noisy | 5 | 27 | 71.1 | 31.8 |
| Random-7x3-noisy | 7 | 39 | 3,346.8 | 69.4 |

Density matrix simulation is ~1.6x slower than statevector at the same
qubit count due to Kraus channel application on 2^n x 2^n matrices.

## MPS Simulation (Tensor Networks)

Matrix Product State simulation scales polynomially with bounded
entanglement — handles circuits far beyond statevector limits:

| Benchmark | Qubits | Bond dim (chi) | Mean (ms) | Std (ms) |
|-----------|--------|------|-----------|----------|
| MPS-GHZ-10 | 10 | 8 | 10.0 | 16.5 |
| MPS-GHZ-15 | 15 | 8 | 0.88 | 0.33 |
| MPS-GHZ-20 | 20 | 8 | 0.89 | 0.07 |
| MPS-GHZ-20 | 20 | 16 | 0.93 | 0.13 |
| MPS-GHZ-30 | 30 | 8 | 13.1 | 20.2 |

**Key insight:** MPS simulates 30 qubits in 13 ms, while statevector
can barely handle 8. For circuits with bounded entanglement (like GHZ
states), MPS scales linearly in qubit count.

## VQE / QAOA Optimization

Variational algorithm timing includes circuit construction, simulation,
and classical optimization:

| Benchmark | Qubits | Mean (ms) | Std (ms) |
|-----------|--------|-----------|----------|
| VQE-H2 (depth=1, 30 iter) | 2 | 96.5 | 11.7 |
| QAOA-MaxCut-4 (p=1, 20 iter) | 4 | 1,701.4 | 11.7 |
| QAOA-MaxCut-6 (p=1, 20 iter) | 6 | 108,737.9 | 4,578.1 |

VQE converges to H2 ground state energy (~-1.137 Ha) in under 100 ms.
QAOA scaling reflects the exponential cost of statevector simulation
in the inner loop.

## Q+ Shor Factoring

Q+ enhanced Shor's algorithm — factors semiprimes using 54-60% fewer
qubits than standard Shor:

| N | Factors | Qubits (Q+) | Qubits (standard) | Savings | Time (ms) |
|---|---------|-------------|-------------------|---------|-----------|
| 15 | 3 x 5 | 6 | 13 | 54% | 0.45 |
| 21 | 3 x 7 | 7 | 15 | 53% | 0.49 |
| 35 | 5 x 7 | 8 | 17 | 53% | 0.89 |
| 77 | 7 x 11 | 9 | 21 | 57% | 1.81 |
| 143 | 11 x 13 | 10 | 25 | 60% | 17.2 |

**Headline number:** Q+ factors N=143 in 17 ms using only 10 qubits
(vs. 25 for standard Shor) — a 60% qubit reduction on commodity hardware.

## Running Benchmarks

```bash
# Run all benchmarks (3 iterations each)
python benchmarks/bench_simulator.py

# Custom iterations and output
python benchmarks/bench_simulator.py --iterations 5 --output my_results.json
```

## Regression Detection

Compare against the baseline:

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
        print(f"{name}: {base['mean_ms']:.1f} -> "
              f"{curr['mean_ms']:.1f} ms ({ratio:.2f}x) [{flag}]")
```

A >1.5x slowdown on any benchmark warrants investigation.

## Scaling Notes

The SDK's statevector simulator uses full 2^n x 2^n matrix multiplication
per gate, giving O(4^n) per-gate cost. This is an educational design choice
— it keeps the code transparent. For production-scale simulation:

- **MPS** handles 30+ qubits for bounded-entanglement circuits
- **Stabilizer tableaux** simulate Clifford circuits in O(n^2) time
- **Q+** reduces qubit requirements by 54-60% through compression, recycling, and distribution
