# Performance Benchmarks

Measured performance of the Quantumium SDK across simulation backends,
variational algorithms, and Q+ factoring. All results from the standardized
benchmark suite (`benchmarks/bench_simulator.py`).

!!! info "Reference Environment"
    Python 3.11.15, Linux x86_64, NumPy 2.4.6, single core.
    3 iterations per benchmark. Simulator uses O(2^n) tensor contraction.

## Statevector Simulation

Tensor-based gate application: reshape state as rank-n tensor, contract
gate via `np.tensordot`, transpose back. Cost: O(2^n * 4^k) per k-qubit gate.

| Benchmark | Qubits | Gates | Depth | Mean (ms) | Std (ms) |
|-----------|--------|-------|-------|-----------|----------|
| Bell (2q) | 2 | 2 | 2 | 0.06 | 0.04 |
| GHZ-3 | 3 | 3 | 3 | 0.05 | 0.01 |
| GHZ-5 | 5 | 5 | 5 | 0.08 | 0.02 |
| GHZ-7 | 7 | 7 | 7 | 0.11 | 0.01 |
| GHZ-8 | 8 | 8 | 8 | 0.21 | 0.06 |
| GHZ-10 | 10 | 10 | 10 | 0.23 | 0.02 |
| GHZ-12 | 12 | 12 | 12 | 0.47 | 0.03 |
| GHZ-15 | 15 | 15 | 15 | 30.9 | 14.8 |
| GHZ-20 | 20 | 20 | 20 | 970.1 | 74.0 |
| QFT-3 | 3 | 6 | 5 | 0.12 | 0.04 |
| QFT-5 | 5 | 15 | 9 | 0.25 | 0.11 |
| QFT-7 | 7 | 28 | 13 | 0.56 | 0.05 |
| QFT-10 | 10 | 55 | 19 | 3.67 | 1.42 |
| QFT-12 | 12 | 78 | 23 | 4.77 | 1.52 |
| QFT-15 | 15 | 120 | 29 | 86.2 | 14.4 |
| Random-10x5 | 10 | 95 | 15 | 2.70 | 0.15 |
| Random-12x3 | 12 | 69 | 9 | 3.02 | 0.06 |
| Random-15x3 | 15 | 87 | 9 | 61.4 | 21.8 |
| Random-20x3 | 20 | 117 | 9 | 4,968.8 | 248.9 |

**Key numbers:** 12-qubit QFT in 4.8 ms. 15-qubit random circuit in 61 ms.
20-qubit GHZ in under 1 second. Previously infeasible — the old O(4^n)
simulator couldn't handle beyond 8 qubits.

### Speedup vs. v0.3.0 (O(4^n) → O(2^n))

| Benchmark | Old (ms) | New (ms) | Speedup |
|-----------|----------|----------|---------|
| GHZ-5 | 5.98 | 0.08 | **75x** |
| GHZ-7 | 443.8 | 0.11 | **4,035x** |
| GHZ-8 | 1,452.9 | 0.21 | **6,919x** |
| QFT-5 | 42.3 | 0.25 | **169x** |
| QFT-7 | 1,895.8 | 0.56 | **3,385x** |
| Random-7x3 | 2,616.0 | 0.78 | **3,354x** |
| Random-8x3 | 8,059.1 | 0.65 | **12,399x** |
| QAOA-6 | 108,737 | 5,061 | **21x** |

## Density Matrix Simulation (Noisy)

Tensor-based ρ' = UρU† and Kraus channel application.
Depolarizing noise: 1% single-qubit, 2% two-qubit.

| Benchmark | Qubits | Gates | Mean (ms) | Std (ms) |
|-----------|--------|-------|-----------|----------|
| Bell-noisy (2q) | 2 | 2 | 0.51 | 0.04 |
| GHZ-3-noisy | 3 | 3 | 0.95 | 0.12 |
| GHZ-5-noisy | 5 | 5 | 2.93 | 0.78 |
| GHZ-7-noisy | 7 | 7 | 26.0 | 17.4 |
| GHZ-10-noisy | 10 | 10 | 7,001.1 | 104.1 |
| Random-5x3-noisy | 5 | 27 | 14.3 | 12.3 |
| Random-7x3-noisy | 7 | 39 | 53.9 | 13.2 |
| Random-10x3-noisy | 10 | 57 | 23,809.5 | 421.2 |

### Density Matrix Speedup

| Benchmark | Old (ms) | New (ms) | Speedup |
|-----------|----------|----------|---------|
| GHZ-7-noisy | 711.4 | 26.0 | **27x** |
| Random-7x3-noisy | 3,346.8 | 53.9 | **62x** |

## MPS Simulation (Tensor Networks)

Matrix Product State simulation scales polynomially with bounded
entanglement — handles circuits far beyond statevector limits:

| Benchmark | Qubits | Bond dim (chi) | Mean (ms) | Std (ms) |
|-----------|--------|------|-----------|----------|
| MPS-GHZ-10 | 10 | 8 | 0.59 | 0.26 |
| MPS-GHZ-15 | 15 | 8 | 0.65 | 0.01 |
| MPS-GHZ-20 | 20 | 8 | 0.85 | 0.01 |
| MPS-GHZ-20 | 20 | 16 | 0.85 | 0.01 |
| MPS-GHZ-30 | 30 | 8 | 1.29 | 0.02 |

MPS simulates 30 qubits in 1.3 ms. For bounded-entanglement circuits,
MPS is faster than statevector above ~12 qubits.

## VQE / QAOA Optimization

Variational algorithm timing includes circuit construction, simulation,
and classical optimization:

| Benchmark | Qubits | Mean (ms) | Std (ms) |
|-----------|--------|-----------|----------|
| VQE-H2 (depth=1, 30 iter) | 2 | 85.1 | 3.9 |
| QAOA-MaxCut-4 (p=1, 20 iter) | 4 | 422.1 | 17.4 |
| QAOA-MaxCut-6 (p=1, 20 iter) | 6 | 5,061.2 | 262.3 |

QAOA-6 improved from 109 seconds to 5 seconds (21x speedup) thanks to
the tensor-based simulator.

## Q+ Shor Factoring

Q+ enhanced Shor's algorithm — factors semiprimes using 54-60% fewer
qubits than standard Shor:

| N | Factors | Qubits (Q+) | Qubits (standard) | Savings | Time (ms) |
|---|---------|-------------|-------------------|---------|-----------|
| 15 | 3 x 5 | 6 | 13 | 54% | 0.43 |
| 21 | 3 x 7 | 7 | 15 | 53% | 0.52 |
| 35 | 5 x 7 | 8 | 17 | 53% | 0.86 |
| 77 | 7 x 11 | 9 | 21 | 57% | 1.70 |
| 143 | 11 x 13 | 10 | 25 | 60% | 22.5 |

**Headline number:** Q+ factors N=143 in 23 ms using only 10 qubits
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

## Architecture

The simulator optimization (v0.3.0 → current) replaced full 2^n x 2^n
matrix construction with tensor contraction:

- **Old:** Build full unitary via `_expand_gate()` (O(4^n) Python loops),
  then matrix-vector multiply — O(4^n) per gate
- **New:** Reshape state as rank-n tensor, contract gate via
  `np.tensordot` on target qubit axes, transpose back —
  O(2^n * 4^k) per k-qubit gate

For density matrices, the same approach replaces O(8^n) matrix multiply
(U @ rho @ U†) with two O(4^n * 4^k) tensor contractions.

The `_expand_gate()` method is preserved for backward compatibility
(used by quantum counting, process tomography, ZX-calculus conversion).
