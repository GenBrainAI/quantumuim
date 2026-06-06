# SDK Benchmarking Methodology

Systematic approach for benchmarking Quantumium against Qiskit, Cirq, and PennyLane across performance, correctness, and developer experience dimensions.

## Benchmark Dimensions

### 1. Circuit Construction Performance

Measure time to build circuits of varying sizes and complexity.

**Test circuits:**
| Circuit | Parameters | What It Tests |
|---------|-----------|---------------|
| QFT(n) | n = 4, 8, 16, 32, 64 | Gate count scaling, controlled rotation chains |
| Random Clifford | n = 4, 8, 16, 32 | Generic circuit construction overhead |
| VQE ansatz | n = 4, 8, layers = 1, 5, 10 | Parameterized circuit construction |
| QAOA MaxCut | n = 4, 8, 12, p = 1, 3, 5 | Problem → circuit compilation |

**Metrics:**
- Wall-clock time (median of 100 runs)
- Memory allocation (peak RSS)
- Gate count in output circuit

**Protocol:**
```python
import time
import tracemalloc

def bench_circuit_construction(sdk, circuit_fn, n, repeats=100):
    tracemalloc.start()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        circuit_fn(n)
        times.append(time.perf_counter() - t0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "median_ms": sorted(times)[repeats // 2] * 1000,
        "peak_memory_mb": peak / 1024 / 1024,
    }
```

### 2. Simulation Performance

Compare statevector simulation speed for identical circuits.

**Test matrix:**
| Simulator | SDK | Type |
|-----------|-----|------|
| Quantumium local | Quantumium | Python statevector |
| Aer statevector | Qiskit | C++ backend |
| Cirq simulator | Cirq | NumPy-based |
| default.qubit | PennyLane | NumPy/autograd |

**Circuits:**
- Bell state (2 qubits) — baseline overhead measurement
- GHZ(n) for n = 4, 8, 12, 16, 20 — scaling with qubit count
- QFT(n) for n = 4, 8, 12, 16 — gate-heavy circuits
- Random circuits (n qubits, d depth) — n = {4,8,12,16}, d = {10,50,100}

**Metrics:**
- Simulation wall-clock time
- Shots/second for sampling
- Memory usage vs qubit count

**Fair comparison notes:**
- Quantumium's simulator is pure Python/NumPy — expected to be slower than Qiskit Aer (C++)
- Focus on correctness first, then identify where performance is competitive
- Document the performance gap honestly — users choose SDKs for multiple reasons
- Include Qiskit Aer with `method='statevector'` as the C++ reference point

### 3. Transpilation Performance

Compare circuit optimization passes.

**Input circuits:**
- Unoptimized QFT(n) — lots of redundant gates
- Random circuits with cancellable gate pairs
- VQE circuits with identity subsequences

**Metrics:**
- Gate count reduction (%)
- Two-qubit gate count reduction (%)
- T-count reduction (%) — Quantumium advantage with ZX-calculus
- Transpilation wall-clock time
- Depth reduction (%)

**Expected Quantumium differentiator:**
- ZX-calculus optimization (Phase 27) should outperform gate-level-only transpilers for T-count
- Compare: Quantumium ZX → gate-level vs Qiskit optimization level 3

### 4. Correctness Verification

Verify all SDKs produce equivalent results.

**Method:**
1. Build same logical circuit in each SDK
2. Extract unitary matrix (small circuits, n ≤ 6)
3. Compare unitaries up to global phase: |tr(U†V)|/2^n ≈ 1
4. For larger circuits: compare measurement distributions (Hellinger fidelity)

**Test cases:**
- All standard gates: H, X, Y, Z, S, T, CNOT, CZ, SWAP, Toffoli
- Parameterized rotations: Rx(π/4), Ry(π/3), Rz(π/6)
- Controlled gates: CRx, CRy, CRz
- Algorithm circuits: Grover(3), QFT(4), VQE ansatz(4)

### 5. Developer Experience

Qualitative comparison across SDKs.

| Dimension | What to Evaluate |
|-----------|-----------------|
| Lines of code | Implement same algorithm in each SDK, count LOC |
| API clarity | How intuitive is the circuit construction API? |
| Documentation | Quality of docs, tutorials, examples |
| Error messages | Clarity of error messages for common mistakes |
| Installation | Time from `pip install` to first circuit run |
| Learning curve | Steps to go from "hello quantum" to VQE |

**Standardized implementations for LOC comparison:**
1. Bell state creation + measurement
2. Grover's search (3 qubits, 1 marked state)
3. VQE for H2 molecule
4. QAOA for MaxCut on 4-node graph

## Benchmark Infrastructure

### Directory Structure

```
benchmarks/
    conftest.py              # Shared fixtures, skip decorators
    bench_construction.py    # Circuit construction benchmarks
    bench_simulation.py      # Simulation performance
    bench_transpilation.py   # Optimization pass comparison
    bench_correctness.py     # Cross-SDK correctness verification
    results/                 # JSON output files
    plots/                   # Generated comparison charts
    README.md                # How to run and interpret
```

### Output Format

```json
{
    "benchmark": "simulation",
    "circuit": "qft",
    "n_qubits": 16,
    "results": {
        "quantumium": {"median_ms": 245.3, "memory_mb": 12.1},
        "qiskit_aer": {"median_ms": 18.7, "memory_mb": 8.4},
        "cirq": {"median_ms": 198.2, "memory_mb": 11.8},
        "pennylane": {"median_ms": 312.1, "memory_mb": 14.2}
    },
    "timestamp": "2026-06-06T12:00:00Z",
    "python_version": "3.11.9",
    "hardware": "Apple M2, 16GB RAM"
}
```

### Reporting

Generate comparison tables and charts:

```
┌─────────────────────────────────────────────────────┐
│ Statevector Simulation — QFT(16), median of 100     │
├──────────────┬──────────┬───────────┬──────────────┤
│ SDK          │ Time(ms) │ Mem (MB)  │ vs Quantumium│
├──────────────┼──────────┼───────────┼──────────────┤
│ Quantumium   │ 245.3    │ 12.1      │ 1.0x         │
│ Qiskit Aer   │ 18.7     │ 8.4       │ 13.1x faster │
│ Cirq         │ 198.2    │ 11.8      │ 1.2x faster  │
│ PennyLane    │ 312.1    │ 14.2      │ 0.8x slower  │
└──────────────┴──────────┴───────────┴──────────────┘
```

## Honest Positioning

Quantumium is a pure-Python educational SDK. Performance comparisons should be honest:

**Where Quantumium is expected to trail:**
- Raw simulation speed (vs Qiskit Aer C++ backend)
- Large-scale circuit compilation (vs mature transpilers)

**Where Quantumium can lead:**
- T-count optimization via ZX-calculus (Phase 27)
- Module breadth: QSP/QSVT, quantum walks, adiabatic, tensor networks, stabilizer sim
- API simplicity and code brevity
- Educational value: every algorithm includes tests and documentation
- Backend portability: single circuit runs on IBM, Google, and Amazon

**Messaging:**
"Quantumium prioritizes breadth, education, and portability over raw simulation speed. For production workloads requiring maximum performance, use Quantumium's circuit construction with Qiskit Aer or cloud backends."

## Quantumium Benchmarking API

```python
from quantumium.benchmarks import (
    BenchmarkSuite,
    compare_sdks,
    generate_report,
)

# Run full benchmark suite
suite = BenchmarkSuite(
    sdks=["quantumium", "qiskit", "cirq", "pennylane"],
    circuits=["bell", "ghz", "qft", "grover", "vqe"],
    qubit_range=[4, 8, 12, 16],
    repeats=100,
)
results = suite.run()
report = generate_report(results, output_dir="benchmarks/results/")
```

## MkDocs Documentation Site

Phase 30 includes a documentation site. Architecture:

```
docs/
    index.md                 # Landing page
    getting-started.md       # Quick start (exists)
    tutorials/
        bell-state.md        # Step-by-step tutorials
        grover.md
        vqe.md
    api/                     # Auto-generated API reference
        core.md
        algorithms.md
        backends.md
        ...
    concepts/                # Conceptual guides (from KB)
        quantum-gates.md
        noise-models.md
        error-correction.md
    benchmarks/              # Benchmark results and methodology
        methodology.md
        results.md
mkdocs.yml                  # MkDocs configuration
```

**Tools:**
- MkDocs with Material theme
- mkdocstrings for Python API docs (reads docstrings)
- mike for version management
- GitHub Pages for hosting

**mkdocs.yml skeleton:**
```yaml
site_name: Quantumium
site_description: Comprehensive quantum computing SDK for Python
repo_url: https://github.com/GenBrainAI/quantumuim
theme:
  name: material
  palette:
    primary: deep purple
    accent: cyan
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - Tutorials: ...
  - API Reference: ...
  - Benchmarks: ...
```
