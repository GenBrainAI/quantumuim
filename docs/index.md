# Quantumium

**Comprehensive quantum computing SDK for Python.**

Build quantum circuits, run algorithms, simulate noise, and execute on real hardware from IBM, Google, and Amazon — all through a single, clean API.

**26 modules** | **12 algorithms** | **4 hardware backends** | **1,709 tests**

## Quick Start

```python
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT

qc = QuantumCircuit(2)
qc.apply(H, 0)
qc.apply(CNOT, 0, 1)
print(qc.measure_all(shots=1000))
# {'00': ~500, '11': ~500}
```

## Installation

```bash
pip install quantumium
```

With hardware backend support:

```bash
pip install quantumium[qiskit]       # IBM Quantum
pip install quantumium[cirq]         # Google Quantum AI
pip install quantumium[braket]       # Amazon Braket
pip install quantumium[all-backends] # All backends
```

## Key Features

- **Write once, run anywhere** — unified backend interface for IBM Quantum, Google Quantum AI, and Amazon Braket
- **Full algorithm library** — Grover, VQE, QAOA, Shor, HHL, QPE, QFT, and more
- **Realistic simulation** — density matrix evolution with depolarizing, amplitude damping, and custom noise models
- **Domain applications** — quantum chemistry, finance, optimization, cryptography, and machine learning
- **Advanced simulation** — tensor networks (MPS), stabilizer tableaux, open quantum systems, adiabatic computing
- **Production tooling** — transpiler, OpenQASM 2.0 import/export, error correction, error mitigation, benchmarking

## Q+ Algorithm

Quantumium includes the Q+ algorithm — a three-layer strategy for solving
problems larger than available qubit hardware:

```python
from quantumium.qplus.shor import ShorQPlus

sqp = ShorQPlus()
result = sqp.factor(143)
print(f"Factors: {result.factors}")       # (11, 13)
print(f"Qubits used: {result.qubits_used}")  # 10 (vs 25 standard)
print(f"Savings: {result.savings_pct:.0f}%")  # 60%
```

Factors N=143 in **17 ms** using only **10 qubits** — a 60% reduction vs.
standard Shor's algorithm. See the [Q+ tutorial](tutorials/qplus.md) and
[benchmarks](benchmarks.md).

## What's Next?

- [Getting Started](getting-started.md) — step-by-step tutorial
- [Tutorials](tutorials/bell-state.md) — Bell state, VQE, Q+ walkthroughs
- [Examples](examples.md) — 35 runnable scripts covering every module
- [API Reference](api/core.md) — full module documentation
- [Benchmarks](benchmarks.md) — performance data across all backends
