# Quantumium

**Comprehensive quantum computing SDK for Python.**

Build quantum circuits, run algorithms, simulate noise, and execute on real hardware from IBM, Google, and Amazon — all through a single, clean API.

**26 modules** | **12 algorithms** | **4 hardware backends** | **1095+ tests**

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

## What's Next?

- [Getting Started](getting-started.md) — step-by-step tutorial
- [Examples](examples.md) — 33 runnable scripts covering every module
- [API Reference](api/core.md) — full module documentation
