# Quantumium

[![CI](https://github.com/GenBrainAI/quantumuim/actions/workflows/ci.yml/badge.svg)](https://github.com/GenBrainAI/quantumuim/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1709%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-report-yellow.svg)](https://github.com/GenBrainAI/quantumuim/actions/workflows/ci.yml)

A comprehensive quantum computing SDK for Python. Build quantum circuits, run
algorithms, simulate noise, and execute on real hardware from IBM, Google, and
Amazon — all through a single, clean API.

**30 packages** | **12 algorithms** | **4 hardware backends** | **1,709 tests**

### Key Features

- **Write once, run anywhere** — unified backend interface for IBM Quantum, Google Quantum AI, and Amazon Braket
- **Full algorithm library** — Grover, VQE, QAOA, Shor, HHL, QPE, QFT, and more
- **Realistic simulation** — density matrix evolution with depolarizing, amplitude damping, and custom noise models
- **Domain applications** — quantum chemistry, finance, optimization, cryptography, and machine learning
- **Advanced simulation** — tensor networks (MPS), stabilizer tableaux, open quantum systems, adiabatic computing
- **Production tooling** — transpiler, OpenQASM 2.0 import/export, error correction, error mitigation, benchmarking

### Q+ Algorithm — 54–60% Qubit Reduction

Q+ factors integers using dramatically fewer qubits than standard Shor's algorithm:

```python
from quantumium.qplus.shor import ShorQPlus

result = ShorQPlus().factor(143)
print(result.factors)   # (11, 13)
print(result.qubits)    # 10 qubits (vs 25 standard — 60% savings)
```

| N | Qubits (Q+) | Standard Shor | Savings |
|---|-------------|---------------|---------|
| 15 | 6 | 13 | 54% |
| 77 | 9 | 21 | 57% |
| 143 | 10 | 25 | **60%** |

### Simulator Performance

Tensor-based statevector simulation — 3,000–12,000x faster than v0.2.0:

| Circuit | v0.2.0 | v0.3.0 | Speedup |
|---------|--------|--------|---------|
| GHZ-8 | 1,453 ms | 0.21 ms | **6,919x** |
| QFT-7 | 1,896 ms | 0.56 ms | **3,385x** |
| Random-8x3 | 8,059 ms | 0.65 ms | **12,399x** |

20-qubit simulation now runs in under 1 second — previously infeasible.

> **New to Quantumium?** Start with the [Getting Started tutorial](docs/getting-started.md).

## Installation

```bash
pip install quantumium
```

With hardware backend support:

```bash
pip install quantumium[qiskit]       # IBM Quantum
pip install quantumium[cirq]         # Google Quantum AI
pip install quantumium[braket]       # Amazon Braket (IonQ, Rigetti, OQC, QuEra)
pip install quantumium[all-backends] # All backends
```

For development:

```bash
git clone https://github.com/GenBrainAI/quantumuim.git
cd quantumuim
pip install -e ".[dev]"
python -m pytest tests/ -q
```

## Quick Start

```python
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT

# Create a Bell state
qc = QuantumCircuit(2)
qc.apply(H, 0)
qc.apply(CNOT, 0, 1)
print(qc.measure_all(shots=1000))
# {'00': ~500, '11': ~500}
```

### Noisy Simulation

```python
from quantumium.core.noise import NoiseModel, depolarizing

noise = NoiseModel(default_single_qubit_noise=depolarizing(0.05))
rho = qc.density_matrix(noise_model=noise)
counts = qc.noisy_measure(noise, shots=1000)
```

### Circuit Visualization

```python
from quantumium.utils.visualization import circuit_to_ascii
print(circuit_to_ascii(qc))
# q0: ──H──●──
#          │
# q1: ─────⊕──
```

### Parameterized Circuits

```python
from quantumium.core.parameter import Parameter, ParameterizedCircuit

theta = Parameter("theta")
pqc = ParameterizedCircuit(1)
pqc.ry(theta, 0)
qc = pqc.bind({theta: 1.57})
```

## Modules

### Core (`quantumium.core`)
- **Qubit**: State vector representation with measurement
- **Gate**: Full gate library — I, X, Y, Z, H, S, T, CNOT, CZ, CCX (Toffoli), SWAP, Rx, Ry, Rz, CRx, CRy, CRz, Rxx, Rzz, plus `controlled()` factory
- **QuantumCircuit**: Statevector and density matrix simulation with tensor contraction, noisy measurement
- **Noise**: Bit flip, phase flip, depolarizing, amplitude damping channels with NoiseModel
- **Parameter**: Symbolic parameterized circuits for variational workflows

### Q+ Algorithm (`quantumium.qplus`)
- **ShorQPlus**: Integer factoring with 54–60% qubit reduction
- **compress**: Semiclassical QFT, modular arithmetic optimization
- **recycle**: Mid-circuit measurement and qubit reuse
- **cut**: Circuit cutting with Pauli decomposition
- **distribute**: Multi-QPU orchestration

### Algorithms (`quantumium.algorithms`)
| Algorithm | Module | Description |
|-----------|--------|-------------|
| Grover's Search | `grover` | Unstructured search with quadratic speedup |
| Deutsch-Jozsa | `deutsch_jozsa` | Constant vs balanced oracle classification |
| Bernstein-Vazirani | `bernstein_vazirani` | Hidden string recovery |
| Simon's | `simon` | Hidden subgroup / period finding |
| QFT | `qft` | Quantum Fourier Transform and inverse |
| QPE | `qpe` | Quantum Phase Estimation |
| VQE | `vqe` | Variational Quantum Eigensolver with Adam optimizer |
| QAOA | `qaoa` | Approximate optimization (MaxCut) |
| Hamiltonian Simulation | `hamiltonian_sim` | Trotter-Suzuki decomposition |
| Shor's Algorithm | `shor` | Integer factoring (educational) |
| HHL | `hhl` | Quantum linear systems solver (Ax=b) |

### Applications (`quantumium.applications`)
- **Chemistry**: Molecular simulation (H2, LiH, HeH+), potential energy surfaces via VQE
- **Optimization**: MaxCut, TSP, portfolio optimization via QAOA
- **QRNG**: Quantum random number generation (bits, ints, floats, bytes, shuffle)

### Quantum Machine Learning (`quantumium.qml`)
- **Encoding**: Amplitude, angle, and basis encoding of classical data
- **Classifier**: Variational quantum classifier (VQC)
- **Kernel**: Quantum kernel methods for SVM

### Cryptography (`quantumium.crypto`)
- **BB84**: Quantum key distribution protocol simulation
- **QRNG**: Cryptographic quantum random number generation

### Finance (`quantumium.finance`)
- **Option Pricing**: Quantum amplitude estimation for European options
- **Risk Analysis**: Value-at-Risk (VaR) and Conditional VaR
- **Portfolio Optimization**: Quantum-enhanced portfolio selection

### Fermionic Simulation (`quantumium.fermion`)
- **FermionOperator**: Ladder operator algebra with complex coefficients
- **Jordan-Wigner transform**: O(n) Z-string anti-commutation enforcement
- **Bravyi-Kitaev transform**: O(log n) Pauli weight via Fenwick tree
- **Hamiltonian builders**: Fermi-Hubbard, molecular Hamiltonian from integrals

### Dynamical Decoupling (`quantumium.dd`)
- **DDSequence**: Spin echo, CPMG, XY-4, XY-8, UDD pulse sequences
- **DDScheduler**: Automatic idle window detection and DD insertion
- **Filter function**: Noise frequency suppression analysis

### Resource Estimation (`quantumium.resource_estimation`)
- **CircuitAnalyzer**: T-gate counting, T-depth, rotation decomposition costs
- **ResourceEstimator**: End-to-end circuit → physical qubit/runtime estimates
- **MagicStateFactory**: 15-to-1 distillation cost model

### Classical Shadows (`quantumium.shadows`)
- Random Pauli measurement protocol, median-of-means estimation
- Pauli expectation values, fidelity, purity estimation

### Transpiler (`quantumium.transpiler`)
- **Optimization passes**: Inverse cancellation, gate merging, CNOT cancellation, commutation, SWAP decomposition
- **Qubit routing**: SABRE algorithm for topology-aware SWAP insertion (linear, grid, heavy-hex)
- **OpenQASM 2.0**: Import/export with round-trip fidelity
- **JSON serialization**: Save/load circuits to file

### Error Correction & Mitigation
- Bit-flip, phase-flip, Shor 9-qubit codes (`quantumium.error_correction`)
- Surface code with MWPM decoder (`quantumium.surface_code`)
- ZNE, measurement error mitigation, Pauli twirling (`quantumium.mitigation`)

### Advanced Simulation
- **Tensor Networks** (`quantumium.tensor_networks`): MPS simulation
- **Stabilizer** (`quantumium.stabilizer`): Clifford tableau
- **Open Systems** (`quantumium.open_systems`): Lindblad master equation
- **Adiabatic** (`quantumium.adiabatic`): Adiabatic quantum computing
- **Quantum Walks** (`quantumium.quantum_walks`): Discrete and continuous walks
- **QSP/QSVT** (`quantumium.qsp`): Quantum Signal Processing
- **Optimal Control** (`quantumium.optimal_control`): GRAPE pulse optimization
- **ZX-Calculus** (`quantumium.zx`): ZX-diagram optimization, T-count reduction

### Utilities & Information Theory
- Entanglement measures: partial trace, von Neumann entropy, concurrence, negativity, Schmidt decomposition
- Quantum channels: Kraus, Choi, superoperator representations
- State/process tomography, ASCII circuit visualization

### Hardware Backends
| Backend | Provider | Devices |
|---------|----------|---------|
| `LocalSimulatorBackend` | Built-in | Statevector simulator |
| `QiskitBackend` | IBM | Aer simulator, IBM Quantum hardware |
| `CirqBackend` | Google | Cirq simulator, Google Quantum Engine |
| `BraketBackend` | Amazon | IonQ, Rigetti, OQC, QuEra, SV1/DM1/TN1 |

## Benchmarks

44 benchmarks across 5 categories. Run them yourself:

```bash
python benchmarks/bench_simulator.py
```

See [Benchmarks](docs/benchmarks.md) for full results.

## Examples

37 runnable example scripts covering every module:

```bash
python examples/bell_state.py               # Bell state basics
python examples/grover_search.py            # Grover's search
python examples/shor_factoring.py           # Shor's factoring
python examples/quantum_finance_demo.py     # Finance applications
python examples/qml_demo.py                 # Machine learning
python examples/fermionic_demo.py           # Fermionic simulation
python examples/dd_demo.py                  # Dynamical decoupling
python examples/resource_estimation_demo.py # Resource estimation
```

See `examples/` for the full list.

## Testing

```bash
python -m pytest tests/ -q    # 1,709 tests
```

## Documentation

- [Documentation Site](https://genbrainai.github.io/quantumuim) — full API reference and guides
- [Getting Started Tutorial](docs/getting-started.md) — step-by-step introduction
- [Tutorials](docs/tutorials/) — Bell state, VQE, Q+ factoring
- [Example Gallery](docs/examples.md) — 37 runnable scripts organized by category
- [Benchmarks](docs/benchmarks.md) — performance data across all simulators
- [Jupyter Notebooks](notebooks/) — interactive tutorials
- [Contributing Guide](CONTRIBUTING.md) — development setup and workflow
- [Changelog](CHANGELOG.md) — version history

Build the docs locally:

```bash
pip install quantumium[docs]
mkdocs serve
```

## License

[MIT](LICENSE) — Copyright (c) 2024-2026 Quantumium Contributors
