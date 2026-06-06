# Quantumium

[![CI](https://github.com/GenBrainAI/quantumuim/actions/workflows/ci.yml/badge.svg)](https://github.com/GenBrainAI/quantumuim/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1095%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-report-yellow.svg)](https://github.com/GenBrainAI/quantumuim/actions/workflows/ci.yml)

A comprehensive quantum computing SDK for Python. Build quantum circuits, run
algorithms, simulate noise, and execute on real hardware from IBM, Google, and
Amazon — all through a single, clean API.

**26 modules** | **12 algorithms** | **4 hardware backends** | **1095+ tests**

### Key Features

- **Write once, run anywhere** — unified backend interface for IBM Quantum, Google Quantum AI, and Amazon Braket
- **Full algorithm library** — Grover, VQE, QAOA, Shor, HHL, QPE, QFT, and more
- **Realistic simulation** — density matrix evolution with depolarizing, amplitude damping, and custom noise models
- **Domain applications** — quantum chemistry, finance, optimization, cryptography, and machine learning
- **Advanced simulation** — tensor networks (MPS), stabilizer tableaux, open quantum systems, adiabatic computing
- **Production tooling** — transpiler, OpenQASM 2.0 import/export, error correction, error mitigation, benchmarking

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
- **QuantumCircuit**: Statevector and density matrix simulation, noisy measurement
- **Noise**: Bit flip, phase flip, depolarizing, amplitude damping channels with NoiseModel
- **Parameter**: Symbolic parameterized circuits for variational workflows

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

### Transpiler (`quantumium.transpiler`)
- **Optimization passes**: Inverse cancellation, gate merging, CNOT cancellation, commutation, SWAP decomposition
- **OpenQASM 2.0**: Import/export with round-trip fidelity
- **JSON serialization**: Save/load circuits to file

### Error Correction (`quantumium.error_correction`)
- Bit-flip code, phase-flip code, Shor 9-qubit code

### Surface Code (`quantumium.surface_code`)
- Rotated planar surface code (d=3, 5, 7) with stabilizer orthogonality verification
- MWPM decoder (self-contained, no external dependencies)
- Logical X/Z operators, syndrome extraction, Monte Carlo threshold estimation

### ZX-Calculus (`quantumium.zx`)
- ZX-diagram representation of quantum circuits (Z/X spiders, Hadamard edges)
- Spider fusion, identity removal, color change simplification rules
- T-count optimization via phase merging through CNOT controls
- One-liner API: `zx_optimize(circuit)` for automatic circuit simplification

### Quantum Networking (`quantumium.networking`)
- Network topology simulation (star, chain, mesh) with quantum channels
- Entanglement generation, swapping through repeater chains, BBPSSW purification
- Quantum teleportation with depolarizing noise model
- One-liner API: `distribute_entanglement(network, source, target)`

### Error Mitigation (`quantumium.mitigation`)
- Zero-Noise Extrapolation (ZNE), measurement error mitigation, Pauli twirling

### Tomography (`quantumium.tomography`)
- State tomography via linear inversion and maximum likelihood estimation (MLE)

### Quantum Information (`quantumium.quantum_info`)
- Quantum channels (Kraus, Choi, superoperator), distance measures (trace distance, fidelity), process tomography

### Benchmarks (`quantumium.benchmarks`)
- Randomized benchmarking, quantum volume, fidelity estimation

### Advanced Simulation
- **Tensor Networks** (`quantumium.tensor_networks`): Matrix Product State (MPS) simulation
- **Stabilizer** (`quantumium.stabilizer`): Clifford tableau for efficient Clifford circuit simulation
- **Open Systems** (`quantumium.open_systems`): Lindblad master equation for dissipative dynamics
- **Adiabatic** (`quantumium.adiabatic`): Adiabatic quantum computing with customizable Hamiltonians
- **Quantum Walks** (`quantumium.quantum_walks`): Discrete and continuous walks on arbitrary graphs
- **QSP/QSVT** (`quantumium.qsp`): Quantum Signal Processing and Quantum Singular Value Transformation
- **Optimal Control** (`quantumium.optimal_control`): GRAPE pulse optimization for gate synthesis

### Utilities (`quantumium.utils`)
- **Entanglement**: Partial trace, von Neumann entropy, purity, concurrence, negativity, Schmidt decomposition
- **Visualization**: ASCII circuit diagrams

### Hardware Backends (`quantumium.backends`)
| Backend | Provider | Devices |
|---------|----------|---------|
| `LocalSimulatorBackend` | Built-in | Statevector simulator |
| `QiskitBackend` | IBM | Aer simulator, IBM Quantum hardware |
| `CirqBackend` | Google | Cirq simulator, Google Quantum Engine |
| `BraketBackend` | Amazon | IonQ, Rigetti, OQC, QuEra, SV1/DM1/TN1 |

### Hardware Validation (`tests/test_backends/`)
- Comprehensive test suite for all 4 backends: Bell, GHZ, Grover, QFT round-trip
- Statistical validation: chi-squared goodness-of-fit and Hellinger fidelity
- Vendor tests skip gracefully when SDK not installed
- Noise model calibration: depolarizing degradation, monotonicity, gate-specific noise
- Cross-backend comparison with automated fidelity checks

### Circuit Templates (`quantumium.circuits`)
Bell pair, GHZ state, quantum teleportation, QFT / inverse QFT

## Jupyter Notebooks

Interactive tutorials for learning quantum computing with Quantumium:

```bash
jupyter notebook notebooks/intro-to-quantum.ipynb      # Qubits, gates, circuits
jupyter notebook notebooks/quantum-algorithms.ipynb     # Grover, VQE, QAOA
jupyter notebook notebooks/quantum-finance.ipynb        # Option pricing, portfolios
```

## Examples

31 runnable example scripts covering every module:

```bash
python examples/bell_state.py               # Bell state basics
python examples/grover_search.py            # Grover's search
python examples/shor_factoring.py           # Shor's factoring
python examples/quantum_finance_demo.py     # Finance applications
python examples/qml_demo.py                 # Machine learning
python examples/adiabatic_demo.py           # Adiabatic optimization
python examples/stabilizer_demo.py          # Clifford simulation
python examples/benchmarking_demo.py        # Hardware benchmarking
python examples/tomography_demo.py          # State reconstruction
python examples/tensor_network_demo.py      # MPS simulation
python examples/surface_code_demo.py        # Surface code QEC
python examples/zx_calculus_demo.py         # ZX-calculus optimization
python examples/networking_demo.py          # Quantum networking
python examples/hardware_validation_demo.py # Backend validation
python examples/algorithms/vqe_h2_example.py    # VQE for H2
python examples/algorithms/qaoa_maxcut_example.py  # QAOA MaxCut
```

See `examples/` for the full list.

## Testing

```bash
python -m pytest tests/ -q    # 1095 tests
```

## Documentation

- [Getting Started Tutorial](docs/getting-started.md) — step-by-step introduction
- [Jupyter Notebooks](notebooks/) — interactive tutorials
- [Contributing Guide](CONTRIBUTING.md) — development setup and workflow
- [Changelog](CHANGELOG.md) — version history

## Knowledge Base

The `kb/` directory contains 23 reference articles on quantum computing:
- Quantum mechanics primer, linear algebra essentials
- Quantum information theory, complexity theory
- Error correction deep-dive, surface codes guide, ZX-calculus guide, classical shadows guide
- Algorithm catalog, variational algorithms guide
- Quantum programming patterns
- Hardware platform comparison, hardware validation guide, qubit routing guide
- SDK benchmarking methodology, v0.3.0 roadmap candidates
- Domain guides: QML, cryptography, finance, optimization, simulation, networking

## License

MIT
