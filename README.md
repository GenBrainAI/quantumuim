# Quantumium

Quantum computing SDK for Python — simulation, algorithms, hardware backends, error mitigation, and quantum applications for NISQ and beyond.

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

### Transpiler (`quantumium.transpiler`)
- **Optimization passes**: Inverse cancellation, gate merging, CNOT cancellation, commutation, SWAP decomposition
- **OpenQASM 2.0**: Import/export with round-trip fidelity
- **JSON serialization**: Save/load circuits to file

### Error Correction (`quantumium.error_correction`)
- Bit-flip code, phase-flip code, Shor 9-qubit code

### Error Mitigation (`quantumium.mitigation`)
- Zero-Noise Extrapolation (ZNE), measurement error mitigation, Pauli twirling

### Tomography (`quantumium.tomography`)
- State tomography via linear inversion and maximum likelihood estimation (MLE)

### Benchmarks (`quantumium.benchmarks`)
- Randomized benchmarking, quantum volume, fidelity estimation

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

### Circuit Templates (`quantumium.circuits`)
Bell pair, GHZ state, quantum teleportation, QFT / inverse QFT

## Examples

```bash
python examples/bell_state.py
python examples/grover_search.py
python examples/ghz_entanglement.py
python examples/deutsch_jozsa_demo.py
python examples/noisy_simulation.py
python examples/shor_factoring.py
python examples/error_correction_demo.py
python examples/algorithms/vqe_h2_example.py
python examples/algorithms/qaoa_maxcut_example.py
python examples/algorithms/qpe_example.py
python examples/algorithms/qft_example.py
python examples/algorithms/ising_simulation.py
```

## Testing

```bash
python -m pytest tests/ -q    # 500+ tests
```

## Knowledge Base

The `kb/` directory contains reference material on quantum computing:
- Quantum mechanics primer, linear algebra essentials
- Quantum information theory, complexity theory
- Error correction deep-dive
- Algorithm catalog
- Hardware platform comparison

## License

MIT
