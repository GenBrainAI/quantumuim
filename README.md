# Quantumium

Quantum computing SDK — algorithms, hardware backends, and error mitigation for NISQ devices.

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

## What's Included

### Core (`quantumium.core`)
- **Qubit**: State vector representation with measurement
- **Gate**: Unitary gate library (I, X, Y, Z, H, S, T, CNOT, SWAP, Rx, Ry, Rz)
- **QuantumCircuit**: Circuit construction, statevector simulation, measurement
- **Noise**: Bit flip, phase flip, depolarizing, amplitude damping channels

### Algorithms (`quantumium.algorithms`)
| Algorithm | Module | Description |
|-----------|--------|-------------|
| Grover's Search | `grover` | Unstructured search with quadratic speedup |
| Deutsch-Jozsa | `deutsch_jozsa` | Constant vs balanced oracle classification |
| Bernstein-Vazirani | `bernstein_vazirani` | Hidden string recovery |
| QFT | `qft` | Quantum Fourier Transform and inverse |
| QPE | `qpe` | Quantum Phase Estimation |
| VQE | `vqe` | Variational Quantum Eigensolver (H₂ ground state) |
| QAOA | `qaoa` | Approximate optimization (MaxCut) |
| Hamiltonian Simulation | `hamiltonian_sim` | Trotter-Suzuki decomposition |
| Shor's Algorithm | `shor` | Integer factoring (educational, N ≤ 21) |

### Hardware Backends (`quantumium.backends`)
| Backend | Provider | Devices |
|---------|----------|---------|
| `LocalSimulatorBackend` | Built-in | Statevector simulator |
| `QiskitBackend` | IBM | Aer simulator, IBM Quantum hardware |
| `CirqBackend` | Google | Cirq simulator, Google Quantum Engine |
| `BraketBackend` | Amazon | IonQ Aria/Forte, Rigetti Ankaa, OQC Lucy, QuEra Aquila, SV1/DM1/TN1 |

### Error Mitigation (`quantumium.mitigation`)
- **Zero-Noise Extrapolation (ZNE)**: Unitary folding + Richardson/linear extrapolation
- **Measurement Error Mitigation**: Calibration matrix inversion
- **Pauli Twirling**: Coherent-to-stochastic noise conversion

### Domain Applications (`quantumium.applications`)
- **Chemistry**: Molecular simulation (H₂, LiH, HeH⁺), potential energy surfaces
- **Optimization**: MaxCut, TSP, portfolio optimization via QAOA

### Circuit Templates (`quantumium.circuits`)
Bell pair, GHZ state, quantum teleportation, QFT / inverse QFT

## Examples

```bash
python examples/bell_state.py
python examples/grover_search.py
python examples/vqe_h2_example.py
python examples/qaoa_maxcut_example.py
python examples/shor_factoring.py
python examples/ising_simulation.py
```

## Testing

```bash
python -m pytest tests/ -q     # 175 tests
```

## Hardware Platforms

| Platform | SDK | Status |
|----------|-----|--------|
| IBM Quantum | Qiskit | Backend ready |
| Google Quantum AI | Cirq | Backend ready |
| Amazon Braket | Braket SDK | Backend ready |
| IonQ | via Braket | Backend ready |
| Rigetti | via Braket | Backend ready |

## License

MIT
