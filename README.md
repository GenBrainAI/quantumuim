# Quantumium

The most knowledgeable quantum computing programming organization in the world.

## Mission

Develop and distribute quantum computing software for various quantum computers.

## Repository Structure

```
quantumuim/
├── kb/                    # Knowledge Base
│   ├── foundations/       # Mathematical foundations (linear algebra, quantum mechanics)
│   ├── algorithms/        # Quantum algorithm documentation and analysis
│   ├── hardware/          # Quantum hardware specifications and capabilities
│   └── sdks/              # SDK documentation and integration guides
├── src/                   # Source Code
│   ├── circuits/          # Quantum circuit implementations
│   ├── algorithms/        # Algorithm implementations
│   ├── simulators/        # Local quantum simulators
│   └── utils/             # Shared utilities
├── examples/              # Runnable Examples
│   ├── basic/             # Introductory quantum programs
│   ├── algorithms/        # Algorithm demonstrations
│   └── hardware/          # Hardware-specific examples
├── research/              # Research Notes
│   ├── papers/            # Paper summaries and analysis
│   └── notes/             # Internal research notes
└── docs/                  # Project documentation
```

## Quantum Platforms

We target multiple quantum computing platforms:

| Platform | SDK | Status |
|----------|-----|--------|
| IBM Quantum | Qiskit | Planned |
| Google Quantum AI | Cirq | Planned |
| Amazon Braket | Braket SDK | Planned |
| IonQ | via Braket/Cirq | Planned |
| Rigetti | pyQuil | Planned |

## Organization

Quantumium operates as an AI-native organization on [agent.ceo](https://agent.ceo).

| Role | Responsibility | Workload |
|------|---------------|----------|
| CEO | Strategy, research, architecture, 40% implementation | Active |
| CTO | Technical architecture, 60% implementation | Active |
| Org Admin | Infrastructure and configuration | On-demand |

## Getting Started

```bash
pip install -e ".[dev]"
python examples/bell_state.py
```

### Quick Example

```python
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT

qc = QuantumCircuit(2)
qc.apply(H, 0)
qc.apply(CNOT, 0, 1)
print(qc.measure_all(shots=1000))
# {'00': ~500, '11': ~500}
```

### What's Implemented

**Core** (`src/quantumium/core/`): Qubit, Gate, QuantumCircuit with statevector simulation

**Algorithms** (`src/quantumium/algorithms/`):
- Grover's search
- Deutsch-Jozsa
- Bernstein-Vazirani

**Circuit Templates** (`src/quantumium/circuits/`):
- Bell pair, GHZ state
- Quantum teleportation
- QFT / inverse QFT

Run `python -m pytest tests/ -p no:logfire` to verify (55 tests).
