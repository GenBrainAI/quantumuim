# Getting Started with Quantumium

This tutorial walks you through the fundamentals of quantum computing with
Quantumium. By the end you will know how to create qubits, build circuits,
run measurements, use built-in algorithms, and connect to real hardware
backends.

## Installation

```bash
pip install quantumium
```

For hardware backend support, install the optional extras:

```bash
pip install quantumium[qiskit]       # IBM Quantum
pip install quantumium[cirq]         # Google Quantum AI
pip install quantumium[braket]       # Amazon Braket (IonQ, Rigetti, OQC, QuEra)
pip install quantumium[all-backends] # All backends
```

For development (includes testing and linting tools):

```bash
git clone https://github.com/GenBrainAI/quantumuim.git
cd quantumuim
pip install -e ".[dev]"
```

## 1. Working with Qubits

A qubit is the basic unit of quantum information. Unlike a classical bit
(0 or 1), a qubit can exist in a **superposition** of both states
simultaneously.

```python
from quantumium.core.qubit import Qubit

# Create a qubit in the |0> state (default)
q = Qubit.zero()
print(q)                # Qubit(alpha=1.0000+0.0000j, beta=0.0000+0.0000j)
print(q.probabilities)  # (1.0, 0.0) — 100% chance of measuring 0

# Create a qubit in superposition (|+> state)
q_plus = Qubit.plus()
print(q_plus.probabilities)  # (0.5, 0.5) — equal chance of 0 or 1

# Measure the qubit — collapses the superposition
result = q_plus.measure()
print(result)  # 0 or 1 (random)
```

You can also create qubits with arbitrary amplitudes:

```python
import numpy as np

# Custom superposition: 30% chance of |0>, 70% chance of |1>
q = Qubit(alpha=np.sqrt(0.3), beta=np.sqrt(0.7))
print(q.probabilities)  # (0.3, 0.7)
```

## 2. Quantum Gates

Gates transform qubit states. Quantumium provides all standard gates:

```python
from quantumium.core.gate import H, X, Y, Z, S, T, CNOT, CZ, SWAP, CCX
from quantumium.core.gate import Rx, Ry, Rz, CRx, CRy, CRz, Rxx, Rzz
from quantumium.core.gate import controlled

# Gates are unitary matrices
print(H.name)        # "H" (Hadamard)
print(H.num_qubits)  # 1
print(H.matrix)      # 2x2 numpy array

# Rotation gates take an angle parameter
rx_gate = Rx(np.pi / 2)  # 90-degree X rotation

# Create a controlled version of any single-qubit gate
cH = controlled(H)  # Controlled-Hadamard
```

### Gate Reference

| Gate | Qubits | Description |
|------|--------|-------------|
| `I` | 1 | Identity (no-op) |
| `X` | 1 | Pauli-X (bit flip, quantum NOT) |
| `Y` | 1 | Pauli-Y |
| `Z` | 1 | Pauli-Z (phase flip) |
| `H` | 1 | Hadamard (creates superposition) |
| `S` | 1 | Phase gate (sqrt of Z) |
| `T` | 1 | T gate (fourth root of Z) |
| `Rx(theta)` | 1 | X-axis rotation by theta |
| `Ry(theta)` | 1 | Y-axis rotation by theta |
| `Rz(theta)` | 1 | Z-axis rotation by theta |
| `CNOT` | 2 | Controlled-NOT (entanglement) |
| `CZ` | 2 | Controlled-Z |
| `SWAP` | 2 | Swap two qubits |
| `CCX` | 3 | Toffoli (controlled-controlled-NOT) |
| `CRx(theta)` | 2 | Controlled X-rotation |
| `CRy(theta)` | 2 | Controlled Y-rotation |
| `CRz(theta)` | 2 | Controlled Z-rotation |
| `Rxx(theta)` | 2 | XX Ising interaction |
| `Rzz(theta)` | 2 | ZZ Ising interaction |

## 3. Building Circuits

A `QuantumCircuit` is a sequence of gate operations on qubits:

```python
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT

# Create a 2-qubit circuit
qc = QuantumCircuit(2)

# Apply gates — qubits are indexed from 0
qc.apply(H, 0)       # Hadamard on qubit 0
qc.apply(CNOT, 0, 1) # CNOT with control=0, target=1

# This creates a Bell state: (|00> + |11>) / sqrt(2)
```

### Inspecting Circuits

```python
print(qc)                    # QuantumCircuit(qubits=2, gates=2)
print(qc.depth)              # 2
print(len(qc.instructions))  # 2

# Get the full statevector
sv = qc.statevector()
print(sv)  # [0.707+0j, 0+0j, 0+0j, 0.707+0j]
```

### Visualizing Circuits

```python
from quantumium.utils.visualization import circuit_to_ascii

print(circuit_to_ascii(qc))
# q0: ──H──●──
#          │
# q1: ─────⊕──
```

### Measuring

Measurement samples from the output probability distribution:

```python
# Run 1000 measurements
counts = qc.measure_all(shots=1000)
print(counts)
# {'00': 512, '11': 488}  (approximately 50/50)
```

## 4. Noisy Simulation

Real quantum hardware is noisy. Quantumium lets you simulate realistic
noise to understand how errors affect your circuits:

```python
from quantumium.core.noise import NoiseModel, depolarizing, amplitude_damping

# Create a noise model with 5% depolarizing noise on all single-qubit gates
noise = NoiseModel(default_single_qubit_noise=depolarizing(0.05))

# Get the density matrix under noise
rho = qc.density_matrix(noise_model=noise)

# Sample measurement outcomes with noise
noisy_counts = qc.noisy_measure(noise, shots=1000)
print(noisy_counts)
# {'00': 467, '01': 23, '10': 19, '11': 491}
#  — notice small counts for '01' and '10' due to noise
```

### Available Noise Channels

```python
from quantumium.core.noise import bit_flip, phase_flip, depolarizing, amplitude_damping

bit_flip(p=0.01)         # Flips |0> <-> |1> with probability p
phase_flip(p=0.01)       # Applies Z with probability p
depolarizing(p=0.05)     # Replaces state with maximally mixed state
amplitude_damping(g=0.1) # Models T1 energy relaxation
```

## 5. Parameterized Circuits

For variational algorithms (VQE, QAOA), you need circuits with tunable
parameters:

```python
from quantumium.core.parameter import Parameter, ParameterizedCircuit
from quantumium.core.gate import H, CNOT

# Define symbolic parameters
theta = Parameter("theta")
phi = Parameter("phi")

# Build a parameterized circuit
pqc = ParameterizedCircuit(2)
pqc.ry(theta, 0)
pqc.ry(phi, 1)
pqc.apply(CNOT, 0, 1)

# Bind concrete values to get a regular QuantumCircuit
qc = pqc.bind({theta: 1.57, phi: 0.78})
counts = qc.measure_all(shots=1000)
```

### Hardware-Efficient Ansatz

Quantumium provides a ready-made ansatz template for variational algorithms:

```python
from quantumium.core.parameter import hardware_efficient_template

pqc, params = hardware_efficient_template(num_qubits=4, depth=2)
print(f"{len(params)} parameters")  # 16 parameters
```

## 6. Circuit Templates

Common circuit patterns are available as pre-built templates:

```python
from quantumium.circuits.templates import bell_pair, ghz_state, teleportation_circuit

# Bell pair (2-qubit entanglement)
bell = bell_pair()
print(bell.measure_all(shots=1000))

# GHZ state (N-qubit entanglement)
ghz = ghz_state(5)
print(ghz.measure_all(shots=1000))
# {'00000': ~500, '11111': ~500}
```

## 7. Using Algorithms

Quantumium includes implementations of all major quantum algorithms.

### Grover's Search

Find a marked item in an unsorted database with quadratic speedup:

```python
from quantumium.algorithms.grover import grover_search

# Search for state |5> = |101> in a 3-qubit space
qc = grover_search(num_qubits=3, marked_states=[5])
counts = qc.measure_all(shots=1024)
print(counts)
# '101' should appear with highest frequency
```

### VQE (Variational Quantum Eigensolver)

Find the ground state energy of a quantum system:

```python
from quantumium.algorithms.vqe import vqe, h2_hamiltonian

# Find the ground state energy of H2
hamiltonian = h2_hamiltonian()
result = vqe(num_qubits=2, hamiltonian=hamiltonian, ansatz_depth=2)
print(f"Ground state energy: {result.optimal_energy:.4f} Ha")
# Ground state energy: ~-1.137 Ha
```

### QAOA (Quantum Approximate Optimization)

Solve combinatorial optimization problems:

```python
from quantumium.algorithms.qaoa import qaoa_maxcut

# MaxCut on a triangle graph
edges = [(0, 1), (1, 2), (2, 0)]
result = qaoa_maxcut(num_qubits=3, edges=edges, p=2)
print(f"Best cut value: {result.optimal_value:.2f}")
print(f"Best bitstring: {result.optimal_bitstring}")
```

### Shor's Algorithm

Factor integers (educational implementation):

```python
from quantumium.algorithms.shor import shor_factor

result = shor_factor(15)
print(f"Factors: {result.factors}")  # e.g., (3, 5)
```

### All Available Algorithms

| Algorithm | Module | Use Case |
|-----------|--------|----------|
| Grover's Search | `quantumium.algorithms.grover` | Unstructured search |
| Deutsch-Jozsa | `quantumium.algorithms.deutsch_jozsa` | Oracle classification |
| Bernstein-Vazirani | `quantumium.algorithms.bernstein_vazirani` | Hidden string recovery |
| Simon's | `quantumium.algorithms.simon` | Period finding |
| QFT | `quantumium.algorithms.qft` | Fourier transform |
| QPE | `quantumium.algorithms.qpe` | Phase estimation |
| VQE | `quantumium.algorithms.vqe` | Ground state energy |
| QAOA | `quantumium.algorithms.qaoa` | Combinatorial optimization |
| HHL | `quantumium.algorithms.hhl` | Linear systems (Ax=b) |
| Shor's | `quantumium.algorithms.shor` | Integer factoring |
| Hamiltonian Simulation | `quantumium.algorithms.hamiltonian_sim` | Time evolution |
| Quantum Counting | `quantumium.algorithms.quantum_counting` | Amplitude estimation |

## 8. Transpilation and Export

### Circuit Optimization

Reduce gate count and circuit depth before running on hardware:

```python
from quantumium.transpiler.passes import transpile

optimized = transpile(qc)
print(f"Before: {len(qc.instructions)} gates")
print(f"After:  {len(optimized.instructions)} gates")
```

### OpenQASM Export

Export circuits to OpenQASM 2.0 for use with other frameworks:

```python
from quantumium.transpiler.qasm import to_qasm, from_qasm

qasm_str = to_qasm(qc)
print(qasm_str)
# OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[2];
# h q[0];
# cx q[0],q[1];

# Round-trip: import QASM back to a QuantumCircuit
qc_back = from_qasm(qasm_str)
```

### JSON Serialization

Save and load circuits:

```python
from quantumium.transpiler.serialization import to_json, from_json

json_str = to_json(qc)
qc_loaded = from_json(json_str)
```

## 9. Connecting to Hardware Backends

Quantumium supports multiple quantum hardware providers through a unified
backend interface.

### Local Simulator (Default)

```python
from quantumium.backends.base import LocalSimulatorBackend

backend = LocalSimulatorBackend()
result = backend.run(qc, shots=1000)
print(result)
```

### IBM Quantum (via Qiskit)

```bash
pip install quantumium[qiskit]
```

```python
from quantumium.backends.qiskit_backend import QiskitBackend

# Connect to IBM Quantum (requires account)
backend = QiskitBackend(backend_name="ibm_brisbane")
result = backend.run(qc, shots=1000)
```

### Google Quantum AI (via Cirq)

```bash
pip install quantumium[cirq]
```

```python
from quantumium.backends.cirq_backend import CirqBackend

backend = CirqBackend()
result = backend.run(qc, shots=1000)
```

### Amazon Braket (IonQ, Rigetti, OQC, QuEra)

```bash
pip install quantumium[braket]
```

```python
from quantumium.backends.braket_backend import BraketBackend

backend = BraketBackend(device_arn="arn:aws:braket:::device/qpu/ionq/Aria-1")
result = backend.run(qc, shots=1000)
```

## 10. Error Correction and Mitigation

### Error Correction Codes

```python
import numpy as np
from quantumium.error_correction.bit_flip import (
    bit_flip_encode, bit_flip_syndrome, bit_flip_correct, bit_flip_decode
)

# Encode a logical |0> state into 3 physical qubits
logical_zero = np.array([1, 0], dtype=complex)
encoded = bit_flip_encode(logical_zero)
print(f"Encoded state: {encoded.shape[0]} amplitudes")  # 8 amplitudes (3 qubits)

# Simulate a bit-flip error on qubit 1
from quantumium.core.gate import X
error = np.eye(8, dtype=complex)
error = np.kron(np.eye(2), np.kron(X.matrix, np.eye(2)))
corrupted = error @ encoded

# Detect and correct the error
syndrome = bit_flip_syndrome(corrupted)
corrected = bit_flip_correct(corrupted, syndrome)
decoded = bit_flip_decode(corrected)
```

### Error Mitigation

```python
from quantumium.mitigation.zne import noise_scaled_circuits, richardson_extrapolate

# Create noise-scaled versions of a circuit for ZNE
scale_factors = [1, 2, 3]
scaled = noise_scaled_circuits(qc, scale_factors)

# Measure expectation values at each noise level, then extrapolate to zero noise
# expectation_values = [measure(c) for c in scaled]
# mitigated = richardson_extrapolate(scale_factors, expectation_values)
```

## Next Steps

- Browse the `examples/` directory for 24 runnable demos
- Explore domain applications: chemistry, finance, optimization, cryptography
- Read the `kb/` directory for quantum computing theory and references
- Check the [README](../README.md) for the full module reference
- Run the test suite: `python -m pytest tests/ -q`
