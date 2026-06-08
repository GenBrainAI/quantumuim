# Cross-SDK Interoperability

Quantumium can convert circuits to and from **Qiskit** and **Cirq**, making it
easy to adopt Quantumium in existing workflows or benchmark against other
frameworks.

## Installation

Install the backend extra you need:

```bash
pip install quantumium[qiskit]   # Qiskit conversion
pip install quantumium[cirq]     # Cirq conversion
```

## Qiskit Conversion

```python
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT
from quantumium.interop import to_qiskit, from_qiskit

# Build a Bell state in Quantumium
qc = QuantumCircuit(2)
qc.apply(H, 0)
qc.apply(CNOT, 0, 1)

# Convert to Qiskit
qiskit_circuit = to_qiskit(qc)
print(qiskit_circuit)

# Convert back
roundtrip = from_qiskit(qiskit_circuit)
```

## Cirq Conversion

```python
from quantumium.interop import to_cirq, from_cirq

# Convert to Cirq
cirq_circuit = to_cirq(qc)
print(cirq_circuit)

# Convert back
roundtrip = from_cirq(cirq_circuit)
```

## Supported Gates

| Gate | Qiskit | Cirq |
|------|--------|------|
| H, X, Y, Z, S, T | Yes | Yes |
| CNOT, CZ, SWAP | Yes | Yes |
| CCX (Toffoli) | Yes | Yes |
| Rx, Ry, Rz | Yes | Yes |
| CRx, CRy, CRz | Yes | Yes (via MatrixGate) |
| Rxx, Rzz | Yes | Yes (via MatrixGate) |

Unsupported gates raise `ValueError` with a clear message listing available gates.

## Graceful Dependency Handling

If a backend package isn't installed, importing from `quantumium.interop` still
works — only the missing converter is unavailable:

```python
import quantumium.interop

# These are only in __all__ if the dep is installed
print(quantumium.interop.__all__)
```
