# Visualization

Quantumium provides publication-quality quantum state and circuit
visualizations powered by matplotlib.

## Installation

matplotlib is included as a dependency — no extra install needed.

## Bloch Sphere

Visualize single-qubit states on the Bloch sphere:

```python
import numpy as np
from quantumium.visualization import plot_bloch_sphere

# |+⟩ state
sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
fig = plot_bloch_sphere(sv)
fig.savefig("bloch.png")
```

Axis labels show the six cardinal states: |0⟩, |1⟩, |+⟩, |-⟩, |i⟩, |-i⟩.

### Multi-qubit Bloch Vectors

For multi-qubit states, `plot_bloch_multivector` shows one Bloch sphere per
qubit using reduced density matrices:

```python
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT
from quantumium.visualization import plot_bloch_multivector

qc = QuantumCircuit(2)
qc.apply(H, 0)
qc.apply(CNOT, 0, 1)
fig = plot_bloch_multivector(qc.statevector())
```

For a Bell state, both qubits appear at the center of the Bloch sphere
(maximally mixed reduced states), reflecting entanglement.

## Measurement Histograms

Plot measurement counts as a probability bar chart:

```python
from quantumium.visualization import plot_histogram

counts = {"00": 512, "11": 488}
fig = plot_histogram(counts, title="Bell State Measurements")
```

Probability labels appear above each bar. Works with any number of outcomes.

## State City Plot

Visualize the density matrix as 3D bar charts (real and imaginary parts):

```python
from quantumium.visualization import plot_state_city

qc = QuantumCircuit(2)
qc.apply(H, 0)
qc.apply(CNOT, 0, 1)
fig = plot_state_city(qc.statevector())
```

Accepts both statevectors and density matrices. Positive values are blue,
negative values are orange.

## Hinton Diagram

Squares sized by matrix element magnitude, colored by sign:

```python
from quantumium.visualization import plot_state_hinton

fig = plot_state_hinton(qc.density_matrix(), title="Bell State")
```

## Circuit Diagram

Render circuits as matplotlib figures with proper gate symbols:

```python
from quantumium.visualization import plot_circuit

qc = QuantumCircuit(3)
qc.apply(H, 0)
qc.apply(CNOT, 0, 1)
qc.apply(CNOT, 1, 2)
fig = plot_circuit(qc, title="GHZ Preparation")
```

Supported gate renderings:

| Gate | Symbol |
|------|--------|
| H, X, Y, Z, S, T | Box with label |
| CNOT | Control dot + target circle |
| CZ | Two control dots |
| SWAP | Cross symbols |
| CCX (Toffoli) | Two controls + target circle |
| Rx, Ry, Rz | Box with gate name and angle |

## Composability

All plot functions return a `matplotlib.figure.Figure`, so you can:

- Save to file: `fig.savefig("plot.png", dpi=150)`
- Display in Jupyter: figures render inline automatically
- Customize: access axes via `fig.get_axes()` for further styling
