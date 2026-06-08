# Tutorial: Bell State and Entanglement

Learn to create, simulate, and measure a Bell state — the simplest example
of quantum entanglement.

## What You'll Build

A 2-qubit circuit that produces the Bell state
|&Phi;+&rangle; = (|00&rangle; + |11&rangle;) / &radic;2.
When measured, it always gives correlated outcomes: both qubits are 0 or
both are 1, never mismatched.

## Step 1 — Create the Circuit

```python
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT

qc = QuantumCircuit(2)
qc.apply(H, 0)        # put qubit 0 in superposition
qc.apply(CNOT, 0, 1)  # entangle qubit 1 with qubit 0
```

The Hadamard gate creates an equal superposition of |0&rangle; and |1&rangle;.
The CNOT then flips qubit 1 conditioned on qubit 0, creating entanglement.

## Step 2 — Inspect the Statevector

```python
sv = qc.statevector()
print(sv)
# [0.707+0j, 0+0j, 0+0j, 0.707+0j]
```

The nonzero amplitudes at indices 0 (`|00>`) and 3 (`|11>`) confirm the
Bell state. Both have amplitude 1/&radic;2 &asymp; 0.707.

## Step 3 — Measure

```python
counts = qc.measure_all(shots=1000)
print(counts)
# {'00': ~500, '11': ~500}
```

You'll only ever see `00` or `11` — never `01` or `10`. That's entanglement:
measuring one qubit instantly determines the other.

## Step 4 — Visualize

```python
from quantumium.utils.visualization import circuit_to_ascii

print(circuit_to_ascii(qc))
# q0: ──H──●──
#          │
# q1: ─────⊕──
```

## Step 5 — Add Noise

Real hardware introduces errors. Simulate a noisy Bell state:

```python
from quantumium.core.noise import NoiseModel, depolarizing

noise = NoiseModel()
noise.add_gate_noise("H", depolarizing(0.02))
noise.add_gate_noise("CNOT", depolarizing(0.05))

rho = qc.density_matrix(noise_model=noise)
noisy_counts = qc.noisy_measure(noise, shots=1000)
print(noisy_counts)
# {'00': ~470, '01': ~15, '10': ~15, '11': ~500}
```

Small counts for `01` and `10` appear due to depolarizing noise.

## Step 6 — Export to OpenQASM

```python
from quantumium.transpiler.qasm import to_qasm

print(to_qasm(qc))
# OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[2];
# h q[0];
# cx q[0],q[1];
```

## Next Steps

- [VQE Tutorial](vqe.md) — variational quantum chemistry
- [Q+ Tutorial](qplus.md) — factor integers with fewer qubits
- [API Reference](../api/core.md) — full QuantumCircuit documentation
