# Tutorial: VQE for Molecular Ground States

Use the Variational Quantum Eigensolver (VQE) to find the ground state
energy of the hydrogen molecule (H2).

## Background

VQE is a hybrid quantum-classical algorithm: a parameterized quantum
circuit prepares a trial state, a classical optimizer tunes the
parameters to minimize the energy expectation value. It's one of the
most practical near-term quantum algorithms.

## Step 1 — Define the Hamiltonian

The H2 molecule's electronic Hamiltonian in the STO-3G basis,
mapped to 2 qubits via Jordan-Wigner transformation:

```python
from quantumium.applications.chemistry import _build_hamiltonian

H2_coeffs = {
    "II": -0.8105,
    "ZI": +0.1721,
    "IZ": -0.2257,
    "ZZ": +0.1709,
    "XX": +0.0453,
    "YY": +0.0453,
}
hamiltonian = _build_hamiltonian(H2_coeffs, num_qubits=2)
```

## Step 2 — Run VQE

```python
from quantumium.algorithms.vqe import vqe

result = vqe(
    num_qubits=2,
    hamiltonian=hamiltonian,
    ansatz_depth=2,
    max_iter=100,
    num_starts=3,
)

print(f"Ground state energy: {result.optimal_energy:.4f} Ha")
print(f"Optimal parameters: {result.optimal_params}")
# Ground state energy: ~-1.137 Ha (exact: -1.1373 Ha)
```

The `num_starts` parameter runs multiple random initializations to
avoid local minima.

## Step 3 — Inspect the Ansatz

VQE uses a hardware-efficient ansatz — layers of single-qubit rotations
followed by entangling CNOT gates:

```python
from quantumium.core.parameter import hardware_efficient_template

pqc, params = hardware_efficient_template(num_qubits=2, depth=2)
print(f"Ansatz has {len(params)} parameters")
# Ansatz has 8 parameters
```

## Step 4 — Noisy VQE

On real hardware, noise affects the energy estimate. You can simulate
this to understand the impact:

```python
from quantumium.core.noise import NoiseModel, depolarizing

noise = NoiseModel(default_single_qubit_noise=depolarizing(0.01))

result_noisy = vqe(
    num_qubits=2,
    hamiltonian=hamiltonian,
    ansatz_depth=2,
    max_iter=100,
)
```

## Performance

From our [benchmark suite](../benchmarks.md):

| Benchmark | Time |
|-----------|------|
| VQE-H2 (2q, depth=1, 30 iter) | ~96 ms |

## Next Steps

- [QAOA for MaxCut](../getting-started.md#7-using-algorithms) — combinatorial optimization
- [Q+ Tutorial](qplus.md) — beyond-qubit-count factoring
- [Algorithms API](../api/algorithms.md) — all algorithm implementations
