# Quantumium SDK Examples

Runnable examples demonstrating the Quantumium quantum computing SDK.

## Getting Started

```bash
# Run any example from the repo root:
python examples/bell_state.py
```

## Core Examples

| Example | Description |
|---------|-------------|
| [bell_state.py](bell_state.py) | Create and measure a Bell state |
| [ghz_entanglement.py](ghz_entanglement.py) | GHZ entanglement across multiple qubits |
| [noisy_simulation.py](noisy_simulation.py) | Simulate circuits with noise models |
| [quantum_channels.py](quantum_channels.py) | Quantum noise channel operations |
| [routing_demo.py](routing_demo.py) | Qubit routing and mapping to hardware |

## Quantum Algorithms

| Example | Description |
|---------|-------------|
| [algorithms/qft_example.py](algorithms/qft_example.py) | Quantum Fourier Transform |
| [algorithms/qpe_example.py](algorithms/qpe_example.py) | Quantum Phase Estimation |
| [algorithms/vqe_h2_example.py](algorithms/vqe_h2_example.py) | VQE for H2 molecular energy |
| [algorithms/qaoa_maxcut_example.py](algorithms/qaoa_maxcut_example.py) | QAOA for MaxCut optimization |
| [algorithms/ising_simulation.py](algorithms/ising_simulation.py) | Ising model simulation |
| [shor_factoring.py](shor_factoring.py) | Shor's factoring algorithm |
| [grover_search.py](grover_search.py) | Grover's search algorithm |
| [deutsch_jozsa_demo.py](deutsch_jozsa_demo.py) | Deutsch-Jozsa algorithm |
| [hhl_linear_solver.py](hhl_linear_solver.py) | HHL linear systems solver |

## Error Correction & Mitigation

| Example | Description |
|---------|-------------|
| [error_correction_demo.py](error_correction_demo.py) | Quantum error correction codes |
| [error_mitigation_demo.py](error_mitigation_demo.py) | Error mitigation techniques |
| [stabilizer_demo.py](stabilizer_demo.py) | Stabilizer formalism |
| [surface_code_demo.py](surface_code_demo.py) | Surface code error correction |
| [dd_demo.py](dd_demo.py) | Dynamical decoupling |

## Advanced Topics

| Example | Description |
|---------|-------------|
| [tomography_demo.py](tomography_demo.py) | Quantum state and process tomography |
| [classical_shadows_demo.py](classical_shadows_demo.py) | Classical shadow estimation |
| [benchmarking_demo.py](benchmarking_demo.py) | Quantum benchmarking protocols |
| [hardware_validation_demo.py](hardware_validation_demo.py) | Hardware validation workflows |
| [resource_estimation_demo.py](resource_estimation_demo.py) | Quantum resource estimation |
| [tensor_network_demo.py](tensor_network_demo.py) | Tensor network simulation |
| [zx_calculus_demo.py](zx_calculus_demo.py) | ZX-calculus optimization |
| [qsp_qsvt_demo.py](qsp_qsvt_demo.py) | Quantum signal processing / QSVT |
| [open_systems_demo.py](open_systems_demo.py) | Open quantum systems |
| [optimal_control_demo.py](optimal_control_demo.py) | Quantum optimal control |
| [adiabatic_demo.py](adiabatic_demo.py) | Adiabatic quantum computing |

## Applications

| Example | Description |
|---------|-------------|
| [qml_demo.py](qml_demo.py) | Quantum machine learning |
| [quantum_finance_demo.py](quantum_finance_demo.py) | Quantum finance algorithms |
| [quantum_walks_demo.py](quantum_walks_demo.py) | Quantum random walks |
| [quantum_counting_demo.py](quantum_counting_demo.py) | Quantum counting algorithm |
| [bb84_key_distribution.py](bb84_key_distribution.py) | BB84 quantum key distribution |
| [networking_demo.py](networking_demo.py) | Quantum networking protocols |
| [fermionic_demo.py](fermionic_demo.py) | Fermionic simulation |

## Q+ Algorithm (Beyond-Qubit-Count Computing)

Q+ enables solving quantum problems larger than available hardware. Examples are in `src/quantumium/qplus/examples/`.

| Example | Description |
|---------|-------------|
| [getting_started.py](../src/quantumium/qplus/examples/getting_started.py) | Tutorial: semiclassical QFT compression |
| [multi_qpu_example.py](../src/quantumium/qplus/examples/multi_qpu_example.py) | Multi-QPU distribution with all strategies |
| [circuit_cutting_example.py](../src/quantumium/qplus/examples/circuit_cutting_example.py) | Circuit splitting and recombination |
| [qplus_demo.py](../src/quantumium/qplus/examples/qplus_demo.py) | All Q+ layers demonstrated |
| [shor_benchmark.py](../src/quantumium/qplus/examples/shor_benchmark.py) | Factor semiprimes with 54-60% fewer qubits |
