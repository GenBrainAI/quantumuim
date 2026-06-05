# Changelog

All notable changes to the Quantumium SDK are documented in this file.

## [0.1.0] - 2025

Initial release of the Quantumium quantum computing SDK.

### Core Framework
- Qubit state vector representation with measurement and superposition
- Full quantum gate library: I, X, Y, Z, H, S, T, CNOT, CZ, SWAP, CCX (Toffoli)
- Rotation gates: Rx, Ry, Rz, CRx, CRy, CRz, Rxx, Rzz
- `controlled()` factory for creating controlled versions of any single-qubit gate
- `QuantumCircuit` with statevector and density matrix simulation
- Noise models: bit flip, phase flip, depolarizing, amplitude damping channels
- Noisy measurement simulation via density matrix evolution
- Parameterized circuits with symbolic `Parameter` objects for variational workflows
- Hardware-efficient ansatz template generator

### Algorithms
- Grover's search algorithm with quadratic speedup
- Deutsch-Jozsa algorithm for constant vs balanced oracle classification
- Bernstein-Vazirani algorithm for hidden string recovery
- Simon's algorithm for hidden subgroup / period finding
- Quantum Fourier Transform (QFT) and inverse QFT
- Quantum Phase Estimation (QPE)
- Variational Quantum Eigensolver (VQE) with Adam optimizer
- Quantum Approximate Optimization Algorithm (QAOA) for MaxCut
- Hamiltonian simulation via Trotter-Suzuki decomposition
- Shor's factoring algorithm (educational implementation)
- HHL algorithm for quantum linear systems (Ax=b)
- Quantum counting and amplitude estimation

### Hardware Backends
- `LocalSimulatorBackend` — built-in statevector simulator
- `QiskitBackend` — IBM Quantum via Qiskit Runtime (Aer simulator + real hardware)
- `CirqBackend` — Google Quantum AI via Cirq
- `BraketBackend` — Amazon Braket (IonQ, Rigetti, OQC, QuEra, SV1/DM1/TN1)

### Domain Applications
- **Quantum Chemistry:** Molecular simulation (H2, LiH, HeH+), potential energy surfaces via VQE
- **Optimization:** MaxCut, TSP, portfolio optimization via QAOA
- **Quantum Random Number Generation:** bits, integers, floats, bytes, shuffle
- **Quantum Finance:** Option pricing (amplitude estimation), risk analysis (VaR/CVaR), portfolio optimization
- **Quantum Cryptography:** BB84 quantum key distribution simulation, cryptographic QRNG

### Quantum Machine Learning
- Amplitude, angle, and basis encoding of classical data
- Variational quantum classifier (VQC) with training loop
- Quantum kernel methods for SVM

### Transpiler
- Circuit optimization passes: inverse cancellation, gate merging, CNOT cancellation, commutation, SWAP decomposition
- OpenQASM 2.0 import and export with round-trip fidelity
- JSON circuit serialization (save/load)

### Error Correction
- 3-qubit bit-flip code
- 3-qubit phase-flip code
- Shor 9-qubit code

### Error Mitigation
- Zero-Noise Extrapolation (ZNE) with Richardson extrapolation
- Measurement error mitigation via calibration matrices
- Pauli twirling for noise symmetrization

### Quantum Information Theory
- Quantum channels: Kraus, Choi, superoperator representations
- Distance measures: trace distance, fidelity, diamond norm
- Process tomography via linear inversion

### Advanced Simulation
- **Tensor Networks:** Matrix Product State (MPS) simulation with configurable bond dimension
- **Stabilizer Simulation:** Clifford tableau for efficient simulation of Clifford circuits
- **Open Quantum Systems:** Lindblad master equation solver for dissipative dynamics
- **Adiabatic Computing:** Adiabatic evolution with customizable Hamiltonians

### Additional Modules
- Quantum walks (discrete and continuous) on arbitrary graphs
- Quantum signal processing (QSP) and Quantum Singular Value Transformation (QSVT)
- Quantum optimal control via GRAPE pulse optimization
- State tomography via linear inversion and maximum likelihood estimation
- Benchmarking: randomized benchmarking, quantum volume, fidelity estimation
- Circuit templates: Bell pair, GHZ state, quantum teleportation
- Utilities: partial trace, von Neumann entropy, purity, concurrence, negativity, Schmidt decomposition, ASCII circuit diagrams

### Infrastructure
- PyPI-ready packaging with `pyproject.toml`
- Optional dependency groups: `[qiskit]`, `[cirq]`, `[braket]`, `[all-backends]`, `[dev]`
- GitHub Actions CI workflow
- 877+ tests across 41 test files
- 24 runnable example scripts
- Quantum computing knowledge base with theory references
