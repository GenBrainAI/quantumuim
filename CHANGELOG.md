# Changelog

All notable changes to the Quantumium SDK are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-06-08

### Q+ Algorithm — Beyond-Qubit-Count Quantum Framework
- `ShorQPlus`: Factors semiprimes using 54–60% fewer qubits than standard Shor
- Three-layer architecture: algorithmic compression, qubit recycling, circuit cutting
- `compress`: Semiclassical QFT, modular arithmetic optimization (Beauregard/Häner)
- `recycle`: Mid-circuit measurement and qubit reuse with feed-forward control
- `cut`: Circuit cutting with Pauli decomposition and classical recombination
- `distribute`: Multi-QPU orchestration (classical, LOCC, teleport strategies)
- Benchmark: N=143 factored in 23 ms using only 10 qubits (vs 25 standard)
- 70+ tests, demo scripts, Shor benchmark suite

### Statevector Simulator Optimization
- Replaced O(4^n) full matrix multiplication with O(2^n) tensor contraction
- `np.tensordot` gate application: reshape state as rank-n tensor, contract on target axes
- Density matrix optimization: two O(4^n × 4^k) contractions replace O(8^n) multiply
- Practical simulation range extended from ~8 qubits to 20+
- Speedups: 75x (5q) → 6,919x (8q) → 12,399x (Random-8x3)
- Density matrix speedups: 27x (GHZ-7-noisy) → 62x (Random-7x3-noisy)
- Fixed qubit-ordering bug in `_apply_single_qubit_channel` (kron MSB vs LSB)
- `_expand_gate()` preserved for backward compatibility (6 external callers)
- 44 benchmarks across 5 categories (statevector, density matrix, MPS, VQE/QAOA, Q+ Shor)

### MkDocs Documentation Site
- Material theme with dark/light toggle, deep purple/cyan palette
- Auto-generated API reference via mkdocstrings for all modules
- Tutorials: Bell state, VQE for H2, Q+ factoring
- Q+ API reference, transpiler API reference
- Example gallery, benchmarks page with real performance data
- `mkdocs build --strict` passes with zero warnings

### Qubit Routing & Mapping (Phase 31)
- `CouplingMap`: physical qubit connectivity with BFS distance matrix
- Factory constructors for linear, grid, and heavy-hex topologies
- `SABRERouter`: forward+backward passes with extended-set lookahead
- `route(circuit, coupling_map)` one-liner API
- 44 tests

### Classical Shadow Tomography (Phase 32)
- `ClassicalShadow`: random Pauli measurement protocol
- Pauli expectation estimation, fidelity estimation, purity estimation
- Median-of-means robust statistics, shadow norm calculation
- `required_snapshots()` sample complexity with O(log M) scaling
- 54 tests

### Fermionic Simulation (Phase 33)
- `FermionOperator`: ladder operator algebra with complex coefficients
- `QubitOperator`: Pauli string algebra with full matrix construction
- `jordan_wigner()`: O(n) Z-string anti-commutation enforcement
- `bravyi_kitaev()`: O(log n) Pauli weight via Fenwick tree
- `fermi_hubbard()`, `molecular_hamiltonian()` builder functions
- JW and BK produce identical eigenspectra — verified across all test Hamiltonians
- 69 tests

### Dynamical Decoupling (Phase 34)
- `DDSequence`: spin echo, CPMG, XY-4, XY-8, UDD sequences
- `DDScheduler`: automatic DD pulse insertion into circuit idle windows
- Filter function computation for noise spectral analysis
- XY-8 achieves >99% fidelity under 1/f noise
- Configurable gate durations and minimum idle window thresholds
- 50 tests

### Resource Estimation (Phase 35)
- `CircuitAnalyzer`: T-gate counting, T-depth, rotation decomposition costs
- `ResourceEstimator`: end-to-end circuit → physical qubit/runtime pipeline
- `MagicStateFactory`: 15-to-1 distillation cost model with multi-round support
- Surface code distance selection from target error rate
- Toffoli = 7T verified, rotation synthesis O(log(1/ε)) confirmed
- 52 tests

### Release Polish
- LICENSE: MIT with copyright 2024-2026 Quantumium Contributors
- README: PyPI-ready with badges, Q+ showcase, performance highlights
- pyproject.toml: verified classifiers, URLs, keywords, optional dependencies
- Clean wheel/sdist generation via `python -m build`

### Stats
- 30 packages, 1,709 tests, 37 examples, 3 notebooks, 28+ KB articles
- 44 performance benchmarks, MkDocs documentation site
- v0.3.0 milestone: Phases 31–35 complete + Q+ algorithm + simulator optimization

---

## [0.2.0] - 2026

### Surface Code Error Correction (Phase 26)
- Rotated planar surface code for distances d=3, 5, 7
- Self-contained MWPM decoder using BFS-based graph distances
- Syndrome extraction, logical X/Z operators, Monte Carlo threshold estimation
- 56 tests

### ZX-Calculus Optimization Engine (Phase 27)
- `ZXDiagram` with Z/X spiders and Hadamard edges
- Circuit-to-ZX and ZX-to-circuit round-trip conversion
- Spider fusion, identity removal, color change, Hadamard cancellation
- T-count reduction via phase merging through CNOT controls
- `zx_optimize(circuit)` one-liner API
- 58 tests

### Quantum Networking Simulation (Phase 28)
- `QuantumNetwork` graph topology (star, chain, mesh)
- Entanglement generation, swapping, BBPSSW purification
- Quantum teleportation with depolarizing noise
- `distribute_entanglement(network, source, target)` one-liner API
- 50 tests

### Real Hardware Validation (Phase 29)
- Backend-specific tests: Qiskit, Cirq, Braket (graceful skip when absent)
- Cross-backend statistical validation: chi-squared, Hellinger fidelity
- Noise model calibration: depolarizing degradation monotonicity
- 96 tests (70 local, 26 skip for absent vendor SDKs)

### Documentation Site & Community (Phase 30)
- MkDocs Material theme documentation site
- Auto-generated API reference via mkdocstrings
- Example gallery, GitHub contribution templates
- `BenchmarkSuite` for performance tracking

### Stats
- 28 packages, 1,095 tests, 33 examples, 3 notebooks, 23 KB articles
- v0.2.0 milestone: Phases 26–30 complete

---

## [0.1.0] - 2025

Initial release of the Quantumium quantum computing SDK.

### Core Framework
- `QuantumCircuit` with statevector and density matrix simulation
- Full gate library: I, X, Y, Z, H, S, T, CNOT, CZ, SWAP, CCX, Rx, Ry, Rz, CRx, CRy, CRz, Rxx, Rzz
- `controlled()` factory for arbitrary controlled gates
- Noise models: bit flip, phase flip, depolarizing, amplitude damping
- Parameterized circuits with symbolic `Parameter` objects

### Algorithms
- Grover's search, Deutsch-Jozsa, Bernstein-Vazirani, Simon's
- QFT, QPE, VQE (Adam optimizer), QAOA (MaxCut)
- Hamiltonian simulation (Trotter-Suzuki), Shor's factoring, HHL

### Hardware Backends
- `LocalSimulatorBackend` — built-in statevector simulator
- `QiskitBackend` — IBM Quantum (Aer + hardware)
- `CirqBackend` — Google Quantum AI
- `BraketBackend` — Amazon Braket (IonQ, Rigetti, OQC, QuEra, SV1/DM1/TN1)

### Domain Applications
- Quantum chemistry: H2, LiH, HeH+ molecular simulation
- Finance: option pricing, VaR/CVaR, portfolio optimization
- Optimization: MaxCut, TSP via QAOA
- Quantum machine learning: amplitude/angle/basis encoding, VQC, quantum kernels
- Cryptography: BB84 QKD, cryptographic QRNG

### Advanced Simulation
- Tensor networks (MPS), stabilizer tableaux, Lindblad master equation
- Adiabatic computing, quantum walks, QSP/QSVT, GRAPE optimal control

### Transpiler & Error Handling
- Optimization passes: inverse cancellation, gate merging, CNOT cancellation
- OpenQASM 2.0 import/export, JSON serialization
- Error correction: bit-flip, phase-flip, Shor 9-qubit codes
- Error mitigation: ZNE, measurement error mitigation, Pauli twirling
- State tomography: linear inversion and MLE

### Infrastructure
- PyPI-ready `pyproject.toml` with optional dependency groups
- GitHub Actions CI workflow
- 877+ tests, 24 examples
