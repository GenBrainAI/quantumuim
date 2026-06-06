# Quantumium v0.3.0 — Roadmap Candidates

Strategic analysis of high-impact features for the next major release cycle.

## Selection Criteria

1. **Differentiation**: Does this set Quantumium apart from Qiskit/Cirq/PennyLane?
2. **Educational value**: Does this deepen understanding of quantum computing?
3. **Practical utility**: Would real users need this for hardware or research work?
4. **Buildability**: Can we implement this with numpy-only (no external quantum deps)?

## Tier 1 — High Impact

### Classical Shadows
- Efficient quantum state learning from randomized measurements
- Predict many observables from few measurements (exponential speedup over full tomography)
- Key papers: Huang, Kueng, Preskill (2020)
- Fits naturally alongside our `tomography` module
- **Why**: No competing SDK has a clean, educational implementation

### Dynamical Decoupling
- Pulse sequences (XY-4, CPMG, Uhrig) that suppress decoherence
- Integrates with `open_systems` and `optimal_control`
- Essential for real hardware — IBM/Google both use DD in production
- **Why**: Bridges gap between simulation and real hardware execution

### Qubit Routing & Mapping
- Map logical qubits to physical device topology (coupling map)
- SWAP insertion, SABRE algorithm, noise-aware routing
- Prerequisite for realistic hardware execution
- **Why**: Our transpiler does gate optimization but lacks topology awareness

### Fermionic Simulation
- Jordan-Wigner, Bravyi-Kitaev, parity transformations
- Map fermionic Hamiltonians to qubit operators
- Deepens our `applications.chemistry` module significantly
- **Why**: Required for serious quantum chemistry work

## Tier 2 — Medium Impact

### Advanced VQA Variants
- ADAPT-VQE (adaptive ansatz construction)
- Variational Quantum Deflation (VQD) for excited states
- Quantum Natural Gradient optimizer
- Extends existing VQE/QAOA infrastructure

### Color Codes & Toric Codes
- Additional topological error correction codes
- Extends `surface_code` module with alternative code families
- Toric code is pedagogically important (Kitaev)

### Quantum Resource Estimation
- Estimate T-gate count, qubit count, circuit depth for fault-tolerant algorithms
- Leverages our ZX-calculus T-count optimization
- Practical tool for researchers planning real experiments

### Quantum Reservoir Computing
- Classical reservoir computing with quantum kernel
- Lightweight extension of `qml` module
- Growing research area with practical near-term applications

## Tier 3 — Nice to Have

### Bloch Sphere Visualization
- 3D Bloch sphere rendering (matplotlib optional dep)
- Single-qubit state trajectory animation
- High educational value, moderate implementation effort

### QASM 3.0 Support
- OpenQASM 3.0 parsing and generation
- Classical control flow, real-time classical computation
- Future-proofing for next-gen hardware APIs

### Quantum Channels Visualization
- Process matrix, Pauli transfer matrix visualization
- Chi matrix bar charts
- Extends `quantum_info` with visual diagnostics

## Recommended v0.3.0 Scope

Based on differentiation and educational value, recommended phases:

| Phase | Feature | Size | Differentiator |
|-------|---------|------|----------------|
| 31 | Classical Shadows | M | First clean educational impl |
| 32 | Dynamical Decoupling | S | Hardware-readiness |
| 33 | Qubit Routing & Mapping | L | Completes transpiler story |
| 34 | Fermionic Simulation | M | Serious chemistry support |
| 35 | Resource Estimation | S | Leverages ZX-calculus |

This scope would bring Quantumium to ~33 packages and position it as the most educationally complete quantum SDK available.
