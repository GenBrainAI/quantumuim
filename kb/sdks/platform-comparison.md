# Quantum Computing Platforms & SDKs

## Platform Comparison Matrix

| Feature | IBM Qiskit | Google Cirq | Amazon Braket | Rigetti pyQuil | IonQ |
|---------|-----------|-------------|---------------|----------------|------|
| Language | Python | Python | Python | Python | Python/API |
| Simulator | Aer (local+cloud) | Built-in | Local+managed | QVM | Cloud |
| Real Hardware | Yes (127+ qubits) | Yes (72 qubits) | Via partners | Yes (80+ qubits) | Yes (36+ qubits) |
| Free Tier | Yes (10 min/month) | Simulator only | Free sim, pay HW | Limited | Via Braket |
| Qubit Type | Superconducting | Superconducting | Multiple | Superconducting | Trapped Ion |
| Gate Set | Standard | Flexible | Standard | Quilc-compiled | Native |
| Error Mitigation | Yes (built-in) | Limited | Limited | Limited | Hardware-level |
| Community | Largest | Large | Growing | Medium | Growing |

## 1. IBM Qiskit

**Best for:** Broadest ecosystem, education, largest hardware fleet

### Architecture
- **Qiskit Terra:** Core circuits, transpilation
- **Qiskit Aer:** High-performance simulators
- **Qiskit Runtime:** Cloud execution with sessions and primitives
- **Qiskit Nature/ML/Finance/Optimization:** Domain-specific modules

### Key Features
- Transpiler with optimization levels 0-3
- Primitive-based execution (Sampler, Estimator)
- Dynamic circuits (mid-circuit measurement and classical control)
- Error suppression, mitigation, and (emerging) correction
- 1000+ qubit roadmap (Condor, Flamingo)

### Install
```bash
pip install qiskit qiskit-ibm-runtime qiskit-aer
```

## 2. Google Cirq

**Best for:** Research, custom gate sets, noise simulation

### Architecture
- Flexible circuit construction with custom gates
- Noise models closely matching Google hardware
- Integration with TensorFlow Quantum

### Key Features
- Custom device specifications and noise models
- QASM import/export
- Optimized for Google Sycamore processor
- OpenFermion integration for quantum chemistry

### Install
```bash
pip install cirq
```

## 3. Amazon Braket

**Best for:** Multi-vendor access from single SDK

### Architecture
- Unified SDK for multiple hardware backends
- Managed simulators (SV1, DM1, TN1)
- Hybrid jobs for variational algorithms

### Key Features
- Access to IonQ, Rigetti, OQC, QuEra hardware
- Pay-per-shot pricing
- Built-in noise simulation
- Hybrid classical-quantum workflows via Braket Hybrid Jobs

### Install
```bash
pip install amazon-braket-sdk
```

## 4. Rigetti pyQuil

**Best for:** Quil instruction language, parametric compilation

### Key Features
- Quil language (quantum instruction language)
- Quilc optimizing compiler
- QVM (Quantum Virtual Machine) simulator
- Active reset and mid-circuit measurement

### Install
```bash
pip install pyquil
```

## 5. IonQ (Trapped Ion)

**Best for:** High gate fidelity, all-to-all connectivity

### Key Features
- Trapped-ion architecture — long coherence times
- All-to-all qubit connectivity (no SWAP overhead)
- Accessible via Braket, Cirq, Qiskit
- IonQ Aria (25 qubits), Forte (36 qubits)

## Recommended Starting Strategy

1. **Primary:** IBM Qiskit — largest ecosystem, best docs, free hardware access
2. **Secondary:** Google Cirq — research flexibility, noise modeling
3. **Multi-vendor:** Amazon Braket — when we need to compare across hardware types
4. **Specialized:** IonQ via Braket — when we need high-fidelity trapped-ion results

## Cross-Platform Abstraction

Consider building our own thin abstraction layer that lets us write circuits once and run on any backend. Key interfaces:
- CircuitBuilder: hardware-agnostic circuit construction
- Backend: unified execution interface
- ResultParser: standardized measurement result format
