"""Quantumium — Quantum computing SDK for Python.

Provides quantum circuit construction, simulation, algorithms, hardware
backend connectors, and domain applications for NISQ and beyond.

Subpackages
-----------
core
    Qubits, gates, circuits, noise models, and parameterized circuits.
algorithms
    Grover, VQE, QAOA, Shor, HHL, QPE, QFT, and more.
backends
    Unified interface for IBM Quantum, Google, and Amazon Braket hardware.
applications
    Quantum chemistry, optimization, and random number generation.
circuits
    Pre-built circuit templates (Bell, GHZ, teleportation).
transpiler
    Circuit optimization, OpenQASM 2.0, and JSON serialization.
error_correction
    Bit-flip, phase-flip, and Shor 9-qubit codes.
mitigation
    ZNE, measurement error mitigation, and Pauli twirling.
qml
    Quantum machine learning (encoding, classifiers, kernels).
crypto
    BB84 quantum key distribution and cryptographic QRNG.
finance
    Option pricing, risk analysis, and portfolio optimization.
qsp
    Quantum signal processing and QSVT.
quantum_info
    Quantum channels, distance measures, and process tomography.
quantum_walks
    Discrete and continuous quantum walks on graphs.
tensor_networks
    Matrix product state (MPS) simulation.
stabilizer
    Clifford tableau simulation.
open_systems
    Lindblad master equation for dissipative dynamics.
optimal_control
    GRAPE pulse optimization.
adiabatic
    Adiabatic quantum computing.
tomography
    State and process tomography.
benchmarks
    Randomized benchmarking, quantum volume, and fidelity estimation.
utils
    Entanglement measures and ASCII circuit visualization.
"""

__version__ = "0.1.0"
