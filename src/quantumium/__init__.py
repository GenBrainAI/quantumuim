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

__version__ = "0.3.0"

from .core import QuantumCircuit, Gate, Qubit, Instruction
from .core import H, X, Y, Z, S, T, I, CNOT, SWAP, CZ, CCX
from .core import Rx, Ry, Rz
from .core import NoiseChannel, NoiseModel, bit_flip, phase_flip, depolarizing, amplitude_damping
from .core import Parameter, ParameterVector, ParameterizedCircuit

__all__ = [
    "QuantumCircuit",
    "Gate",
    "Qubit",
    "Instruction",
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "T",
    "I",
    "CNOT",
    "SWAP",
    "CZ",
    "CCX",
    "Rx",
    "Ry",
    "Rz",
    "NoiseChannel",
    "NoiseModel",
    "bit_flip",
    "phase_flip",
    "depolarizing",
    "amplitude_damping",
    "Parameter",
    "ParameterVector",
    "ParameterizedCircuit",
]
