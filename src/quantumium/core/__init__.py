"""Core quantum computing primitives — qubits, gates, and measurements."""

from .gate import (
    Gate, I, X, Y, Z, H, S, T, CNOT, SWAP, CZ, CCX,
    Rx, Ry, Rz, CRx, CRy, CRz, Rxx, Rzz,
)
from .circuit import QuantumCircuit, Instruction
from .qubit import Qubit
from .noise import (
    NoiseChannel, NoiseModel,
    bit_flip, phase_flip, depolarizing, amplitude_damping,
)
from .parameter import Parameter, ParameterVector, ParameterizedCircuit, hardware_efficient_template

__all__ = [
    "Gate", "I", "X", "Y", "Z", "H", "S", "T", "CNOT", "SWAP", "CZ", "CCX",
    "Rx", "Ry", "Rz", "CRx", "CRy", "CRz", "Rxx", "Rzz",
    "QuantumCircuit", "Instruction",
    "Qubit",
    "NoiseChannel", "NoiseModel",
    "bit_flip", "phase_flip", "depolarizing", "amplitude_damping",
    "Parameter", "ParameterVector", "ParameterizedCircuit", "hardware_efficient_template",
]
