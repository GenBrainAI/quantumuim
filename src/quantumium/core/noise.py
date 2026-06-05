"""Quantum noise models for realistic simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class NoiseChannel:
    """A quantum noise channel defined by Kraus operators."""

    name: str
    kraus_operators: list[NDArray[np.complexfloating]]

    def __post_init__(self) -> None:
        total = sum(k.conj().T @ k for k in self.kraus_operators)
        if not np.allclose(total, np.eye(total.shape[0]), atol=1e-10):
            raise ValueError(f"Kraus operators for '{self.name}' do not satisfy completeness")

    def apply(self, density_matrix: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        return sum(k @ density_matrix @ k.conj().T for k in self.kraus_operators)  # type: ignore[return-value]


def bit_flip(p: float) -> NoiseChannel:
    """Bit flip channel: flips |0> <-> |1> with probability p."""
    if not 0 <= p <= 1:
        raise ValueError(f"Probability must be in [0, 1], got {p}")
    k0 = np.sqrt(1 - p) * np.eye(2, dtype=complex)
    k1 = np.sqrt(p) * np.array([[0, 1], [1, 0]], dtype=complex)
    return NoiseChannel("bit_flip", [k0, k1])


def phase_flip(p: float) -> NoiseChannel:
    """Phase flip channel: applies Z with probability p."""
    if not 0 <= p <= 1:
        raise ValueError(f"Probability must be in [0, 1], got {p}")
    k0 = np.sqrt(1 - p) * np.eye(2, dtype=complex)
    k1 = np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=complex)
    return NoiseChannel("phase_flip", [k0, k1])


def depolarizing(p: float) -> NoiseChannel:
    """Depolarizing channel: replaces state with maximally mixed with probability p."""
    if not 0 <= p <= 1:
        raise ValueError(f"Probability must be in [0, 1], got {p}")
    k0 = np.sqrt(1 - 3 * p / 4) * np.eye(2, dtype=complex)
    k1 = np.sqrt(p / 4) * np.array([[0, 1], [1, 0]], dtype=complex)
    k2 = np.sqrt(p / 4) * np.array([[0, -1j], [1j, 0]], dtype=complex)
    k3 = np.sqrt(p / 4) * np.array([[1, 0], [0, -1]], dtype=complex)
    return NoiseChannel("depolarizing", [k0, k1, k2, k3])


def amplitude_damping(gamma: float) -> NoiseChannel:
    """Amplitude damping channel: models energy relaxation (T1 decay)."""
    if not 0 <= gamma <= 1:
        raise ValueError(f"Gamma must be in [0, 1], got {gamma}")
    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return NoiseChannel("amplitude_damping", [k0, k1])


@dataclass
class NoiseModel:
    """Noise model that maps gate names to noise channels."""

    gate_noise: dict[str, NoiseChannel] = field(default_factory=dict)
    default_single_qubit_noise: NoiseChannel | None = None
    default_two_qubit_noise: NoiseChannel | None = None

    def add_gate_noise(self, gate_name: str, channel: NoiseChannel) -> None:
        self.gate_noise[gate_name] = channel

    def get_noise(self, gate_name: str, num_qubits: int) -> NoiseChannel | None:
        if gate_name in self.gate_noise:
            return self.gate_noise[gate_name]
        if num_qubits == 1 and self.default_single_qubit_noise:
            return self.default_single_qubit_noise
        if num_qubits == 2 and self.default_two_qubit_noise:
            return self.default_two_qubit_noise
        return None
