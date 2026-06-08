"""Entanglement generation and swapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import QuantumNetwork


@dataclass
class EntangledPair:
    node_a: str
    node_b: str
    fidelity: float
    density_matrix: np.ndarray | None = None

    def __post_init__(self):  # type: ignore[no-untyped-def]
        if self.density_matrix is None:
            self.density_matrix = _werner_state(self.fidelity)

    @property
    def is_entangled(self) -> bool:
        return self.fidelity > 0.5


def _werner_state(fidelity: float) -> np.ndarray:
    phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    proj = np.outer(phi_plus, phi_plus.conj())
    identity4 = np.eye(4, dtype=complex) / 4
    return fidelity * proj + (1 - fidelity) * identity4


def _fidelity_from_dm(rho: np.ndarray) -> float:
    phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return float(np.real(phi_plus.conj() @ rho @ phi_plus))


def generate_entanglement(network: QuantumNetwork, node_a: str, node_b: str) -> EntangledPair:
    channel = network.get_channel(node_a, node_b)
    fidelity = channel.channel_fidelity
    return EntangledPair(node_a, node_b, fidelity)


def entanglement_swap(pair1: EntangledPair, pair2: EntangledPair) -> EntangledPair:
    if pair1.node_b != pair2.node_a:
        if pair1.node_a == pair2.node_a:
            pair1 = EntangledPair(pair1.node_b, pair1.node_a, pair1.fidelity, pair1.density_matrix)
        elif pair1.node_b == pair2.node_b:
            pair2 = EntangledPair(pair2.node_b, pair2.node_a, pair2.fidelity, pair2.density_matrix)
        elif pair1.node_a == pair2.node_b:
            pair1 = EntangledPair(pair1.node_b, pair1.node_a, pair1.fidelity, pair1.density_matrix)
            pair2 = EntangledPair(pair2.node_b, pair2.node_a, pair2.fidelity, pair2.density_matrix)

    if pair1.node_b != pair2.node_a:
        raise ValueError(
            f"Pairs must share a middle node for swapping. "
            f"Got {pair1.node_a}-{pair1.node_b} and {pair2.node_a}-{pair2.node_b}"
        )

    f1 = pair1.fidelity
    f2 = pair2.fidelity
    f_swap = f1 * f2 + (1 - f1) * (1 - f2) / 3
    return EntangledPair(pair1.node_a, pair2.node_b, f_swap)
