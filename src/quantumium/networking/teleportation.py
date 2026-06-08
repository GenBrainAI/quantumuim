"""Quantum teleportation over noisy channels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .entanglement import EntangledPair


@dataclass
class TeleportationResult:
    input_state: np.ndarray
    output_state: np.ndarray
    fidelity: float
    channel_fidelity: float


def teleport(state: np.ndarray | list, pair: EntangledPair) -> TeleportationResult:  # type: ignore[type-arg]
    psi = np.asarray(state, dtype=complex)
    if psi.shape == (2,):
        psi = psi / np.linalg.norm(psi)
        rho_in = np.outer(psi, psi.conj())
    elif psi.shape == (2, 2):
        rho_in = psi
    else:
        raise ValueError("State must be a 2-element vector or 2x2 density matrix")

    f_channel = pair.fidelity
    rho_out = _depolarizing_teleport(rho_in, f_channel)
    tel_fidelity = float(np.real(np.trace(rho_in @ rho_out)))

    return TeleportationResult(
        input_state=rho_in,
        output_state=rho_out,
        fidelity=tel_fidelity,
        channel_fidelity=f_channel,
    )


def teleportation_fidelity(channel_fidelity: float) -> float:
    return (2 * channel_fidelity + 1) / 3


def _depolarizing_teleport(rho_in: np.ndarray, f_channel: float) -> np.ndarray:
    p = (4 * f_channel - 1) / 3
    p = max(0.0, min(1.0, p))
    identity2 = np.eye(2, dtype=complex) / 2
    return p * rho_in + (1 - p) * identity2
