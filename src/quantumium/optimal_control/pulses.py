"""Pulse sequence definitions for quantum control.

Provides standard pulse shapes (constant, Gaussian, DRAG) and a
container for multi-channel time-discretized pulse sequences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PulseSequence:
    """Time-discretized pulse sequence for multiple control channels.

    Attributes:
        amplitudes: Array of shape (num_steps, num_controls) with
            the control amplitude for each time step and channel.
        dt: Time step duration.
        num_steps: Number of time steps.
        num_controls: Number of control channels.
    """
    amplitudes: NDArray[np.floating]
    dt: float

    @property
    def num_steps(self) -> int:
        return self.amplitudes.shape[0]

    @property
    def num_controls(self) -> int:
        return self.amplitudes.shape[1]

    @property
    def total_time(self) -> float:
        return self.num_steps * self.dt

    def copy(self) -> PulseSequence:
        return PulseSequence(
            amplitudes=self.amplitudes.copy(),
            dt=self.dt,
        )


def constant_pulse(
    amplitude: float,
    num_steps: int,
    dt: float = 0.1,
    num_controls: int = 1,
) -> PulseSequence:
    """Create a constant pulse on the first control channel.

    Args:
        amplitude: Constant amplitude value.
        num_steps: Number of time steps.
        dt: Time step duration.
        num_controls: Number of control channels.
    """
    amps = np.zeros((num_steps, num_controls))
    amps[:, 0] = amplitude
    return PulseSequence(amplitudes=amps, dt=dt)


def gaussian_pulse(
    amplitude: float,
    sigma: float,
    num_steps: int,
    dt: float = 0.1,
    num_controls: int = 1,
    channel: int = 0,
) -> PulseSequence:
    """Create a Gaussian-shaped pulse.

    u(t) = amplitude * exp(-(t - t_center)² / (2σ²))

    Args:
        amplitude: Peak amplitude.
        sigma: Gaussian width (in time units).
        num_steps: Number of time steps.
        dt: Time step duration.
        num_controls: Number of control channels.
        channel: Which control channel to put the pulse on.
    """
    amps = np.zeros((num_steps, num_controls))
    t_center = (num_steps - 1) * dt / 2
    for i in range(num_steps):
        t = i * dt
        amps[i, channel] = amplitude * math.exp(-(t - t_center)**2 / (2 * sigma**2))
    return PulseSequence(amplitudes=amps, dt=dt)


def drag_pulse(
    amplitude: float,
    sigma: float,
    beta: float,
    num_steps: int,
    dt: float = 0.1,
) -> PulseSequence:
    """Create a DRAG (Derivative Removal by Adiabatic Gate) pulse.

    X channel: Gaussian envelope
    Y channel: -β * d(Gaussian)/dt (derivative correction)

    DRAG suppresses leakage to non-computational states in transmons.

    Args:
        amplitude: Peak amplitude of the Gaussian component.
        sigma: Gaussian width.
        beta: DRAG parameter (anharmonicity correction scale).
        num_steps: Number of time steps.
        dt: Time step duration.

    Returns:
        PulseSequence with 2 control channels (X, Y).
    """
    amps = np.zeros((num_steps, 2))
    t_center = (num_steps - 1) * dt / 2

    for i in range(num_steps):
        t = i * dt
        gauss = amplitude * math.exp(-(t - t_center)**2 / (2 * sigma**2))
        d_gauss = gauss * (-(t - t_center) / sigma**2)
        amps[i, 0] = gauss
        amps[i, 1] = -beta * d_gauss

    return PulseSequence(amplitudes=amps, dt=dt)
