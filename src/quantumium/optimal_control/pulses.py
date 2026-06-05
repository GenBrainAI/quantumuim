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
        """Return the number of time steps in the pulse sequence.

        Returns:
            int: Number of time steps.
        """
        return self.amplitudes.shape[0]

    @property
    def num_controls(self) -> int:
        """Return the number of control channels.

        Returns:
            int: Number of control channels.
        """
        return self.amplitudes.shape[1]

    @property
    def total_time(self) -> float:
        """Return the total duration of the pulse sequence.

        Returns:
            float: Total time (num_steps * dt).
        """
        return self.num_steps * self.dt

    def copy(self) -> PulseSequence:
        """Create a deep copy of this pulse sequence.

        Returns:
            PulseSequence: A new PulseSequence with independently copied
                amplitude data.
        """
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
    """Create a constant-amplitude pulse on the first control channel.

    All other control channels are set to zero amplitude.

    Args:
        amplitude (float): Constant amplitude value for channel 0.
        num_steps (int): Number of time steps in the pulse.
        dt (float): Time step duration. Defaults to 0.1.
        num_controls (int): Total number of control channels.
            Defaults to 1.

    Returns:
        PulseSequence: A pulse sequence with constant amplitude on
            channel 0 and zero on all other channels.
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
    """Create a Gaussian-shaped pulse centered in the time window.

    The pulse envelope is u(t) = amplitude * exp(-(t - t_center)^2 / (2*sigma^2)),
    where t_center is the midpoint of the pulse duration.

    Args:
        amplitude (float): Peak amplitude at the center of the Gaussian.
        sigma (float): Standard deviation (width) of the Gaussian in
            time units.
        num_steps (int): Number of time steps in the pulse.
        dt (float): Time step duration. Defaults to 0.1.
        num_controls (int): Total number of control channels.
            Defaults to 1.
        channel (int): Index of the control channel for the Gaussian
            pulse. Defaults to 0.

    Returns:
        PulseSequence: A pulse sequence with the Gaussian envelope on
            the specified channel and zero on all others.
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
