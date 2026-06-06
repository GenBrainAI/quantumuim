"""Noise filter functions and fidelity estimation for DD sequences."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from quantumium.dd.sequences import DDSequence


DEFAULT_DURATIONS: dict[str, int] = {
    "I": 1,
    "X": 1,
    "Y": 1,
    "Z": 0,
    "H": 1,
    "S": 1,
    "T": 1,
    "CNOT": 2,
    "CZ": 2,
    "SWAP": 6,
}


def filter_function(
    sequence: DDSequence,
    frequencies: NDArray[np.floating],
    total_time: float,
) -> NDArray[np.floating]:
    """Compute the noise filter function |F(omega)|^2 for a DD sequence.

    The filter function determines which noise frequencies are suppressed.
    Lower values = better suppression at that frequency.

    For a sequence with pulses at times t_1, ..., t_N in [0, T]:
        y(t) = (-1)^{number of pulses before t}
        F(omega) = integral_0^T y(t) e^{i*omega*t} dt
        |F(omega)|^2 is the filter function

    Args:
        sequence: DD pulse sequence.
        frequencies: Angular frequencies omega (rad/s).
        total_time: Total evolution time T.

    Returns:
        |F(omega)|^2 array of same shape as frequencies.
    """
    pulse_times_norm = sequence.pulse_times()
    pulse_times = [t * total_time for t in pulse_times_norm]

    omega = np.asarray(frequencies, dtype=float)

    safe = np.abs(omega) > 1e-15

    boundaries = [0.0] + pulse_times + [total_time]
    n_intervals = len(boundaries) - 1

    F = np.zeros_like(omega, dtype=complex)
    for idx in range(n_intervals):
        sign = (-1.0) ** idx
        t_start = boundaries[idx]
        t_end = boundaries[idx + 1]

        w_safe = omega[safe]
        contrib = np.zeros_like(omega, dtype=complex)
        contrib[safe] = sign * (
            np.exp(1j * w_safe * t_end) - np.exp(1j * w_safe * t_start)
        ) / (1j * w_safe)
        contrib[~safe] = sign * (t_end - t_start)
        F += contrib

    result: NDArray[np.floating[Any]] = np.abs(F) ** 2
    return result


def fid_filter(
    frequencies: NDArray[np.floating], total_time: float
) -> NDArray[np.floating]:
    """Filter function for free induction decay (no pulses)."""
    no_dd = DDSequence(name="fid", pulses=[], spacings=[1.0])
    return filter_function(no_dd, frequencies, total_time)


def dd_fidelity(
    sequence: DDSequence,
    noise_spectrum: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
    total_time: float,
    n_points: int = 10000,
    omega_max: float = 100.0,
) -> float:
    """Estimate fidelity after DD sequence under given noise spectrum.

    Uses the overlap integral:
        chi(T) = exp(-1/pi * integral S(omega) * |F(omega)|^2 / omega^2 d_omega)

    Args:
        sequence: DD pulse sequence (or None for FID).
        noise_spectrum: S(omega) function, takes array of frequencies.
        total_time: Total evolution time.
        n_points: Integration resolution.
        omega_max: Upper frequency cutoff.

    Returns:
        Estimated fidelity in [0, 1].
    """
    omega = np.linspace(1e-6, omega_max, n_points)
    d_omega = omega[1] - omega[0]

    F2 = filter_function(sequence, omega, total_time)
    S: NDArray[np.floating[Any]] = np.asarray(noise_spectrum(omega), dtype=np.float64)

    integrand: NDArray[np.floating[Any]] = S * F2 / (omega**2)
    chi = np.exp(-(1.0 / np.pi) * np.trapezoid(integrand, dx=d_omega))

    return float(np.clip(chi, 0.0, 1.0))
