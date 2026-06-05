"""Adiabatic evolution — time-dependent Schrödinger equation solver.

Evolves a quantum state under a time-dependent Hamiltonian
H(s(t)) = (1-s)H_init + s·H_final using the specified schedule s(t).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


@dataclass
class AdiabaticResult:
    """Result of adiabatic quantum evolution.

    Attributes:
        final_state: Final quantum state vector.
        ground_state_prob: Probability of being in the final Hamiltonian's
            ground state.
        schedule_values: Values of s(t) at each time step.
        energy_history: Expectation value of H(s) at each time step.
        gap_history: Spectral gap at each schedule point.
        optimal_bitstring: Computational basis state with highest probability.
    """
    final_state: NDArray[np.complexfloating]
    ground_state_prob: float
    schedule_values: NDArray[np.floating]
    energy_history: NDArray[np.floating]
    gap_history: NDArray[np.floating]
    optimal_bitstring: str


def linear_schedule(t: float, T: float) -> float:
    """Linear annealing schedule s(t) = t/T."""
    return t / T


def polynomial_schedule(t: float, T: float, power: float = 2.0) -> float:
    """Polynomial schedule s(t) = (t/T)^p.

    Higher power spends more time near s=0 (useful when the gap
    is smallest near the end of the anneal).
    """
    return (t / T) ** power


def adiabatic_evolve(
    H_init: NDArray[np.complexfloating],
    H_final: NDArray[np.complexfloating],
    total_time: float = 10.0,
    num_steps: int = 1000,
    schedule: Callable[[float, float], float] | None = None,
    initial_state: NDArray[np.complexfloating] | None = None,
    num_sample_points: int = 50,
) -> AdiabaticResult:
    """Simulate adiabatic quantum evolution.

    Evolves from the ground state of H_init to (approximately) the
    ground state of H_final using the time-dependent Hamiltonian:

        H(t) = (1 - s(t)) H_init + s(t) H_final

    Uses piecewise-constant evolution with matrix exponential.

    Args:
        H_init: Initial Hamiltonian.
        H_final: Final (problem) Hamiltonian.
        total_time: Total anneal time T. Longer → more adiabatic.
        num_steps: Number of time steps for integration.
        schedule: Schedule function s(t, T) → [0, 1]. Default: linear.
        initial_state: Initial state (default: ground state of H_init).
        num_sample_points: Number of points for energy/gap history.

    Returns:
        AdiabaticResult with the final state and diagnostics.
    """
    H_init = np.asarray(H_init, dtype=complex)
    H_final = np.asarray(H_final, dtype=complex)
    dim = H_init.shape[0]

    if schedule is None:
        schedule = linear_schedule

    # Initialize state
    if initial_state is not None:
        psi = np.asarray(initial_state, dtype=complex).copy()
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(H_init)
        psi = eigenvectors[:, 0].copy()

    dt = total_time / num_steps
    sample_interval = max(1, num_steps // num_sample_points)
    schedule_values = []
    energy_history = []
    gap_history = []

    for step in range(num_steps):
        t = step * dt
        s = schedule(t, total_time)

        # Record diagnostics at sample points
        if step % sample_interval == 0:
            H_t = (1 - s) * H_init + s * H_final
            schedule_values.append(s)
            energy_history.append(float(np.real(psi.conj() @ H_t @ psi)))
            eigenvalues = np.linalg.eigvalsh(H_t)
            gap_history.append(float(eigenvalues[1] - eigenvalues[0]) if dim > 1 else 0.0)

        # Evolve: |ψ(t+dt)> = exp(-i H(t) dt) |ψ(t)>
        H_t = (1 - s) * H_init + s * H_final
        eigenvalues_t, V = np.linalg.eigh(H_t)
        exp_diag = np.exp(-1j * eigenvalues_t * dt)
        psi = V @ (exp_diag * (V.conj().T @ psi))

    # Final diagnostics
    s_final = schedule(total_time, total_time)
    H_t = (1 - s_final) * H_init + s_final * H_final
    schedule_values.append(s_final)
    energy_history.append(float(np.real(psi.conj() @ H_t @ psi)))
    eigenvalues_final = np.linalg.eigvalsh(H_t)
    gap_history.append(float(eigenvalues_final[1] - eigenvalues_final[0]) if dim > 1 else 0.0)

    # Ground state of final Hamiltonian
    eigenvalues_f, eigvecs_f = np.linalg.eigh(H_final)
    ground_state = eigvecs_f[:, 0]
    ground_state_prob = float(abs(np.dot(ground_state.conj(), psi)) ** 2)

    # Find optimal bitstring
    probs = np.abs(psi) ** 2
    num_qubits = int(np.log2(dim))
    optimal_idx = int(np.argmax(probs))
    optimal_bitstring = format(optimal_idx, f"0{num_qubits}b")

    return AdiabaticResult(
        final_state=psi,
        ground_state_prob=ground_state_prob,
        schedule_values=np.array(schedule_values),
        energy_history=np.array(energy_history),
        gap_history=np.array(gap_history),
        optimal_bitstring=optimal_bitstring,
    )


def spectral_gap(
    H_init: NDArray[np.complexfloating],
    H_final: NDArray[np.complexfloating],
    num_points: int = 100,
    schedule: Callable[[float, float], float] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the spectral gap along the annealing path.

    Returns:
        (s_values, gaps) — schedule parameter values and corresponding
        energy gaps between ground and first excited states.
    """
    if schedule is None:
        schedule = linear_schedule

    s_values = []
    gaps = []
    T = 1.0  # Arbitrary, only s matters

    for i in range(num_points + 1):
        t = i * T / num_points
        s = schedule(t, T)
        H_s = (1 - s) * H_init + s * H_final
        eigenvalues = np.linalg.eigvalsh(H_s)
        s_values.append(s)
        gaps.append(float(eigenvalues[1] - eigenvalues[0]))

    return np.array(s_values), np.array(gaps)
