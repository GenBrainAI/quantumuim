"""GRAPE — Gradient Ascent Pulse Engineering.

Optimizes control pulse amplitudes to maximize the gate fidelity
between the realized propagator and a target unitary.

The algorithm:
1. Compute forward propagators P_k = U_k · ... · U_1
2. Compute backward propagators L_k = U_N · ... · U_{k+1}
3. Gradient: ∂F/∂u_j(k) = (2/d²) Re[Tr(V† L_k (-iH_j) P_{k+1}) · conj(Tr(V† U))]
4. Update: u_j(k) ← u_j(k) + ε · ∂F/∂u_j(k)

References:
    Khaneja et al., "Optimal control of coupled spin dynamics"
    (Journal of Magnetic Resonance 172, 296-305, 2005)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .hamiltonians import ControlHamiltonian
from .pulses import PulseSequence
from .fidelity import propagator_sequence, gate_fidelity


@dataclass
class GRAPEResult:
    """Result from GRAPE pulse optimization.

    Attributes:
        optimal_pulse (PulseSequence): The optimized pulse sequence that
            best approximates the target unitary.
        fidelity (float): Final gate fidelity achieved, in [0, 1].
        fidelity_history (list[float]): Fidelity value at each iteration,
            useful for monitoring convergence.
        num_iterations (int): Total number of optimization iterations
            performed (including the final evaluation).
        converged (bool): Whether the optimization reached the convergence
            threshold before exhausting iterations.
    """

    optimal_pulse: PulseSequence
    fidelity: float
    fidelity_history: list[float]
    num_iterations: int
    converged: bool


def grape_optimize(
    hamiltonian: ControlHamiltonian,
    target: NDArray[np.complexfloating],
    num_steps: int = 100,
    dt: float = 0.1,
    num_iterations: int = 500,
    learning_rate: float = 0.1,
    convergence_threshold: float = 1 - 1e-8,
    initial_pulse: PulseSequence | None = None,
    max_amplitude: float | None = None,
) -> GRAPEResult:
    """Optimize control pulses using GRAPE.

    Finds pulse amplitudes u_j(t_k) that maximize the gate fidelity
    F = |Tr(U_target† · U)|² / d² between the propagator U and
    the target unitary.

    Args:
        hamiltonian: Control Hamiltonian with drift and control terms.
        target: Target unitary matrix (d × d).
        num_steps: Number of time steps for discretization.
        dt: Time step duration.
        num_iterations: Maximum optimization iterations.
        learning_rate: Step size for gradient ascent.
        convergence_threshold: Stop when fidelity exceeds this.
        initial_pulse: Starting pulse (random if None).
        max_amplitude: Clip amplitudes to [-max, max] if set.

    Returns:
        GRAPEResult with optimized pulse and convergence info.
    """
    target = np.asarray(target, dtype=complex)
    dim = hamiltonian.dim
    num_controls = hamiltonian.num_controls

    if target.shape != (dim, dim):
        raise ValueError(
            f"Target unitary shape {target.shape} doesn't match Hamiltonian dimension {dim}"
        )

    # Initialize pulse
    if initial_pulse is not None:
        pulse = initial_pulse.copy()
    else:
        rng = np.random.default_rng(42)
        amps = rng.uniform(-1.0, 1.0, size=(num_steps, num_controls))
        pulse = PulseSequence(amplitudes=amps, dt=dt)

    fidelity_history = []
    converged = False

    for iteration in range(num_iterations):
        step_unitaries = propagator_sequence(hamiltonian, pulse)

        # P[0] = I, P[k+1] = U_k @ P[k]
        # So P[k] = U_{k-1} @ ... @ U_0 (forward product, not including step k)
        P = [np.eye(dim, dtype=complex)]
        for k in range(num_steps):
            P.append(step_unitaries[k] @ P[-1])
        U_total = P[num_steps]

        fid = gate_fidelity(U_total, target)
        fidelity_history.append(fid)

        if fid >= convergence_threshold:
            converged = True
            break

        # L[k] = U_{N-1} @ ... @ U_{k+1} (backward, not including step k)
        # L[N-1] = I, L[k] = L[k+1] @ U_{k+1}
        L = [None] * num_steps
        L[num_steps - 1] = np.eye(dim, dtype=complex)
        for k in range(num_steps - 2, -1, -1):
            L[k] = L[k + 1] @ step_unitaries[k + 1]

        overlap = np.trace(target.conj().T @ U_total)
        overlap_conj = np.conj(overlap)
        Vdag = target.conj().T

        grad = np.zeros((num_steps, num_controls))
        for k in range(num_steps):
            for j in range(num_controls):
                H_j = hamiltonian.controls[j]
                # dt-normalized gradient of F = |Tr(V† U)|²/d²
                dM = Vdag @ L[k] @ (-1j * H_j) @ P[k + 1]
                grad[k, j] = 2 * float(np.real(np.trace(dM) * overlap_conj)) / dim**2

        # Gradient ascent update
        pulse.amplitudes = pulse.amplitudes + learning_rate * grad

        # Amplitude clipping
        if max_amplitude is not None:
            pulse.amplitudes = np.clip(pulse.amplitudes, -max_amplitude, max_amplitude)

    # Final fidelity
    step_unitaries = propagator_sequence(hamiltonian, pulse)
    U_final = np.eye(dim, dtype=complex)
    for U_k in step_unitaries:
        U_final = U_k @ U_final
    final_fid = gate_fidelity(U_final, target)
    fidelity_history.append(final_fid)

    return GRAPEResult(
        optimal_pulse=pulse,
        fidelity=final_fid,
        fidelity_history=fidelity_history,
        num_iterations=len(fidelity_history),
        converged=converged,
    )
