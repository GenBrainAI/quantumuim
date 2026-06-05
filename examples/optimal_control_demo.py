"""Quantum Optimal Control (GRAPE) Demo.

Demonstrates:
1. Control Hamiltonian setup for a qubit
2. GRAPE optimization for X and Hadamard gates
3. Pulse shapes (Gaussian, DRAG)
4. Transmon Hamiltonian with anharmonicity
"""

import math
import numpy as np

from quantumium.optimal_control import (
    ControlHamiltonian,
    drift_hamiltonian,
    control_hamiltonian_x,
    control_hamiltonian_y,
    transmon_hamiltonian,
    grape_optimize,
    gate_fidelity,
    propagator,
    constant_pulse,
    gaussian_pulse,
    drag_pulse,
)


X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)


def demo_hamiltonian_setup():
    print("=" * 60)
    print("  Control Hamiltonian Setup")
    print("=" * 60)

    H0 = drift_hamiltonian(5.0)
    Hx = control_hamiltonian_x()
    Hy = control_hamiltonian_y()
    print(f"\n  Drift H0 = (ω/2)σz with ω = 5.0 GHz")
    print(f"    H0 =\n      {H0[0]}\n      {H0[1]}")

    ch = ControlHamiltonian(drift=H0, controls=[Hx, Hy])
    print(f"\n  System: {ch.dim}D, {ch.num_controls} control channels (X, Y)")

    amps = np.array([1.0, 0.5])
    H_total = ch.total_hamiltonian(amps)
    print(f"  H(u=[1.0, 0.5]) =\n      {H_total[0]}\n      {H_total[1]}")


def demo_pulse_shapes():
    print("\n" + "=" * 60)
    print("  Pulse Shapes")
    print("=" * 60)

    num_steps = 20
    dt = 0.05

    p_const = constant_pulse(1.0, num_steps=num_steps, dt=dt)
    print(f"\n  Constant pulse: {p_const.num_steps} steps, T = {p_const.total_time:.2f}")

    p_gauss = gaussian_pulse(1.0, sigma=0.3, num_steps=num_steps, dt=dt)
    peak_idx = np.argmax(p_gauss.amplitudes[:, 0])
    print(f"  Gaussian pulse: peak = {p_gauss.amplitudes[peak_idx, 0]:.4f} at step {peak_idx}")

    p_drag = drag_pulse(1.0, sigma=0.3, beta=0.2, num_steps=num_steps, dt=dt)
    print(f"  DRAG pulse: {p_drag.num_controls} channels (X + derivative Y)")

    print(f"\n  DRAG pulse shape (first 10 steps):")
    print(f"  {'Step':>6} {'X channel':>10} {'Y channel':>10}")
    for i in range(min(10, num_steps)):
        print(f"  {i:>6} {p_drag.amplitudes[i, 0]:>10.4f} {p_drag.amplitudes[i, 1]:>10.4f}")


def demo_grape_x_gate():
    print("\n" + "=" * 60)
    print("  GRAPE: Synthesizing X Gate")
    print("=" * 60)

    H0 = np.zeros((2, 2), dtype=complex)
    Hx = control_hamiltonian_x()
    Hy = control_hamiltonian_y()
    ch = ControlHamiltonian(drift=H0, controls=[Hx, Hy])

    print("\n  Optimizing pulses for X gate (NOT gate)...")
    result = grape_optimize(
        ch, X_GATE,
        num_steps=50, dt=0.02,
        num_iterations=200, learning_rate=1.0,
    )

    print(f"  Final fidelity: {result.fidelity:.8f}")
    print(f"  Converged:      {result.converged}")
    print(f"  Iterations:     {result.num_iterations}")

    milestones = [0, 10, 50, 100, len(result.fidelity_history) - 1]
    print(f"\n  Fidelity progression:")
    for i in milestones:
        if i < len(result.fidelity_history):
            print(f"    Iter {i:>4}: F = {result.fidelity_history[i]:.6f}")


def demo_grape_hadamard():
    print("\n" + "=" * 60)
    print("  GRAPE: Synthesizing Hadamard Gate")
    print("=" * 60)

    H0 = np.zeros((2, 2), dtype=complex)
    Hx = control_hamiltonian_x()
    Hy = control_hamiltonian_y()
    ch = ControlHamiltonian(drift=H0, controls=[Hx, Hy])

    print("\n  Optimizing pulses for Hadamard gate...")
    result = grape_optimize(
        ch, H_GATE,
        num_steps=50, dt=0.02,
        num_iterations=300, learning_rate=1.0,
    )

    print(f"  Final fidelity: {result.fidelity:.8f}")
    print(f"  Converged:      {result.converged}")

    U = propagator(ch, result.optimal_pulse)
    print(f"\n  Realized unitary (up to global phase):")
    phase = np.exp(-1j * np.angle(U[0, 0]))
    U_norm = U * phase
    for i in range(2):
        row = "  ".join(f"{U_norm[i, j].real:+.4f}" for j in range(2))
        print(f"    [{row}]")
    print(f"  Target Hadamard:")
    for i in range(2):
        row = "  ".join(f"{H_GATE[i, j].real:+.4f}" for j in range(2))
        print(f"    [{row}]")


def demo_transmon():
    print("\n" + "=" * 60)
    print("  Transmon Hamiltonian")
    print("=" * 60)

    omega = 5.0
    alpha = -0.3
    H0, Hx, Hy = transmon_hamiltonian(omega=omega, anharmonicity=alpha, num_levels=3)

    print(f"\n  Transmon: ω = {omega} GHz, α = {alpha} GHz, 3 levels")
    print(f"  Energy levels (diagonal of H0):")
    for n in range(3):
        print(f"    |{n}> : E = {H0[n, n].real:.4f} GHz")

    print(f"\n  Transition frequencies:")
    print(f"    |0> → |1> : {(H0[1, 1] - H0[0, 0]).real:.4f} GHz")
    print(f"    |1> → |2> : {(H0[2, 2] - H0[1, 1]).real:.4f} GHz  (shifted by α = {alpha})")


if __name__ == "__main__":
    demo_hamiltonian_setup()
    demo_pulse_shapes()
    demo_grape_x_gate()
    demo_grape_hadamard()
    demo_transmon()
    print("\n" + "=" * 60)
    print("  All optimal control demos completed!")
    print("=" * 60)
