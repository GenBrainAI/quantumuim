"""Open Quantum Systems (Lindblad) Demo.

Demonstrates:
1. T1 decay (amplitude damping)
2. T2 dephasing
3. Rabi oscillations with decoherence
4. Thermal equilibrium
5. Steady-state computation
"""

import math
import numpy as np

from quantumium.open_systems import (
    LindbladSystem,
    mesolve,
    steady_state,
    decay_operator,
    dephasing_operator,
    thermal_operator,
    expect,
    purity,
    von_neumann_entropy,
    coherence,
)


SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
KET0 = np.array([[1], [0]], dtype=complex)
KET1 = np.array([[0], [1]], dtype=complex)
RHO0 = KET0 @ KET0.conj().T
RHO1 = KET1 @ KET1.conj().T
RHO_PLUS = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)


def demo_t1_decay():
    print("=" * 60)
    print("  T1 Decay — Amplitude Damping")
    print("=" * 60)

    gamma = 0.5  # 1/T1
    system = LindbladSystem(
        hamiltonian=np.zeros((2, 2), dtype=complex),
        collapse_ops=[decay_operator(gamma)],
    )

    tlist = np.linspace(0, 6, 31)
    result = mesolve(system, RHO1, tlist, observables={"<σz>": SZ})

    print(f"\n  Excited state |1> decaying with T1 = {1/gamma:.1f}")
    print(f"  {'Time':>6}  {'<σz>':>8}  {'P(|1>)':>8}  {'Purity':>8}")
    for i in [0, 5, 10, 15, 20, 30]:
        if i < len(tlist):
            t = tlist[i]
            sz = result.expect_values["<σz>"][i]
            p1 = float(np.real(result.states[i][1, 1]))
            pur = purity(result.states[i])
            print(f"  {t:6.2f}  {sz:8.4f}  {p1:8.4f}  {pur:8.4f}")

    p1_theory = math.exp(-gamma * tlist[-1])
    print(f"\n  Final P(|1>) = {float(np.real(result.states[-1][1,1])):.6f}")
    print(f"  Theory:        {p1_theory:.6f}")


def demo_dephasing():
    print("\n" + "=" * 60)
    print("  T2 Dephasing — Pure Dephasing")
    print("=" * 60)

    gamma_phi = 1.0
    system = LindbladSystem(
        hamiltonian=np.zeros((2, 2), dtype=complex),
        collapse_ops=[dephasing_operator(gamma_phi)],
    )

    tlist = np.linspace(0, 4, 21)
    result = mesolve(system, RHO_PLUS, tlist)

    print(f"\n  |+> state dephasing with γ_φ = {gamma_phi}")
    print(f"  {'Time':>6}  {'|ρ01|':>8}  {'Purity':>8}  {'Entropy':>8}")
    for i in [0, 5, 10, 15, 20]:
        if i < len(tlist):
            t = tlist[i]
            coh = abs(result.states[i][0, 1])
            pur = purity(result.states[i])
            ent = von_neumann_entropy(result.states[i])
            print(f"  {t:6.2f}  {coh:8.4f}  {pur:8.4f}  {ent:8.4f}")

    print(f"\n  Populations preserved: P(|0>) = {float(np.real(result.states[-1][0,0])):.4f}")


def demo_rabi_with_decay():
    print("\n" + "=" * 60)
    print("  Rabi Oscillations with Decoherence")
    print("=" * 60)

    omega = 2 * math.pi  # Rabi frequency
    gamma = 0.2
    H = omega / 2 * SX

    system = LindbladSystem(
        hamiltonian=H,
        collapse_ops=[decay_operator(gamma)],
    )

    tlist = np.linspace(0, 8, 81)
    result = mesolve(system, RHO0, tlist, observables={"<σz>": SZ})

    print(f"\n  Ω = 2π, γ = {gamma}")
    print(f"  Damped Rabi oscillations (first 10 points):")
    print(f"  {'Time':>6}  {'<σz>':>8}")
    for i in range(0, 20, 2):
        t = tlist[i]
        sz = result.expect_values["<σz>"][i]
        print(f"  {t:6.2f}  {sz:8.4f}")

    rho_ss = steady_state(system)
    print(f"\n  Steady state <σz> = {float(np.real(expect(rho_ss, SZ))):.4f}")


def demo_thermal_equilibrium():
    print("\n" + "=" * 60)
    print("  Thermal Equilibrium")
    print("=" * 60)

    n_thermal = 0.3
    gamma = 1.0
    system = LindbladSystem(
        hamiltonian=np.zeros((2, 2), dtype=complex),
        collapse_ops=thermal_operator(gamma, n_thermal),
    )

    rho_ss = steady_state(system)
    p_excited = float(np.real(rho_ss[1, 1]))
    p_expected = n_thermal / (2 * n_thermal + 1)

    print(f"\n  Thermal bath: n̄ = {n_thermal}, γ = {gamma}")
    print(f"  Steady-state P(|1>) = {p_excited:.6f}")
    print(f"  Expected:            {p_expected:.6f}")
    print(f"  Entropy:             {von_neumann_entropy(rho_ss):.4f} bits")


if __name__ == "__main__":
    demo_t1_decay()
    demo_dephasing()
    demo_rabi_with_decay()
    demo_thermal_equilibrium()
    print("\n" + "=" * 60)
    print("  All open systems demos completed!")
    print("=" * 60)
