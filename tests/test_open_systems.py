"""Tests for quantumium.open_systems — Lindblad master equation dynamics."""

import math
import numpy as np
import pytest

from quantumium.open_systems import (
    LindbladSystem,
    lindblad_rhs,
    lindblad_evolve,
    mesolve,
    steady_state,
    decay_operator,
    dephasing_operator,
    thermal_operator,
    sigma_minus,
    sigma_plus,
    number_operator,
    annihilation_operator,
    creation_operator,
    EvolutionResult,
    expect,
    purity,
    von_neumann_entropy,
    coherence,
)


SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
I2 = np.eye(2, dtype=complex)
KET0 = np.array([[1], [0]], dtype=complex)
KET1 = np.array([[0], [1]], dtype=complex)
RHO0 = KET0 @ KET0.conj().T
RHO1 = KET1 @ KET1.conj().T
RHO_PLUS = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)


# ============================================================
# Operators
# ============================================================

class TestOperators:
    def test_sigma_minus(self):
        sm = sigma_minus()
        assert sm.shape == (2, 2)
        np.testing.assert_allclose(sm, [[0, 1], [0, 0]])

    def test_sigma_plus(self):
        sp = sigma_plus()
        np.testing.assert_allclose(sp, [[0, 0], [1, 0]])

    def test_sigma_minus_plus_conjugate(self):
        sm = sigma_minus()
        sp = sigma_plus()
        np.testing.assert_allclose(sm, sp.conj().T)

    def test_multi_qubit_sigma(self):
        sm = sigma_minus(num_qubits=2, qubit=1)
        assert sm.shape == (4, 4)
        expected = np.kron(I2, np.array([[0, 1], [0, 0]], dtype=complex))
        np.testing.assert_allclose(sm, expected)

    def test_decay_operator(self):
        gamma, op = decay_operator(0.1)
        assert gamma == 0.1
        np.testing.assert_allclose(op, sigma_minus())

    def test_dephasing_operator(self):
        gamma, op = dephasing_operator(0.05)
        assert gamma == 0.05
        np.testing.assert_allclose(op, SZ / 2)

    def test_thermal_operators_zero_temp(self):
        ops = thermal_operator(0.1, n_thermal=0.0)
        assert len(ops) == 1
        assert ops[0][0] == 0.1

    def test_thermal_operators_finite_temp(self):
        ops = thermal_operator(0.1, n_thermal=0.5)
        assert len(ops) == 2
        assert ops[0][0] == pytest.approx(0.15)  # γ(1+n̄)
        assert ops[1][0] == pytest.approx(0.05)   # γn̄

    def test_annihilation_operator(self):
        a = annihilation_operator(3)
        assert a.shape == (3, 3)
        assert a[0, 1] == pytest.approx(1.0)
        assert a[1, 2] == pytest.approx(math.sqrt(2))
        assert a[0, 0] == 0

    def test_creation_annihilation_conjugate(self):
        a = annihilation_operator(4)
        adag = creation_operator(4)
        np.testing.assert_allclose(adag, a.conj().T)

    def test_number_operator(self):
        n = number_operator(4)
        np.testing.assert_allclose(n, np.diag([0, 1, 2, 3]))

    def test_number_from_creation_annihilation(self):
        a = annihilation_operator(4)
        adag = creation_operator(4)
        n = number_operator(4)
        np.testing.assert_allclose(adag @ a, n, atol=1e-12)


# ============================================================
# Dynamics utilities
# ============================================================

class TestDynamics:
    def test_expect_pure(self):
        val = expect(RHO0, SZ)
        assert val == pytest.approx(1.0)

    def test_expect_excited(self):
        val = expect(RHO1, SZ)
        assert val == pytest.approx(-1.0)

    def test_expect_superposition(self):
        val = expect(RHO_PLUS, SZ)
        assert val == pytest.approx(0.0, abs=1e-12)

    def test_purity_pure_state(self):
        assert purity(RHO0) == pytest.approx(1.0)

    def test_purity_mixed_state(self):
        rho_mixed = I2 / 2
        assert purity(rho_mixed) == pytest.approx(0.5)

    def test_entropy_pure(self):
        assert von_neumann_entropy(RHO0) == pytest.approx(0.0, abs=1e-10)

    def test_entropy_maximally_mixed(self):
        rho = I2 / 2
        assert von_neumann_entropy(rho) == pytest.approx(1.0)  # 1 bit

    def test_coherence_diagonal(self):
        assert coherence(RHO0) == pytest.approx(0.0, abs=1e-12)

    def test_coherence_superposition(self):
        assert coherence(RHO_PLUS) == pytest.approx(1.0)


# ============================================================
# Lindblad RHS
# ============================================================

class TestLindbladRHS:
    def test_closed_system_unitary(self):
        H = SZ / 2
        system = LindbladSystem(hamiltonian=H)
        drho = lindblad_rhs(RHO0, system)
        np.testing.assert_allclose(drho, np.zeros((2, 2)), atol=1e-14)

    def test_closed_system_off_diagonal(self):
        H = SZ / 2
        system = LindbladSystem(hamiltonian=H)
        drho = lindblad_rhs(RHO_PLUS, system)
        assert abs(drho[0, 1]) > 0

    def test_decay_ground_state_stable(self):
        system = LindbladSystem(
            hamiltonian=np.zeros((2, 2), dtype=complex),
            collapse_ops=[decay_operator(1.0)],
        )
        drho = lindblad_rhs(RHO0, system)
        np.testing.assert_allclose(drho, np.zeros((2, 2)), atol=1e-14)

    def test_decay_excited_state(self):
        system = LindbladSystem(
            hamiltonian=np.zeros((2, 2), dtype=complex),
            collapse_ops=[decay_operator(1.0)],
        )
        drho = lindblad_rhs(RHO1, system)
        assert drho[0, 0] > 0
        assert drho[1, 1] < 0

    def test_trace_preservation(self):
        H = 0.5 * SX
        system = LindbladSystem(
            hamiltonian=H,
            collapse_ops=[decay_operator(0.1), dephasing_operator(0.05)],
        )
        drho = lindblad_rhs(RHO_PLUS, system)
        assert np.trace(drho) == pytest.approx(0.0, abs=1e-12)

    def test_time_dependent_hamiltonian(self):
        def H_t(t):
            return np.cos(t) * SZ / 2
        system = LindbladSystem(hamiltonian=H_t)
        drho0 = lindblad_rhs(RHO_PLUS, system, t=0.0)
        drho_pi2 = lindblad_rhs(RHO_PLUS, system, t=math.pi / 2)
        assert np.linalg.norm(drho0) > np.linalg.norm(drho_pi2)


# ============================================================
# Lindblad evolution
# ============================================================

class TestLindbladEvolve:
    def test_closed_system_preserves_purity(self):
        H = SZ / 2
        system = LindbladSystem(hamiltonian=H)
        rho_final = lindblad_evolve(RHO_PLUS, system, dt=0.01, num_steps=100)
        assert purity(rho_final) == pytest.approx(1.0, abs=1e-6)

    def test_closed_system_preserves_trace(self):
        H = SX / 2
        system = LindbladSystem(hamiltonian=H)
        rho_final = lindblad_evolve(RHO0, system, dt=0.01, num_steps=200)
        assert np.trace(rho_final) == pytest.approx(1.0, abs=1e-8)

    def test_t1_decay_to_ground(self):
        gamma = 1.0
        system = LindbladSystem(
            hamiltonian=np.zeros((2, 2), dtype=complex),
            collapse_ops=[decay_operator(gamma)],
        )
        rho_final = lindblad_evolve(RHO1, system, dt=0.01, num_steps=1000)
        assert rho_final[0, 0] == pytest.approx(1.0, abs=1e-3)

    def test_t1_decay_rate(self):
        gamma = 0.5
        T = 2.0
        system = LindbladSystem(
            hamiltonian=np.zeros((2, 2), dtype=complex),
            collapse_ops=[decay_operator(gamma)],
        )
        rho_final = lindblad_evolve(RHO1, system, dt=0.01, num_steps=int(T / 0.01))
        p_excited = float(np.real(rho_final[1, 1]))
        expected = math.exp(-gamma * T)
        assert p_excited == pytest.approx(expected, abs=0.01)

    def test_dephasing_destroys_coherence(self):
        gamma_phi = 2.0
        system = LindbladSystem(
            hamiltonian=np.zeros((2, 2), dtype=complex),
            collapse_ops=[dephasing_operator(gamma_phi)],
        )
        rho_final = lindblad_evolve(RHO_PLUS, system, dt=0.01, num_steps=500)
        assert coherence(rho_final) < 0.01

    def test_dephasing_preserves_populations(self):
        gamma_phi = 2.0
        system = LindbladSystem(
            hamiltonian=np.zeros((2, 2), dtype=complex),
            collapse_ops=[dephasing_operator(gamma_phi)],
        )
        rho_final = lindblad_evolve(RHO_PLUS, system, dt=0.01, num_steps=500)
        assert rho_final[0, 0] == pytest.approx(0.5, abs=1e-6)
        assert rho_final[1, 1] == pytest.approx(0.5, abs=1e-6)


# ============================================================
# mesolve
# ============================================================

class TestMesolve:
    def test_returns_evolution_result(self):
        system = LindbladSystem(hamiltonian=SZ / 2)
        tlist = np.linspace(0, 1, 11)
        result = mesolve(system, RHO0, tlist)
        assert isinstance(result, EvolutionResult)
        assert len(result.states) == 11
        assert len(result.times) == 11

    def test_expectation_values(self):
        system = LindbladSystem(hamiltonian=SZ / 2)
        tlist = np.linspace(0, 1, 11)
        result = mesolve(system, RHO0, tlist, observables={"sz": SZ})
        assert "sz" in result.expect_values
        np.testing.assert_allclose(result.expect_values["sz"], 1.0, atol=1e-6)

    def test_decay_trajectory(self):
        gamma = 1.0
        system = LindbladSystem(
            hamiltonian=np.zeros((2, 2), dtype=complex),
            collapse_ops=[decay_operator(gamma)],
        )
        tlist = np.linspace(0, 5, 51)
        result = mesolve(system, RHO1, tlist, observables={"pop": RHO1})
        expected = np.exp(-gamma * tlist)
        np.testing.assert_allclose(result.expect_values["pop"], expected, atol=0.02)

    def test_rabi_oscillation_with_decay(self):
        omega = 2 * math.pi
        gamma = 0.1
        H = omega / 2 * SX
        system = LindbladSystem(hamiltonian=H, collapse_ops=[decay_operator(gamma)])
        tlist = np.linspace(0, 5, 101)
        result = mesolve(system, RHO0, tlist, observables={"sz": SZ})
        sz_vals = result.expect_values["sz"]
        # Should oscillate but damp toward steady state
        assert np.std(sz_vals[:20]) > np.std(sz_vals[-20:])


# ============================================================
# Steady state
# ============================================================

class TestSteadyState:
    def test_decay_steady_state_is_ground(self):
        system = LindbladSystem(
            hamiltonian=np.zeros((2, 2), dtype=complex),
            collapse_ops=[decay_operator(1.0)],
        )
        rho_ss = steady_state(system)
        np.testing.assert_allclose(rho_ss, RHO0, atol=1e-8)

    def test_thermal_steady_state(self):
        n_th = 0.5
        gamma = 1.0
        system = LindbladSystem(
            hamiltonian=np.zeros((2, 2), dtype=complex),
            collapse_ops=thermal_operator(gamma, n_th),
        )
        rho_ss = steady_state(system)
        p_excited_expected = n_th / (2 * n_th + 1)
        assert rho_ss[1, 1] == pytest.approx(p_excited_expected, abs=1e-6)

    def test_steady_state_is_stationary(self):
        H = SZ / 2
        system = LindbladSystem(
            hamiltonian=H,
            collapse_ops=[decay_operator(0.5), dephasing_operator(0.2)],
        )
        rho_ss = steady_state(system)
        drho = lindblad_rhs(rho_ss, system)
        np.testing.assert_allclose(drho, np.zeros((2, 2)), atol=1e-8)

    def test_steady_state_trace_one(self):
        system = LindbladSystem(
            hamiltonian=SX / 2,
            collapse_ops=[decay_operator(0.3)],
        )
        rho_ss = steady_state(system)
        assert np.trace(rho_ss) == pytest.approx(1.0, abs=1e-10)

    def test_steady_state_positive(self):
        system = LindbladSystem(
            hamiltonian=SZ,
            collapse_ops=[decay_operator(0.5), dephasing_operator(0.5)],
        )
        rho_ss = steady_state(system)
        eigvals = np.linalg.eigvalsh(rho_ss)
        assert all(v >= -1e-10 for v in eigvals)

    def test_steady_state_evolve_method(self):
        system = LindbladSystem(
            hamiltonian=np.zeros((2, 2), dtype=complex),
            collapse_ops=[decay_operator(1.0)],
        )
        rho_ss = steady_state(system, method="evolve")
        np.testing.assert_allclose(rho_ss, RHO0, atol=1e-3)

    def test_time_dependent_raises(self):
        def H_t(t):
            return t * SZ
        system = LindbladSystem(hamiltonian=H_t)
        with pytest.raises(ValueError, match="time-independent"):
            steady_state(system)


# ============================================================
# Multi-qubit
# ============================================================

class TestMultiQubit:
    def test_two_qubit_independent_decay(self):
        H = np.zeros((4, 4), dtype=complex)
        system = LindbladSystem(
            hamiltonian=H,
            collapse_ops=[
                decay_operator(1.0, num_qubits=2, qubit=0),
                decay_operator(1.0, num_qubits=2, qubit=1),
            ],
        )
        rho11 = np.zeros((4, 4), dtype=complex)
        rho11[3, 3] = 1.0  # |11>
        rho_final = lindblad_evolve(rho11, system, dt=0.01, num_steps=1000)
        assert rho_final[0, 0] == pytest.approx(1.0, abs=1e-2)

    def test_two_qubit_trace_preservation(self):
        H = np.kron(SZ, I2) / 2
        system = LindbladSystem(
            hamiltonian=H,
            collapse_ops=[
                decay_operator(0.5, num_qubits=2, qubit=0),
                dephasing_operator(0.3, num_qubits=2, qubit=1),
            ],
        )
        rho0 = np.zeros((4, 4), dtype=complex)
        rho0[3, 3] = 1.0
        rho_final = lindblad_evolve(rho0, system, dt=0.01, num_steps=200)
        assert np.trace(rho_final) == pytest.approx(1.0, abs=1e-8)
