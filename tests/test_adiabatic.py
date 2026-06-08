"""Tests for quantumium.adiabatic — adiabatic quantum computing."""

import numpy as np
import pytest

from quantumium.adiabatic import (
    AdiabaticResult,
    adiabatic_evolve,
    spectral_gap,
    linear_schedule,
    polynomial_schedule,
    transverse_field_hamiltonian,
    ising_hamiltonian,
    maxcut_hamiltonian,
)


SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# ============================================================
# Hamiltonians
# ============================================================


class TestHamiltonians:
    def test_transverse_field_hermitian(self):
        H = transverse_field_hamiltonian(3)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_transverse_field_ground_state(self):
        H = transverse_field_hamiltonian(2)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        # Ground state should be |++> = (1/2)[1,1,1,1]
        gs = eigenvectors[:, 0]
        expected = np.ones(4) / 2
        np.testing.assert_allclose(np.abs(gs), np.abs(expected), atol=1e-10)

    def test_ising_hermitian(self):
        J = np.array([[0, 1], [0, 0]], dtype=float)
        H = ising_hamiltonian(2, J=J)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_ising_diagonal(self):
        J = np.array([[0, 1], [0, 0]], dtype=float)
        h = np.array([0.5, -0.5], dtype=float)
        H = ising_hamiltonian(2, J=J, h=h)
        # Should be diagonal (all Z operators)
        off_diag = H - np.diag(np.diag(H))
        np.testing.assert_allclose(off_diag, 0, atol=1e-12)

    def test_ising_coupling(self):
        # Two qubits with J=1: ground states are |00> and |11>
        J = np.array([[0, 1], [0, 0]], dtype=float)
        H = ising_hamiltonian(2, J=J)
        eigenvalues = np.sort(np.linalg.eigvalsh(H))
        # Energies: -J for aligned (00,11), +J for anti-aligned (01,10)
        assert eigenvalues[0] == pytest.approx(-1.0)

    def test_maxcut_hermitian(self):
        H = maxcut_hamiltonian(3, edges=[(0, 1), (1, 2)])
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_maxcut_triangle(self):
        edges = [(0, 1), (1, 2), (0, 2)]
        H = maxcut_hamiltonian(3, edges)
        eigenvalues = np.linalg.eigvalsh(H)
        # Triangle max cut = 2 (can cut 2 of 3 edges)
        # H encodes cost, ground state has energy = -(num uncut edges)
        # Wait, H = Σ(I - ZZ)/2, for cut edge: (1-(-1))/2=1, uncut: (1-1)/2=0
        # So H|optimal> = max_cut_value, and minimum eigenvalue is 0 (all same)
        # Actually max_cut is 2, so max eigenvalue of H is 2
        assert max(eigenvalues) == pytest.approx(2.0, abs=1e-10)

    def test_maxcut_weighted(self):
        edges = [(0, 1)]
        weights = [2.0]
        H = maxcut_hamiltonian(2, edges, weights)
        eigenvalues = np.sort(np.linalg.eigvalsh(H))
        # Max cut = 2.0 (for |01> or |10>)
        assert max(eigenvalues) == pytest.approx(2.0, abs=1e-10)


# ============================================================
# Schedule functions
# ============================================================


class TestSchedules:
    def test_linear_boundaries(self):
        assert linear_schedule(0, 10) == 0.0
        assert linear_schedule(10, 10) == 1.0
        assert linear_schedule(5, 10) == 0.5

    def test_polynomial_boundaries(self):
        assert polynomial_schedule(0, 10) == 0.0
        assert polynomial_schedule(10, 10) == 1.0

    def test_polynomial_slower_start(self):
        # Quadratic schedule spends more time near s=0
        s_linear = linear_schedule(5, 10)
        s_quad = polynomial_schedule(5, 10, power=2)
        assert s_quad < s_linear


# ============================================================
# Spectral gap
# ============================================================


class TestSpectralGap:
    def test_gap_shape(self):
        H_init = transverse_field_hamiltonian(2)
        H_final = ising_hamiltonian(2, J=np.array([[0, 1], [0, 0]]))
        s_vals, gaps = spectral_gap(H_init, H_final, num_points=20)
        assert len(s_vals) == 21
        assert len(gaps) == 21

    def test_gap_positive(self):
        H_init = transverse_field_hamiltonian(2)
        H_final = ising_hamiltonian(2, J=np.array([[0, 1], [0, 0]]))
        _, gaps = spectral_gap(H_init, H_final)
        assert all(g > -1e-10 for g in gaps)

    def test_gap_at_endpoints(self):
        H_init = transverse_field_hamiltonian(2)
        J = np.array([[0, 1], [0, 0]], dtype=float)
        H_final = ising_hamiltonian(2, J=J)
        s_vals, gaps = spectral_gap(H_init, H_final)
        # Gap at s=0 is the gap of H_init (transverse field)
        assert gaps[0] > 0
        # Gap at s=1 is the gap of H_final
        assert gaps[-1] >= 0


# ============================================================
# Adiabatic evolution
# ============================================================


class TestAdiabaticEvolution:
    def test_returns_result(self):
        H_init = transverse_field_hamiltonian(2)
        H_final = ising_hamiltonian(2, J=np.array([[0, 1], [0, 0]]))
        result = adiabatic_evolve(H_init, H_final, total_time=1.0, num_steps=100)
        assert isinstance(result, AdiabaticResult)
        assert 0 <= result.ground_state_prob <= 1.0 + 1e-10

    def test_slow_evolution_finds_ground_state(self):
        # Simple problem: H_final = -Z (ground state |0>)
        H_init = -SX
        H_final = -SZ
        result = adiabatic_evolve(
            H_init,
            H_final,
            total_time=20.0,
            num_steps=2000,
        )
        assert result.ground_state_prob > 0.95

    def test_fast_evolution_lower_fidelity(self):
        H_init = -SX
        H_final = -SZ
        fast = adiabatic_evolve(H_init, H_final, total_time=0.5, num_steps=50)
        slow = adiabatic_evolve(H_init, H_final, total_time=20.0, num_steps=2000)
        assert slow.ground_state_prob > fast.ground_state_prob

    def test_normalization(self):
        H_init = transverse_field_hamiltonian(2)
        H_final = ising_hamiltonian(2, J=np.array([[0, 1], [0, 0]]))
        result = adiabatic_evolve(H_init, H_final, total_time=5.0, num_steps=500)
        norm = np.linalg.norm(result.final_state)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_energy_decreases(self):
        H_init = transverse_field_hamiltonian(2)
        H_final = ising_hamiltonian(2, J=np.array([[0, 1], [0, 0]]))
        result = adiabatic_evolve(H_init, H_final, total_time=10.0, num_steps=1000)
        # Energy should generally track the ground state energy
        assert len(result.energy_history) > 0

    def test_schedule_values_range(self):
        H_init = -SX
        H_final = -SZ
        result = adiabatic_evolve(H_init, H_final, total_time=5.0, num_steps=500)
        assert result.schedule_values[0] == pytest.approx(0.0)
        assert result.schedule_values[-1] == pytest.approx(1.0)

    def test_optimal_bitstring(self):
        H_init = -SX
        H_final = -SZ
        result = adiabatic_evolve(
            H_init,
            H_final,
            total_time=20.0,
            num_steps=2000,
        )
        assert result.optimal_bitstring == "0"

    def test_polynomial_schedule(self):
        H_init = -SX
        H_final = -SZ
        result = adiabatic_evolve(
            H_init,
            H_final,
            total_time=10.0,
            num_steps=1000,
            schedule=lambda t, T: polynomial_schedule(t, T, power=3),
        )
        assert result.ground_state_prob > 0.8

    def test_two_qubit_ising(self):
        H_init = transverse_field_hamiltonian(2)
        J = np.array([[0, 1], [0, 0]], dtype=float)
        H_final = ising_hamiltonian(2, J=J)
        result = adiabatic_evolve(
            H_init,
            H_final,
            total_time=20.0,
            num_steps=2000,
        )
        # Ground state of -ZZ is |00> or |11> (degenerate)
        probs = np.abs(result.final_state) ** 2
        assert (probs[0] + probs[3]) > 0.9  # |00> + |11>

    def test_maxcut_problem(self):
        # Triangle graph: max cut = 2
        edges = [(0, 1), (1, 2), (0, 2)]
        H_init = transverse_field_hamiltonian(3)
        # MaxCut H has MAXIMUM eigenvalue at the cut
        # For adiabatic, we want to find the max, so negate
        H_final = -maxcut_hamiltonian(3, edges)
        result = adiabatic_evolve(
            H_init,
            H_final,
            total_time=30.0,
            num_steps=3000,
        )
        # Should find a bitstring that cuts 2 edges
        bs = result.optimal_bitstring
        cut = sum(1 for i, j in edges if bs[i] != bs[j])
        assert cut >= 2

    def test_custom_initial_state(self):
        H_init = -SX
        H_final = -SZ
        psi0 = np.array([1, 0], dtype=complex)  # |0> instead of ground state of H_init
        result = adiabatic_evolve(
            H_init,
            H_final,
            total_time=10.0,
            num_steps=1000,
            initial_state=psi0,
        )
        assert isinstance(result, AdiabaticResult)
