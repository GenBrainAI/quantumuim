"""Tests for quantumium.optimal_control — GRAPE, Hamiltonians, pulses, fidelity."""

import math
import numpy as np
import pytest

from quantumium.optimal_control.hamiltonians import (
    ControlHamiltonian,
    drift_hamiltonian,
    control_hamiltonian_x,
    control_hamiltonian_y,
    transmon_hamiltonian,
)
from quantumium.optimal_control.pulses import (
    PulseSequence,
    constant_pulse,
    gaussian_pulse,
    drag_pulse,
)
from quantumium.optimal_control.fidelity import (
    gate_fidelity,
    process_fidelity,
    propagator,
    propagator_sequence,
)
from quantumium.optimal_control.grape import (
    GRAPEResult,
    grape_optimize,
)


# Standard gates for testing
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)
H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
I_GATE = np.eye(2, dtype=complex)


# ============================================================
# Hamiltonians
# ============================================================

class TestHamiltonians:
    def test_drift_hermitian(self):
        H = drift_hamiltonian(5.0)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_drift_shape(self):
        H = drift_hamiltonian(5.0, num_qubits=2)
        assert H.shape == (4, 4)

    def test_control_x_hermitian(self):
        Hx = control_hamiltonian_x()
        np.testing.assert_allclose(Hx, Hx.conj().T, atol=1e-12)

    def test_control_y_hermitian(self):
        Hy = control_hamiltonian_y()
        np.testing.assert_allclose(Hy, Hy.conj().T, atol=1e-12)

    def test_control_x_multi_qubit(self):
        Hx = control_hamiltonian_x(qubit=1, num_qubits=2)
        assert Hx.shape == (4, 4)
        np.testing.assert_allclose(Hx, Hx.conj().T, atol=1e-12)

    def test_total_hamiltonian(self):
        H0 = drift_hamiltonian(1.0)
        Hx = control_hamiltonian_x()
        Hy = control_hamiltonian_y()
        ch = ControlHamiltonian(drift=H0, controls=[Hx, Hy])

        H_total = ch.total_hamiltonian(np.array([0.5, 0.3]))
        expected = H0 + 0.5 * Hx + 0.3 * Hy
        np.testing.assert_allclose(H_total, expected, atol=1e-12)

    def test_control_hamiltonian_properties(self):
        H0 = drift_hamiltonian(2.0)
        Hx = control_hamiltonian_x()
        ch = ControlHamiltonian(drift=H0, controls=[Hx])
        assert ch.dim == 2
        assert ch.num_controls == 1


class TestTransmon:
    def test_shapes(self):
        H0, Hx, Hy = transmon_hamiltonian(num_levels=3)
        assert H0.shape == (3, 3)
        assert Hx.shape == (3, 3)
        assert Hy.shape == (3, 3)

    def test_drift_diagonal(self):
        H0, _, _ = transmon_hamiltonian(omega=5.0, anharmonicity=-0.3)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(H0[i, j]) < 1e-15

    def test_ground_state_energy_zero(self):
        H0, _, _ = transmon_hamiltonian()
        assert abs(H0[0, 0]) < 1e-15

    def test_control_hermitian(self):
        _, Hx, Hy = transmon_hamiltonian()
        np.testing.assert_allclose(Hx, Hx.conj().T, atol=1e-12)
        np.testing.assert_allclose(Hy, Hy.conj().T, atol=1e-12)


# ============================================================
# Pulses
# ============================================================

class TestPulses:
    def test_constant_pulse(self):
        p = constant_pulse(1.0, num_steps=10)
        assert p.num_steps == 10
        assert p.num_controls == 1
        assert np.all(p.amplitudes[:, 0] == 1.0)

    def test_gaussian_pulse(self):
        p = gaussian_pulse(1.0, sigma=0.5, num_steps=20, dt=0.1)
        assert p.num_steps == 20
        # Peak at center
        center = 10
        assert p.amplitudes[center, 0] > p.amplitudes[0, 0]
        assert p.amplitudes[center, 0] > p.amplitudes[-1, 0]

    def test_gaussian_symmetry(self):
        p = gaussian_pulse(1.0, sigma=0.5, num_steps=21, dt=0.1)
        np.testing.assert_allclose(
            p.amplitudes[:, 0], p.amplitudes[::-1, 0], atol=1e-12
        )

    def test_drag_pulse_two_channels(self):
        p = drag_pulse(1.0, sigma=0.5, beta=0.1, num_steps=20)
        assert p.num_controls == 2

    def test_drag_x_channel_gaussian(self):
        p = drag_pulse(1.0, sigma=0.5, beta=0.0, num_steps=20)
        p_gauss = gaussian_pulse(1.0, sigma=0.5, num_steps=20, num_controls=2)
        np.testing.assert_allclose(p.amplitudes[:, 0], p_gauss.amplitudes[:, 0], atol=1e-12)

    def test_total_time(self):
        p = constant_pulse(1.0, num_steps=10, dt=0.05)
        assert abs(p.total_time - 0.5) < 1e-12

    def test_copy_independent(self):
        p = constant_pulse(1.0, num_steps=5)
        p2 = p.copy()
        p2.amplitudes[0, 0] = 99.0
        assert p.amplitudes[0, 0] == 1.0


# ============================================================
# Fidelity
# ============================================================

class TestFidelity:
    def test_identity_fidelity(self):
        F = gate_fidelity(I_GATE, I_GATE)
        assert abs(F - 1.0) < 1e-12

    def test_same_gate(self):
        F = gate_fidelity(X_GATE, X_GATE)
        assert abs(F - 1.0) < 1e-12

    def test_orthogonal_gates(self):
        F = gate_fidelity(X_GATE, Z_GATE)
        assert F < 0.1

    def test_global_phase(self):
        U = np.exp(1j * 0.3) * X_GATE
        F = gate_fidelity(U, X_GATE)
        assert abs(F - 1.0) < 1e-10

    def test_process_fidelity_equals_gate(self):
        F_gate = gate_fidelity(H_GATE, H_GATE)
        F_proc = process_fidelity(H_GATE, H_GATE)
        assert abs(F_gate - F_proc) < 1e-12

    def test_fidelity_bounded(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            U = np.linalg.qr(rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)))[0]
            V = np.linalg.qr(rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)))[0]
            F = gate_fidelity(U, V)
            assert 0 <= F <= 1 + 1e-10


# ============================================================
# Propagator
# ============================================================

class TestPropagator:
    def test_identity_evolution(self):
        H0 = np.zeros((2, 2), dtype=complex)
        Hx = control_hamiltonian_x()
        ch = ControlHamiltonian(drift=H0, controls=[Hx])
        pulse = constant_pulse(0.0, num_steps=10)

        U = propagator(ch, pulse)
        np.testing.assert_allclose(U, I_GATE, atol=1e-10)

    def test_unitarity(self):
        H0 = drift_hamiltonian(1.0)
        Hx = control_hamiltonian_x()
        ch = ControlHamiltonian(drift=H0, controls=[Hx])
        pulse = gaussian_pulse(1.0, sigma=0.3, num_steps=20)

        U = propagator(ch, pulse)
        np.testing.assert_allclose(U @ U.conj().T, I_GATE, atol=1e-8)

    def test_x_rotation(self):
        H0 = np.zeros((2, 2), dtype=complex)
        Hx = control_hamiltonian_x()
        ch = ControlHamiltonian(drift=H0, controls=[Hx])

        # exp(-i u Hx T) = exp(-i u σx/2 T). For X gate: u*T/2 = π/2 → u = π/T
        num_steps = 100
        dt = 0.01
        amp = math.pi / (num_steps * dt)
        pulse = constant_pulse(amp, num_steps=num_steps, dt=dt)

        U = propagator(ch, pulse)
        F = gate_fidelity(U, X_GATE)
        assert F > 0.99

    def test_propagator_sequence_length(self):
        H0 = np.zeros((2, 2), dtype=complex)
        Hx = control_hamiltonian_x()
        ch = ControlHamiltonian(drift=H0, controls=[Hx])
        pulse = constant_pulse(1.0, num_steps=5)

        seq = propagator_sequence(ch, pulse)
        assert len(seq) == 5


# ============================================================
# GRAPE
# ============================================================

class TestGRAPE:
    def test_returns_result(self):
        H0 = np.zeros((2, 2), dtype=complex)
        Hx = control_hamiltonian_x()
        Hy = control_hamiltonian_y()
        ch = ControlHamiltonian(drift=H0, controls=[Hx, Hy])

        result = grape_optimize(
            ch, X_GATE, num_steps=20, dt=0.05,
            num_iterations=10, learning_rate=0.5
        )
        assert isinstance(result, GRAPEResult)
        assert 0 <= result.fidelity <= 1.0 + 1e-10

    def test_x_gate_optimization(self):
        H0 = np.zeros((2, 2), dtype=complex)
        Hx = control_hamiltonian_x()
        Hy = control_hamiltonian_y()
        ch = ControlHamiltonian(drift=H0, controls=[Hx, Hy])

        result = grape_optimize(
            ch, X_GATE, num_steps=50, dt=0.02,
            num_iterations=200, learning_rate=1.0
        )
        assert result.fidelity > 0.9

    def test_fidelity_improves(self):
        H0 = np.zeros((2, 2), dtype=complex)
        Hx = control_hamiltonian_x()
        ch = ControlHamiltonian(drift=H0, controls=[Hx])

        result = grape_optimize(
            ch, X_GATE, num_steps=30, dt=0.05,
            num_iterations=50, learning_rate=0.5
        )
        # Fidelity should improve from initial to final
        assert result.fidelity_history[-1] >= result.fidelity_history[0] - 0.01

    def test_amplitude_clipping(self):
        H0 = np.zeros((2, 2), dtype=complex)
        Hx = control_hamiltonian_x()
        ch = ControlHamiltonian(drift=H0, controls=[Hx])

        result = grape_optimize(
            ch, X_GATE, num_steps=20, dt=0.05,
            num_iterations=50, max_amplitude=5.0
        )
        assert np.all(np.abs(result.optimal_pulse.amplitudes) <= 5.0 + 1e-10)

    def test_dimension_mismatch_raises(self):
        H0 = np.zeros((2, 2), dtype=complex)
        Hx = control_hamiltonian_x()
        ch = ControlHamiltonian(drift=H0, controls=[Hx])
        target_3d = np.eye(3, dtype=complex)

        with pytest.raises(ValueError, match="dimension"):
            grape_optimize(ch, target_3d, num_steps=10)

    def test_initial_pulse(self):
        H0 = np.zeros((2, 2), dtype=complex)
        Hx = control_hamiltonian_x()
        ch = ControlHamiltonian(drift=H0, controls=[Hx])
        init = constant_pulse(1.0, num_steps=20, dt=0.05)

        result = grape_optimize(
            ch, X_GATE, num_steps=20, dt=0.05,
            num_iterations=20, initial_pulse=init
        )
        assert isinstance(result, GRAPEResult)

    def test_hadamard_optimization(self):
        H0 = np.zeros((2, 2), dtype=complex)
        Hx = control_hamiltonian_x()
        Hy = control_hamiltonian_y()
        ch = ControlHamiltonian(drift=H0, controls=[Hx, Hy])

        result = grape_optimize(
            ch, H_GATE, num_steps=50, dt=0.02,
            num_iterations=300, learning_rate=1.0
        )
        assert result.fidelity > 0.85
