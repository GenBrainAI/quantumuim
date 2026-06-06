"""Tests for the dynamical decoupling module."""

import math

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, Y, Z, CNOT, SWAP
from quantumium.dd.sequences import DDSequence
from quantumium.dd.scheduler import DDScheduler
from quantumium.dd.filters import filter_function, fid_filter, dd_fidelity


# ──────────────────────────────────────────────
# DDSequence construction
# ──────────────────────────────────────────────

class TestDDSequenceConstruction:
    def test_spin_echo(self):
        seq = DDSequence.spin_echo()
        assert seq.name == "spin_echo"
        assert seq.pulses == ["X"]
        assert len(seq.spacings) == 2
        assert abs(sum(seq.spacings) - 1.0) < 1e-10

    def test_cpmg_2(self):
        seq = DDSequence.cpmg(2)
        assert seq.pulses == ["Y", "Y"]
        assert len(seq.spacings) == 3
        assert abs(sum(seq.spacings) - 1.0) < 1e-10

    def test_cpmg_4(self):
        seq = DDSequence.cpmg(4)
        assert len(seq.pulses) == 4
        assert all(p == "Y" for p in seq.pulses)

    def test_xy4(self):
        seq = DDSequence.xy4()
        assert seq.pulses == ["X", "Y", "X", "Y"]
        assert len(seq.spacings) == 5
        assert abs(sum(seq.spacings) - 1.0) < 1e-10

    def test_xy8(self):
        seq = DDSequence.xy8()
        assert seq.pulses == ["X", "Y", "X", "Y", "Y", "X", "Y", "X"]
        assert len(seq.spacings) == 9
        assert abs(sum(seq.spacings) - 1.0) < 1e-10

    def test_udd_1(self):
        seq = DDSequence.udd(1)
        assert len(seq.pulses) == 1
        assert abs(sum(seq.spacings) - 1.0) < 1e-10

    def test_udd_4(self):
        seq = DDSequence.udd(4)
        assert len(seq.pulses) == 4
        assert abs(sum(seq.spacings) - 1.0) < 1e-10

    def test_spacings_length_mismatch(self):
        with pytest.raises(ValueError, match="spacings length"):
            DDSequence(name="bad", pulses=["X"], spacings=[0.5, 0.3, 0.2])

    def test_spacings_sum_not_one(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DDSequence(name="bad", pulses=["X"], spacings=[0.3, 0.3])

    def test_cpmg_zero_pulses(self):
        with pytest.raises(ValueError):
            DDSequence.cpmg(0)

    def test_udd_zero_pulses(self):
        with pytest.raises(ValueError):
            DDSequence.udd(0)


# ──────────────────────────────────────────────
# Pulse times
# ──────────────────────────────────────────────

class TestPulseTimes:
    def test_spin_echo_midpoint(self):
        times = DDSequence.spin_echo().pulse_times()
        assert len(times) == 1
        assert abs(times[0] - 0.5) < 1e-10

    def test_cpmg_symmetric(self):
        times = DDSequence.cpmg(4).pulse_times()
        assert len(times) == 4
        for i, expected in enumerate([0.125, 0.375, 0.625, 0.875]):
            assert abs(times[i] - expected) < 1e-10

    def test_xy4_times(self):
        times = DDSequence.xy4().pulse_times()
        assert len(times) == 4

    def test_udd_times_in_unit_interval(self):
        times = DDSequence.udd(6).pulse_times()
        assert all(0 < t < 1 for t in times)
        assert times == sorted(times)

    def test_udd_formula(self):
        n = 4
        seq = DDSequence.udd(n)
        times = seq.pulse_times()
        for j in range(1, n + 1):
            expected = math.sin(math.pi * j / (2 * n + 2)) ** 2
            assert abs(times[j - 1] - expected) < 1e-10


# ──────────────────────────────────────────────
# DDScheduler — idle window detection
# ──────────────────────────────────────────────

class TestDDSchedulerIdleWindows:
    def test_bell_state_idle(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        sched = DDScheduler(DDSequence.xy4())
        windows = sched.find_idle_windows(qc)
        assert len(windows[0]) == 0
        assert len(windows[1]) >= 1

    def test_no_idle_single_qubit(self):
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        qc.apply(Y, 0)
        sched = DDScheduler(DDSequence.spin_echo())
        windows = sched.find_idle_windows(qc)
        assert len(windows[0]) == 0

    def test_three_qubit_idle(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(X, 0)
        qc.apply(CNOT, 0, 2)
        sched = DDScheduler(DDSequence.cpmg(2))
        windows = sched.find_idle_windows(qc)
        assert 2 in windows

    def test_empty_circuit(self):
        qc = QuantumCircuit(2)
        sched = DDScheduler(DDSequence.xy4())
        windows = sched.find_idle_windows(qc)
        assert len(windows[0]) == 0
        assert len(windows[1]) == 0


# ──────────────────────────────────────────────
# DDScheduler — pulse insertion
# ──────────────────────────────────────────────

class TestDDSchedulerInsert:
    def test_insert_increases_gates(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(X, 0)
        qc.apply(CNOT, 0, 2)
        qc.apply(X, 1)
        qc.apply(H, 2)

        original_count = len(qc.instructions)
        sched = DDScheduler(DDSequence.cpmg(2))
        dd_count = sched.count_dd_pulses(qc)

        if dd_count > 0:
            new_qc = sched.schedule(qc)
            assert len(new_qc.instructions) > original_count

    def test_count_matches_insertion(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(X, 0)
        qc.apply(CNOT, 0, 2)
        qc.apply(X, 1)

        sched = DDScheduler(DDSequence.cpmg(2))
        count = sched.count_dd_pulses(qc)
        if count > 0:
            new_qc = sched.schedule(qc)
            added = len(new_qc.instructions) - len(qc.instructions)
            assert added == count

    def test_min_idle_duration_filter(self):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        sched_strict = DDScheduler(DDSequence.spin_echo(), min_idle_duration=10)
        assert sched_strict.count_dd_pulses(qc) == 0

    def test_schedule_preserves_num_qubits(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(CNOT, 1, 2)
        sched = DDScheduler(DDSequence.xy4())
        new_qc = sched.schedule(qc)
        assert new_qc.num_qubits == qc.num_qubits

    def test_original_circuit_unchanged(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(X, 0)
        qc.apply(CNOT, 0, 2)
        original_count = len(qc.instructions)
        sched = DDScheduler(DDSequence.cpmg(2))
        sched.schedule(qc)
        assert len(qc.instructions) == original_count


# ──────────────────────────────────────────────
# Identity preservation
# ──────────────────────────────────────────────

class TestIdentityPreservation:
    def _dd_fidelity_noiseless(self, seq: DDSequence) -> float:
        """Apply DD sequence directly to |0⟩ and check fidelity."""
        qc = QuantumCircuit(1)
        for pulse_name in seq.pulses:
            if pulse_name == "X":
                qc.apply(X, 0)
            elif pulse_name == "Y":
                qc.apply(Y, 0)
            elif pulse_name == "Z":
                qc.apply(Z, 0)
        sv = qc.statevector()
        return float(abs(sv[0]) ** 2)

    def test_spin_echo_flips(self):
        # Spin echo has 1 X pulse (odd), so X|0⟩ = |1⟩; fidelity with |0⟩ is 0.
        # Identity preservation requires even repetitions (XX = I).
        assert abs(self._dd_fidelity_noiseless(DDSequence.spin_echo())) < 1e-10

    def test_cpmg_2_identity(self):
        assert abs(self._dd_fidelity_noiseless(DDSequence.cpmg(2)) - 1.0) < 1e-10

    def test_cpmg_4_identity(self):
        assert abs(self._dd_fidelity_noiseless(DDSequence.cpmg(4)) - 1.0) < 1e-10

    def test_xy4_identity(self):
        assert abs(self._dd_fidelity_noiseless(DDSequence.xy4()) - 1.0) < 1e-10

    def test_xy8_identity(self):
        assert abs(self._dd_fidelity_noiseless(DDSequence.xy8()) - 1.0) < 1e-10

    def test_udd_2_identity(self):
        assert abs(self._dd_fidelity_noiseless(DDSequence.udd(2)) - 1.0) < 1e-10

    def test_udd_4_identity(self):
        assert abs(self._dd_fidelity_noiseless(DDSequence.udd(4)) - 1.0) < 1e-10


# ──────────────────────────────────────────────
# Filter function
# ──────────────────────────────────────────────

class TestFilterFunction:
    def test_fid_filter_nonzero(self):
        omega = np.linspace(0.1, 10, 100)
        F2 = fid_filter(omega, total_time=1.0)
        assert F2.shape == omega.shape
        assert np.all(F2 >= 0)

    def test_spin_echo_suppresses_dc(self):
        omega = np.array([0.01])
        fid = fid_filter(omega, 1.0)
        se = filter_function(DDSequence.spin_echo(), omega, 1.0)
        assert se[0] < fid[0]

    def test_cpmg_filter_shape(self):
        omega = np.linspace(0.1, 50, 500)
        F2 = filter_function(DDSequence.cpmg(4), omega, total_time=5.0)
        assert F2.shape == omega.shape
        assert np.all(F2 >= 0)

    def test_filter_nonnegative(self):
        omega = np.linspace(0.01, 100, 1000)
        for seq in [DDSequence.spin_echo(), DDSequence.xy4(), DDSequence.udd(3)]:
            F2 = filter_function(seq, omega, 1.0)
            assert np.all(F2 >= -1e-10)

    def test_zero_frequency(self):
        omega = np.array([0.0])
        F2 = fid_filter(omega, total_time=1.0)
        assert F2.shape == (1,)

    def test_xy8_better_than_xy4_at_low_freq(self):
        omega = np.array([0.01, 0.05, 0.1])
        xy4_f = filter_function(DDSequence.xy4(), omega, 10.0)
        xy8_f = filter_function(DDSequence.xy8(), omega, 10.0)
        assert np.mean(xy8_f) < np.mean(xy4_f)


# ──────────────────────────────────────────────
# DD fidelity
# ──────────────────────────────────────────────

class TestDDFidelity:
    def _noise_1f(self, omega: np.ndarray) -> np.ndarray:
        return 0.01 / (omega + 1e-10)

    def test_fidelity_in_range(self):
        fid = dd_fidelity(DDSequence.cpmg(4), self._noise_1f, total_time=10.0)
        assert 0.0 <= fid <= 1.0

    def test_dd_better_than_fid(self):
        no_dd = DDSequence(name="fid", pulses=[], spacings=[1.0])
        fid_none = dd_fidelity(no_dd, self._noise_1f, total_time=10.0)
        fid_xy4 = dd_fidelity(DDSequence.xy4(), self._noise_1f, total_time=10.0)
        assert fid_xy4 > fid_none

    def test_more_pulses_better(self):
        fid_2 = dd_fidelity(DDSequence.cpmg(2), self._noise_1f, total_time=10.0)
        fid_4 = dd_fidelity(DDSequence.cpmg(4), self._noise_1f, total_time=10.0)
        assert fid_4 > fid_2

    def test_white_noise_fidelity(self):
        def white(omega: np.ndarray) -> np.ndarray:
            return np.ones_like(omega) * 0.001

        fid = dd_fidelity(DDSequence.xy4(), white, total_time=1.0)
        assert 0.0 <= fid <= 1.0


# ──────────────────────────────────────────────
# Noise hierarchy
# ──────────────────────────────────────────────

class TestNoiseHierarchy:
    def test_xy8_best_under_1f(self):
        def noise_1f(omega: np.ndarray) -> np.ndarray:
            return 0.01 / (omega + 1e-10)

        fid_cpmg = dd_fidelity(DDSequence.cpmg(4), noise_1f, total_time=10.0)
        fid_xy8 = dd_fidelity(DDSequence.xy8(), noise_1f, total_time=10.0)
        assert fid_xy8 > fid_cpmg


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────

class TestEdgeCases:
    def test_num_pulses_property(self):
        assert DDSequence.spin_echo().num_pulses == 1
        assert DDSequence.xy4().num_pulses == 4
        assert DDSequence.xy8().num_pulses == 8
        assert DDSequence.cpmg(6).num_pulses == 6

    def test_custom_sequence(self):
        seq = DDSequence(name="custom", pulses=["X", "X"], spacings=[0.25, 0.5, 0.25])
        assert seq.num_pulses == 2
        assert abs(sum(seq.spacings) - 1.0) < 1e-10

    def test_single_qubit_no_idle(self):
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        sched = DDScheduler(DDSequence.xy4())
        assert sched.count_dd_pulses(qc) == 0

    def test_custom_gate_duration(self):
        durations = {"H": 2, "CNOT": 4, "X": 1, "Y": 1}
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        sched = DDScheduler(DDSequence.spin_echo(), durations=durations)
        windows = sched.find_idle_windows(qc)
        assert 1 in windows


# ──────────────────────────────────────────────
# Integration
# ──────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self):
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(X, 0)
        qc.apply(CNOT, 0, 2)

        seq = DDSequence.cpmg(2)
        sched = DDScheduler(seq)
        new_qc = sched.schedule(qc)
        assert new_qc.num_qubits == 3
        assert len(new_qc.instructions) >= len(qc.instructions)

    def test_filter_then_fidelity(self):
        seq = DDSequence.xy4()
        omega = np.linspace(0.1, 50, 200)
        F2 = filter_function(seq, omega, total_time=5.0)
        assert np.all(F2 >= 0)

        def noise(w: np.ndarray) -> np.ndarray:
            return 0.005 / (w + 0.1)

        fid = dd_fidelity(seq, noise, total_time=5.0)
        assert 0.0 <= fid <= 1.0

    def test_udd_scaling(self):
        def noise_1f(omega: np.ndarray) -> np.ndarray:
            return 0.01 / (omega + 1e-10)

        fids = []
        for n in [2, 4, 6]:
            fid = dd_fidelity(DDSequence.udd(n), noise_1f, total_time=10.0)
            fids.append(fid)
        assert fids[1] > fids[0]
        assert fids[2] > fids[1]
