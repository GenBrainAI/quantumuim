"""Tests for Layer 2: qubit recycling and feedforward."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quantumium.qplus.recycle import (
    QubitRecycler,
    FeedForwardController,
    RecycleEvent,
)
from quantumium.qplus.compress import ModularArithmeticOptimizer


class TestFeedForwardController:
    def test_record_and_get_measurement(self) -> None:
        ffc = FeedForwardController()
        ffc.record_measurement(0, 1)
        assert ffc.get_measurement(0) == 1
        assert ffc.get_measurement(1) is None

    def test_measurements_property(self) -> None:
        ffc = FeedForwardController()
        ffc.record_measurement(0, 0)
        ffc.record_measurement(1, 1)
        assert ffc.measurements == {0: 0, 1: 1}

    def test_add_correction(self) -> None:
        ffc = FeedForwardController()
        ffc.add_correction(0, 1, "phase", np.pi / 4)
        corrections = ffc.get_corrections(1)
        assert len(corrections) == 1
        assert corrections[0].operation == "phase"
        assert abs(corrections[0].angle - np.pi / 4) < 1e-10

    def test_get_corrections_empty(self) -> None:
        ffc = FeedForwardController()
        assert ffc.get_corrections(0) == []

    def test_compute_phase_correction_no_bits(self) -> None:
        ffc = FeedForwardController()
        correction = ffc.compute_phase_correction(0, [])
        assert abs(correction) < 1e-15

    def test_compute_phase_correction_with_bits(self) -> None:
        ffc = FeedForwardController()
        correction = ffc.compute_phase_correction(2, [1, 0])
        expected = -np.pi / 4  # -pi / 2^(2-0) = -pi/4
        assert abs(correction - expected) < 1e-10

    def test_compute_phase_correction_all_ones(self) -> None:
        ffc = FeedForwardController()
        correction = ffc.compute_phase_correction(3, [1, 1, 1])
        expected = -np.pi / 8 - np.pi / 4 - np.pi / 2
        assert abs(correction - expected) < 1e-10

    def test_reset(self) -> None:
        ffc = FeedForwardController()
        ffc.record_measurement(0, 1)
        ffc.add_correction(0, 1, "phase", 0.5)
        ffc.reset()
        assert ffc.measurements == {}
        assert ffc.get_corrections(1) == []


class TestQubitRecycler:
    def test_init(self) -> None:
        qr = QubitRecycler(3)
        assert qr.n_physical == 3
        assert qr.available_count == 3
        assert qr.in_use_count == 0

    def test_init_invalid(self) -> None:
        with pytest.raises(ValueError, match="n_physical must be >= 1"):
            QubitRecycler(0)

    def test_allocate(self) -> None:
        qr = QubitRecycler(3)
        phys = qr.allocate()
        assert phys == 0
        assert qr.available_count == 2
        assert qr.in_use_count == 1

    def test_allocate_exhaustion(self) -> None:
        qr = QubitRecycler(2)
        qr.allocate()
        qr.allocate()
        with pytest.raises(RuntimeError, match="No physical qubits available"):
            qr.allocate()

    def test_measure_and_recycle_zero_state(self) -> None:
        qr = QubitRecycler(2)
        phys = qr.allocate()
        state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        outcome, new_state = qr.measure_and_recycle(phys, state, 0, 2)
        assert outcome == 0
        assert qr.available_count == 2
        assert qr.total_recycles == 1

    def test_measure_and_recycle_one_state(self) -> None:
        qr = QubitRecycler(2)
        phys = qr.allocate()
        state = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩
        outcome, new_state = qr.measure_and_recycle(phys, state, 0, 2)
        assert outcome == 1
        assert qr.total_recycles == 1

    def test_recycle_events(self) -> None:
        qr = QubitRecycler(2)
        phys = qr.allocate()
        state = np.array([1, 0, 0, 0], dtype=complex)
        qr.measure_and_recycle(phys, state, 0, 2)
        events = qr.events
        assert len(events) == 1
        assert isinstance(events[0], RecycleEvent)
        assert events[0].qubit_id == phys

    def test_allocate_after_recycle(self) -> None:
        qr = QubitRecycler(1)
        phys = qr.allocate()
        state = np.array([1, 0], dtype=complex)
        qr.measure_and_recycle(phys, state, 0, 1)
        phys2 = qr.allocate()
        assert phys2 == phys

    def test_feedforward_property(self) -> None:
        qr = QubitRecycler(2)
        assert isinstance(qr.feedforward, FeedForwardController)

    def test_recycled_phase_estimation_identity(self) -> None:
        """Identity unitary should give phase ~0."""
        qr = QubitRecycler(3)

        def ctrl_u(power: int, state: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
            return np.array(state, dtype=complex)

        phase, bits = qr.simulate_recycled_phase_estimation(ctrl_u, 2, 4)
        assert 0.0 <= phase < 1.0
        assert len(bits) == 4

    def test_recycled_phase_estimation_z_gate(self) -> None:
        """Z eigenvalue -1 -> phase 0.5."""
        results: list[float] = []
        for _ in range(20):
            qr = QubitRecycler(2)

            def ctrl_z(
                power: int, state: NDArray[np.complexfloating]
            ) -> NDArray[np.complexfloating]:
                out = np.array(state, dtype=complex)
                out[1] *= (-1) ** power
                return out

            phase, bits = qr.simulate_recycled_phase_estimation(ctrl_z, 1, 5)
            results.append(phase)
        mean_phase = np.mean(results)
        assert abs(mean_phase - 0.5) < 0.2

    def test_recycled_order_finding_15(self) -> None:
        """Order of 7 mod 15 should be 4."""
        from quantumium.qplus.compress import _extract_order

        n_bits = 4
        successes = 0
        for _ in range(30):
            qr = QubitRecycler(n_bits + 1)

            def ctrl_u(
                power: int, state: NDArray[np.complexfloating]
            ) -> NDArray[np.complexfloating]:
                perm = ModularArithmeticOptimizer.mod_exp_permutation(7, power, 15, n_bits)
                return perm @ state

            phase, bits = qr.simulate_recycled_phase_estimation(ctrl_u, n_bits, 9)
            order = _extract_order(phase, 7, 15)
            if order == 4:
                successes += 1
        assert successes >= 3, f"Only {successes}/30 found order 4"
