"""Layer 2: Qubit recycling — measure, reset, and reuse qubits.

Implements the measure-reset-reuse pattern that enables sequential
circuit execution with fewer physical qubits. Combined with semiclassical
QFT, reduces Shor's algorithm from 2n+3 to n+1 qubits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RecycleEvent:
    """Record of a qubit recycle operation."""

    qubit_id: int
    measured_value: int
    recycled_at_step: int
    reused_at_step: int | None = None


@dataclass
class FeedForwardOp:
    """A classically-controlled operation based on a measurement outcome."""

    source_qubit: int
    target_qubit: int
    operation: str
    angle: float = 0.0


class FeedForwardController:
    """Manages classical feedforward for mid-circuit measurements.

    Tracks measurement outcomes and determines corrective operations
    to apply on subsequent qubits based on prior measurements.
    """

    def __init__(self) -> None:
        self._measurements: dict[int, int] = {}
        self._pending_corrections: list[FeedForwardOp] = []

    @property
    def measurements(self) -> dict[int, int]:
        return dict(self._measurements)

    def record_measurement(self, qubit: int, outcome: int) -> None:
        """Record a measurement outcome for feedforward."""
        self._measurements[qubit] = outcome

    def get_measurement(self, qubit: int) -> int | None:
        """Get the recorded measurement for a qubit."""
        return self._measurements.get(qubit)

    def add_correction(
        self,
        source_qubit: int,
        target_qubit: int,
        operation: str = "phase",
        angle: float = 0.0,
    ) -> None:
        """Schedule a classically-controlled correction."""
        self._pending_corrections.append(
            FeedForwardOp(source_qubit, target_qubit, operation, angle)
        )

    def get_corrections(self, target_qubit: int) -> list[FeedForwardOp]:
        """Get all pending corrections for a target qubit."""
        return [c for c in self._pending_corrections if c.target_qubit == target_qubit]

    def compute_phase_correction(self, target_step: int, measured_bits: list[int]) -> float:
        """Compute cumulative phase correction from prior measurements.

        For semiclassical QFT, qubit k needs correction:
        θ = -Σ_{j<k} m_j · π/2^(k-j)
        where m_j is the measurement outcome of qubit j.
        """
        correction = 0.0
        for j, bit in enumerate(measured_bits):
            if bit:
                correction -= np.pi / (2 ** (target_step - j))
        return correction

    def reset(self) -> None:
        """Clear all recorded measurements and corrections."""
        self._measurements.clear()
        self._pending_corrections.clear()


class QubitRecycler:
    """Manages qubit recycling: measure → reset → reuse.

    Tracks a pool of logical qubits and their lifecycle. When a qubit is
    measured and no longer needed, it can be reset to |0⟩ and reassigned
    to a new logical role.
    """

    def __init__(self, n_physical: int) -> None:
        if n_physical < 1:
            raise ValueError("n_physical must be >= 1")
        self.n_physical = n_physical
        self._available: list[int] = list(range(n_physical))
        self._in_use: dict[int, int] = {}
        self._events: list[RecycleEvent] = []
        self._step: int = 0
        self._feedforward = FeedForwardController()

    @property
    def feedforward(self) -> FeedForwardController:
        return self._feedforward

    @property
    def available_count(self) -> int:
        return len(self._available)

    @property
    def in_use_count(self) -> int:
        return len(self._in_use)

    @property
    def events(self) -> list[RecycleEvent]:
        return list(self._events)

    @property
    def total_recycles(self) -> int:
        return len(self._events)

    def allocate(self, logical_id: int | None = None) -> int:
        """Allocate a physical qubit.

        Returns the physical qubit index. Raises if no qubits available.
        """
        if not self._available:
            raise RuntimeError("No physical qubits available for allocation")
        phys = self._available.pop(0)
        lid = logical_id if logical_id is not None else phys
        self._in_use[phys] = lid
        return phys

    def measure_and_recycle(
        self,
        phys_qubit: int,
        state: NDArray[np.complexfloating],
        qubit_index: int,
        n_total: int,
    ) -> tuple[int, NDArray[np.complexfloating]]:
        """Measure a qubit, collapse state, reset to |0⟩, and return to pool.

        Parameters
        ----------
        phys_qubit
            The physical qubit ID being recycled.
        state
            The full statevector of the system.
        qubit_index
            Index of this qubit in the statevector (position in tensor product).
        n_total
            Total number of qubits in the system.

        Returns
        -------
        (measurement_outcome, collapsed_state)
        """
        dim = 2**n_total
        prob_0 = 0.0
        for i in range(dim):
            if not ((i >> (n_total - 1 - qubit_index)) & 1):
                prob_0 += abs(state[i]) ** 2

        outcome = 0 if np.random.random() < prob_0 else 1

        new_state = np.zeros_like(state)
        for i in range(dim):
            bit = (i >> (n_total - 1 - qubit_index)) & 1
            if bit == outcome:
                new_state[i] = state[i]

        norm = np.linalg.norm(new_state)
        if norm > 1e-15:
            new_state /= norm

        if outcome == 1:
            new_state = self._flip_qubit(new_state, qubit_index, n_total)

        self._feedforward.record_measurement(phys_qubit, outcome)

        event = RecycleEvent(
            qubit_id=phys_qubit,
            measured_value=outcome,
            recycled_at_step=self._step,
        )
        self._events.append(event)
        self._step += 1

        if phys_qubit in self._in_use:
            del self._in_use[phys_qubit]
        self._available.append(phys_qubit)

        return outcome, new_state

    def _flip_qubit(
        self,
        state: NDArray[np.complexfloating],
        qubit_index: int,
        n_total: int,
    ) -> NDArray[np.complexfloating]:
        """Apply X gate to reset qubit from |1⟩ to |0⟩ in statevector."""
        dim = 2**n_total
        new_state = np.zeros_like(state)
        bit_mask = 1 << (n_total - 1 - qubit_index)
        for i in range(dim):
            j = i ^ bit_mask
            new_state[j] = state[i]
        return new_state

    def simulate_recycled_phase_estimation(
        self,
        controlled_u_fn: Callable[[int, NDArray[np.complexfloating]], NDArray[np.complexfloating]],
        n_work: int,
        n_counting_bits: int,
    ) -> tuple[float, list[int]]:
        """Run phase estimation using 1 recycled counting qubit.

        Sequentially: prepare |+⟩, apply C-U^(2^k), correct phases
        from prior measurements, measure, recycle.

        Parameters
        ----------
        controlled_u_fn
            Callable(power, work_state) -> work_state.
        n_work
            Number of work qubits.
        n_counting_bits
            Number of phase bits to extract.

        Returns
        -------
        (phase, measured_bits)
        """
        dim_work = 2**n_work
        work_state = np.zeros(dim_work, dtype=complex)
        work_state[1] = 1.0

        measured_bits: list[int] = []
        self._feedforward.reset()

        for k in range(n_counting_bits - 1, -1, -1):
            power = 2**k

            state_0_work = work_state / np.sqrt(2)
            state_1_work = controlled_u_fn(power, work_state) / np.sqrt(2)

            correction = self._feedforward.compute_phase_correction(
                len(measured_bits), measured_bits
            )
            if abs(correction) > 1e-15:
                state_1_work *= np.exp(1j * correction)

            prob_0 = float(np.sum(np.abs(state_0_work) ** 2))
            prob_1 = float(np.sum(np.abs(state_1_work) ** 2))
            total = prob_0 + prob_1
            if total > 0:
                prob_0 /= total

            bit = int(np.random.random() >= prob_0)
            measured_bits.append(bit)

            if bit == 0 and prob_0 > 1e-15:
                work_state = state_0_work * np.sqrt(2)
            elif prob_1 > 1e-15:
                work_state = state_1_work * np.sqrt(2)

            self._feedforward.record_measurement(len(measured_bits) - 1, bit)
            self._step += 1

        measured_bits.reverse()
        phase = sum(b * 2 ** -(i + 1) for i, b in enumerate(measured_bits))
        return phase, measured_bits
