"""DDScheduler — insert DD sequences into circuit idle windows."""

from __future__ import annotations

from quantumium.core.circuit import QuantumCircuit, Instruction
from quantumium.core.gate import Gate, X, Y, Z
from quantumium.dd.sequences import DDSequence


DEFAULT_DURATIONS: dict[str, int] = {
    "I": 1,
    "X": 1,
    "Y": 1,
    "Z": 0,
    "H": 1,
    "S": 1,
    "T": 1,
    "RX": 1,
    "RY": 1,
    "RZ": 0,
    "CNOT": 2,
    "CZ": 2,
    "SWAP": 6,
    "MEASURE": 1,
}

_GATE_MAP: dict[str, Gate] = {
    "X": X,
    "Y": Y,
    "Z": Z,
}


class DDScheduler:
    """Insert dynamical decoupling sequences into circuit idle windows.

    The scheduler identifies idle periods on each qubit (gaps where no
    gate acts on that qubit while other qubits are busy) and fills them
    with DD pulse sequences.
    """

    def __init__(
        self,
        sequence: DDSequence,
        min_idle_duration: int = 2,
        durations: dict[str, int] | None = None,
    ) -> None:
        self.sequence = sequence
        self.min_idle_duration = min_idle_duration
        self.durations = durations or dict(DEFAULT_DURATIONS)

    def _gate_duration(self, gate_name: str) -> int:
        return self.durations.get(gate_name, 1)

    def _compute_schedule(
        self, circuit: QuantumCircuit
    ) -> tuple[list[tuple[int, int, Instruction]], list[list[tuple[int, int]]]]:
        """Compute start/end times for each instruction and idle windows per qubit.

        Returns:
            (scheduled_instructions, idle_windows)
            scheduled_instructions: [(start, end, instruction), ...]
            idle_windows: per-qubit list of (start, end) idle intervals
        """
        qubit_available: list[int] = [0] * circuit.num_qubits
        scheduled: list[tuple[int, int, Instruction]] = []

        for inst in circuit.instructions:
            start = max(qubit_available[q] for q in inst.target_qubits)
            duration = self._gate_duration(inst.gate.name)
            end = start + duration
            scheduled.append((start, end, inst))
            for q in inst.target_qubits:
                qubit_available[q] = end

        total_time = max(qubit_available) if qubit_available else 0

        qubit_events: list[list[tuple[int, int]]] = [[] for _ in range(circuit.num_qubits)]
        for start, end, inst in scheduled:
            for q in inst.target_qubits:
                qubit_events[q].append((start, end))

        idle_windows: list[list[tuple[int, int]]] = []
        for q in range(circuit.num_qubits):
            events = sorted(qubit_events[q])
            windows: list[tuple[int, int]] = []
            cursor = 0
            for start, end in events:
                if start > cursor:
                    windows.append((cursor, start))
                cursor = max(cursor, end)
            if cursor < total_time:
                windows.append((cursor, total_time))
            idle_windows.append(windows)

        return scheduled, idle_windows

    def schedule(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Insert DD pulses into idle windows of the circuit.

        Returns a new circuit with DD sequences inserted during idle periods.
        The original circuit is not modified.
        """
        scheduled, idle_windows = self._compute_schedule(circuit)
        total_time = 0
        for s, e, _ in scheduled:
            total_time = max(total_time, e)

        dd_insertions: list[tuple[int, int, str]] = []

        for qubit in range(circuit.num_qubits):
            for win_start, win_end in idle_windows[qubit]:
                window_len = win_end - win_start
                if window_len < self.min_idle_duration:
                    continue
                if window_len < self.sequence.num_pulses:
                    continue

                pulse_times = self.sequence.pulse_times()
                for i, rel_time in enumerate(pulse_times):
                    abs_time = win_start + rel_time * window_len
                    dd_insertions.append((int(round(abs_time)), qubit, self.sequence.pulses[i]))

        all_events: list[tuple[int, int, str, tuple[int, ...]]] = []
        for start, end, inst in scheduled:
            all_events.append((start, end, inst.gate.name, inst.target_qubits))
        for t, q, gate_name in dd_insertions:
            all_events.append((t, t + 1, gate_name, (q,)))

        all_events.sort(key=lambda e: (e[0], e[1]))

        new_circuit = QuantumCircuit(circuit.num_qubits)
        for _, _, gate_name, qubits in all_events:
            if gate_name in _GATE_MAP:
                gate = _GATE_MAP[gate_name]
            else:
                orig_inst = None
                for s, e, inst in scheduled:
                    if inst.gate.name == gate_name and inst.target_qubits == qubits:
                        orig_inst = inst
                        break
                if orig_inst is not None:
                    gate = orig_inst.gate
                else:
                    continue
            new_circuit.apply(gate, *qubits)

        return new_circuit

    def count_dd_pulses(self, circuit: QuantumCircuit) -> int:
        """Count how many DD pulses would be inserted."""
        _, idle_windows = self._compute_schedule(circuit)
        count = 0
        for qubit in range(circuit.num_qubits):
            for win_start, win_end in idle_windows[qubit]:
                window_len = win_end - win_start
                if window_len < self.min_idle_duration:
                    continue
                if window_len < self.sequence.num_pulses:
                    continue
                count += self.sequence.num_pulses
        return count

    def find_idle_windows(self, circuit: QuantumCircuit) -> dict[int, list[tuple[int, int]]]:
        """Return idle windows per qubit for inspection."""
        _, idle_windows = self._compute_schedule(circuit)
        return {q: windows for q, windows in enumerate(idle_windows)}
