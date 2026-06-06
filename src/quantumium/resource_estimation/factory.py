"""Magic state distillation factory cost model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MagicStateFactory:
    """15-to-1 magic state distillation factory.

    Models the cost of producing high-fidelity |T> states from noisy ones
    using the standard 15-to-1 distillation protocol.

    Attributes:
        code_distance: Surface code distance.
        physical_error_rate: Physical qubit error rate.
        n_distillation_rounds: Number of distillation rounds (default 1).
    """

    code_distance: int
    physical_error_rate: float = 1e-3
    n_distillation_rounds: int = 1

    @property
    def input_states(self) -> int:
        return int(15 ** self.n_distillation_rounds)

    @property
    def output_error_rate(self) -> float:
        p = self.physical_error_rate
        for _ in range(self.n_distillation_rounds):
            p = 35.0 * p**3
        return p

    @property
    def qubits_per_factory(self) -> int:
        d = self.code_distance
        qubits_per_logical = 2 * d * d - 1
        return 15 * qubits_per_logical

    @property
    def cycles_per_t_state(self) -> int:
        return 5 * self.code_distance * self.n_distillation_rounds

    def throughput_per_us(self, code_cycle_time_us: float = 1.0) -> float:
        cycle_time = self.cycles_per_t_state * code_cycle_time_us
        if cycle_time <= 0:
            return 0.0
        return 1.0 / cycle_time

    @staticmethod
    def rounds_needed(
        physical_error_rate: float, target_t_error: float
    ) -> int:
        p = physical_error_rate
        rounds = 0
        while p > target_t_error and rounds < 10:
            p = 35.0 * p**3
            rounds += 1
        return max(1, rounds)

    def summary(self) -> str:
        return (
            f"MagicStateFactory(d={self.code_distance}, "
            f"rounds={self.n_distillation_rounds})\n"
            f"  Qubits per factory:   {self.qubits_per_factory:,}\n"
            f"  Output error rate:    {self.output_error_rate:.2e}\n"
            f"  Cycles per T-state:   {self.cycles_per_t_state}\n"
            f"  Input states needed:  {self.input_states}"
        )
