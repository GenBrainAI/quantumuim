"""Data classes for logical and physical resource estimates."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LogicalResources:
    """Logical resource counts extracted from a quantum circuit."""

    n_qubits: int
    n_t_gates: int
    t_depth: int
    n_clifford: int
    n_measurements: int
    circuit_depth: int
    gate_counts: dict[str, int] = field(default_factory=dict)
    n_rotation_gates: int = 0
    rotation_t_cost: int = 0

    @property
    def total_t_budget(self) -> int:
        return self.n_t_gates + self.rotation_t_cost

    def summary(self) -> str:
        lines = [
            "Logical Resources",
            f"  Qubits:           {self.n_qubits}",
            f"  T-gates:          {self.n_t_gates}",
            f"  Rotation gates:   {self.n_rotation_gates}",
            f"  Rotation T-cost:  {self.rotation_t_cost}",
            f"  Total T-budget:   {self.total_t_budget}",
            f"  T-depth:          {self.t_depth}",
            f"  Clifford gates:   {self.n_clifford}",
            f"  Measurements:     {self.n_measurements}",
            f"  Circuit depth:    {self.circuit_depth}",
        ]
        if self.gate_counts:
            lines.append("  Gate breakdown:")
            for gate, count in sorted(self.gate_counts.items()):
                lines.append(f"    {gate}: {count}")
        return "\n".join(lines)


@dataclass
class PhysicalEstimate:
    """Physical resource estimate for fault-tolerant execution."""

    logical: LogicalResources
    code_distance: int
    physical_qubits_per_logical: int
    data_qubits: int
    n_factories: int
    factory_qubits: int
    total_qubits: int
    runtime_us: float
    logical_error_rate: float
    physical_error_rate: float
    routing_overhead: float

    @property
    def runtime_seconds(self) -> float:
        return self.runtime_us / 1e6

    @property
    def runtime_hours(self) -> float:
        return self.runtime_us / 3.6e9

    def summary(self) -> str:
        lines = [
            "Physical Resource Estimate",
            f"  Code distance:        d = {self.code_distance}",
            f"  Physical error rate:  {self.physical_error_rate:.1e}",
            f"  Logical error rate:   {self.logical_error_rate:.2e}",
            f"  Qubits per logical:   {self.physical_qubits_per_logical}",
            f"  Data qubits:          {self.data_qubits:,}",
            f"  Magic state factories:{self.n_factories}",
            f"  Factory qubits:       {self.factory_qubits:,}",
            f"  Total physical qubits:{self.total_qubits:,}",
            f"  Runtime:              {self.runtime_seconds:.2f} s "
            f"({self.runtime_hours:.4f} h)",
            f"  Routing overhead:     {self.routing_overhead:.1f}x",
            "",
            self.logical.summary(),
        ]
        return "\n".join(lines)
