"""Abstract backend interface for quantum circuit execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..core.circuit import QuantumCircuit


@dataclass
class ExecutionResult:
    """Standardized result from running a quantum circuit."""
    counts: dict[str, int]
    shots: int
    backend_name: str
    metadata: dict = field(default_factory=dict)

    @property
    def probabilities(self) -> dict[str, float]:
        return {k: v / self.shots for k, v in self.counts.items()}

    @property
    def most_frequent(self) -> str:
        return max(self.counts, key=lambda k: self.counts[k])


class Backend(ABC):
    """Abstract base class for quantum execution backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> ExecutionResult:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...


class LocalSimulatorBackend(Backend):
    """Backend using our built-in statevector simulator."""

    @property
    def name(self) -> str:
        return "quantumium-local-sim"

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> ExecutionResult:
        counts = circuit.measure_all(shots=shots)
        return ExecutionResult(
            counts=counts,
            shots=shots,
            backend_name=self.name,
        )

    def is_available(self) -> bool:
        return True
