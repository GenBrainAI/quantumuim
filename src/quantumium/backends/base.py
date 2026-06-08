"""Abstract backend interface for quantum circuit execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..core.circuit import QuantumCircuit


@dataclass
class ExecutionResult:
    """Standardized result from running a quantum circuit.

    Encapsulates measurement outcomes and metadata returned by any backend,
    providing convenience accessors for probabilities and the most likely
    bitstring.

    Attributes:
        counts: Mapping of measured bitstrings to the number of times each
            was observed (e.g., ``{"00": 512, "11": 512}``).
        shots: Total number of measurement repetitions performed.
        backend_name: Identifier of the backend that produced this result.
        metadata: Arbitrary backend-specific metadata (device ARN, timing
            information, etc.).
    """

    counts: dict[str, int]
    shots: int
    backend_name: str
    metadata: dict = field(default_factory=dict)  # type: ignore[type-arg]

    @property
    def probabilities(self) -> dict[str, float]:
        """Compute the empirical probability distribution over bitstrings.

        Returns:
            dict[str, float]: Mapping of each observed bitstring to its
                relative frequency (count / total shots).
        """
        return {k: v / self.shots for k, v in self.counts.items()}

    @property
    def most_frequent(self) -> str:
        """Return the bitstring that was observed most often.

        Returns:
            str: The bitstring with the highest count. Ties are broken
                arbitrarily.
        """
        return max(self.counts, key=lambda k: self.counts[k])


class Backend(ABC):
    """Abstract base class for quantum execution backends.

    Every concrete backend must implement three members:

    * ``name`` -- a human-readable identifier for the backend.
    * ``run`` -- translate and execute a Quantumium circuit.
    * ``is_available`` -- report whether the required SDK is installed.

    Subclasses typically translate a :class:`QuantumCircuit` into the
    vendor-specific circuit representation, dispatch execution to either a
    local simulator or cloud hardware, and wrap the raw results in an
    :class:`ExecutionResult`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable name of this backend.

        Returns:
            str: Backend identifier (e.g., ``"qiskit-aer-simulator"``).
        """
        ...

    @abstractmethod
    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> ExecutionResult:
        """Execute a quantum circuit on this backend.

        Args:
            circuit (QuantumCircuit): The Quantumium circuit to execute.
            shots (int): Number of measurement repetitions. Defaults to 1024.

        Returns:
            ExecutionResult: Measurement counts and associated metadata.

        Raises:
            RuntimeError: If the backend SDK is not installed or the device
                is unreachable.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the backend's SDK is installed and usable.

        Returns:
            bool: ``True`` if the backend can accept circuits for execution,
                ``False`` otherwise.
        """
        ...


class LocalSimulatorBackend(Backend):
    """Backend using Quantumium's built-in statevector simulator.

    This backend requires no external dependencies and is always available.
    It delegates measurement to :meth:`QuantumCircuit.measure_all`, which
    performs ideal statevector simulation followed by probabilistic sampling.
    """

    @property
    def name(self) -> str:
        """Return the backend name.

        Returns:
            str: Always ``"quantumium-local-sim"``.
        """
        return "quantumium-local-sim"

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> ExecutionResult:
        """Simulate the circuit using the built-in statevector engine.

        Args:
            circuit (QuantumCircuit): The circuit to simulate.
            shots (int): Number of measurement samples to draw. Defaults to 1024.

        Returns:
            ExecutionResult: Sampled measurement counts from ideal simulation.
        """
        counts = circuit.measure_all(shots=shots)
        return ExecutionResult(
            counts=counts,
            shots=shots,
            backend_name=self.name,
        )

    def is_available(self) -> bool:
        """Check availability of the local simulator.

        Returns:
            bool: Always ``True`` since no external SDK is required.
        """
        return True
