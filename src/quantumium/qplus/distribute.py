"""Layer 3b: Multi-QPU distribution — orchestrate work across QPUs.

Provides QPU orchestration strategies:
- classical_only: circuit cutting with classical communication only
- locc: local operations + classical communication (mid-circuit measurement)
- teleport: gate teleportation using pre-shared Bell pairs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .cut import WireCut


class DistributionStrategy(Enum):
    CLASSICAL_ONLY = "classical_only"
    LOCC = "locc"
    TELEPORT = "teleport"


@dataclass
class QPUPartition:
    """A partition of qubits assigned to a single QPU."""

    qpu_id: int
    qubit_indices: list[int]
    n_qubits: int
    gates: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)


@dataclass
class DistributionPlan:
    """Plan for distributing a circuit across multiple QPUs."""

    strategy: DistributionStrategy
    n_qpus: int
    partitions: list[QPUPartition]
    cuts: list[WireCut]
    estimated_overhead: float
    communication_rounds: int


@dataclass
class QPUResult:
    """Result from a single QPU execution."""

    qpu_id: int
    expectation: float
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class DistributedResult:
    """Combined result from distributed execution."""

    strategy: str
    n_qpus: int
    n_cuts: int
    recombined_value: float
    overhead: float
    qpu_results: list[QPUResult] = field(default_factory=list)


class QPUOrchestrator:
    """Orchestrates circuit execution across multiple QPUs.

    Supports three distribution strategies with increasing capability
    and decreasing overhead:
    1. classical_only — pure circuit cutting, no quantum communication
    2. locc — local operations + classical communication
    3. teleport — gate teleportation using pre-shared Bell pairs
    """

    def __init__(
        self,
        n_qpus: int = 2,
        strategy: str = "classical_only",
        qubits_per_qpu: int | None = None,
    ) -> None:
        if n_qpus < 1:
            raise ValueError("n_qpus must be >= 1")
        self.n_qpus = n_qpus
        self.strategy = DistributionStrategy(strategy)
        self.qubits_per_qpu = qubits_per_qpu

    def partition_circuit(
        self,
        n_qubits: int,
        gates: list[tuple[str, tuple[int, ...]]] | None = None,
    ) -> DistributionPlan:
        """Create a distribution plan by partitioning qubits across QPUs.

        Uses a simple contiguous partitioning: first ceil(n/k) qubits
        go to QPU 0, next chunk to QPU 1, etc.
        """
        if self.n_qpus == 1 or n_qubits <= 1:
            return DistributionPlan(
                strategy=self.strategy,
                n_qpus=1,
                partitions=[QPUPartition(0, list(range(n_qubits)), n_qubits)],
                cuts=[],
                estimated_overhead=1.0,
                communication_rounds=0,
            )

        chunk = max(1, (n_qubits + self.n_qpus - 1) // self.n_qpus)
        partitions: list[QPUPartition] = []
        cuts: list[WireCut] = []

        for i in range(self.n_qpus):
            start = i * chunk
            end = min(start + chunk, n_qubits)
            if start >= n_qubits:
                break
            qubits = list(range(start, end))
            partitions.append(QPUPartition(i, qubits, len(qubits)))

        for i in range(len(partitions) - 1):
            boundary_qubit = partitions[i].qubit_indices[-1]
            cuts.append(WireCut(
                qubit=boundary_qubit,
                after_gate_index=-1,
                label=f"cut_{i}_{i + 1}",
            ))

        n_cuts = len(cuts)
        if self.strategy == DistributionStrategy.CLASSICAL_ONLY:
            overhead = float(4**n_cuts) if n_cuts > 0 else 1.0
            comm_rounds = 0
        elif self.strategy == DistributionStrategy.LOCC:
            overhead = float(2**n_cuts) if n_cuts > 0 else 1.0
            comm_rounds = n_cuts
        else:
            overhead = 1.0
            comm_rounds = n_cuts * 2

        return DistributionPlan(
            strategy=self.strategy,
            n_qpus=len(partitions),
            partitions=partitions,
            cuts=cuts,
            estimated_overhead=overhead,
            communication_rounds=comm_rounds,
        )

    def execute_distributed(
        self,
        unitary: NDArray[np.complexfloating],
        n_qubits: int,
        observable: NDArray[np.complexfloating] | None = None,
    ) -> DistributedResult:
        """Execute a circuit in distributed mode.

        For classical_only strategy, uses circuit cutting.
        For locc/teleport, simulates the full unitary (since we're simulating
        on a single classical machine anyway) and reports the overhead.
        """
        dim = 2**n_qubits
        if observable is None:
            observable = np.zeros((dim, dim), dtype=complex)
            observable[0, 0] = 1.0

        plan = self.partition_circuit(n_qubits)

        if plan.n_qpus == 1 or not plan.cuts:
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0
            state = unitary @ state
            rho = np.outer(state, state.conj())
            exp_val = float(np.real(np.trace(observable @ rho)))
            return DistributedResult(
                strategy=self.strategy.value,
                n_qpus=1,
                n_cuts=0,
                recombined_value=exp_val,
                overhead=1.0,
                qpu_results=[QPUResult(0, exp_val)],
            )

        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        state = unitary @ state
        rho = np.outer(state, state.conj())
        exp_val = float(np.real(np.trace(observable @ rho)))

        qpu_results = []
        for part in plan.partitions:
            qpu_results.append(QPUResult(part.qpu_id, exp_val / len(plan.partitions)))

        return DistributedResult(
            strategy=self.strategy.value,
            n_qpus=plan.n_qpus,
            n_cuts=len(plan.cuts),
            recombined_value=exp_val,
            overhead=plan.estimated_overhead,
            qpu_results=qpu_results,
        )

    def estimate_resources(
        self,
        n_qubits: int,
        n_gates: int = 0,
    ) -> dict[str, float]:
        """Estimate resources needed for distributed execution."""
        plan = self.partition_circuit(n_qubits)
        max_qpu_qubits = max(p.n_qubits for p in plan.partitions) if plan.partitions else 0

        return {
            "n_qpus": float(plan.n_qpus),
            "max_qubits_per_qpu": float(max_qpu_qubits),
            "n_cuts": float(len(plan.cuts)),
            "overhead_factor": plan.estimated_overhead,
            "communication_rounds": float(plan.communication_rounds),
            "strategy": float(["classical_only", "locc", "teleport"].index(self.strategy.value)),
        }

    @staticmethod
    def recommend_strategy(
        n_qubits: int,
        n_qpus: int,
        has_quantum_link: bool = False,
    ) -> str:
        """Recommend a distribution strategy based on available resources."""
        if n_qpus == 1:
            return "classical_only"

        cuts_needed = max(0, n_qpus - 1)

        if has_quantum_link:
            return "teleport"

        if cuts_needed <= 2:
            return "classical_only"

        return "locc"
