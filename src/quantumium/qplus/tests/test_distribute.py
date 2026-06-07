"""Tests for Layer 3b: multi-QPU distribution."""

from __future__ import annotations

import numpy as np
import pytest

from quantumium.qplus.distribute import (
    QPUOrchestrator,
    DistributionStrategy,
    DistributedResult,
    QPUPartition,
)


class TestQPUOrchestrator:
    def test_init_default(self):
        orch = QPUOrchestrator()
        assert orch.n_qpus == 2
        assert orch.strategy == DistributionStrategy.CLASSICAL_ONLY

    def test_init_custom(self):
        orch = QPUOrchestrator(n_qpus=4, strategy="locc")
        assert orch.n_qpus == 4
        assert orch.strategy == DistributionStrategy.LOCC

    def test_init_invalid_qpus(self):
        with pytest.raises(ValueError, match="n_qpus must be >= 1"):
            QPUOrchestrator(n_qpus=0)

    def test_partition_single_qpu(self):
        orch = QPUOrchestrator(n_qpus=1)
        plan = orch.partition_circuit(4)
        assert plan.n_qpus == 1
        assert len(plan.cuts) == 0
        assert plan.estimated_overhead == 1.0

    def test_partition_two_qpus(self):
        orch = QPUOrchestrator(n_qpus=2)
        plan = orch.partition_circuit(4)
        assert plan.n_qpus == 2
        assert len(plan.partitions) == 2
        assert len(plan.cuts) == 1

    def test_partition_qubit_coverage(self):
        orch = QPUOrchestrator(n_qpus=2)
        plan = orch.partition_circuit(6)
        all_qubits = []
        for p in plan.partitions:
            all_qubits.extend(p.qubit_indices)
        assert sorted(all_qubits) == list(range(6))

    def test_partition_three_qpus(self):
        orch = QPUOrchestrator(n_qpus=3)
        plan = orch.partition_circuit(6)
        assert plan.n_qpus == 3
        assert len(plan.cuts) == 2

    def test_partition_more_qpus_than_qubits(self):
        orch = QPUOrchestrator(n_qpus=10)
        plan = orch.partition_circuit(3)
        assert plan.n_qpus == 3

    def test_overhead_classical_only(self):
        orch = QPUOrchestrator(n_qpus=2, strategy="classical_only")
        plan = orch.partition_circuit(4)
        assert plan.estimated_overhead == 4.0
        assert plan.communication_rounds == 0

    def test_overhead_locc(self):
        orch = QPUOrchestrator(n_qpus=2, strategy="locc")
        plan = orch.partition_circuit(4)
        assert plan.estimated_overhead == 2.0
        assert plan.communication_rounds == 1

    def test_overhead_teleport(self):
        orch = QPUOrchestrator(n_qpus=2, strategy="teleport")
        plan = orch.partition_circuit(4)
        assert plan.estimated_overhead == 1.0
        assert plan.communication_rounds == 2

    def test_execute_single_qpu(self):
        orch = QPUOrchestrator(n_qpus=1)
        unitary = np.eye(4, dtype=complex)
        result = orch.execute_distributed(unitary, 2)
        assert isinstance(result, DistributedResult)
        assert result.n_qpus == 1
        assert abs(result.recombined_value - 1.0) < 0.01

    def test_execute_distributed_hadamard(self):
        orch = QPUOrchestrator(n_qpus=1)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        result = orch.execute_distributed(H, 1)
        assert isinstance(result, DistributedResult)
        assert abs(result.recombined_value - 0.5) < 0.01

    def test_execute_multi_qpu(self):
        orch = QPUOrchestrator(n_qpus=2)
        unitary = np.eye(4, dtype=complex)
        result = orch.execute_distributed(unitary, 2)
        assert result.n_qpus == 2
        assert result.n_cuts == 1

    def test_estimate_resources(self):
        orch = QPUOrchestrator(n_qpus=2)
        resources = orch.estimate_resources(4)
        assert resources["n_qpus"] == 2.0
        assert resources["n_cuts"] == 1.0
        assert resources["max_qubits_per_qpu"] >= 2.0

    def test_recommend_strategy_single_qpu(self):
        assert QPUOrchestrator.recommend_strategy(4, 1) == "classical_only"

    def test_recommend_strategy_with_quantum_link(self):
        assert QPUOrchestrator.recommend_strategy(4, 2, has_quantum_link=True) == "teleport"

    def test_recommend_strategy_few_cuts(self):
        assert QPUOrchestrator.recommend_strategy(4, 2) == "classical_only"

    def test_recommend_strategy_many_cuts(self):
        result = QPUOrchestrator.recommend_strategy(10, 5)
        assert result in ("classical_only", "locc")


class TestDistributionStrategy:
    def test_enum_values(self):
        assert DistributionStrategy.CLASSICAL_ONLY.value == "classical_only"
        assert DistributionStrategy.LOCC.value == "locc"
        assert DistributionStrategy.TELEPORT.value == "teleport"

    def test_from_string(self):
        assert DistributionStrategy("classical_only") == DistributionStrategy.CLASSICAL_ONLY


class TestQPUPartition:
    def test_partition_creation(self):
        p = QPUPartition(qpu_id=0, qubit_indices=[0, 1], n_qubits=2)
        assert p.qpu_id == 0
        assert p.n_qubits == 2
