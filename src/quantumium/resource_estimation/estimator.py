"""ResourceEstimator — end-to-end physical resource estimation."""

from __future__ import annotations

import math

from quantumium.core.circuit import QuantumCircuit
from quantumium.resource_estimation.analyzer import CircuitAnalyzer
from quantumium.resource_estimation.factory import MagicStateFactory
from quantumium.resource_estimation.resources import PhysicalEstimate

P_THRESHOLD = 0.01


class ResourceEstimator:
    """End-to-end resource estimation from circuit to physical requirements.

    Computes surface code distance, physical qubit count, magic state
    factory allocation, and runtime for fault-tolerant execution.
    """

    def __init__(
        self,
        physical_error_rate: float = 1e-3,
        code_cycle_time_us: float = 1.0,
        routing_overhead: float = 1.5,
        rotation_precision: float = 1e-10,
    ) -> None:
        self.physical_error_rate = physical_error_rate
        self.code_cycle_time_us = code_cycle_time_us
        self.routing_overhead = routing_overhead
        self.analyzer = CircuitAnalyzer(rotation_precision=rotation_precision)

    def code_distance(
        self, target_error: float, n_logical_ops: int
    ) -> int:
        """Compute required surface code distance.

        Uses: p_L ≈ 0.1 * (p_phys / p_threshold)^((d+1)/2)
        The per-operation error must be target_error / n_logical_ops.
        """
        if n_logical_ops < 1:
            n_logical_ops = 1
        per_op_error = target_error / n_logical_ops

        ratio = self.physical_error_rate / P_THRESHOLD
        if ratio >= 1.0:
            return 3

        log_ratio = math.log(ratio)
        target_val = per_op_error / 0.1
        if target_val <= 0:
            return 3

        exponent = math.log(target_val) / log_ratio
        d_float = 2.0 * exponent - 1.0
        d = int(math.ceil(d_float))
        if d < 3:
            d = 3
        if d % 2 == 0:
            d += 1
        return d

    def factory_count(
        self,
        t_count: int,
        t_depth: int,
        code_distance: int,
    ) -> int:
        """How many magic state factories are needed for throughput matching."""
        if t_count == 0:
            return 0
        if t_depth == 0:
            return 1
        factories = math.ceil(t_count / t_depth)
        return max(1, factories)

    def estimate(
        self,
        circuit: QuantumCircuit,
        target_error: float = 1e-6,
    ) -> PhysicalEstimate:
        logical = self.analyzer.analyze(circuit)
        total_t = logical.total_t_budget
        n_ops = logical.circuit_depth + total_t
        if n_ops < 1:
            n_ops = 1

        d = self.code_distance(target_error, n_ops)

        qubits_per_logical = 2 * d * d - 1
        data_qubits = int(
            math.ceil(logical.n_qubits * qubits_per_logical * self.routing_overhead)
        )

        n_factories = self.factory_count(total_t, logical.t_depth, d)
        factory = MagicStateFactory(
            code_distance=d,
            physical_error_rate=self.physical_error_rate,
        )
        factory_qubits = n_factories * factory.qubits_per_factory

        total_qubits = data_qubits + factory_qubits

        circuit_time = logical.circuit_depth * d * self.code_cycle_time_us
        if n_factories > 0 and total_t > 0:
            t_time = (total_t / n_factories) * factory.cycles_per_t_state * self.code_cycle_time_us
        else:
            t_time = 0.0
        runtime_us = max(circuit_time, t_time)

        ratio = self.physical_error_rate / P_THRESHOLD
        logical_error_per_op = 0.1 * ratio ** ((d + 1) / 2.0)
        total_logical_error = logical_error_per_op * n_ops

        return PhysicalEstimate(
            logical=logical,
            code_distance=d,
            physical_qubits_per_logical=qubits_per_logical,
            data_qubits=data_qubits,
            n_factories=n_factories,
            factory_qubits=factory_qubits,
            total_qubits=total_qubits,
            runtime_us=runtime_us,
            logical_error_rate=total_logical_error,
            physical_error_rate=self.physical_error_rate,
            routing_overhead=self.routing_overhead,
        )
