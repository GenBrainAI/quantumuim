"""Backend connectors for quantum hardware and simulators.

This package provides a unified interface for executing Quantumium quantum
circuits on a variety of backends, including local simulators and cloud-hosted
quantum hardware from IBM, Google, and Amazon.

Exported classes:
    Backend: Abstract base class defining the backend interface.
    ExecutionResult: Standardized container for circuit execution results.
    LocalSimulatorBackend: Built-in statevector simulator backend.
"""

from .base import Backend, ExecutionResult, LocalSimulatorBackend

__all__ = [
    "Backend",
    "ExecutionResult",
    "LocalSimulatorBackend",
]
