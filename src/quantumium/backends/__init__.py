"""Backend connectors for quantum hardware and simulators."""

from .base import Backend, ExecutionResult, LocalSimulatorBackend

__all__ = [
    "Backend",
    "ExecutionResult",
    "LocalSimulatorBackend",
]
