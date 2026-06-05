"""Quantum benchmarking -- hardware characterization and performance metrics.

This package provides standard benchmarking protocols for characterizing
quantum hardware quality, including randomized benchmarking for gate error
rates, quantum volume estimation, and fidelity metrics for states and
processes.

Modules:
    randomized: Standard single-qubit randomized benchmarking (RB) protocol.
    quantum_volume: Quantum volume (QV) estimation via random model circuits.
    fidelity: State fidelity, process fidelity, and average gate fidelity metrics.
"""

from .randomized import randomized_benchmarking, RBResult
from .quantum_volume import quantum_volume, QVResult
from .fidelity import state_fidelity, process_fidelity, average_gate_fidelity

__all__ = [
    "randomized_benchmarking",
    "RBResult",
    "quantum_volume",
    "QVResult",
    "state_fidelity",
    "process_fidelity",
    "average_gate_fidelity",
]
