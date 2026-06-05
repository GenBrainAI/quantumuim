"""Quantum optimal control — pulse-level gate synthesis.

Implements GRAPE (Gradient Ascent Pulse Engineering) and related
algorithms for finding control pulses that realize target unitaries
on quantum hardware.

References:
    Khaneja et al., "Optimal control of coupled spin dynamics"
    (Journal of Magnetic Resonance 172, 296-305, 2005)
    de Fouquières et al., "Second order gradient ascent pulse engineering"
    (Journal of Magnetic Resonance 212, 412-417, 2011)
"""

from .grape import (
    GRAPEResult,
    grape_optimize,
)
from .hamiltonians import (
    ControlHamiltonian,
    drift_hamiltonian,
    control_hamiltonian_x,
    control_hamiltonian_y,
    transmon_hamiltonian,
)
from .pulses import (
    PulseSequence,
    constant_pulse,
    gaussian_pulse,
    drag_pulse,
)
from .fidelity import (
    gate_fidelity,
    process_fidelity,
    propagator,
)

__all__ = [
    "GRAPEResult",
    "grape_optimize",
    "ControlHamiltonian",
    "drift_hamiltonian",
    "control_hamiltonian_x",
    "control_hamiltonian_y",
    "transmon_hamiltonian",
    "PulseSequence",
    "constant_pulse",
    "gaussian_pulse",
    "drag_pulse",
    "gate_fidelity",
    "process_fidelity",
    "propagator",
]
