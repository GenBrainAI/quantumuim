"""Quantum adiabatic computing and annealing simulation.

Simulates adiabatic quantum computation where the system Hamiltonian
is slowly evolved from an initial (easy) ground state to a final
(problem) Hamiltonian whose ground state encodes the solution.

    H(s) = (1 - s) H_init + s H_final,  s ∈ [0, 1]

The adiabatic theorem guarantees the system remains in the ground
state if the evolution is sufficiently slow compared to the minimum
spectral gap.

References:
    Farhi et al., "Quantum Computation by Adiabatic Evolution"
    (arXiv:quant-ph/0001106, 2000)
    Albash & Lidar, "Adiabatic quantum computation"
    (Reviews of Modern Physics 90, 015002, 2018)
"""

from .evolution import (
    AdiabaticResult,
    adiabatic_evolve,
    spectral_gap,
    linear_schedule,
    polynomial_schedule,
)
from .hamiltonians import (
    transverse_field_hamiltonian,
    ising_hamiltonian,
    maxcut_hamiltonian,
)

__all__ = [
    "AdiabaticResult",
    "adiabatic_evolve",
    "spectral_gap",
    "linear_schedule",
    "polynomial_schedule",
    "transverse_field_hamiltonian",
    "ising_hamiltonian",
    "maxcut_hamiltonian",
]
