"""Quantum algorithm implementations.

This package provides implementations of major quantum algorithms, including:

- **Quantum Fourier Transform (QFT)**: Basis transformation between computational
  and Fourier bases, used as a subroutine in many quantum algorithms.
- **QAOA**: Quantum Approximate Optimization Algorithm for combinatorial
  optimization problems such as MaxCut.
- **VQE**: Variational Quantum Eigensolver for finding ground-state energies
  of molecular and spin Hamiltonians.
- **Hamiltonian Simulation**: Trotter-Suzuki decomposition for simulating
  quantum system evolution under a Hamiltonian.
- **Shor's Algorithm**: Integer factoring via quantum order-finding.
- **Simon's Algorithm**: Finding hidden periods of black-box functions with
  exponential quantum speedup.
- **HHL Algorithm**: Quantum linear systems solver for Ax = b.
- **Quantum Counting & Amplitude Estimation**: Estimating the number of
  solutions or the amplitude of a target subspace.

All algorithms operate on :class:`~quantumium.core.circuit.QuantumCircuit`
instances and use the gate primitives from :mod:`quantumium.core.gate`.
"""

from .qft import qft, inverse_qft
from .qaoa import qaoa_circuit, qaoa_expectation, qaoa_maxcut, maxcut_cost_operator
from .vqe import vqe, hardware_efficient_ansatz, h2_hamiltonian, hamiltonian_expectation
from .hamiltonian_sim import (
    simulate_evolution,
    first_order_trotter,
    second_order_trotter,
    transverse_field_ising,
)
from .shor import shor_factor, quantum_order_finding
from .simon import simon_algorithm
from .hhl import hhl_solve
from .quantum_counting import (
    quantum_counting,
    amplitude_estimation,
    iterative_amplitude_estimation,
)

__all__ = [
    "qft",
    "inverse_qft",
    "qaoa_circuit",
    "qaoa_expectation",
    "qaoa_maxcut",
    "maxcut_cost_operator",
    "vqe",
    "hardware_efficient_ansatz",
    "h2_hamiltonian",
    "hamiltonian_expectation",
    "simulate_evolution",
    "first_order_trotter",
    "second_order_trotter",
    "transverse_field_ising",
    "shor_factor",
    "quantum_order_finding",
    "simon_algorithm",
    "hhl_solve",
    "quantum_counting",
    "amplitude_estimation",
    "iterative_amplitude_estimation",
]
