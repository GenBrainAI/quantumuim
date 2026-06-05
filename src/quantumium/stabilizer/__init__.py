"""Stabilizer simulation via Clifford tableau.

Provides efficient simulation of Clifford circuits using the
Aaronson-Gottesman stabilizer formalism. Clifford circuits (composed of
H, S, CNOT, CZ, and Pauli gates) can be simulated in O(n²) time per
gate, compared to O(2^n) for general quantum circuits.

References:
    Aaronson & Gottesman, "Improved simulation of stabilizer circuits"
    (Phys. Rev. A 70, 052328, 2004)
    Gottesman, "The Heisenberg representation of quantum computers"
    (arXiv:quant-ph/9807006, 1998)
"""

from .tableau import (
    StabilizerTableau,
    hadamard,
    phase_gate,
    cnot,
    cz,
    pauli_x,
    pauli_y,
    pauli_z,
    measure,
    to_statevector,
)

__all__ = [
    "StabilizerTableau",
    "hadamard",
    "phase_gate",
    "cnot",
    "cz",
    "pauli_x",
    "pauli_y",
    "pauli_z",
    "measure",
    "to_statevector",
]
