"""Fermionic simulation module for quantum chemistry and condensed matter.

Provides FermionOperator algebra, fermion-to-qubit transforms
(Jordan-Wigner, Bravyi-Kitaev), and Hamiltonian builders.
"""

from quantumium.fermion.operators import FermionOperator, QubitOperator
from quantumium.fermion.transforms import jordan_wigner, bravyi_kitaev
from quantumium.fermion.hamiltonians import (
    number_operator,
    hopping_operator,
    coulomb_operator,
    fermi_hubbard,
    molecular_hamiltonian,
)

__all__ = [
    "FermionOperator",
    "QubitOperator",
    "jordan_wigner",
    "bravyi_kitaev",
    "number_operator",
    "hopping_operator",
    "coulomb_operator",
    "fermi_hubbard",
    "molecular_hamiltonian",
]
