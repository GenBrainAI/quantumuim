"""Hamiltonian builders for quantum chemistry and condensed matter."""

from __future__ import annotations

import numpy as np
from quantumium.fermion.operators import FermionOperator


def number_operator(mode: int) -> FermionOperator:
    """n_i = a†_i a_i"""
    return FermionOperator.creation(mode) * FermionOperator.annihilation(mode)


def hopping_operator(i: int, j: int, coeff: complex = -1.0) -> FermionOperator:
    """coeff * (a†_i a_j + a†_j a_i)"""
    hop = FermionOperator.creation(i) * FermionOperator.annihilation(j)
    hop = hop + FermionOperator.creation(j) * FermionOperator.annihilation(i)
    return hop * coeff


def coulomb_operator(i: int, j: int, coeff: complex = 1.0) -> FermionOperator:
    """coeff * n_i * n_j"""
    return number_operator(i) * number_operator(j) * coeff


def fermi_hubbard(
    n_sites: int,
    t: float = 1.0,
    U: float = 2.0,
    periodic: bool = False,
) -> FermionOperator:
    """1D Fermi-Hubbard model.

    H = -t Σ_{<i,j>,σ} (a†_{i,σ} a_{j,σ} + h.c.) + U Σ_i n_{i,↑} n_{i,↓}

    Mode ordering: spin-up 0..n_sites-1, spin-down n_sites..2*n_sites-1
    """
    H = FermionOperator.zero()

    # Hopping terms
    for i in range(n_sites - 1):
        # Spin-up hopping
        H = H + hopping_operator(i, i + 1, -t)
        # Spin-down hopping
        H = H + hopping_operator(i + n_sites, i + 1 + n_sites, -t)

    if periodic and n_sites > 2:
        H = H + hopping_operator(n_sites - 1, 0, -t)
        H = H + hopping_operator(2 * n_sites - 1, n_sites, -t)

    # On-site interaction
    for i in range(n_sites):
        H = H + coulomb_operator(i, i + n_sites, U)

    return H


def molecular_hamiltonian(
    one_body: np.ndarray,
    two_body: np.ndarray,
    nuclear_repulsion: float = 0.0,
) -> FermionOperator:
    """Build molecular Hamiltonian from one- and two-body integrals.

    H = Σ_{pq} h_{pq} a†_p a_q
      + (1/2) Σ_{pqrs} h_{pqrs} a†_p a†_r a_s a_q
      + E_nuc

    Uses physicist's notation for two-body integrals: h_{pqrs} = <pq|rs>.
    """
    n = one_body.shape[0]
    H = FermionOperator.zero()

    if abs(nuclear_repulsion) > 1e-15:
        H = H + FermionOperator.identity() * nuclear_repulsion

    # One-body terms
    for p in range(n):
        for q in range(n):
            if abs(one_body[p, q]) > 1e-15:
                term = FermionOperator.creation(p) * FermionOperator.annihilation(q)
                H = H + term * one_body[p, q]

    # Two-body terms
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    if abs(two_body[p, q, r, s]) > 1e-15:
                        term = (
                            FermionOperator.creation(p)
                            * FermionOperator.creation(r)
                            * FermionOperator.annihilation(s)
                            * FermionOperator.annihilation(q)
                        )
                        H = H + term * (0.5 * two_body[p, q, r, s])

    return H
