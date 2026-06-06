#!/usr/bin/env python3
"""Fermionic simulation demo — Jordan-Wigner, Bravyi-Kitaev, and Hamiltonian builders."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantumium.fermion import (
    FermionOperator,
    QubitOperator,
    jordan_wigner,
    bravyi_kitaev,
    number_operator,
    hopping_operator,
    coulomb_operator,
    fermi_hubbard,
    molecular_hamiltonian,
)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    # ── 1. FermionOperator algebra ──
    section("1. FermionOperator Algebra")

    c0 = FermionOperator.creation(0)
    a0 = FermionOperator.annihilation(0)
    n0 = c0 * a0
    print(f"a†_0 = {c0}")
    print(f"a_0  = {a0}")
    print(f"n_0 = a†_0 a_0 = {n0}")
    print(f"n_0 is Hermitian: {n0.is_hermitian}")

    anti = a0 * c0
    no = anti.normal_order()
    print(f"\na_0 a†_0 (normal ordered) = {no}")
    print("  (Should be 1 - a†_0 a_0)")

    # ── 2. Jordan-Wigner transform ──
    section("2. Jordan-Wigner Transform")

    n0_qubit = jordan_wigner(n0, 2)
    print(f"JW(n_0) for 2 qubits:")
    for pauli, coeff in sorted(n0_qubit.terms.items()):
        print(f"  {coeff:+.4f} {pauli}")

    mat = n0_qubit.to_matrix(2)
    eigs = np.linalg.eigvalsh(mat).real
    print(f"\nEigenvalues: {sorted(eigs)}")
    print("  (Expected: [0, 0, 1, 1] — counts occupation of mode 0)")

    # ── 3. Anti-commutation verification ──
    section("3. Anti-Commutation Relations")

    n = 3
    print(f"Verifying {{a†_i, a_j}} = δ_ij for {n} modes (JW)...")
    all_ok = True
    for i in range(n):
        for j in range(n):
            ci = FermionOperator.creation(i)
            aj = FermionOperator.annihilation(j)
            anti = ci * aj + aj * ci
            q = jordan_wigner(anti, n)
            mat = q.to_matrix(n)
            expected = np.eye(2**n) if i == j else np.zeros((2**n, 2**n))
            if not np.allclose(mat, expected, atol=1e-10):
                print(f"  FAIL: {{a†_{i}, a_{j}}}")
                all_ok = False
    print(f"  All anti-commutation relations: {'PASS' if all_ok else 'FAIL'}")

    # ── 4. Bravyi-Kitaev transform ──
    section("4. Bravyi-Kitaev Transform")

    hop = hopping_operator(0, 1, -1.0)
    jw_hop = jordan_wigner(hop, 4)
    bk_hop = bravyi_kitaev(hop, 4)

    print("JW hopping (0,1) Pauli terms:")
    for pauli, coeff in sorted(jw_hop.terms.items()):
        if abs(coeff) > 1e-10:
            print(f"  {coeff:+.4f} {pauli}")

    print("\nBK hopping (0,1) Pauli terms:")
    for pauli, coeff in sorted(bk_hop.terms.items()):
        if abs(coeff) > 1e-10:
            print(f"  {coeff:+.4f} {pauli}")

    jw_mat = jw_hop.to_matrix(4)
    bk_mat = bk_hop.to_matrix(4)
    jw_eigs = sorted(np.linalg.eigvalsh(jw_mat).real)
    bk_eigs = sorted(np.linalg.eigvalsh(bk_mat).real)
    print(f"\nJW eigenvalues: {[f'{e:.4f}' for e in jw_eigs[:4]]}...")
    print(f"BK eigenvalues: {[f'{e:.4f}' for e in bk_eigs[:4]]}...")
    print(f"Spectra match: {np.allclose(jw_eigs, bk_eigs, atol=1e-10)}")

    # ── 5. Fermi-Hubbard model ──
    section("5. Fermi-Hubbard Model (2-site)")

    t_val, U_val = 1.0, 2.0
    H_hub = fermi_hubbard(2, t=t_val, U=U_val)
    print(f"Parameters: t={t_val}, U={U_val}")
    print(f"Number of modes: {H_hub.num_modes}")
    print(f"Hermitian: {H_hub.is_hermitian}")

    mat_hub = jordan_wigner(H_hub, 4).to_matrix(4)
    eigs_hub = sorted(np.linalg.eigvalsh(mat_hub).real)
    E_exact = U_val / 2 - np.sqrt((U_val / 2) ** 2 + 4 * t_val**2)
    print(f"\nGround state energy: {eigs_hub[0]:.6f}")
    print(f"Exact analytical:    {E_exact:.6f}")
    print(f"Match: {abs(eigs_hub[0] - E_exact) < 1e-10}")

    print(f"\nFull spectrum: {[f'{e:.4f}' for e in eigs_hub]}")

    # JW vs BK comparison
    bk_mat_hub = bravyi_kitaev(H_hub, 4).to_matrix(4)
    bk_eigs_hub = sorted(np.linalg.eigvalsh(bk_mat_hub).real)
    print(f"JW-BK spectra match: {np.allclose(eigs_hub, bk_eigs_hub, atol=1e-10)}")

    # ── 6. Molecular Hamiltonian ──
    section("6. Molecular Hamiltonian (H₂-like)")

    h1 = np.array([
        [-1.2526, -0.4752],
        [-0.4752, -0.4752],
    ])
    h2 = np.zeros((2, 2, 2, 2))
    h2[0, 0, 1, 1] = 0.6746
    h2[1, 1, 0, 0] = 0.6746
    h2[0, 1, 1, 0] = 0.6746
    h2[1, 0, 0, 1] = 0.6746
    E_nuc = 0.7137

    H_mol = molecular_hamiltonian(h1, h2, nuclear_repulsion=E_nuc)
    mat_mol = jordan_wigner(H_mol, 2).to_matrix(2)
    eigs_mol = sorted(np.linalg.eigvalsh(mat_mol).real)

    print(f"Nuclear repulsion: {E_nuc}")
    print(f"Hermitian: {np.allclose(mat_mol, mat_mol.conj().T)}")
    print(f"Ground state energy: {eigs_mol[0]:.6f}")
    print(f"Full spectrum: {[f'{e:.4f}' for e in eigs_mol]}")

    # ── 7. Pauli weight comparison ──
    section("7. Pauli Weight: JW vs BK")

    for n_modes in [4, 6, 8]:
        hop = hopping_operator(0, n_modes - 1, -1.0)
        jw_op = jordan_wigner(hop, n_modes)
        bk_op = bravyi_kitaev(hop, n_modes)

        def max_weight(op):
            return max(
                sum(1 for c in k if c != "I")
                for k in op.terms
            ) if op.terms else 0

        jw_w = max_weight(jw_op)
        bk_w = max_weight(bk_op)
        print(f"  n={n_modes}: JW max weight={jw_w}, BK max weight={bk_w}")

    print("\n  BK achieves O(log n) Pauli weight vs JW's O(n)")

    # ── Summary ──
    section("Summary")
    print("All fermionic simulation features demonstrated:")
    print("  - FermionOperator algebra (creation, annihilation, normal ordering)")
    print("  - Jordan-Wigner transform with Z-string anti-commutation")
    print("  - Bravyi-Kitaev transform with O(log n) Pauli weight")
    print("  - Anti-commutation relation verification")
    print("  - Fermi-Hubbard model with exact ground state energy")
    print("  - Molecular Hamiltonian from one/two-body integrals")
    print("  - JW/BK eigenspectrum equivalence")


if __name__ == "__main__":
    main()
