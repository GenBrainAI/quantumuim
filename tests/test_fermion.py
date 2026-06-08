"""Tests for fermionic simulation module."""

import numpy as np

from quantumium.fermion.operators import (
    FermionOperator,
    QubitOperator,
    _multiply_pauli_strings,
)
from quantumium.fermion.transforms import (
    jordan_wigner,
    bravyi_kitaev,
    _bk_update_set,
    _bk_parity_set,
    _bk_children_set,
)
from quantumium.fermion.hamiltonians import (
    number_operator,
    hopping_operator,
    coulomb_operator,
    fermi_hubbard,
    molecular_hamiltonian,
)


# ──────────────────────────────────────────────
# FermionOperator basics
# ──────────────────────────────────────────────


class TestFermionOperator:
    def test_creation_annihilation(self):
        c = FermionOperator.creation(0)
        a = FermionOperator.annihilation(0)
        assert ((0, True),) in c.terms
        assert ((0, False),) in a.terms

    def test_identity_zero(self):
        I = FermionOperator.identity()
        Z = FermionOperator.zero()
        assert () in I.terms
        assert len(Z.terms) == 0

    def test_addition(self):
        c0 = FermionOperator.creation(0)
        c1 = FermionOperator.creation(1)
        s = c0 + c1
        assert len(s.terms) == 2

    def test_subtraction(self):
        c = FermionOperator.creation(0)
        diff = c - c
        diff = diff.compress()
        assert len(diff.terms) == 0

    def test_scalar_multiply(self):
        c = FermionOperator.creation(0)
        scaled = c * 3.0
        assert abs(scaled.terms[((0, True),)] - 3.0) < 1e-10

    def test_operator_multiply(self):
        c0 = FermionOperator.creation(0)
        a0 = FermionOperator.annihilation(0)
        prod = c0 * a0
        assert ((0, True), (0, False)) in prod.terms

    def test_dagger(self):
        c = FermionOperator.creation(0)
        d = c.dagger()
        assert ((0, False),) in d.terms

    def test_dagger_product(self):
        op = FermionOperator.creation(0) * FermionOperator.annihilation(1)
        dag = op.dagger()
        assert ((1, True), (0, False)) in dag.terms

    def test_num_modes(self):
        op = FermionOperator.creation(3) * FermionOperator.annihilation(1)
        assert op.num_modes == 4

    def test_hermitian_check(self):
        n = number_operator(0)
        assert n.is_hermitian


# ──────────────────────────────────────────────
# Normal ordering
# ──────────────────────────────────────────────


class TestNormalOrder:
    def test_already_normal(self):
        op = FermionOperator.creation(0) * FermionOperator.annihilation(0)
        no = op.normal_order()
        assert ((0, True), (0, False)) in no.terms

    def test_single_swap(self):
        # a_0 a†_0 = 1 - a†_0 a_0
        op = FermionOperator.annihilation(0) * FermionOperator.creation(0)
        no = op.normal_order()
        assert () in no.terms
        assert abs(no.terms[()] - 1.0) < 1e-10
        assert ((0, True), (0, False)) in no.terms
        assert abs(no.terms[((0, True), (0, False))] + 1.0) < 1e-10

    def test_different_modes(self):
        # a_0 a†_1 = -a†_1 a_0
        op = FermionOperator.annihilation(0) * FermionOperator.creation(1)
        no = op.normal_order()
        assert ((1, True), (0, False)) in no.terms
        assert abs(no.terms[((1, True), (0, False))] + 1.0) < 1e-10

    def test_double_creation(self):
        # a†_0 a†_0 = 0
        op = FermionOperator.creation(0) * FermionOperator.creation(0)
        no = op.normal_order()
        assert len(no.compress().terms) == 0

    def test_double_annihilation(self):
        op = FermionOperator.annihilation(0) * FermionOperator.annihilation(0)
        no = op.normal_order()
        assert len(no.compress().terms) == 0

    def test_number_operator_normal(self):
        n = number_operator(0)
        no = n.normal_order()
        assert ((0, True), (0, False)) in no.terms


# ──────────────────────────────────────────────
# QubitOperator
# ──────────────────────────────────────────────


class TestQubitOperator:
    def test_identity(self):
        I = QubitOperator.identity()
        assert "I" in I.terms

    def test_zero(self):
        Z = QubitOperator.zero()
        assert len(Z.terms) == 0

    def test_addition(self):
        x = QubitOperator({"X": 1.0})
        z = QubitOperator({"Z": 1.0})
        s = x + z
        assert len(s.terms) == 2

    def test_pauli_multiply(self):
        x = QubitOperator({"X": 1.0})
        y = QubitOperator({"Y": 1.0})
        xy = x * y
        assert "Z" in xy.terms
        assert abs(xy.terms["Z"] - 1j) < 1e-10

    def test_to_matrix_x(self):
        x = QubitOperator({"X": 1.0})
        mat = x.to_matrix(1)
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        np.testing.assert_allclose(mat, expected)

    def test_to_matrix_identity(self):
        I = QubitOperator.identity()
        mat = I.to_matrix(2)
        np.testing.assert_allclose(mat, np.eye(4, dtype=complex))

    def test_compress(self):
        op = QubitOperator({"XI": 1.0, "X": 2.0})
        c = op.compress()
        assert "X" in c.terms
        assert abs(c.terms["X"] - 3.0) < 1e-10


# ──────────────────────────────────────────────
# Pauli multiplication
# ──────────────────────────────────────────────


class TestPauliMultiplication:
    def test_xx_is_i(self):
        s, p = _multiply_pauli_strings("X", "X")
        assert s == "I" and abs(p - 1) < 1e-10

    def test_xy_is_iz(self):
        s, p = _multiply_pauli_strings("X", "Y")
        assert s == "Z" and abs(p - 1j) < 1e-10

    def test_yz_is_ix(self):
        s, p = _multiply_pauli_strings("Y", "Z")
        assert s == "X" and abs(p - 1j) < 1e-10

    def test_multi_qubit(self):
        s, p = _multiply_pauli_strings("XY", "YX")
        assert s == "ZZ"
        assert abs(p - 1) < 1e-10

    def test_identity_pass(self):
        s, p = _multiply_pauli_strings("I", "X")
        assert s == "X" and abs(p - 1) < 1e-10


# ──────────────────────────────────────────────
# Jordan-Wigner transform
# ──────────────────────────────────────────────


class TestJordanWigner:
    def test_number_operator(self):
        n0 = number_operator(0)
        q = jordan_wigner(n0, 2)
        mat = q.to_matrix(2)
        eigs = sorted(np.linalg.eigvalsh(mat).real)
        np.testing.assert_allclose(eigs, [0, 0, 1, 1], atol=1e-10)

    def test_anti_commutation_2mode(self):
        n = 2
        for i in range(n):
            for j in range(n):
                ci = FermionOperator.creation(i)
                aj = FermionOperator.annihilation(j)
                anti = ci * aj + aj * ci
                q = jordan_wigner(anti, n)
                mat = q.to_matrix(n)
                expected = np.eye(2**n) if i == j else np.zeros((2**n, 2**n))
                np.testing.assert_allclose(mat, expected, atol=1e-10)

    def test_anti_commutation_creation(self):
        n = 2
        for i in range(n):
            for j in range(n):
                ci = FermionOperator.creation(i)
                cj = FermionOperator.creation(j)
                anti = ci * cj + cj * ci
                q = jordan_wigner(anti, n)
                mat = q.to_matrix(n)
                np.testing.assert_allclose(mat, np.zeros((2**n, 2**n)), atol=1e-10)

    def test_hermitian_number(self):
        n = number_operator(0)
        q = jordan_wigner(n, 2)
        mat = q.to_matrix(2)
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-10)

    def test_hopping_eigenvalues(self):
        hop = hopping_operator(0, 1, -1.0)
        q = jordan_wigner(hop, 2)
        mat = q.to_matrix(2)
        eigs = sorted(np.linalg.eigvalsh(mat).real)
        np.testing.assert_allclose(eigs, [-1, 0, 0, 1], atol=1e-10)

    def test_fock_matrix_match(self):
        n0 = number_operator(0)
        q = jordan_wigner(n0, 1)
        mat = q.to_matrix(1)
        expected = np.diag([0.0, 1.0])
        np.testing.assert_allclose(mat, expected, atol=1e-10)


# ──────────────────────────────────────────────
# Bravyi-Kitaev transform
# ──────────────────────────────────────────────


class TestBravyiKitaev:
    def test_anti_commutation_4mode(self):
        n = 4
        for i in range(n):
            for j in range(n):
                ci = FermionOperator.creation(i)
                aj = FermionOperator.annihilation(j)
                anti = ci * aj + aj * ci
                q = bravyi_kitaev(anti, n)
                mat = q.to_matrix(n)
                expected = np.eye(2**n) if i == j else np.zeros((2**n, 2**n))
                np.testing.assert_allclose(
                    mat, expected, atol=1e-10, err_msg=f"{{a†_{i}, a_{j}}} failed"
                )

    def test_anti_commutation_6mode(self):
        n = 6
        for i in range(n):
            for j in range(n):
                ci = FermionOperator.creation(i)
                aj = FermionOperator.annihilation(j)
                anti = ci * aj + aj * ci
                q = bravyi_kitaev(anti, n)
                mat = q.to_matrix(n)
                expected = np.eye(2**n) if i == j else np.zeros((2**n, 2**n))
                np.testing.assert_allclose(
                    mat, expected, atol=1e-10, err_msg=f"{{a†_{i}, a_{j}}} failed for n=6"
                )

    def test_hopping_eigenspectrum(self):
        hop = hopping_operator(0, 1, -1.0)
        jw_mat = jordan_wigner(hop, 2).to_matrix(2)
        bk_mat = bravyi_kitaev(hop, 2).to_matrix(2)
        jw_eigs = sorted(np.linalg.eigvalsh(jw_mat).real)
        bk_eigs = sorted(np.linalg.eigvalsh(bk_mat).real)
        np.testing.assert_allclose(jw_eigs, bk_eigs, atol=1e-10)

    def test_hubbard_2site_eigenspectrum(self):
        H = fermi_hubbard(2, t=1.0, U=2.0)
        jw_mat = jordan_wigner(H, 4).to_matrix(4)
        bk_mat = bravyi_kitaev(H, 4).to_matrix(4)
        jw_eigs = sorted(np.linalg.eigvalsh(jw_mat).real)
        bk_eigs = sorted(np.linalg.eigvalsh(bk_mat).real)
        np.testing.assert_allclose(jw_eigs, bk_eigs, atol=1e-10)

    def test_hubbard_3site_eigenspectrum(self):
        H = fermi_hubbard(3, t=1.0, U=3.0)
        n = 6
        jw_mat = jordan_wigner(H, n).to_matrix(n)
        bk_mat = bravyi_kitaev(H, n).to_matrix(n)
        jw_eigs = sorted(np.linalg.eigvalsh(jw_mat).real)
        bk_eigs = sorted(np.linalg.eigvalsh(bk_mat).real)
        np.testing.assert_allclose(jw_eigs, bk_eigs, atol=1e-10)

    def test_anti_commutation_creation_4mode(self):
        n = 4
        for i in range(n):
            for j in range(n):
                ci = FermionOperator.creation(i)
                cj = FermionOperator.creation(j)
                anti = ci * cj + cj * ci
                q = bravyi_kitaev(anti, n)
                mat = q.to_matrix(n)
                np.testing.assert_allclose(mat, np.zeros((2**n, 2**n)), atol=1e-10)


# ──────────────────────────────────────────────
# BK helper sets
# ──────────────────────────────────────────────


class TestBKSets:
    def test_update_set_mode0(self):
        u = _bk_update_set(0, 8)
        assert 1 in u
        assert 3 in u
        assert 7 in u

    def test_parity_set_mode3(self):
        p = _bk_parity_set(3)
        assert 2 in p

    def test_children_set_mode3(self):
        c = _bk_children_set(3)
        assert 2 in c

    def test_update_parity_disjoint(self):
        for mode in range(8):
            u = _bk_update_set(mode, 8)
            p = _bk_parity_set(mode)
            assert u.isdisjoint(p), f"U∩P non-empty for mode {mode}"

    def test_children_subset_parity(self):
        for mode in range(8):
            p = _bk_parity_set(mode)
            c = _bk_children_set(mode)
            assert c.issubset(p), f"C⊄P for mode {mode}"


# ──────────────────────────────────────────────
# JW vs BK equivalence
# ──────────────────────────────────────────────


class TestTransformEquivalence:
    def test_single_mode_number(self):
        n0 = number_operator(0)
        jw = jordan_wigner(n0, 4).to_matrix(4)
        bk = bravyi_kitaev(n0, 4).to_matrix(4)
        jw_eigs = sorted(np.linalg.eigvalsh(jw).real)
        bk_eigs = sorted(np.linalg.eigvalsh(bk).real)
        np.testing.assert_allclose(jw_eigs, bk_eigs, atol=1e-10)

    def test_two_body_coulomb(self):
        op = coulomb_operator(0, 1, 2.0)
        jw = jordan_wigner(op, 4).to_matrix(4)
        bk = bravyi_kitaev(op, 4).to_matrix(4)
        jw_eigs = sorted(np.linalg.eigvalsh(jw).real)
        bk_eigs = sorted(np.linalg.eigvalsh(bk).real)
        np.testing.assert_allclose(jw_eigs, bk_eigs, atol=1e-10)

    def test_all_hopping_pairs(self):
        n = 4
        for i in range(n):
            for j in range(i + 1, n):
                hop = hopping_operator(i, j, -1.0)
                jw = jordan_wigner(hop, n).to_matrix(n)
                bk = bravyi_kitaev(hop, n).to_matrix(n)
                jw_eigs = sorted(np.linalg.eigvalsh(jw).real)
                bk_eigs = sorted(np.linalg.eigvalsh(bk).real)
                np.testing.assert_allclose(
                    jw_eigs, bk_eigs, atol=1e-10, err_msg=f"Hopping ({i},{j}) mismatch"
                )


# ──────────────────────────────────────────────
# Hamiltonian builders
# ──────────────────────────────────────────────


class TestNumberOperator:
    def test_eigenvalues(self):
        n0 = number_operator(0)
        q = jordan_wigner(n0, 2)
        mat = q.to_matrix(2)
        eigs = sorted(np.linalg.eigvalsh(mat).real)
        np.testing.assert_allclose(eigs, [0, 0, 1, 1], atol=1e-10)

    def test_hermitian(self):
        n0 = number_operator(0)
        assert n0.is_hermitian


class TestHoppingOperator:
    def test_hermitian(self):
        hop = hopping_operator(0, 1)
        assert hop.is_hermitian

    def test_coefficient(self):
        hop = hopping_operator(0, 1, -2.0)
        q = jordan_wigner(hop, 2)
        mat = q.to_matrix(2)
        eigs = sorted(np.linalg.eigvalsh(mat).real)
        np.testing.assert_allclose(eigs, [-2, 0, 0, 2], atol=1e-10)

    def test_symmetric(self):
        h01 = hopping_operator(0, 1)
        h10 = hopping_operator(1, 0)
        q01 = jordan_wigner(h01, 2).to_matrix(2)
        q10 = jordan_wigner(h10, 2).to_matrix(2)
        np.testing.assert_allclose(q01, q10, atol=1e-10)


class TestCoulombOperator:
    def test_eigenvalues(self):
        c = coulomb_operator(0, 1, 2.0)
        q = jordan_wigner(c, 2)
        mat = q.to_matrix(2)
        eigs = sorted(np.linalg.eigvalsh(mat).real)
        np.testing.assert_allclose(eigs, [0, 0, 0, 2], atol=1e-10)


# ──────────────────────────────────────────────
# Fermi-Hubbard model
# ──────────────────────────────────────────────


class TestFermiHubbard:
    def test_hermitian(self):
        H = fermi_hubbard(2, t=1.0, U=2.0)
        assert H.is_hermitian

    def test_ground_energy_2site(self):
        t, U = 1.0, 2.0
        H = fermi_hubbard(2, t=t, U=U)
        mat = jordan_wigner(H, 4).to_matrix(4)
        E_gs = np.linalg.eigvalsh(mat).real.min()
        E_exact = U / 2 - np.sqrt((U / 2) ** 2 + 4 * t**2)
        np.testing.assert_allclose(E_gs, E_exact, atol=1e-10)

    def test_num_modes(self):
        H = fermi_hubbard(3, t=1.0, U=2.0)
        assert H.num_modes == 6

    def test_periodic_vs_open(self):
        H_open = fermi_hubbard(3, t=1.0, U=2.0, periodic=False)
        H_periodic = fermi_hubbard(3, t=1.0, U=2.0, periodic=True)
        m_open = jordan_wigner(H_open, 6).to_matrix(6)
        m_periodic = jordan_wigner(H_periodic, 6).to_matrix(6)
        assert not np.allclose(m_open, m_periodic)

    def test_U0_free_fermions(self):
        H = fermi_hubbard(2, t=1.0, U=0.0)
        mat = jordan_wigner(H, 4).to_matrix(4)
        eigs = sorted(np.linalg.eigvalsh(mat).real)
        assert abs(eigs[0] - (-2.0)) < 1e-10


# ──────────────────────────────────────────────
# Molecular Hamiltonian
# ──────────────────────────────────────────────


class TestMolecularHamiltonian:
    def test_one_body_only(self):
        h1 = np.array([[1.0, 0.5], [0.5, 2.0]])
        h2 = np.zeros((2, 2, 2, 2))
        H = molecular_hamiltonian(h1, h2)
        mat = jordan_wigner(H, 2).to_matrix(2)
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-10)

    def test_nuclear_repulsion(self):
        h1 = np.zeros((2, 2))
        h2 = np.zeros((2, 2, 2, 2))
        H = molecular_hamiltonian(h1, h2, nuclear_repulsion=5.0)
        mat = jordan_wigner(H, 2).to_matrix(2)
        np.testing.assert_allclose(mat, 5.0 * np.eye(4), atol=1e-10)

    def test_hermitian(self):
        h1 = np.array([[1.0, 0.3], [0.3, 2.0]])
        h2 = np.zeros((2, 2, 2, 2))
        h2[0, 1, 0, 1] = 0.5
        h2[1, 0, 1, 0] = 0.5
        H = molecular_hamiltonian(h1, h2)
        mat = jordan_wigner(H, 2).to_matrix(2)
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-10)


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_operator_jw(self):
        op = FermionOperator.zero()
        q = jordan_wigner(op, 2)
        mat = q.to_matrix(2)
        np.testing.assert_allclose(mat, np.zeros((4, 4)), atol=1e-10)

    def test_identity_jw(self):
        op = FermionOperator.identity()
        q = jordan_wigner(op, 2)
        mat = q.to_matrix(2)
        np.testing.assert_allclose(mat, np.eye(4), atol=1e-10)

    def test_complex_coefficients(self):
        op = FermionOperator.creation(0) * 1j
        q = jordan_wigner(op, 1)
        mat = q.to_matrix(1)
        assert mat.dtype == complex

    def test_large_mode_index(self):
        n5 = number_operator(5)
        q = jordan_wigner(n5, 6)
        mat = q.to_matrix(6)
        eigs = sorted(np.linalg.eigvalsh(mat).real)
        assert abs(eigs[0]) < 1e-10
        assert abs(eigs[-1] - 1.0) < 1e-10


# ──────────────────────────────────────────────
# Integration tests
# ──────────────────────────────────────────────


class TestIntegration:
    def test_hubbard_jw_bk_full(self):
        H = fermi_hubbard(2, t=1.0, U=4.0)
        jw = jordan_wigner(H, 4).to_matrix(4)
        bk = bravyi_kitaev(H, 4).to_matrix(4)
        jw_eigs = sorted(np.linalg.eigvalsh(jw).real)
        bk_eigs = sorted(np.linalg.eigvalsh(bk).real)
        np.testing.assert_allclose(jw_eigs, bk_eigs, atol=1e-10)

    def test_molecular_transform_equivalence(self):
        h1 = np.array([[1.0, 0.2], [0.2, 1.5]])
        h2 = np.zeros((2, 2, 2, 2))
        h2[0, 0, 1, 1] = 0.3
        h2[1, 1, 0, 0] = 0.3
        H = molecular_hamiltonian(h1, h2, nuclear_repulsion=1.0)
        jw = jordan_wigner(H, 2).to_matrix(2)
        bk = bravyi_kitaev(H, 2).to_matrix(2)
        jw_eigs = sorted(np.linalg.eigvalsh(jw).real)
        bk_eigs = sorted(np.linalg.eigvalsh(bk).real)
        np.testing.assert_allclose(jw_eigs, bk_eigs, atol=1e-10)

    def test_end_to_end_h2_like(self):
        h1 = np.array(
            [
                [-1.2526, -0.4752],
                [-0.4752, -0.4752],
            ]
        )
        h2 = np.zeros((2, 2, 2, 2))
        h2[0, 0, 1, 1] = 0.6746
        h2[1, 1, 0, 0] = 0.6746
        h2[0, 1, 1, 0] = 0.6746
        h2[1, 0, 0, 1] = 0.6746
        H = molecular_hamiltonian(h1, h2, nuclear_repulsion=0.7137)
        mat = jordan_wigner(H, 2).to_matrix(2)
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-10)
        eigs = sorted(np.linalg.eigvalsh(mat).real)
        assert eigs[0] < 0
