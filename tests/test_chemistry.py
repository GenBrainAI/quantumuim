"""Tests for quantum chemistry applications."""

import numpy as np

from quantumium.applications.chemistry import (
    get_molecular_hamiltonian,
    simulate_molecule,
    _build_hamiltonian,
    MOLECULAR_HAMILTONIANS,
)


class TestBuildHamiltonian:
    def test_identity(self):
        """II should produce identity matrix."""
        H = _build_hamiltonian({"II": 1.0}, 2)
        np.testing.assert_allclose(H, np.eye(4, dtype=complex), atol=1e-10)

    def test_single_z(self):
        """ZI should be Z⊗I."""
        H = _build_hamiltonian({"ZI": 1.0}, 2)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        expected = np.kron(Z, np.eye(2, dtype=complex))
        np.testing.assert_allclose(H, expected, atol=1e-10)

    def test_hermitian(self):
        """Built Hamiltonian should always be Hermitian."""
        H = _build_hamiltonian({"XX": 0.5, "YY": 0.3, "ZZ": 0.2, "II": -1.0}, 2)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-10)

    def test_wrong_length_raises(self):
        try:
            _build_hamiltonian({"XYZ": 1.0}, 2)
            assert False, "Should raise ValueError"
        except ValueError:
            pass


class TestMolecularHamiltonians:
    def test_h2_available(self):
        H, n = get_molecular_hamiltonian("H2")
        assert H.shape == (4, 4)
        assert n == 2

    def test_heh_plus_available(self):
        H, n = get_molecular_hamiltonian("HeH+")
        assert H.shape == (4, 4)
        assert n == 2

    def test_lih_available(self):
        H, n = get_molecular_hamiltonian("LiH")
        assert H.shape == (16, 16)
        assert n == 4

    def test_unknown_raises(self):
        try:
            get_molecular_hamiltonian("Unknown")
            assert False
        except ValueError as e:
            assert "Unknown" in str(e)

    def test_all_hermitian(self):
        for name in MOLECULAR_HAMILTONIANS:
            H, _ = get_molecular_hamiltonian(name)
            np.testing.assert_allclose(H, H.conj().T, atol=1e-10, err_msg=f"{name} not Hermitian")


class TestSimulateMolecule:
    def test_h2_simulation(self):
        """VQE should find H2 ground state reasonably close to exact."""
        result = simulate_molecule("H2", ansatz_depth=2, max_iter=100, num_starts=3)
        assert result.molecule == "H2"
        assert result.error < 0.1
        assert result.exact_energy < -1.0
        assert result.ground_energy < -1.0

    def test_heh_plus_simulation(self):
        """HeH+ should also converge."""
        result = simulate_molecule("HeH+", ansatz_depth=2, max_iter=100, num_starts=3)
        assert result.molecule == "HeH+"
        assert result.error < 0.15
        assert result.exact_energy < -2.0
