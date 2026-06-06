"""Tests for surface code error correction with MWPM decoder."""

import numpy as np
import pytest

from quantumium.surface_code.lattice import SurfaceCodeLattice
from quantumium.surface_code.syndrome import apply_error, extract_syndrome, random_error
from quantumium.surface_code.decoder import MWPMDecoder
from quantumium.surface_code.logical import (
    logical_x, logical_z, apply_logical_x, apply_logical_z, has_logical_error,
)
from quantumium.surface_code.threshold import logical_error_rate


# --- Lattice construction ---

class TestSurfaceCodeLattice:
    def test_d3_data_qubits(self):
        lattice = SurfaceCodeLattice(3)
        assert lattice.num_data == 9
        assert len(lattice.data_qubits) == 9

    def test_d5_data_qubits(self):
        lattice = SurfaceCodeLattice(5)
        assert lattice.num_data == 25

    def test_d7_data_qubits(self):
        lattice = SurfaceCodeLattice(7)
        assert lattice.num_data == 49

    def test_stabilizer_count_d3(self):
        lattice = SurfaceCodeLattice(3)
        assert lattice.num_x_stabilizers == 4
        assert lattice.num_z_stabilizers == 4

    def test_stabilizer_count_d5(self):
        lattice = SurfaceCodeLattice(5)
        assert lattice.num_x_stabilizers == 12
        assert lattice.num_z_stabilizers == 12

    def test_stabilizer_count_d7(self):
        lattice = SurfaceCodeLattice(7)
        assert lattice.num_x_stabilizers == 24
        assert lattice.num_z_stabilizers == 24

    def test_invalid_distance_even(self):
        with pytest.raises(ValueError):
            SurfaceCodeLattice(4)

    def test_invalid_distance_small(self):
        with pytest.raises(ValueError):
            SurfaceCodeLattice(1)

    def test_stabilizer_matrix_shape_d3(self):
        lattice = SurfaceCodeLattice(3)
        Hx = lattice.stabilizer_matrix_x()
        Hz = lattice.stabilizer_matrix_z()
        assert Hx.shape == (4, 9)
        assert Hz.shape == (4, 9)

    def test_stabilizers_orthogonal(self):
        """X and Z stabilizers must commute: H_x · H_z^T = 0 mod 2."""
        for d in [3, 5, 7]:
            lattice = SurfaceCodeLattice(d)
            Hx = lattice.stabilizer_matrix_x()
            Hz = lattice.stabilizer_matrix_z()
            product = (Hx @ Hz.T) % 2
            np.testing.assert_array_equal(product, 0)


# --- Syndrome extraction ---

class TestSyndrome:
    def test_no_error_syndrome(self):
        lattice = SurfaceCodeLattice(3)
        error = np.zeros(9, dtype=np.int8)
        syndrome = extract_syndrome(lattice, error, "X")
        np.testing.assert_array_equal(syndrome, 0)

    def test_single_x_error_nonzero_syndrome(self):
        lattice = SurfaceCodeLattice(3)
        for q in range(9):
            error = apply_error(lattice, "X", q)
            syndrome = extract_syndrome(lattice, error, "X")
            assert np.sum(syndrome) > 0, f"X error on qubit {q} produced zero syndrome"

    def test_single_z_error_nonzero_syndrome(self):
        lattice = SurfaceCodeLattice(3)
        for q in range(9):
            error = apply_error(lattice, "Z", q)
            syndrome = extract_syndrome(lattice, error, "Z")
            assert np.sum(syndrome) > 0, f"Z error on qubit {q} produced zero syndrome"

    def test_syndrome_is_binary(self):
        lattice = SurfaceCodeLattice(5)
        error = apply_error(lattice, "X", [0, 5, 12])
        syndrome = extract_syndrome(lattice, error, "X")
        assert set(np.unique(syndrome)).issubset({0, 1})

    def test_apply_error_invalid_type(self):
        lattice = SurfaceCodeLattice(3)
        with pytest.raises(ValueError):
            apply_error(lattice, "Y", 0)

    def test_apply_error_invalid_qubit(self):
        lattice = SurfaceCodeLattice(3)
        with pytest.raises(ValueError):
            apply_error(lattice, "X", 99)


# --- Decoder ---

class TestMWPMDecoder:
    def test_no_error_no_correction(self):
        lattice = SurfaceCodeLattice(3)
        decoder = MWPMDecoder(lattice)
        syndrome = np.zeros(lattice.num_z_stabilizers, dtype=np.int8)
        correction = decoder.decode_x_errors(syndrome)
        np.testing.assert_array_equal(correction, 0)

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_correct_single_x_errors(self, d):
        lattice = SurfaceCodeLattice(d)
        decoder = MWPMDecoder(lattice)
        for q in range(lattice.num_data):
            error = apply_error(lattice, "X", q)
            syndrome = extract_syndrome(lattice, error, "X")
            correction = decoder.decode_x_errors(syndrome)
            assert not has_logical_error(error, correction, lattice, "X"), \
                f"d={d}: Failed to correct X error on qubit {q}"

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_correct_single_z_errors(self, d):
        lattice = SurfaceCodeLattice(d)
        decoder = MWPMDecoder(lattice)
        for q in range(lattice.num_data):
            error = apply_error(lattice, "Z", q)
            syndrome = extract_syndrome(lattice, error, "Z")
            correction = decoder.decode_z_errors(syndrome)
            assert not has_logical_error(error, correction, lattice, "Z"), \
                f"d={d}: Failed to correct Z error on qubit {q}"

    def test_correction_clears_syndrome_d3(self):
        lattice = SurfaceCodeLattice(3)
        decoder = MWPMDecoder(lattice)
        for q in range(9):
            error = apply_error(lattice, "X", q)
            syndrome = extract_syndrome(lattice, error, "X")
            correction = decoder.decode_x_errors(syndrome)
            residual = (error + correction) % 2
            new_syndrome = extract_syndrome(lattice, residual.astype(np.int8), "X")
            np.testing.assert_array_equal(new_syndrome, 0,
                err_msg=f"Correction did not clear syndrome for X error on qubit {q}")


# --- Logical operators ---

class TestLogicalOperators:
    def test_logical_x_support(self):
        lattice = SurfaceCodeLattice(3)
        lx = logical_x(lattice)
        assert np.sum(lx) == 3
        assert lx[lattice.data_index[(0, 0)]] == 1
        assert lx[lattice.data_index[(1, 0)]] == 1
        assert lx[lattice.data_index[(2, 0)]] == 1

    def test_logical_z_support(self):
        lattice = SurfaceCodeLattice(3)
        lz = logical_z(lattice)
        assert np.sum(lz) == 3
        assert lz[lattice.data_index[(0, 0)]] == 1
        assert lz[lattice.data_index[(0, 1)]] == 1
        assert lz[lattice.data_index[(0, 2)]] == 1

    def test_logical_x_commutes_with_stabilizers(self):
        for d in [3, 5, 7]:
            lattice = SurfaceCodeLattice(d)
            lx = logical_x(lattice)
            Hz = lattice.stabilizer_matrix_z()
            overlaps = (Hz @ lx) % 2
            np.testing.assert_array_equal(overlaps, 0)

    def test_logical_z_commutes_with_stabilizers(self):
        for d in [3, 5, 7]:
            lattice = SurfaceCodeLattice(d)
            lz = logical_z(lattice)
            Hx = lattice.stabilizer_matrix_x()
            overlaps = (Hx @ lz) % 2
            np.testing.assert_array_equal(overlaps, 0)

    def test_logical_x_z_anticommute(self):
        for d in [3, 5, 7]:
            lattice = SurfaceCodeLattice(d)
            lx = logical_x(lattice)
            lz = logical_z(lattice)
            overlap = int(np.sum(lx * lz)) % 2
            assert overlap == 1, f"Logical X and Z should anticommute for d={d}"

    def test_apply_logical_x(self):
        lattice = SurfaceCodeLattice(3)
        error = np.zeros(9, dtype=np.int8)
        result = apply_logical_x(error, lattice)
        assert np.sum(result) == 3

    def test_apply_logical_z(self):
        lattice = SurfaceCodeLattice(3)
        error = np.zeros(9, dtype=np.int8)
        result = apply_logical_z(error, lattice)
        assert np.sum(result) == 3

    def test_double_logical_x_is_identity(self):
        lattice = SurfaceCodeLattice(3)
        error = np.zeros(9, dtype=np.int8)
        result = apply_logical_x(apply_logical_x(error, lattice), lattice)
        np.testing.assert_array_equal(result, 0)


# --- Logical error detection ---

class TestHasLogicalError:
    def test_no_error_no_logical_error(self):
        lattice = SurfaceCodeLattice(3)
        error = np.zeros(9, dtype=np.int8)
        correction = np.zeros(9, dtype=np.int8)
        assert not has_logical_error(error, correction, lattice, "X")

    def test_logical_x_is_logical_error(self):
        lattice = SurfaceCodeLattice(3)
        error = logical_x(lattice)
        correction = np.zeros(9, dtype=np.int8)
        assert has_logical_error(error, correction, lattice, "X")

    def test_stabilizer_is_not_logical_error(self):
        """An error that equals a stabilizer should not be a logical error."""
        lattice = SurfaceCodeLattice(3)
        stab_error = np.zeros(9, dtype=np.int8)
        for q in lattice.x_stabilizers[0]:
            stab_error[q] = 1
        correction = np.zeros(9, dtype=np.int8)
        assert not has_logical_error(stab_error, correction, lattice, "Z")


# --- Threshold estimation ---

class TestThreshold:
    def test_low_error_rate_low_logical_rate(self):
        rate = logical_error_rate(3, 0.01, num_trials=200, seed=42)
        assert rate < 0.2

    def test_high_error_rate_high_logical_rate(self):
        rate = logical_error_rate(3, 0.20, num_trials=200, seed=42)
        assert rate > 0.1

    def test_logical_rate_increases_with_p(self):
        rate_low = logical_error_rate(3, 0.02, num_trials=300, seed=42)
        rate_high = logical_error_rate(3, 0.15, num_trials=300, seed=42)
        assert rate_high > rate_low

    def test_random_error_shape(self):
        lattice = SurfaceCodeLattice(5)
        rng = np.random.default_rng(42)
        error = random_error(lattice, "X", 0.1, rng)
        assert error.shape == (25,)
        assert set(np.unique(error)).issubset({0, 1})
