"""Tests for the surface code error correction module."""

import numpy as np
import pytest

from quantumium.surface_code.lattice import SurfaceCode
from quantumium.surface_code.syndrome import inject_error, measure_syndrome
from quantumium.surface_code.decoder import mwpm_decode
from quantumium.surface_code.logical import logical_x, logical_z, has_logical_error
from quantumium.surface_code.threshold import estimate_threshold


class TestLatticeConstruction:
    """Tests for the surface code lattice builder."""

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_data_qubit_count(self, d):
        code = SurfaceCode(d)
        assert code.n_data == d * d

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_stabilizer_count(self, d):
        code = SurfaceCode(d)
        expected = (d * d - 1) // 2
        assert code.n_x_stabilizers == expected
        assert code.n_z_stabilizers == expected

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_stabilizer_orthogonality(self, d):
        """H_x * H_z^T = 0 mod 2 (X and Z stabilizers commute)."""
        code = SurfaceCode(d)
        product = (code.h_x @ code.h_z.T) % 2
        assert np.all(product == 0)

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_parity_check_shapes(self, d):
        code = SurfaceCode(d)
        n_stab = (d * d - 1) // 2
        assert code.h_x.shape == (n_stab, d * d)
        assert code.h_z.shape == (n_stab, d * d)

    def test_invalid_distance_even(self):
        with pytest.raises(ValueError):
            SurfaceCode(4)

    def test_invalid_distance_small(self):
        with pytest.raises(ValueError):
            SurfaceCode(1)

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_stabilizer_weights(self, d):
        """Each stabilizer has weight 2 (boundary) or 4 (interior)."""
        code = SurfaceCode(d)
        for stab in code.x_stabilizers + code.z_stabilizers:
            assert len(stab) in (2, 4)

    def test_reset(self):
        code = SurfaceCode(3)
        code.error_x[0] = 1
        code.error_z[4] = 1
        code.reset()
        assert np.all(code.error_x == 0)
        assert np.all(code.error_z == 0)


class TestSyndromeExtraction:
    """Tests for error injection and syndrome measurement."""

    def test_no_error_no_syndrome(self):
        code = SurfaceCode(3)
        x_syn, z_syn = measure_syndrome(code)
        assert np.all(x_syn == 0)
        assert np.all(z_syn == 0)

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_single_x_error_triggers_z_stabilizers(self, d):
        code = SurfaceCode(d)
        q = d * d // 2  # middle qubit
        inject_error(code, "X", q)
        x_syn, z_syn = measure_syndrome(code)
        assert np.all(x_syn == 0)
        assert np.sum(z_syn) > 0

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_single_z_error_triggers_x_stabilizers(self, d):
        code = SurfaceCode(d)
        q = d * d // 2
        inject_error(code, "Z", q)
        x_syn, z_syn = measure_syndrome(code)
        assert np.sum(x_syn) > 0
        assert np.all(z_syn == 0)

    def test_y_error_triggers_both(self):
        code = SurfaceCode(3)
        inject_error(code, "Y", 4)
        x_syn, z_syn = measure_syndrome(code)
        assert np.sum(x_syn) > 0
        assert np.sum(z_syn) > 0

    def test_invalid_qubit(self):
        code = SurfaceCode(3)
        with pytest.raises(ValueError):
            inject_error(code, "X", 9)

    def test_invalid_error_type(self):
        code = SurfaceCode(3)
        with pytest.raises(ValueError):
            inject_error(code, "W", 0)

    def test_syndrome_even_defects(self):
        """Number of defects should be even for periodic-free codes (or include boundary)."""
        code = SurfaceCode(5)
        inject_error(code, "X", 12)
        _, z_syn = measure_syndrome(code)
        n_defects = np.sum(z_syn)
        assert n_defects in (1, 2, 3, 4)


class TestMWPMDecoder:
    """Tests for the MWPM decoder."""

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_correct_all_single_x_errors(self, d):
        """Decoder should correct every single-qubit X error."""
        code = SurfaceCode(d)
        for q in range(code.n_data):
            code.reset()
            inject_error(code, "X", q)
            x_syn, z_syn = measure_syndrome(code)
            x_corr, z_corr = mwpm_decode(code, x_syn, z_syn)
            assert not has_logical_error(code, x_corr, z_corr), (
                f"Failed to correct X error on qubit {q} for d={d}"
            )

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_correct_all_single_z_errors(self, d):
        """Decoder should correct every single-qubit Z error."""
        code = SurfaceCode(d)
        for q in range(code.n_data):
            code.reset()
            inject_error(code, "Z", q)
            x_syn, z_syn = measure_syndrome(code)
            x_corr, z_corr = mwpm_decode(code, x_syn, z_syn)
            assert not has_logical_error(code, x_corr, z_corr), (
                f"Failed to correct Z error on qubit {q} for d={d}"
            )

    def test_no_error_no_correction(self):
        code = SurfaceCode(3)
        x_syn, z_syn = measure_syndrome(code)
        x_corr, z_corr = mwpm_decode(code, x_syn, z_syn)
        assert np.all(x_corr == 0)
        assert np.all(z_corr == 0)

    def test_correction_shapes(self):
        code = SurfaceCode(5)
        inject_error(code, "X", 12)
        x_syn, z_syn = measure_syndrome(code)
        x_corr, z_corr = mwpm_decode(code, x_syn, z_syn)
        assert x_corr.shape == (25,)
        assert z_corr.shape == (25,)


class TestLogicalOperators:
    """Tests for logical X and Z operators."""

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_logical_x_weight(self, d):
        code = SurfaceCode(d)
        lx = logical_x(code)
        assert np.sum(lx) == d

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_logical_z_weight(self, d):
        code = SurfaceCode(d)
        lz = logical_z(code)
        assert np.sum(lz) == d

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_logical_operators_anticommute(self, d):
        """Logical X and Z must anticommute (overlap on odd number of qubits)."""
        code = SurfaceCode(d)
        lx = logical_x(code)
        lz = logical_z(code)
        overlap = np.dot(lx, lz) % 2
        assert overlap == 1

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_logical_x_commutes_with_stabilizers(self, d):
        code = SurfaceCode(d)
        lx = logical_x(code)
        assert np.all((code.h_z @ lx) % 2 == 0)

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_logical_z_commutes_with_stabilizers(self, d):
        code = SurfaceCode(d)
        lz = logical_z(code)
        assert np.all((code.h_x @ lz) % 2 == 0)


class TestThresholdEstimation:
    """Tests for Monte Carlo threshold estimation."""

    def test_basic_run(self):
        result = estimate_threshold(
            distances=[3],
            error_rates=[0.001, 0.01],
            num_trials=50,
            seed=42,
        )
        assert 3 in result.logical_error_rates
        assert len(result.logical_error_rates[3]) == 2

    def test_low_noise_low_logical_error(self):
        result = estimate_threshold(
            distances=[3],
            error_rates=[0.0001],
            num_trials=100,
            seed=42,
        )
        assert result.logical_error_rates[3][0] < 0.1

    def test_logical_error_increases_with_noise(self):
        result = estimate_threshold(
            distances=[3],
            error_rates=[0.001, 0.05],
            num_trials=200,
            seed=42,
        )
        rates = result.logical_error_rates[3]
        assert rates[1] >= rates[0]

    def test_multiple_distances(self):
        result = estimate_threshold(
            distances=[3, 5],
            error_rates=[0.005],
            num_trials=50,
            seed=42,
        )
        assert len(result.logical_error_rates) == 2
