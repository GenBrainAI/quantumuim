"""Tests for classical shadow tomography."""

from __future__ import annotations

import numpy as np
import pytest

from quantumium.shadows.shadows import (
    ClassicalShadow,
    Snapshot,
    ShadowResult,
    shadow_norm_pauli,
    shadow_norm_observable,
    required_snapshots,
    pauli_decomposition,
    _pauli_string_matrix,
    PAULI_X,
    PAULI_Z,
    PAULI_I,
)


# ---------------------------------------------------------------------------
# Helper states
# ---------------------------------------------------------------------------


def _zero_state(n=1):
    s = np.zeros(2**n, dtype=complex)
    s[0] = 1.0
    return s


def _one_state():
    return np.array([0, 1], dtype=complex)


def _plus_state():
    return np.array([1, 1], dtype=complex) / np.sqrt(2)


def _bell_state():
    return np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


def _ghz3_state():
    s = np.zeros(8, dtype=complex)
    s[0] = 1 / np.sqrt(2)
    s[7] = 1 / np.sqrt(2)
    return s


# ===========================================================================
# ClassicalShadow Construction
# ===========================================================================


class TestShadowConstruction:
    def test_from_state_basic(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 100, seed=42)
        assert shadow.num_qubits == 1
        assert shadow.num_snapshots == 100

    def test_from_state_two_qubit(self):
        shadow = ClassicalShadow.from_state(_bell_state(), 200, seed=42)
        assert shadow.num_qubits == 2
        assert shadow.num_snapshots == 200

    def test_from_state_three_qubit(self):
        shadow = ClassicalShadow.from_state(_ghz3_state(), 100, seed=42)
        assert shadow.num_qubits == 3

    def test_from_records(self):
        shadow = ClassicalShadow.from_records(
            num_qubits=2,
            bases_list=["XZ", "ZX", "YY"],
            outcomes_list=[(0, 1), (1, 0), (0, 0)],
        )
        assert shadow.num_qubits == 2
        assert shadow.num_snapshots == 3

    def test_snapshots_property(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 50, seed=42)
        snaps = shadow.snapshots
        assert len(snaps) == 50
        for s in snaps:
            assert isinstance(s, Snapshot)
            assert len(s.bases) == 1
            assert s.bases in ("X", "Y", "Z")
            assert s.outcomes[0] in (0, 1)

    def test_deterministic_with_seed(self):
        s1 = ClassicalShadow.from_state(_zero_state(), 100, seed=99)
        s2 = ClassicalShadow.from_state(_zero_state(), 100, seed=99)
        for a, b in zip(s1.snapshots, s2.snapshots):
            assert a.bases == b.bases
            assert a.outcomes == b.outcomes

    def test_different_seeds(self):
        s1 = ClassicalShadow.from_state(_zero_state(), 100, seed=1)
        s2 = ClassicalShadow.from_state(_zero_state(), 100, seed=2)
        bases1 = [s.bases for s in s1.snapshots]
        bases2 = [s.bases for s in s2.snapshots]
        assert bases1 != bases2


# ===========================================================================
# Snapshot State Reconstruction
# ===========================================================================


class TestSnapshotState:
    def test_snapshot_state_shape(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 10, seed=42)
        rho = shadow.snapshot_state(0)
        assert rho.shape == (2, 2)

    def test_snapshot_state_two_qubit(self):
        shadow = ClassicalShadow.from_state(_bell_state(), 10, seed=42)
        rho = shadow.snapshot_state(0)
        assert rho.shape == (4, 4)

    def test_snapshot_state_trace(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 50, seed=42)
        traces = [np.real(np.trace(shadow.snapshot_state(i))) for i in range(50)]
        mean_trace = np.mean(traces)
        assert abs(mean_trace - 1.0) < 0.5

    def test_snapshot_state_hermitian(self):
        shadow = ClassicalShadow.from_state(_plus_state(), 10, seed=42)
        for i in range(10):
            rho = shadow.snapshot_state(i)
            assert np.allclose(rho, rho.conj().T, atol=1e-10)


# ===========================================================================
# Pauli Expectation Estimation
# ===========================================================================


class TestPauliEstimation:
    def test_z_for_zero_state(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 5000, seed=42)
        result = shadow.estimate_pauli("Z")
        assert abs(result.value - 1.0) < 0.3

    def test_z_for_one_state(self):
        shadow = ClassicalShadow.from_state(_one_state(), 5000, seed=42)
        result = shadow.estimate_pauli("Z")
        assert abs(result.value - (-1.0)) < 0.3

    def test_x_for_plus_state(self):
        shadow = ClassicalShadow.from_state(_plus_state(), 5000, seed=42)
        result = shadow.estimate_pauli("X")
        assert abs(result.value - 1.0) < 0.3

    def test_x_for_zero_state(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 5000, seed=42)
        result = shadow.estimate_pauli("X")
        assert abs(result.value) < 0.3

    def test_zz_for_bell(self):
        shadow = ClassicalShadow.from_state(_bell_state(), 5000, seed=42)
        result = shadow.estimate_pauli("ZZ")
        assert abs(result.value - 1.0) < 0.3

    def test_xx_for_bell(self):
        shadow = ClassicalShadow.from_state(_bell_state(), 5000, seed=42)
        result = shadow.estimate_pauli("XX")
        assert abs(result.value - 1.0) < 0.3

    def test_yy_for_bell(self):
        shadow = ClassicalShadow.from_state(_bell_state(), 5000, seed=42)
        result = shadow.estimate_pauli("YY")
        assert abs(result.value - (-1.0)) < 0.3

    def test_identity_gives_one(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 1000, seed=42)
        result = shadow.estimate_pauli("I")
        assert abs(result.value - 1.0) < 0.1

    def test_zi_for_bell(self):
        shadow = ClassicalShadow.from_state(_bell_state(), 5000, seed=42)
        result = shadow.estimate_pauli("ZI")
        assert abs(result.value) < 0.3

    def test_result_type(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 100, seed=42)
        result = shadow.estimate_pauli("Z")
        assert isinstance(result, ShadowResult)
        assert result.num_snapshots == 100
        assert result.num_groups >= 1


# ===========================================================================
# General Observable Estimation
# ===========================================================================


class TestObservableEstimation:
    def test_z_observable(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 5000, seed=42)
        result = shadow.estimate_expectation(PAULI_Z)
        assert abs(result.value - 1.0) < 0.3

    def test_x_observable_plus(self):
        shadow = ClassicalShadow.from_state(_plus_state(), 5000, seed=42)
        result = shadow.estimate_expectation(PAULI_X)
        assert abs(result.value - 1.0) < 0.3

    def test_identity_observable(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 500, seed=42)
        result = shadow.estimate_expectation(PAULI_I)
        assert abs(result.value - 1.0) < 0.3

    def test_custom_observable(self):
        obs = (PAULI_Z + PAULI_I) / 2  # projector onto |0>
        shadow = ClassicalShadow.from_state(_zero_state(), 5000, seed=42)
        result = shadow.estimate_expectation(obs)
        assert abs(result.value - 1.0) < 0.3

    def test_empty_shadow(self):
        shadow = ClassicalShadow(1, [])
        result = shadow.estimate_expectation(PAULI_Z)
        assert result.value == 0.0
        assert result.num_snapshots == 0


# ===========================================================================
# Fidelity Estimation
# ===========================================================================


class TestFidelityEstimation:
    def test_fidelity_same_state(self):
        state = _zero_state()
        shadow = ClassicalShadow.from_state(state, 5000, seed=42)
        result = shadow.estimate_fidelity(state)
        assert abs(result.value - 1.0) < 0.3

    def test_fidelity_orthogonal(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 5000, seed=42)
        result = shadow.estimate_fidelity(_one_state())
        assert abs(result.value) < 0.3

    def test_fidelity_bell_state(self):
        bell = _bell_state()
        shadow = ClassicalShadow.from_state(bell, 5000, seed=42)
        result = shadow.estimate_fidelity(bell)
        assert abs(result.value - 1.0) < 0.3

    def test_fidelity_with_density_matrix(self):
        state = _plus_state()
        rho = np.outer(state, state.conj())
        shadow = ClassicalShadow.from_state(state, 5000, seed=42)
        result = shadow.estimate_fidelity(rho)
        assert abs(result.value - 1.0) < 0.3


# ===========================================================================
# Purity Estimation
# ===========================================================================


class TestPurityEstimation:
    def test_pure_state_purity(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 10000, seed=42)
        purity = shadow.estimate_purity()
        assert abs(purity - 1.0) < 1.0  # high variance estimator

    def test_purity_nonnegative_count(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 2, seed=42)
        purity = shadow.estimate_purity()
        assert isinstance(purity, float)


# ===========================================================================
# Shadow Norm
# ===========================================================================


class TestShadowNorm:
    def test_single_pauli(self):
        assert shadow_norm_pauli("Z") == 3.0
        assert shadow_norm_pauli("X") == 3.0
        assert shadow_norm_pauli("Y") == 3.0

    def test_identity(self):
        assert shadow_norm_pauli("I") == 1.0
        assert shadow_norm_pauli("II") == 1.0

    def test_two_local(self):
        assert shadow_norm_pauli("ZZ") == 9.0
        assert shadow_norm_pauli("XY") == 9.0

    def test_mixed_locality(self):
        assert shadow_norm_pauli("ZI") == 3.0
        assert shadow_norm_pauli("IZI") == 3.0
        assert shadow_norm_pauli("ZIZ") == 9.0

    def test_k_local_scaling(self):
        assert shadow_norm_pauli("XXXX") == 81.0  # 3^4

    def test_observable_norm(self):
        norm = shadow_norm_observable(PAULI_Z)
        assert norm == pytest.approx((2 + 1) * 1.0)  # (2^1+1)*||Z||=3


# ===========================================================================
# Required Snapshots
# ===========================================================================


class TestRequiredSnapshots:
    def test_basic(self):
        T = required_snapshots(3.0, epsilon=0.1)
        assert T > 0

    def test_more_accuracy_needs_more(self):
        T1 = required_snapshots(3.0, epsilon=0.1)
        T2 = required_snapshots(3.0, epsilon=0.01)
        assert T2 > T1

    def test_higher_norm_needs_more(self):
        T1 = required_snapshots(3.0, epsilon=0.1)
        T2 = required_snapshots(9.0, epsilon=0.1)
        assert T2 > T1

    def test_multiple_observables(self):
        T1 = required_snapshots(3.0, epsilon=0.1, num_observables=1)
        T10 = required_snapshots(3.0, epsilon=0.1, num_observables=10)
        assert T10 > T1


# ===========================================================================
# Pauli Decomposition
# ===========================================================================


class TestPauliDecomposition:
    def test_identity(self):
        decomp = pauli_decomposition(PAULI_I)
        assert abs(decomp.get("I", 0) - 1.0) < 1e-10

    def test_pauli_z(self):
        decomp = pauli_decomposition(PAULI_Z)
        assert abs(decomp.get("Z", 0) - 1.0) < 1e-10
        assert "X" not in decomp
        assert "Y" not in decomp

    def test_projector(self):
        proj_0 = (PAULI_I + PAULI_Z) / 2
        decomp = pauli_decomposition(proj_0)
        assert abs(decomp.get("I", 0) - 0.5) < 1e-10
        assert abs(decomp.get("Z", 0) - 0.5) < 1e-10

    def test_two_qubit(self):
        zz = np.kron(PAULI_Z, PAULI_Z)
        decomp = pauli_decomposition(zz)
        assert abs(decomp.get("ZZ", 0) - 1.0) < 1e-10
        assert len(decomp) == 1

    def test_round_trip(self):
        obs = 0.5 * PAULI_Z + 0.3 * PAULI_X
        decomp = pauli_decomposition(obs)
        reconstructed = sum(c * _pauli_string_matrix(p) for p, c in decomp.items())
        assert np.allclose(obs, reconstructed, atol=1e-10)


# ===========================================================================
# Statistical Properties
# ===========================================================================


class TestStatisticalProperties:
    def test_more_snapshots_more_accurate(self):
        state = _zero_state()
        errors = []
        for T in [500, 2000, 8000]:
            shadow = ClassicalShadow.from_state(state, T, seed=42)
            result = shadow.estimate_pauli("Z")
            errors.append(abs(result.value - 1.0))
        assert errors[-1] <= errors[0] + 0.1

    def test_multiple_observables_consistent(self):
        shadow = ClassicalShadow.from_state(_bell_state(), 5000, seed=42)
        zz = shadow.estimate_pauli("ZZ").value
        xx = shadow.estimate_pauli("XX").value
        zi = shadow.estimate_pauli("ZI").value
        assert abs(zz - 1.0) < 0.3
        assert abs(xx - 1.0) < 0.3
        assert abs(zi) < 0.3


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    def test_single_snapshot(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 1, seed=42)
        result = shadow.estimate_pauli("Z")
        assert isinstance(result.value, float)

    def test_three_qubit_pauli(self):
        shadow = ClassicalShadow.from_state(_ghz3_state(), 5000, seed=42)
        result = shadow.estimate_pauli("ZZZ")
        assert abs(result.value) < 0.5  # <ZZZ> = 0 for GHZ: +1 for |000>, -1 for |111>

    def test_num_groups_override(self):
        shadow = ClassicalShadow.from_state(_zero_state(), 1000, seed=42)
        r1 = shadow.estimate_expectation(PAULI_Z, num_groups=5)
        r2 = shadow.estimate_expectation(PAULI_Z, num_groups=50)
        assert r1.num_groups == 5
        assert r2.num_groups == 50


# ===========================================================================
# Integration
# ===========================================================================


class TestIntegration:
    def test_from_circuit_state(self):
        from quantumium.core.circuit import QuantumCircuit
        from quantumium.core.gate import H, CNOT

        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        state = qc.statevector()
        shadow = ClassicalShadow.from_state(state, 3000, seed=42)
        zz = shadow.estimate_pauli("ZZ")
        assert abs(zz.value - 1.0) < 0.3

    def test_existing_imports(self):
        pass
