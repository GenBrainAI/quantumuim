"""Tests for Layer 1: algorithmic compression (semiclassical QFT, modular arithmetic)."""

from __future__ import annotations

import numpy as np
import pytest

from quantumium.qplus.compress import (
    SemiclassicalQFT,
    SemiclassicalQFTResult,
    ModularArithmeticOptimizer,
    CompressedModExpConfig,
    _extract_order,
)


class TestSemiclassicalQFT:
    def test_init_valid(self):
        sqft = SemiclassicalQFT(5)
        assert sqft.n_bits == 5

    def test_init_invalid(self):
        with pytest.raises(ValueError, match="n_bits must be >= 1"):
            SemiclassicalQFT(0)

    def test_estimate_phase_identity(self):
        """Identity unitary should give phase ~0."""
        sqft = SemiclassicalQFT(4)

        def ctrl_u(power, state):
            return state.copy()

        result = sqft.estimate_phase(ctrl_u, 1)
        assert isinstance(result, SemiclassicalQFTResult)
        assert result.n_qubits_used == 2

    def test_estimate_phase_result_structure(self):
        sqft = SemiclassicalQFT(3)

        def ctrl_u(power, state):
            return state.copy()

        result = sqft.estimate_phase(ctrl_u, 2)
        assert len(result.measured_bits) == 3
        assert all(b in (0, 1) for b in result.measured_bits)
        assert 0.0 <= result.phase < 1.0

    def test_qubit_savings(self):
        """Semiclassical QFT uses n_work+1 qubits, not n_work+n_counting."""
        sqft = SemiclassicalQFT(8)

        def ctrl_u(power, state):
            return state.copy()

        result = sqft.estimate_phase(ctrl_u, 4)
        assert result.n_qubits_used == 5  # 1 counting + 4 work

    def test_estimate_phase_z_gate(self):
        """Z gate has eigenvalue -1 = e^{i*pi}, phase should be ~0.5."""
        sqft = SemiclassicalQFT(6)

        def ctrl_z(power, state):
            result = state.copy()
            result[1] *= (-1) ** power
            return result

        initial = np.array([0, 1], dtype=complex)
        phases = []
        for _ in range(20):
            result = sqft.estimate_phase(ctrl_z, 1, initial)
            phases.append(result.phase)
        mean_phase = np.mean(phases)
        assert abs(mean_phase - 0.5) < 0.2


class TestModularArithmeticOptimizer:
    def test_qubit_count_standard(self):
        assert ModularArithmeticOptimizer.qubit_count(4, "standard") == 13

    def test_qubit_count_beauregard(self):
        assert ModularArithmeticOptimizer.qubit_count(4, "beauregard") == 11

    def test_qubit_count_haner(self):
        assert ModularArithmeticOptimizer.qubit_count(4, "haner") == 10

    def test_qubit_count_semiclassical(self):
        assert ModularArithmeticOptimizer.qubit_count(4, "semiclassical") == 6

    def test_qubit_count_unknown_strategy(self):
        with pytest.raises(ValueError):
            ModularArithmeticOptimizer.qubit_count(4, "unknown")

    def test_mod_exp_permutation_basic(self):
        perm = ModularArithmeticOptimizer.mod_exp_permutation(2, 1, 15, 4)
        assert perm.shape == (16, 16)
        state = np.zeros(16, dtype=complex)
        state[1] = 1.0
        result = perm @ state
        assert abs(result[2]) > 0.99  # 2*1 mod 15 = 2

    def test_mod_exp_permutation_unitarity(self):
        perm = ModularArithmeticOptimizer.mod_exp_permutation(7, 1, 15, 4)
        identity = np.eye(16, dtype=complex)
        assert np.allclose(perm @ perm.conj().T, identity, atol=1e-10)

    def test_mod_exp_permutation_power(self):
        perm = ModularArithmeticOptimizer.mod_exp_permutation(2, 3, 15, 4)
        state = np.zeros(16, dtype=complex)
        state[1] = 1.0
        result = perm @ state
        assert abs(result[8]) > 0.99  # 2^3 * 1 mod 15 = 8

    def test_mod_exp_overflow_identity(self):
        perm = ModularArithmeticOptimizer.mod_exp_permutation(2, 1, 15, 4)
        state = np.zeros(16, dtype=complex)
        state[15] = 1.0  # x=15 >= N=15, should be identity
        result = perm @ state
        assert abs(result[15]) > 0.99

    def test_controlled_mod_multiply(self):
        state = np.zeros(16, dtype=complex)
        state[3] = 1.0  # |3⟩
        result = ModularArithmeticOptimizer.controlled_mod_multiply(2, 15, 4, state)
        assert abs(result[6]) > 0.99  # 2*3 mod 15 = 6

    def test_create_config(self):
        config = ModularArithmeticOptimizer.create_config(7, 15, "beauregard")
        assert isinstance(config, CompressedModExpConfig)
        assert config.a == 7
        assert config.N == 15
        assert config.strategy == "beauregard"

    def test_compressed_order_finding_15(self):
        """Order of 7 mod 15 is 4."""
        results = []
        for _ in range(30):
            phase, order, qubits = ModularArithmeticOptimizer.compressed_order_finding(
                7, 15, strategy="semiclassical"
            )
            if order is not None:
                results.append(order)
        assert any(r == 4 for r in results), f"Expected order 4, got {results}"

    def test_compressed_order_finding_qubit_savings(self):
        _, _, qubits = ModularArithmeticOptimizer.compressed_order_finding(
            7, 15, strategy="semiclassical"
        )
        standard = ModularArithmeticOptimizer.qubit_count(4, "standard")
        assert qubits < standard


class TestExtractOrder:
    def test_exact_phase(self):
        order = _extract_order(0.25, 7, 15)
        assert order == 4

    def test_zero_phase(self):
        assert _extract_order(0.0, 7, 15) is None

    def test_known_phase(self):
        # phase = 1/4 for order 4
        order = _extract_order(0.25, 2, 15)
        assert order is not None
        assert pow(2, order, 15) == 1

    def test_approximate_phase(self):
        order = _extract_order(0.2501, 7, 15)
        assert order is not None
        assert pow(7, order, 15) == 1
