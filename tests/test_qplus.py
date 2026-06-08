"""Tests for Q+ Algorithm — beyond-qubit-count quantum framework.

Tests all layers: compress, recycle, cut, distribute, and full Shor integration.
"""

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
from quantumium.qplus.recycle import (
    QubitRecycler,
    FeedForwardController,
)
from quantumium.qplus.cut import (
    CircuitCutter,
    CutRecombiner,
    WireCut,
    PAULI_MATRICES,
    PAULI_PREP_STATES,
    _expand_single_qubit_op,
)
from quantumium.qplus.distribute import (
    QPUOrchestrator,
    DistributedResult,
)
from quantumium.qplus.shor import ShorQPlus, QplusResult


class TestSemiclassicalQFT:
    def test_init_valid(self):
        sqft = SemiclassicalQFT(5)
        assert sqft.n_bits == 5

    def test_init_invalid(self):
        with pytest.raises(ValueError, match="n_bits must be >= 1"):
            SemiclassicalQFT(0)

    def test_estimate_phase_identity(self):
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
        sqft = SemiclassicalQFT(8)

        def ctrl_u(power, state):
            return state.copy()

        result = sqft.estimate_phase(ctrl_u, 4)
        assert result.n_qubits_used == 5

    def test_estimate_phase_z_gate(self):
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
        assert abs(result[2]) > 0.99

    def test_mod_exp_permutation_unitarity(self):
        perm = ModularArithmeticOptimizer.mod_exp_permutation(7, 1, 15, 4)
        identity = np.eye(16, dtype=complex)
        assert np.allclose(perm @ perm.conj().T, identity, atol=1e-10)

    def test_mod_exp_permutation_power(self):
        perm = ModularArithmeticOptimizer.mod_exp_permutation(2, 3, 15, 4)
        state = np.zeros(16, dtype=complex)
        state[1] = 1.0
        result = perm @ state
        assert abs(result[8]) > 0.99

    def test_mod_exp_overflow_identity(self):
        perm = ModularArithmeticOptimizer.mod_exp_permutation(2, 1, 15, 4)
        state = np.zeros(16, dtype=complex)
        state[15] = 1.0
        result = perm @ state
        assert abs(result[15]) > 0.99

    def test_controlled_mod_multiply(self):
        state = np.zeros(16, dtype=complex)
        state[3] = 1.0
        result = ModularArithmeticOptimizer.controlled_mod_multiply(2, 15, 4, state)
        assert abs(result[6]) > 0.99

    def test_create_config(self):
        config = ModularArithmeticOptimizer.create_config(7, 15, "beauregard")
        assert isinstance(config, CompressedModExpConfig)
        assert config.a == 7
        assert config.N == 15

    def test_compressed_order_finding_15(self):
        results = []
        for _ in range(30):
            phase, order, qubits = ModularArithmeticOptimizer.compressed_order_finding(
                7, 15, strategy="semiclassical"
            )
            if order is not None:
                results.append(order)
        assert any(r == 4 for r in results)

    def test_compressed_order_finding_qubit_savings(self):
        _, _, qubits = ModularArithmeticOptimizer.compressed_order_finding(
            7, 15, strategy="semiclassical"
        )
        standard = ModularArithmeticOptimizer.qubit_count(4, "standard")
        assert qubits < standard


class TestExtractOrder:
    def test_exact_phase(self):
        assert _extract_order(0.25, 7, 15) == 4

    def test_zero_phase(self):
        assert _extract_order(0.0, 7, 15) is None

    def test_known_phase(self):
        order = _extract_order(0.25, 2, 15)
        assert order is not None
        assert pow(2, order, 15) == 1

    def test_approximate_phase(self):
        order = _extract_order(0.2501, 7, 15)
        assert order is not None
        assert pow(7, order, 15) == 1


# --- Layer 2: Recycle ---


class TestFeedForwardController:
    def test_record_and_get_measurement(self):
        ffc = FeedForwardController()
        ffc.record_measurement(0, 1)
        assert ffc.get_measurement(0) == 1
        assert ffc.get_measurement(1) is None

    def test_measurements_property(self):
        ffc = FeedForwardController()
        ffc.record_measurement(0, 0)
        ffc.record_measurement(1, 1)
        assert ffc.measurements == {0: 0, 1: 1}

    def test_add_correction(self):
        ffc = FeedForwardController()
        ffc.add_correction(0, 1, "phase", np.pi / 4)
        corrections = ffc.get_corrections(1)
        assert len(corrections) == 1
        assert abs(corrections[0].angle - np.pi / 4) < 1e-10

    def test_compute_phase_correction_no_bits(self):
        ffc = FeedForwardController()
        assert abs(ffc.compute_phase_correction(0, [])) < 1e-15

    def test_compute_phase_correction_with_bits(self):
        ffc = FeedForwardController()
        correction = ffc.compute_phase_correction(2, [1, 0])
        assert abs(correction - (-np.pi / 4)) < 1e-10

    def test_reset(self):
        ffc = FeedForwardController()
        ffc.record_measurement(0, 1)
        ffc.reset()
        assert ffc.measurements == {}


class TestQubitRecycler:
    def test_init(self):
        qr = QubitRecycler(3)
        assert qr.n_physical == 3
        assert qr.available_count == 3

    def test_init_invalid(self):
        with pytest.raises(ValueError):
            QubitRecycler(0)

    def test_allocate(self):
        qr = QubitRecycler(3)
        phys = qr.allocate()
        assert phys == 0
        assert qr.available_count == 2

    def test_allocate_exhaustion(self):
        qr = QubitRecycler(2)
        qr.allocate()
        qr.allocate()
        with pytest.raises(RuntimeError):
            qr.allocate()

    def test_measure_and_recycle_zero_state(self):
        qr = QubitRecycler(2)
        phys = qr.allocate()
        state = np.array([1, 0, 0, 0], dtype=complex)
        outcome, _ = qr.measure_and_recycle(phys, state, 0, 2)
        assert outcome == 0
        assert qr.available_count == 2

    def test_allocate_after_recycle(self):
        qr = QubitRecycler(1)
        phys = qr.allocate()
        state = np.array([1, 0], dtype=complex)
        qr.measure_and_recycle(phys, state, 0, 1)
        phys2 = qr.allocate()
        assert phys2 == phys

    def test_recycled_phase_estimation_identity(self):
        qr = QubitRecycler(3)

        def ctrl_u(power, state):
            return state.copy()

        phase, bits = qr.simulate_recycled_phase_estimation(ctrl_u, 2, 4)
        assert 0.0 <= phase < 1.0
        assert len(bits) == 4

    def test_recycled_order_finding_15(self):
        from quantumium.qplus.compress import _extract_order

        n_bits = 4
        successes = 0
        for _ in range(30):
            qr = QubitRecycler(n_bits + 1)

            def ctrl_u(power, state):
                perm = ModularArithmeticOptimizer.mod_exp_permutation(7, power, 15, n_bits)
                return perm @ state

            phase, bits = qr.simulate_recycled_phase_estimation(ctrl_u, n_bits, 9)
            order = _extract_order(phase, 7, 15)
            if order == 4:
                successes += 1
        assert successes >= 3


# --- Layer 3a: Cut ---


class TestPauliConstants:
    def test_pauli_matrices_hermitian(self):
        for label, mat in PAULI_MATRICES.items():
            assert np.allclose(mat, mat.conj().T)

    def test_pauli_matrices_unitary(self):
        for label, mat in PAULI_MATRICES.items():
            assert np.allclose(mat @ mat.conj().T, np.eye(2))

    def test_pauli_prep_states_normalized(self):
        for label, state in PAULI_PREP_STATES.items():
            assert abs(np.linalg.norm(state) - 1.0) < 1e-10

    def test_pauli_anticommutation(self):
        X, Y, Z = PAULI_MATRICES["X"], PAULI_MATRICES["Y"], PAULI_MATRICES["Z"]
        assert np.allclose(X @ Y + Y @ X, np.zeros((2, 2)))


class TestCircuitCutter:
    def test_init_empty(self):
        cc = CircuitCutter()
        assert cc.n_cuts == 0
        assert cc.overhead == 1.0

    def test_add_cut(self):
        cc = CircuitCutter()
        cc.add_cut(qubit=0, after_gate_index=5)
        assert cc.n_cuts == 1

    def test_overhead_scaling(self):
        cc = CircuitCutter()
        cc.add_cut(0, 0)
        assert cc.overhead == 4.0
        cc.add_cut(1, 1)
        assert cc.overhead == 16.0

    def test_decompose_no_cuts(self):
        cc = CircuitCutter()
        configs = cc.decompose(2, [])
        assert len(configs) == 1

    def test_decompose_one_cut(self):
        cc = CircuitCutter([WireCut(0, 0)])
        configs = cc.decompose(2, [])
        assert len(configs) == 6

    def test_evaluate_cut_identity(self):
        cc = CircuitCutter([WireCut(0, 0)])
        n = 2
        identity = np.eye(2**n, dtype=complex)
        result = cc.evaluate_cut_circuit(identity, identity, 0, n)
        assert abs(result - 1.0) < 0.1

    def test_estimate_resources(self):
        cc = CircuitCutter()
        resources = cc.estimate_resources(n_cuts=3)
        assert resources["overhead_factor"] == 64.0


class TestCutRecombiner:
    def test_recombine_single(self):
        assert abs(CutRecombiner.recombine([(1.0, 0.5)]) - 0.5) < 1e-10

    def test_recombine_multiple(self):
        pairs = [(0.5, 1.0), (-0.5, 0.5)]
        result = CutRecombiner.recombine(pairs)
        assert abs(result - 0.25) < 1e-10

    def test_verify_reconstruction_pass(self):
        assert CutRecombiner.verify_reconstruction(1.0, 1.05, tolerance=0.1)

    def test_verify_reconstruction_fail(self):
        assert not CutRecombiner.verify_reconstruction(1.0, 1.5, tolerance=0.1)


class TestExpandOp:
    def test_single_qubit(self):
        X = PAULI_MATRICES["X"]
        assert np.allclose(_expand_single_qubit_op(X, 0, 1), X)

    def test_two_qubit_first(self):
        X = PAULI_MATRICES["X"]
        result = _expand_single_qubit_op(X, 0, 2)
        assert np.allclose(result, np.kron(X, np.eye(2)))


# --- Layer 3b: Distribute ---


class TestQPUOrchestrator:
    def test_init_default(self):
        orch = QPUOrchestrator()
        assert orch.n_qpus == 2

    def test_init_invalid(self):
        with pytest.raises(ValueError):
            QPUOrchestrator(n_qpus=0)

    def test_partition_single_qpu(self):
        orch = QPUOrchestrator(n_qpus=1)
        plan = orch.partition_circuit(4)
        assert plan.n_qpus == 1
        assert len(plan.cuts) == 0

    def test_partition_two_qpus(self):
        orch = QPUOrchestrator(n_qpus=2)
        plan = orch.partition_circuit(4)
        assert plan.n_qpus == 2
        assert len(plan.cuts) == 1

    def test_overhead_classical_only(self):
        orch = QPUOrchestrator(n_qpus=2, strategy="classical_only")
        plan = orch.partition_circuit(4)
        assert plan.estimated_overhead == 4.0

    def test_overhead_teleport(self):
        orch = QPUOrchestrator(n_qpus=2, strategy="teleport")
        plan = orch.partition_circuit(4)
        assert plan.estimated_overhead == 1.0

    def test_execute_single(self):
        orch = QPUOrchestrator(n_qpus=1)
        result = orch.execute_distributed(np.eye(4, dtype=complex), 2)
        assert isinstance(result, DistributedResult)
        assert abs(result.recombined_value - 1.0) < 0.01

    def test_recommend_strategy(self):
        assert QPUOrchestrator.recommend_strategy(4, 1) == "classical_only"
        assert QPUOrchestrator.recommend_strategy(4, 2, True) == "teleport"


# --- Full Stack: Shor Q+ ---


class TestShorQPlusInit:
    def test_default(self):
        assert ShorQPlus().strategy == "auto"

    def test_invalid(self):
        with pytest.raises(ValueError):
            ShorQPlus(strategy="magic")


class TestShorQPlusTrivial:
    def test_factor_even(self):
        result = ShorQPlus().factor(6)
        assert result.success
        assert result.factors[0] * result.factors[1] == 6

    def test_factor_4(self):
        result = ShorQPlus().factor(4)
        assert result.success
        assert result.factors == (2, 2)

    def test_too_small(self):
        with pytest.raises(ValueError):
            ShorQPlus().factor(3)

    def test_trivial_gcd(self):
        result = ShorQPlus().factor(15, a=3)
        assert result.success
        assert "trivial_gcd" in result.layers_used


class TestShorQPlusQuantum:
    def test_factor_15_semiclassical(self):
        shor = ShorQPlus(strategy="semiclassical")
        successes = sum(1 for _ in range(20) if shor.factor(15, a=7, max_attempts=1).success)
        assert successes >= 3

    def test_factor_15_recycled(self):
        shor = ShorQPlus(strategy="recycled")
        successes = sum(1 for _ in range(20) if shor.factor(15, a=7, max_attempts=1).success)
        assert successes >= 3

    def test_factor_21(self):
        shor = ShorQPlus(strategy="semiclassical")
        successes = sum(1 for _ in range(20) if shor.factor(21, a=2).success)
        assert successes >= 2

    def test_factor_35(self):
        shor = ShorQPlus(strategy="semiclassical")
        successes = sum(1 for _ in range(20) if shor.factor(35, a=2).success)
        assert successes >= 2

    def test_factor_auto(self):
        assert ShorQPlus(strategy="auto").factor(15, a=3).success


class TestQubitSavings:
    def test_savings_15(self):
        comp = ShorQPlus.qubit_comparison(15)
        assert comp["semiclassical_qft"] < comp["standard_shor"]

    def test_savings_21(self):
        comp = ShorQPlus.qubit_comparison(21)
        assert comp["semiclassical_qft"] == 7
        assert comp["standard_shor"] == 16

    def test_savings_scaling(self):
        for N in [15, 21, 35, 77, 143, 323]:
            comp = ShorQPlus.qubit_comparison(N)
            ratio = comp["semiclassical_qft"] / comp["standard_shor"]
            assert ratio < 0.6


class TestShorQPlusSemiprimes:
    @pytest.mark.parametrize("N", [15, 21])
    def test_reliable_factoring(self, N):
        shor = ShorQPlus(strategy="recycled")
        successes = sum(
            1
            for _ in range(30)
            if (r := shor.factor(N, max_attempts=10)).success
            and r.factors is not None
            and r.factors[0] * r.factors[1] == N
        )
        assert successes >= 3

    def test_factor_77(self):
        shor = ShorQPlus(strategy="semiclassical")
        successes = sum(
            1
            for _ in range(20)
            if (r := shor.factor(77, max_attempts=15)).success
            and r.factors is not None
            and r.factors[0] * r.factors[1] == 77
        )
        assert successes >= 1

    def test_factor_143(self):
        shor = ShorQPlus(strategy="semiclassical")
        successes = sum(
            1
            for _ in range(20)
            if (r := shor.factor(143, max_attempts=15)).success
            and r.factors is not None
            and r.factors[0] * r.factors[1] == 143
        )
        assert successes >= 1


class TestQplusResult:
    def test_structure(self):
        result = QplusResult(
            N=15,
            factors=(3, 5),
            success=True,
            qubits_used=6,
            qubits_standard=13,
            qubit_savings=7,
        )
        assert result.qubit_savings == 7

    def test_defaults(self):
        result = QplusResult(
            N=15,
            factors=None,
            success=False,
            qubits_used=6,
            qubits_standard=13,
            qubit_savings=7,
        )
        assert result.layers_used == []
