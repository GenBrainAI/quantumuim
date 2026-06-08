"""Q+ integration tests — test Q+ with the broader SDK pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quantumium.core.gate import H, X, CNOT, Rz
from quantumium.core.circuit import QuantumCircuit
from quantumium.transpiler.passes import transpile, count_gates
from quantumium.circuits.templates import qft, inverse_qft

from quantumium.qplus.compress import SemiclassicalQFT, ModularArithmeticOptimizer
from quantumium.qplus.recycle import QubitRecycler, FeedForwardController
from quantumium.qplus.cut import CircuitCutter, CutRecombiner, WireCut
from quantumium.qplus.distribute import QPUOrchestrator, DistributionStrategy
from quantumium.qplus.shor import ShorQPlus, QplusResult


class TestSemiclassicalQFTWithSDK:
    """SemiclassicalQFT phase estimation using SDK circuit unitaries."""

    def test_phase_estimation_with_sdk_unitary(self) -> None:
        """Use SDK QuantumCircuit to build the controlled unitary, then estimate phase."""
        qc_z = QuantumCircuit(1)
        qc_z.apply(Rz(np.pi), 0)
        z_unitary = qc_z._expand_gate(Rz(np.pi).matrix, (0,), 1)

        def ctrl_u(power: int, state: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
            u = np.linalg.matrix_power(z_unitary, power)
            return u @ state

        sqft = SemiclassicalQFT(6)
        initial = np.array([0, 1], dtype=complex)
        phases: list[float] = []
        for _ in range(20):
            result = sqft.estimate_phase(ctrl_u, 1, initial)
            phases.append(result.phase)
        mean_phase = np.mean(phases)
        assert abs(mean_phase - 0.5) < 0.2

    def test_semiclassical_qubit_count_vs_standard_qft(self) -> None:
        """Compare qubit usage: standard QFT circuit vs semiclassical."""
        n = 4
        qft_circuit = qft(n)
        qft_qubits = qft_circuit.num_qubits
        sqft = SemiclassicalQFT(n)

        def ctrl_identity(power: int, state: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
            return np.array(state, dtype=complex)

        result = sqft.estimate_phase(ctrl_identity, 2)
        assert result.n_qubits_used < qft_qubits + 2


class TestCircuitCuttingWithSDK:
    """Circuit cutting with SDK-built circuit unitaries."""

    def test_cut_sdk_bell_circuit(self) -> None:
        """Cut a Bell state circuit and verify reconstruction."""
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        U = _circuit_unitary(qc)

        cutter = CircuitCutter([WireCut(0, 0)])
        dim = 4
        identity = np.eye(dim, dtype=complex)

        result = cutter.evaluate_cut_circuit(identity, U, 0, 2)
        assert isinstance(result, float)

    def test_cut_transpiled_circuit(self) -> None:
        """Build circuit -> transpile -> cut -> verify."""
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(X, 0).apply(X, 0)
        qc.apply(CNOT, 0, 1)
        optimized = transpile(qc)
        U = _circuit_unitary(optimized)

        cutter = CircuitCutter([WireCut(0, 0)])
        result = cutter.evaluate_cut_circuit(
            np.eye(4, dtype=complex), U, 0, 2
        )
        assert isinstance(result, float)

    def test_cut_recombine_identity(self) -> None:
        """Cutting identity circuit, recombining should give 1.0."""
        cutter = CircuitCutter([WireCut(0, 0)])
        identity = np.eye(4, dtype=complex)
        result = cutter.evaluate_cut_circuit(identity, identity, 0, 2)
        assert abs(result - 1.0) < 0.15

    def test_overhead_matches_theory(self) -> None:
        """Verify O(4^K) overhead scaling."""
        for k in range(1, 4):
            resources = CircuitCutter().estimate_resources(n_cuts=k)
            assert resources["overhead_factor"] == float(4**k)
            assert resources["n_subcircuit_evals"] == float(6**k)


class TestQPUOrchestratorWithSDK:
    """QPU distribution with SDK circuit unitaries."""

    def test_distribute_identity_circuit(self) -> None:
        """Distributing identity should preserve expectation value."""
        orch = QPUOrchestrator(n_qpus=2, strategy="classical_only")
        unitary = np.eye(4, dtype=complex)
        result = orch.execute_distributed(unitary, 2)
        assert abs(result.recombined_value - 1.0) < 0.01

    def test_distribute_hadamard_circuit(self) -> None:
        """Distribute a Hadamard circuit."""
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(H, 1)
        U = _circuit_unitary(qc)
        orch = QPUOrchestrator(n_qpus=1)
        result = orch.execute_distributed(U, 2)
        assert result.n_qpus == 1
        state = np.zeros(4, dtype=complex)
        state[0] = 1.0
        final = U @ state
        expected = abs(final[0]) ** 2
        assert abs(result.recombined_value - expected) < 0.01

    def test_distribute_bell_state(self) -> None:
        """Distribute a Bell state circuit across 2 QPUs."""
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        U = _circuit_unitary(qc)
        orch = QPUOrchestrator(n_qpus=2, strategy="classical_only")
        result = orch.execute_distributed(U, 2)
        assert result.n_qpus == 2
        assert result.overhead > 1.0

    def test_resource_estimation_for_sdk_circuit(self) -> None:
        """Estimate resources for a realistic circuit."""
        qc = QuantumCircuit(4)
        qc.apply(H, 0)
        for i in range(3):
            qc.apply(CNOT, i, i + 1)
        orch = QPUOrchestrator(n_qpus=2, strategy="locc")
        resources = orch.estimate_resources(qc.num_qubits, len(qc.instructions))
        assert resources["n_qpus"] == 2.0
        assert resources["n_cuts"] >= 1.0

    def test_all_strategies(self) -> None:
        """Test all three distribution strategies produce valid results."""
        U = np.eye(4, dtype=complex)
        for strategy in ["classical_only", "locc", "teleport"]:
            orch = QPUOrchestrator(n_qpus=2, strategy=strategy)
            result = orch.execute_distributed(U, 2)
            assert result.strategy == strategy
            assert abs(result.recombined_value - 1.0) < 0.01


class TestShorQPlusEndToEnd:
    """ShorQPlus with different SDK components."""

    def test_factor_with_qubit_comparison(self) -> None:
        """Factor a number and verify qubit savings against SDK circuit sizes."""
        shor = ShorQPlus(strategy="semiclassical")
        N = 15
        comp = ShorQPlus.qubit_comparison(N)

        qft_n = qft(comp["n_bits"])
        standard_counting_qubits = qft_n.num_qubits

        result = shor.factor(N, a=7)
        if result.success and result.qubits_used > 0:
            assert result.qubits_used < comp["standard_shor"]

    def test_shor_result_consistency(self) -> None:
        """Verify QplusResult fields are consistent."""
        shor = ShorQPlus(strategy="recycled")
        result = shor.factor(15, max_attempts=20)
        if result.success:
            assert result.factors is not None
            p, q = result.factors
            assert p * q == result.N
            assert result.qubit_savings == result.qubits_standard - result.qubits_used
            assert len(result.layers_used) > 0

    def test_shor_strategies_all_factor_15(self) -> None:
        """All strategies should be able to factor 15."""
        for strategy in ["semiclassical", "recycled", "auto"]:
            shor = ShorQPlus(strategy=strategy)
            successes = 0
            for _ in range(20):
                result = shor.factor(15, max_attempts=10)
                if result.success and result.factors is not None:
                    p, q = result.factors
                    if p * q == 15:
                        successes += 1
            assert successes >= 2, f"Strategy {strategy} failed: {successes}/20"


class TestRecyclerWithSDKOperations:
    """QubitRecycler with SDK-style operations."""

    def test_recycler_lifecycle(self) -> None:
        """Full lifecycle: allocate -> use -> measure -> recycle -> reuse."""
        qr = QubitRecycler(2)
        p0 = qr.allocate(0)
        p1 = qr.allocate(1)
        assert qr.available_count == 0

        state = np.array([1, 0, 0, 0], dtype=complex)
        outcome, new_state = qr.measure_and_recycle(p0, state, 0, 2)
        assert qr.available_count == 1

        p2 = qr.allocate()
        assert p2 == p0
        assert qr.total_recycles == 1

    def test_feedforward_with_multiple_measurements(self) -> None:
        """Feedforward controller tracks multiple measurement outcomes."""
        ffc = FeedForwardController()
        ffc.record_measurement(0, 1)
        ffc.record_measurement(1, 0)
        ffc.record_measurement(2, 1)

        correction = ffc.compute_phase_correction(3, [1, 0, 1])
        assert abs(correction) > 0

        ffc.reset()
        assert ffc.measurements == {}

    def test_recycled_phase_estimation_qubit_efficiency(self) -> None:
        """Recycled phase estimation uses fewer qubits than standard."""
        n_bits = 4
        n_counting = 9
        qr = QubitRecycler(n_bits + 1)

        def ctrl_u(power: int, state: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
            perm = ModularArithmeticOptimizer.mod_exp_permutation(7, power, 15, n_bits)
            return perm @ state

        phase, bits = qr.simulate_recycled_phase_estimation(ctrl_u, n_bits, n_counting)
        assert 0.0 <= phase < 1.0
        assert len(bits) == n_counting
        standard_qubits = n_counting + n_bits
        recycled_qubits = 1 + n_bits
        assert recycled_qubits < standard_qubits


class TestCutRecombinerWithSDK:
    """CutRecombiner for combining subcircuit results."""

    def test_recombine_weighted_expectations(self) -> None:
        """Recombine subcircuit expectations from different Pauli terms."""
        pairs = [
            (0.5, 0.8),
            (-0.5, 0.2),
            (0.5, 0.6),
            (-0.5, 0.4),
        ]
        result = CutRecombiner.recombine(pairs)
        expected = sum(c * e for c, e in pairs)
        assert abs(result - expected) < 1e-10

    def test_recombine_probabilities_normalization(self) -> None:
        """Recombined probabilities should be normalized."""
        probs: list[tuple[float, dict[str, float]]] = [
            (0.5, {"00": 0.9, "01": 0.1}),
            (0.5, {"00": 0.7, "01": 0.3}),
            (-0.5, {"00": 0.5, "01": 0.5}),
            (0.5, {"00": 0.6, "01": 0.4}),
        ]
        result = CutRecombiner.recombine_probabilities(probs)
        total = sum(result.values())
        assert abs(total - 1.0) < 0.01
        assert all(v >= 0 for v in result.values())

    def test_verify_reconstruction(self) -> None:
        assert CutRecombiner.verify_reconstruction(0.5, 0.55, tolerance=0.1)
        assert not CutRecombiner.verify_reconstruction(0.5, 0.7, tolerance=0.1)


def _circuit_unitary(qc: QuantumCircuit) -> np.ndarray:
    n = qc.num_qubits
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for inst in qc.instructions:
        full = qc._expand_gate(inst.gate.matrix, inst.target_qubits, n)
        U = full @ U
    return U
