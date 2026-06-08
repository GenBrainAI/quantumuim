"""Comprehensive tests for LocalSimulatorBackend.

Covers Bell state, GHZ states, Grover's search, QFT round-trip,
deterministic circuits, and statistical validation.
"""

from __future__ import annotations


from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, CNOT, CCX
from quantumium.backends import ExecutionResult
from quantumium.backends.base import Backend

from .conftest import chi_squared_test, hellinger_fidelity


SHOTS = 4096


class TestLocalSimulatorBasics:
    def test_name(self, local_backend):
        assert local_backend.name == "quantumium-local-sim"

    def test_is_available(self, local_backend):
        assert local_backend.is_available() is True

    def test_isinstance_backend(self, local_backend):
        assert isinstance(local_backend, Backend)

    def test_result_type(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=100)
        assert isinstance(result, ExecutionResult)

    def test_result_shots(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=500)
        assert result.shots == 500
        assert sum(result.counts.values()) == 500

    def test_result_backend_name(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=100)
        assert result.backend_name == "quantumium-local-sim"


class TestBellState:
    def test_bell_state_outcomes(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=SHOTS)
        valid_outcomes = {"00", "11"}
        for k in result.counts:
            assert k in valid_outcomes, f"Unexpected outcome {k}"

    def test_bell_state_approximately_equal(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=SHOTS)
        p00 = result.probabilities.get("00", 0)
        p11 = result.probabilities.get("11", 0)
        assert abs(p00 - 0.5) < 0.05
        assert abs(p11 - 0.5) < 0.05

    def test_bell_state_chi_squared(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=SHOTS)
        expected = {"00": 0.5, "11": 0.5}
        p_value = chi_squared_test(result.counts, expected, SHOTS)
        assert p_value > 0.01, f"Chi-squared test failed: p={p_value:.4f}"

    def test_bell_state_total_probability(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=SHOTS)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-10


class TestGHZState:
    def test_ghz3_outcomes(self, local_backend, ghz3_circuit):
        result = local_backend.run(ghz3_circuit, shots=SHOTS)
        for k in result.counts:
            assert k in {"000", "111"}, f"Unexpected GHZ3 outcome {k}"

    def test_ghz3_approximately_equal(self, local_backend, ghz3_circuit):
        result = local_backend.run(ghz3_circuit, shots=SHOTS)
        p000 = result.probabilities.get("000", 0)
        p111 = result.probabilities.get("111", 0)
        assert abs(p000 - 0.5) < 0.05
        assert abs(p111 - 0.5) < 0.05

    def test_ghz5_outcomes(self, local_backend, ghz5_circuit):
        result = local_backend.run(ghz5_circuit, shots=SHOTS)
        for k in result.counts:
            assert k in {"00000", "11111"}, f"Unexpected GHZ5 outcome {k}"

    def test_ghz3_chi_squared(self, local_backend, ghz3_circuit):
        result = local_backend.run(ghz3_circuit, shots=SHOTS)
        expected = {"000": 0.5, "111": 0.5}
        p_value = chi_squared_test(result.counts, expected, SHOTS)
        assert p_value > 0.01


class TestGroverSearch:
    def test_grover3_marked_state_dominant(self, local_backend, grover3_circuit):
        result = local_backend.run(grover3_circuit, shots=SHOTS)
        assert result.most_frequent == "101"

    def test_grover3_marked_probability(self, local_backend, grover3_circuit):
        result = local_backend.run(grover3_circuit, shots=SHOTS)
        p_marked = result.probabilities.get("101", 0)
        assert p_marked > 0.7, f"Grover marked state prob too low: {p_marked:.3f}"

    def test_grover3_above_random(self, local_backend, grover3_circuit):
        result = local_backend.run(grover3_circuit, shots=SHOTS)
        p_marked = result.probabilities.get("101", 0)
        random_baseline = 1.0 / 8
        assert p_marked > random_baseline * 3


class TestQFTRoundTrip:
    def test_qft_inverse_returns_zero(self, local_backend, qft_roundtrip_circuit):
        result = local_backend.run(qft_roundtrip_circuit, shots=SHOTS)
        p_zero = result.probabilities.get("000", 0)
        assert p_zero > 0.95, f"QFT round-trip didn't return to |000>: p={p_zero:.3f}"

    def test_qft_most_frequent_is_zero(self, local_backend, qft_roundtrip_circuit):
        result = local_backend.run(qft_roundtrip_circuit, shots=SHOTS)
        assert result.most_frequent == "000"


class TestDeterministicCircuits:
    def test_x_gate_all_ones(self, local_backend, x_gate_circuit):
        result = local_backend.run(x_gate_circuit, shots=1000)
        assert result.counts.get("1", 0) == 1000

    def test_identity_all_zeros(self, local_backend):
        qc = QuantumCircuit(2)
        result = local_backend.run(qc, shots=1000)
        assert result.counts.get("00", 0) == 1000

    def test_double_x_returns_zero(self, local_backend):
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        qc.apply(X, 0)
        result = local_backend.run(qc, shots=1000)
        assert result.counts.get("0", 0) == 1000

    def test_toffoli_111(self, local_backend):
        qc = QuantumCircuit(3)
        qc.apply(X, 0)
        qc.apply(X, 1)
        qc.apply(CCX, 0, 1, 2)
        result = local_backend.run(qc, shots=1000)
        assert result.counts.get("111", 0) == 1000

    def test_toffoli_no_flip_when_control_off(self, local_backend):
        qc = QuantumCircuit(3)
        qc.apply(X, 0)
        qc.apply(CCX, 0, 1, 2)
        result = local_backend.run(qc, shots=1000)
        assert len(result.counts) == 1, f"Expected deterministic output, got {result.counts}"
        outcome = list(result.counts.keys())[0]
        assert outcome[0] == "0", f"Target qubit (q2, leftmost) should not flip: got {outcome}"


class TestReproducibility:
    def test_same_circuit_different_runs_statistical(self, local_backend, bell_circuit):
        r1 = local_backend.run(bell_circuit, shots=SHOTS)
        r2 = local_backend.run(bell_circuit, shots=SHOTS)
        fid = hellinger_fidelity(r1.probabilities, r2.probabilities)
        assert fid > 0.99, f"Two runs differ too much: fidelity={fid:.4f}"


class TestEdgeCases:
    def test_single_shot(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=1)
        assert sum(result.counts.values()) == 1

    def test_large_shots(self, local_backend, x_gate_circuit):
        result = local_backend.run(x_gate_circuit, shots=10000)
        assert result.counts.get("1", 0) == 10000

    def test_many_qubits(self, local_backend):
        qc = QuantumCircuit(8)
        qc.apply(H, 0)
        for i in range(7):
            qc.apply(CNOT, 0, i + 1)
        result = local_backend.run(qc, shots=SHOTS)
        for k in result.counts:
            assert k in {"00000000", "11111111"}


class TestStatisticalValidation:
    def test_hadamard_uniform(self, local_backend):
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        result = local_backend.run(qc, shots=SHOTS)
        expected = {"0": 0.5, "1": 0.5}
        p_value = chi_squared_test(result.counts, expected, SHOTS)
        assert p_value > 0.01

    def test_two_hadamard_uniform(self, local_backend):
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(H, 1)
        result = local_backend.run(qc, shots=SHOTS)
        expected = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
        p_value = chi_squared_test(result.counts, expected, SHOTS)
        assert p_value > 0.01

    def test_hellinger_self_fidelity(self, local_backend, bell_circuit):
        result = local_backend.run(bell_circuit, shots=SHOTS)
        fid = hellinger_fidelity(result.probabilities, result.probabilities)
        assert abs(fid - 1.0) < 1e-10

    def test_hellinger_orthogonal(self):
        p = {"0": 1.0}
        q = {"1": 1.0}
        fid = hellinger_fidelity(p, q)
        assert abs(fid) < 1e-10
