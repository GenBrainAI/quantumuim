"""Property-based tests using Hypothesis for quantum circuit invariants."""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, Y, Z, S, T, CNOT, CZ, SWAP, Rx, Ry, Rz
from quantumium.core.noise import NoiseModel, depolarizing


SINGLE_GATES = [H, X, Y, Z, S, T]
PARAM_GATE_FACTORIES = [Rx, Ry, Rz]
TWO_QUBIT_GATES = [CNOT, CZ, SWAP]


@st.composite
def random_circuit(draw, min_qubits=2, max_qubits=5, min_gates=1, max_gates=10):
    n = draw(st.integers(min_value=min_qubits, max_value=max_qubits))
    num_gates = draw(st.integers(min_value=min_gates, max_value=max_gates))
    qc = QuantumCircuit(n)
    for _ in range(num_gates):
        gate_type = draw(st.integers(min_value=0, max_value=2))
        if gate_type == 0:
            gate = draw(st.sampled_from(SINGLE_GATES))
            qubit = draw(st.integers(min_value=0, max_value=n - 1))
            qc.apply(gate, qubit)
        elif gate_type == 1:
            factory = draw(st.sampled_from(PARAM_GATE_FACTORIES))
            angle = draw(st.floats(min_value=-2 * np.pi, max_value=2 * np.pi, allow_nan=False))
            qubit = draw(st.integers(min_value=0, max_value=n - 1))
            qc.apply(factory(angle), qubit)
        else:
            if n >= 2:
                gate = draw(st.sampled_from(TWO_QUBIT_GATES))
                q0 = draw(st.integers(min_value=0, max_value=n - 1))
                q1 = draw(st.integers(min_value=0, max_value=n - 1).filter(lambda x, q=q0: x != q))
                qc.apply(gate, q0, q1)
            else:
                gate = draw(st.sampled_from(SINGLE_GATES))
                qc.apply(gate, 0)
    return qc


class TestStatevectorInvariants:
    """Property: statevector norm must always be 1."""

    @given(qc=random_circuit())
    @settings(max_examples=50, deadline=5000)
    def test_statevector_norm_is_one(self, qc):
        sv = qc.statevector()
        norm = np.linalg.norm(sv)
        assert abs(norm - 1.0) < 1e-10, f"Norm = {norm}"

    @given(qc=random_circuit())
    @settings(max_examples=50, deadline=5000)
    def test_statevector_length(self, qc):
        sv = qc.statevector()
        assert len(sv) == 2 ** qc.num_qubits


class TestMeasurementInvariants:
    """Property: measurement probabilities must sum to 1."""

    @given(qc=random_circuit(max_qubits=4, max_gates=6))
    @settings(max_examples=30, deadline=10000)
    def test_probabilities_sum_to_one(self, qc):
        counts = qc.measure_all(shots=100)
        total = sum(counts.values())
        assert total == 100

    @given(qc=random_circuit(max_qubits=4, max_gates=6))
    @settings(max_examples=30, deadline=10000)
    def test_measurement_keys_valid(self, qc):
        counts = qc.measure_all(shots=100)
        n = qc.num_qubits
        for key in counts:
            assert len(key) == n
            assert all(c in "01" for c in key)


class TestDensityMatrixInvariants:
    """Property: density matrix must be valid (trace 1, PSD, Hermitian)."""

    @given(qc=random_circuit(max_qubits=4, max_gates=6))
    @settings(max_examples=30, deadline=10000)
    def test_density_matrix_trace_is_one(self, qc):
        rho = qc.density_matrix()
        trace = np.real(np.trace(rho))
        assert abs(trace - 1.0) < 1e-10, f"Trace = {trace}"

    @given(qc=random_circuit(max_qubits=4, max_gates=6))
    @settings(max_examples=30, deadline=10000)
    def test_density_matrix_is_hermitian(self, qc):
        rho = qc.density_matrix()
        assert np.allclose(rho, rho.conj().T, atol=1e-10)

    @given(qc=random_circuit(max_qubits=3, max_gates=5))
    @settings(max_examples=20, deadline=10000)
    def test_density_matrix_positive_semidefinite(self, qc):
        rho = qc.density_matrix()
        eigenvalues = np.linalg.eigvalsh(rho)
        assert all(ev > -1e-10 for ev in eigenvalues), f"Negative eigenvalue: {min(eigenvalues)}"

    @given(qc=random_circuit(max_qubits=3, max_gates=5))
    @settings(max_examples=20, deadline=10000)
    def test_noisy_density_matrix_trace_one(self, qc):
        noise = NoiseModel(default_single_qubit_noise=depolarizing(0.01))
        rho = qc.density_matrix(noise_model=noise)
        trace = np.real(np.trace(rho))
        assert abs(trace - 1.0) < 1e-8, f"Noisy trace = {trace}"


class TestTranspilerInvariants:
    """Property: transpiled circuit produces same statevector as original."""

    @given(qc=random_circuit(max_qubits=4, max_gates=8))
    @settings(max_examples=30, deadline=10000)
    def test_transpiler_preserves_statevector(self, qc):
        from quantumium.transpiler.passes import transpile

        sv_before = qc.statevector()
        optimized = transpile(qc)
        sv_after = optimized.statevector()
        overlap = abs(np.vdot(sv_before, sv_after))
        assert overlap > 1 - 1e-8, f"Overlap = {overlap}"


class TestSerializationRoundtrip:
    """Property: serialize then deserialize produces equivalent circuit."""

    @given(qc=random_circuit(max_qubits=4, max_gates=8))
    @settings(max_examples=30, deadline=10000)
    def test_json_roundtrip_preserves_statevector(self, qc):
        from quantumium.transpiler.serialization import to_json, from_json

        json_str = to_json(qc)
        restored = from_json(json_str)
        sv_orig = qc.statevector()
        sv_restored = restored.statevector()
        overlap = abs(np.vdot(sv_orig, sv_restored))
        assert overlap > 1 - 1e-6, f"JSON roundtrip overlap = {overlap}"


class TestCircuitComposition:
    """Property: statevector after A then B equals applying B's gates to A's output."""

    @given(
        n=st.integers(min_value=1, max_value=3),
        data=st.data(),
    )
    @settings(max_examples=30, deadline=10000)
    def test_sequential_application(self, n, data):
        qc1 = QuantumCircuit(n)
        qc2 = QuantumCircuit(n)
        for _ in range(data.draw(st.integers(min_value=1, max_value=4))):
            gate = data.draw(st.sampled_from(SINGLE_GATES))
            q = data.draw(st.integers(min_value=0, max_value=n - 1))
            qc1.apply(gate, q)
        for _ in range(data.draw(st.integers(min_value=1, max_value=4))):
            gate = data.draw(st.sampled_from(SINGLE_GATES))
            q = data.draw(st.integers(min_value=0, max_value=n - 1))
            qc2.apply(gate, q)

        combined = QuantumCircuit(n)
        for inst in qc1.instructions:
            combined.apply(inst.gate, *inst.target_qubits)
        for inst in qc2.instructions:
            combined.apply(inst.gate, *inst.target_qubits)

        sv_combined = combined.statevector()
        sv1 = qc1.statevector()
        qc2_on_sv1 = QuantumCircuit(n)
        for inst in qc2.instructions:
            qc2_on_sv1.apply(inst.gate, *inst.target_qubits)

        d = 2**n
        U2 = np.eye(d, dtype=complex)
        for inst in qc2.instructions:
            U2 = qc2._expand_gate(inst.gate.matrix, inst.target_qubits, n) @ U2
        sv_sequential = U2 @ sv1

        overlap = abs(np.vdot(sv_combined, sv_sequential))
        assert overlap > 1 - 1e-8


class TestQPlusInvariants:
    """Property: Q+ factoring must return correct factors."""

    @given(N=st.sampled_from([15, 21, 35]))
    @settings(max_examples=6, deadline=30000)
    def test_qplus_factors_are_correct(self, N):
        from quantumium.qplus.shor import ShorQPlus

        result = ShorQPlus().factor(N, max_attempts=20)
        assert result.success
        p, q = result.factors
        assert p * q == N
        assert p > 1 and q > 1

    @given(N=st.sampled_from([15, 21, 35]))
    @settings(max_examples=6, deadline=30000)
    def test_qplus_uses_fewer_qubits(self, N):
        from quantumium.qplus.shor import ShorQPlus

        result = ShorQPlus().factor(N, max_attempts=20)
        assert result.qubits_used < result.qubits_standard
