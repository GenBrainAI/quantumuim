"""Integration tests — cross-module workflows that exercise realistic SDK pipelines."""

from __future__ import annotations

import numpy as np
import pytest

from quantumium.core.gate import H, X, CNOT, SWAP, Rx, Ry, Rz, Rxx, Rzz
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.noise import NoiseModel, bit_flip, depolarizing, amplitude_damping
from quantumium.core.parameter import (
    Parameter,
    ParameterizedCircuit,
    hardware_efficient_template,
)
from quantumium.transpiler.passes import transpile, count_gates
from quantumium.circuits.templates import bell_pair, ghz_state, qft, inverse_qft


def _circuit_unitary(qc: QuantumCircuit) -> np.ndarray:
    n = qc.num_qubits
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for inst in qc.instructions:
        full = qc._expand_gate(inst.gate.matrix, inst.target_qubits, n)
        U = full @ U
    return U


class TestCircuitBuildTranspileSimulate:
    """Build circuit -> transpile -> simulate -> verify unitary preserved."""

    def test_bell_state_pipeline(self) -> None:
        qc = bell_pair()
        U_orig = _circuit_unitary(qc)
        optimized = transpile(qc)
        U_opt = _circuit_unitary(optimized)
        np.testing.assert_allclose(U_orig, U_opt, atol=1e-10)
        state = optimized.statevector()
        assert abs(state[0]) ** 2 == pytest.approx(0.5, abs=1e-10)
        assert abs(state[3]) ** 2 == pytest.approx(0.5, abs=1e-10)

    def test_ghz_transpile_preserves_state(self) -> None:
        qc = ghz_state(4)
        state_orig = qc.statevector()
        optimized = transpile(qc)
        state_opt = optimized.statevector()
        np.testing.assert_allclose(np.abs(state_orig) ** 2, np.abs(state_opt) ** 2, atol=1e-10)

    def test_qft_inverse_qft_pipeline(self) -> None:
        """QFT followed by inverse QFT should be identity."""
        n = 3
        qc_prep = QuantumCircuit(n)
        qc_prep.apply(X, 0)
        qc_prep.apply(H, 1)
        state_before = qc_prep.statevector()

        qc_roundtrip = QuantumCircuit(n)
        qc_roundtrip.apply(X, 0)
        qc_roundtrip.apply(H, 1)
        for inst in qft(n).instructions:
            qc_roundtrip.apply(inst.gate, *inst.target_qubits)
        for inst in inverse_qft(n).instructions:
            qc_roundtrip.apply(inst.gate, *inst.target_qubits)
        state_after = qc_roundtrip.statevector()
        np.testing.assert_allclose(np.abs(state_after) ** 2, np.abs(state_before) ** 2, atol=1e-10)

    def test_transpile_redundant_gates_removed(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(X, 0).apply(X, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(CNOT, 0, 1)
        qc.apply(H, 1).apply(H, 1)
        U_orig = _circuit_unitary(qc)
        optimized = transpile(qc)
        U_opt = _circuit_unitary(optimized)
        np.testing.assert_allclose(U_orig, U_opt, atol=1e-10)
        assert len(optimized.instructions) < len(qc.instructions)

    def test_swap_decompose_and_optimize(self) -> None:
        from quantumium.transpiler.passes import decompose_swap

        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(SWAP, 0, 1)
        qc.apply(H, 1)
        U_orig = _circuit_unitary(qc)
        optimized = transpile(qc, passes=[decompose_swap])
        U_opt = _circuit_unitary(optimized)
        np.testing.assert_allclose(U_orig, U_opt, atol=1e-10)
        counts = count_gates(optimized.instructions)
        assert "SWAP" not in counts

    def test_deep_circuit_transpile(self) -> None:
        qc = QuantumCircuit(3)
        for _ in range(5):
            qc.apply(H, 0)
            qc.apply(H, 0)
            qc.apply(X, 1)
            qc.apply(X, 1)
            qc.apply(CNOT, 0, 2)
            qc.apply(CNOT, 0, 2)
        U_orig = _circuit_unitary(qc)
        optimized = transpile(qc)
        U_opt = _circuit_unitary(optimized)
        np.testing.assert_allclose(U_orig, U_opt, atol=1e-10)
        assert len(optimized.instructions) < len(qc.instructions)


class TestNoisySimulation:
    """Build circuit -> add noise -> measure -> verify statistics."""

    def test_x_with_bit_flip_noise(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        nm = NoiseModel(default_single_qubit_noise=bit_flip(0.05))
        counts = qc.noisy_measure(nm, shots=10000)
        assert counts.get("1", 0) > 8000
        assert counts.get("0", 0) < 2000

    def test_depolarizing_increases_entropy(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        rho_clean = qc.density_matrix()
        nm = NoiseModel(default_single_qubit_noise=depolarizing(0.5))
        rho_noisy = qc.density_matrix(nm)
        purity_clean = float(np.real(np.trace(rho_clean @ rho_clean)))
        purity_noisy = float(np.real(np.trace(rho_noisy @ rho_noisy)))
        assert purity_noisy < purity_clean

    def test_amplitude_damping_decays_to_ground(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        nm = NoiseModel(default_single_qubit_noise=amplitude_damping(0.99))
        rho = qc.density_matrix(nm)
        assert rho[0, 0] > 0.9

    def test_two_qubit_noisy_bell(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        nm = NoiseModel(default_single_qubit_noise=depolarizing(0.01))
        counts = qc.noisy_measure(nm, shots=10000)
        total = sum(counts.values())
        assert total == 10000
        assert counts.get("00", 0) > 3000
        assert counts.get("11", 0) > 3000

    def test_no_noise_density_equals_pure(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        rho = qc.density_matrix()
        sv = qc.statevector()
        expected = np.outer(sv, sv.conj())
        np.testing.assert_allclose(rho, expected, atol=1e-10)


class TestParameterizedPipeline:
    """Parameterized circuit -> bind -> transpile -> measure."""

    def test_vqe_style_workflow(self) -> None:
        pqc, params = hardware_efficient_template(2, depth=1)
        values = {p: np.pi / 4 for p in params}
        qc = pqc.bind(values)
        optimized = transpile(qc)
        state = optimized.statevector()
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10
        U_orig = _circuit_unitary(qc)
        U_opt = _circuit_unitary(optimized)
        np.testing.assert_allclose(U_orig, U_opt, atol=1e-10)

    def test_parameter_sweep(self) -> None:
        pc = ParameterizedCircuit(1)
        p = Parameter("theta")
        pc.ry(p, 0)
        for angle in [0, np.pi / 4, np.pi / 2, np.pi]:
            qc = pc.bind({p: angle})
            state = qc.statevector()
            expected_p1 = np.sin(angle / 2) ** 2
            assert abs(state[1]) ** 2 == pytest.approx(expected_p1, abs=1e-10)

    def test_bind_then_noisy_measure(self) -> None:
        pc = ParameterizedCircuit(1)
        p = Parameter("theta")
        pc.ry(p, 0)
        qc = pc.bind({p: np.pi})
        nm = NoiseModel(default_single_qubit_noise=bit_flip(0.01))
        counts = qc.noisy_measure(nm, shots=1000)
        assert counts.get("1", 0) > 900

    def test_multi_param_bind_array(self) -> None:
        pqc, params = hardware_efficient_template(2, depth=1)
        values = np.zeros(len(params))
        qc = pqc.bind_array(values)
        state = qc.statevector()
        assert abs(state[0]) ** 2 == pytest.approx(1.0, abs=1e-10)


class TestCircuitTemplatesPipeline:
    """Circuit templates -> transpile -> verify correctness."""

    def test_bell_pair_measure_statistics(self) -> None:
        qc = bell_pair()
        counts = qc.measure_all(shots=10000)
        assert counts.get("00", 0) > 3000
        assert counts.get("11", 0) > 3000
        mid = sum(v for k, v in counts.items() if k not in ("00", "11"))
        assert mid < 100

    def test_qft_uniform_output(self) -> None:
        qc = qft(3)
        state = qc.statevector()
        probs = np.abs(state) ** 2
        for p in probs:
            assert p == pytest.approx(1.0 / 8, abs=1e-10)

    def test_ghz_fidelity_after_transpile(self) -> None:
        qc = ghz_state(3)
        optimized = transpile(qc)
        state = optimized.statevector()
        assert abs(state[0]) ** 2 == pytest.approx(0.5, abs=1e-10)
        assert abs(state[7]) ** 2 == pytest.approx(0.5, abs=1e-10)

    def test_rotation_gate_circuit(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(Rx(np.pi / 2), 0)
        qc.apply(Ry(np.pi / 2), 1)
        qc.apply(CNOT, 0, 1)
        qc.apply(Rz(np.pi / 4), 0)
        state = qc.statevector()
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10
        optimized = transpile(qc)
        state_opt = optimized.statevector()
        np.testing.assert_allclose(np.abs(state) ** 2, np.abs(state_opt) ** 2, atol=1e-10)

    def test_two_qubit_rotation_gates(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(H, 1)
        qc.apply(Rxx(np.pi / 4), 0, 1)
        qc.apply(Rzz(np.pi / 4), 0, 1)
        state = qc.statevector()
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10
