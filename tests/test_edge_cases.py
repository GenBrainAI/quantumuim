"""Edge case tests for core SDK modules — boundary conditions, invalid inputs, error paths."""

from __future__ import annotations

import numpy as np
import pytest

from quantumium.core.gate import (
    Gate,
    I,
    X,
    Y,
    Z,
    H,
    S,
    T,
    CNOT,
    SWAP,
    CZ,
    CCX,
    controlled,
    Rx,
    Ry,
    Rz,
    CRx,
    CRy,
    CRz,
    Rxx,
    Rzz,
)
from quantumium.core.qubit import Qubit
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.noise import (
    NoiseChannel,
    NoiseModel,
    bit_flip,
    phase_flip,
    depolarizing,
    amplitude_damping,
)
from quantumium.core.parameter import (
    Parameter,
    ParameterVector,
    ParameterizedCircuit,
    hardware_efficient_template,
)


class TestGateEdgeCases:
    def test_non_square_matrix_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            Gate(np.array([[1, 0, 0], [0, 1, 0]], dtype=complex))

    def test_1x1_gate(self) -> None:
        g = Gate(np.array([[1.0]], dtype=complex), "Phase1")
        assert g.num_qubits == 0
        assert g.matrix.shape == (1, 1)

    def test_gate_repr(self) -> None:
        assert repr(X) == "X"
        assert repr(H) == "H"

    def test_gate_default_name(self) -> None:
        g = Gate(np.eye(2, dtype=complex))
        assert "2x2" in g.name

    def test_controlled_multi_qubit_raises(self) -> None:
        with pytest.raises(ValueError, match="single-qubit"):
            controlled(CNOT)

    def test_controlled_creates_valid_gate(self) -> None:
        cz = controlled(Z, "CZ_custom")
        assert cz.num_qubits == 2
        assert np.allclose(cz.matrix @ cz.matrix.conj().T, np.eye(4), atol=1e-10)

    def test_controlled_x_equals_cnot(self) -> None:
        cx = controlled(X)
        assert np.allclose(cx.matrix, CNOT.matrix, atol=1e-10)

    def test_rotation_zero_angle(self) -> None:
        for gate_fn in [Rx, Ry, Rz]:
            g = gate_fn(0.0)
            assert np.allclose(g.matrix, np.eye(2), atol=1e-10)

    def test_rotation_full_turn(self) -> None:
        for gate_fn in [Rx, Ry, Rz]:
            g = gate_fn(2 * np.pi)
            assert np.allclose(g.matrix, -np.eye(2), atol=1e-10)

    def test_rotation_negative_angle(self) -> None:
        for theta in [-np.pi / 4, -np.pi, -2 * np.pi]:
            for gate_fn in [Rx, Ry, Rz]:
                g = gate_fn(theta)
                assert np.allclose(g.matrix @ g.matrix.conj().T, np.eye(2), atol=1e-10)

    def test_controlled_rotations_are_unitary(self) -> None:
        for theta in [0, np.pi / 3, np.pi]:
            for gate_fn in [CRx, CRy, CRz]:
                g = gate_fn(theta)
                assert g.num_qubits == 2
                assert np.allclose(g.matrix @ g.matrix.conj().T, np.eye(4), atol=1e-10)

    def test_rxx_rzz_are_unitary(self) -> None:
        for theta in [0, np.pi / 4, np.pi / 2, np.pi]:
            for gate_fn in [Rxx, Rzz]:
                g = gate_fn(theta)
                assert g.num_qubits == 2
                assert np.allclose(g.matrix @ g.matrix.conj().T, np.eye(4), atol=1e-10)

    def test_rxx_zero_is_identity(self) -> None:
        g = Rxx(0.0)
        assert np.allclose(g.matrix, np.eye(4), atol=1e-10)

    def test_rzz_zero_is_identity(self) -> None:
        g = Rzz(0.0)
        assert np.allclose(g.matrix, np.eye(4), atol=1e-10)

    def test_standard_gates_are_unitary(self) -> None:
        for gate in [I, X, Y, Z, H, S, T, CNOT, SWAP, CZ, CCX]:
            m = gate.matrix
            dim = m.shape[0]
            assert np.allclose(m @ m.conj().T, np.eye(dim), atol=1e-10), f"{gate.name} not unitary"

    def test_pauli_algebra(self) -> None:
        assert np.allclose(X.matrix @ Y.matrix, 1j * Z.matrix, atol=1e-10)
        assert np.allclose(Y.matrix @ Z.matrix, 1j * X.matrix, atol=1e-10)
        assert np.allclose(Z.matrix @ X.matrix, 1j * Y.matrix, atol=1e-10)

    def test_ccx_is_toffoli(self) -> None:
        assert CCX.num_qubits == 3
        state = np.zeros(8, dtype=complex)
        state[0b110] = 1.0  # |110⟩
        result = CCX.matrix @ state
        assert abs(result[0b111]) > 0.99  # should flip target


class TestQubitEdgeCases:
    def test_minus_state(self) -> None:
        q = Qubit.minus()
        assert q.probabilities == pytest.approx((0.5, 0.5))

    def test_complex_amplitudes(self) -> None:
        q = Qubit(1j, 0.0)
        assert q.probabilities == pytest.approx((1.0, 0.0))

    def test_large_amplitudes_normalize(self) -> None:
        q = Qubit(100.0, 0.0)
        assert q.probabilities == pytest.approx((1.0, 0.0))

    def test_state_property_returns_copy(self) -> None:
        q = Qubit.zero()
        s1 = q.state
        s2 = q.state
        assert np.allclose(s1, s2)
        s1[0] = 999
        assert not np.allclose(q.state, s1)

    def test_repr(self) -> None:
        q = Qubit.zero()
        r = repr(q)
        assert "Qubit" in r
        assert "α" in r or "alpha" in r.lower() or "1.0000" in r

    def test_measure_one_state(self) -> None:
        q = Qubit.one()
        result = q.measure()
        assert result == 1
        assert q.probabilities == pytest.approx((0.0, 1.0))

    def test_near_zero_vector_raises(self) -> None:
        with pytest.raises(ValueError):
            Qubit(1e-11, 1e-11)


class TestCircuitEdgeCases:
    def test_zero_qubit_circuit_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            QuantumCircuit(0)

    def test_negative_qubit_index_raises(self) -> None:
        qc = QuantumCircuit(2)
        with pytest.raises(IndexError):
            qc.apply(X, -1)

    def test_duplicate_qubit_indices_raises(self) -> None:
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="Duplicate"):
            qc.apply(CNOT, 0, 0)

    def test_empty_circuit_statevector(self) -> None:
        qc = QuantumCircuit(3)
        state = qc.statevector()
        assert abs(state[0]) ** 2 == pytest.approx(1.0)
        assert sum(abs(s) ** 2 for s in state[1:]) == pytest.approx(0.0, abs=1e-10)

    def test_empty_circuit_depth(self) -> None:
        qc = QuantumCircuit(2)
        assert qc.depth == 0

    def test_single_gate_depth(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        assert qc.depth == 1

    def test_instructions_returns_copy(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        insts = qc.instructions
        assert len(insts) == 1
        insts.clear()
        assert len(qc.instructions) == 1

    def test_repr(self) -> None:
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        r = repr(qc)
        assert "3" in r
        assert "1" in r

    def test_density_matrix_pure_state(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        rho = qc.density_matrix()
        assert rho.shape == (2, 2)
        sv = qc.statevector()
        expected = np.outer(sv, sv.conj())
        assert np.allclose(rho, expected, atol=1e-10)

    def test_density_matrix_with_noise(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        nm = NoiseModel(default_single_qubit_noise=depolarizing(0.1))
        rho = qc.density_matrix(nm)
        assert rho.shape == (2, 2)
        assert abs(np.trace(rho) - 1.0) < 1e-10
        assert np.allclose(rho, rho.conj().T, atol=1e-10)

    def test_noisy_measure_returns_valid_counts(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(X, 0)
        nm = NoiseModel(default_single_qubit_noise=bit_flip(0.01))
        counts = qc.noisy_measure(nm, shots=1000)
        assert sum(counts.values()) == 1000
        assert counts.get("1", 0) > 900

    def test_measure_all_single_qubit(self) -> None:
        qc = QuantumCircuit(1)
        counts = qc.measure_all(shots=100)
        assert sum(counts.values()) == 100
        assert counts.get("0", 0) == 100

    def test_3_qubit_ghz_measure(self) -> None:
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(CNOT, 1, 2)
        counts = qc.measure_all(shots=10000)
        assert counts.get("000", 0) > 3000
        assert counts.get("111", 0) > 3000
        mid_values = sum(v for k, v in counts.items() if k not in ("000", "111"))
        assert mid_values < 100

    def test_cz_gate_in_circuit(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(H, 1)
        qc.apply(CZ, 0, 1)
        state = qc.statevector()
        assert len(state) == 4

    def test_ccx_in_circuit(self) -> None:
        qc = QuantumCircuit(3)
        qc.apply(X, 0)
        qc.apply(X, 1)
        qc.apply(CCX, 0, 1, 2)
        state = qc.statevector()
        assert abs(state[0b111]) ** 2 == pytest.approx(1.0, abs=1e-10)

    def test_large_circuit(self) -> None:
        qc = QuantumCircuit(1)
        for _ in range(100):
            qc.apply(X, 0)
        state = qc.statevector()
        assert abs(state[0]) ** 2 == pytest.approx(1.0, abs=1e-10)


class TestNoiseEdgeCases:
    def test_bit_flip_invalid_probability(self) -> None:
        with pytest.raises(ValueError):
            bit_flip(-0.1)
        with pytest.raises(ValueError):
            bit_flip(1.1)

    def test_phase_flip_invalid_probability(self) -> None:
        with pytest.raises(ValueError):
            phase_flip(-0.1)
        with pytest.raises(ValueError):
            phase_flip(1.1)

    def test_depolarizing_invalid_probability(self) -> None:
        with pytest.raises(ValueError):
            depolarizing(-0.1)
        with pytest.raises(ValueError):
            depolarizing(1.1)

    def test_amplitude_damping_invalid_gamma(self) -> None:
        with pytest.raises(ValueError):
            amplitude_damping(-0.1)
        with pytest.raises(ValueError):
            amplitude_damping(1.1)

    def test_zero_noise_is_identity(self) -> None:
        for channel_fn in [bit_flip, phase_flip, depolarizing, amplitude_damping]:
            ch = channel_fn(0.0)
            rho = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
            result = ch.apply(rho)
            assert np.allclose(result, rho, atol=1e-10)

    def test_full_bit_flip(self) -> None:
        ch = bit_flip(1.0)
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = ch.apply(rho)
        expected = np.array([[0, 0], [0, 1]], dtype=complex)
        assert np.allclose(result, expected, atol=1e-10)

    def test_full_amplitude_damping(self) -> None:
        ch = amplitude_damping(1.0)
        rho = np.array([[0, 0], [0, 1]], dtype=complex)
        result = ch.apply(rho)
        assert abs(result[0, 0] - 1.0) < 1e-10

    def test_invalid_kraus_operators_raise(self) -> None:
        bad_k = [np.eye(2, dtype=complex), np.eye(2, dtype=complex)]
        with pytest.raises(ValueError, match="completeness"):
            NoiseChannel("bad", bad_k)

    def test_noise_model_gate_noise_priority(self) -> None:
        nm = NoiseModel()
        bf = bit_flip(0.1)
        pf = phase_flip(0.2)
        nm.default_single_qubit_noise = pf
        nm.add_gate_noise("X", bf)
        assert nm.get_noise("X", 1) is bf
        assert nm.get_noise("H", 1) is pf
        assert nm.get_noise("CNOT", 2) is None

    def test_noise_model_two_qubit_default(self) -> None:
        nm = NoiseModel()
        bf = bit_flip(0.05)
        nm.default_two_qubit_noise = bf
        assert nm.get_noise("CNOT", 2) is bf
        assert nm.get_noise("X", 1) is None

    def test_noise_model_no_noise_returns_none(self) -> None:
        nm = NoiseModel()
        assert nm.get_noise("X", 1) is None
        assert nm.get_noise("CNOT", 2) is None


class TestParameterEdgeCases:
    def test_parameter_repr(self) -> None:
        p = Parameter("theta")
        assert "theta" in repr(p)

    def test_parameter_frozen(self) -> None:
        p = Parameter("x")
        with pytest.raises(AttributeError):
            p.name = "y"  # type: ignore[misc]

    def test_parameter_vector_length(self) -> None:
        pv = ParameterVector("w", 5)
        assert len(pv) == 5
        assert pv[0].name == "w[0]"
        assert pv[4].name == "w[4]"

    def test_parameter_vector_iteration(self) -> None:
        pv = ParameterVector("x", 3)
        names = [p.name for p in pv]
        assert names == ["x[0]", "x[1]", "x[2]"]

    def test_parameterized_circuit_zero_qubits_raises(self) -> None:
        with pytest.raises(ValueError):
            ParameterizedCircuit(0)

    def test_bind_missing_parameter_raises(self) -> None:
        pc = ParameterizedCircuit(1)
        p = Parameter("theta")
        pc.rx(p, 0)
        with pytest.raises(ValueError, match="Missing"):
            pc.bind({})

    def test_bind_array_wrong_length_raises(self) -> None:
        pc = ParameterizedCircuit(1)
        p = Parameter("theta")
        pc.rx(p, 0)
        with pytest.raises(ValueError, match="Expected"):
            pc.bind_array([1.0, 2.0])

    def test_bind_produces_correct_circuit(self) -> None:
        pc = ParameterizedCircuit(1)
        p = Parameter("theta")
        pc.ry(p, 0)
        qc = pc.bind({p: np.pi})
        state = qc.statevector()
        assert abs(state[1]) ** 2 == pytest.approx(1.0, abs=1e-10)

    def test_bind_array_produces_correct_circuit(self) -> None:
        pc = ParameterizedCircuit(1)
        p = Parameter("theta")
        pc.rz(p, 0)
        qc = pc.bind_array([np.pi / 2])
        state = qc.statevector()
        assert len(state) == 2

    def test_mixed_fixed_and_parameterized(self) -> None:
        pc = ParameterizedCircuit(2)
        p = Parameter("a")
        pc.apply(H, 0)
        pc.rx(p, 1)
        pc.apply(CNOT, 0, 1)
        qc = pc.bind({p: 0.0})
        state = qc.statevector()
        assert len(state) == 4

    def test_parameters_property(self) -> None:
        pc = ParameterizedCircuit(1)
        p1 = Parameter("a")
        p2 = Parameter("b")
        pc.rx(p1, 0)
        pc.ry(p2, 0)
        assert pc.num_parameters == 2
        assert pc.parameters == {p1, p2}

    def test_repr(self) -> None:
        pc = ParameterizedCircuit(2)
        p = Parameter("x")
        pc.rx(p, 0)
        r = repr(pc)
        assert "2" in r
        assert "1" in r

    def test_hardware_efficient_template(self) -> None:
        pqc, params = hardware_efficient_template(3, depth=2)
        assert pqc.num_qubits == 3
        assert len(params) == 12  # 2 params/qubit/layer * 3 qubits * 2 layers
        values = {p: 0.1 for p in params}
        qc = pqc.bind(values)
        state = qc.statevector()
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10

    def test_hardware_efficient_template_depth_1(self) -> None:
        pqc, params = hardware_efficient_template(2, depth=1)
        assert len(params) == 4  # 2 params/qubit * 2 qubits

    def test_same_parameter_reused(self) -> None:
        pc = ParameterizedCircuit(2)
        p = Parameter("shared")
        pc.rx(p, 0)
        pc.rx(p, 1)
        assert pc.num_parameters == 1
        qc = pc.bind({p: np.pi / 2})
        state = qc.statevector()
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10
