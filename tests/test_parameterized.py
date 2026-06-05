"""Tests for parameterized quantum circuits."""

import numpy as np
import pytest

from quantumium.core.gate import H, X, CNOT, Ry, Rz
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.parameter import (
    Parameter,
    ParameterVector,
    ParameterizedCircuit,
    hardware_efficient_template,
)


class TestParameter:
    def test_creation(self):
        p = Parameter("theta")
        assert p.name == "theta"

    def test_frozen(self):
        p = Parameter("theta")
        with pytest.raises(AttributeError):
            p.name = "phi"

    def test_hashable(self):
        p1 = Parameter("theta")
        p2 = Parameter("theta")
        assert p1 == p2
        assert hash(p1) == hash(p2)
        assert len({p1, p2}) == 1

    def test_distinct(self):
        p1 = Parameter("theta")
        p2 = Parameter("phi")
        assert p1 != p2


class TestParameterVector:
    def test_length(self):
        pv = ParameterVector("theta", 5)
        assert len(pv) == 5

    def test_indexing(self):
        pv = ParameterVector("theta", 3)
        assert pv[0].name == "theta[0]"
        assert pv[2].name == "theta[2]"

    def test_iteration(self):
        pv = ParameterVector("p", 3)
        names = [p.name for p in pv]
        assert names == ["p[0]", "p[1]", "p[2]"]


class TestParameterizedCircuit:
    def test_fixed_gates(self):
        pqc = ParameterizedCircuit(2)
        pqc.apply(H, 0).apply(CNOT, 0, 1)
        assert pqc.num_parameters == 0
        qc = pqc.bind({})
        sv = qc.statevector()
        expected = QuantumCircuit(2)
        expected.apply(H, 0).apply(CNOT, 0, 1)
        np.testing.assert_allclose(sv, expected.statevector(), atol=1e-10)

    def test_parameterized_ry(self):
        theta = Parameter("theta")
        pqc = ParameterizedCircuit(1)
        pqc.ry(theta, 0)
        qc = pqc.bind({theta: np.pi / 2})
        sv = qc.statevector()
        expected = QuantumCircuit(1)
        expected.apply(Ry(np.pi / 2), 0)
        np.testing.assert_allclose(sv, expected.statevector(), atol=1e-10)

    def test_parameterized_rx(self):
        phi = Parameter("phi")
        pqc = ParameterizedCircuit(1)
        pqc.rx(phi, 0)
        assert pqc.num_parameters == 1
        qc = pqc.bind({phi: np.pi})
        sv = qc.statevector()
        assert abs(sv[1]) > 0.99

    def test_parameterized_rz(self):
        alpha = Parameter("alpha")
        pqc = ParameterizedCircuit(1)
        pqc.apply(H, 0)
        pqc.rz(alpha, 0)
        assert pqc.num_parameters == 1
        qc = pqc.bind({alpha: 0.0})
        sv = qc.statevector()
        expected = QuantumCircuit(1)
        expected.apply(H, 0)
        np.testing.assert_allclose(sv, expected.statevector(), atol=1e-10)

    def test_multiple_parameters(self):
        t1 = Parameter("t1")
        t2 = Parameter("t2")
        pqc = ParameterizedCircuit(1)
        pqc.ry(t1, 0).rz(t2, 0)
        assert pqc.num_parameters == 2
        qc = pqc.bind({t1: 0.5, t2: 1.0})
        sv = qc.statevector()
        expected = QuantumCircuit(1)
        expected.apply(Ry(0.5), 0).apply(Rz(1.0), 0)
        np.testing.assert_allclose(sv, expected.statevector(), atol=1e-10)

    def test_reuse_parameter(self):
        theta = Parameter("theta")
        pqc = ParameterizedCircuit(2)
        pqc.ry(theta, 0).ry(theta, 1)
        assert pqc.num_parameters == 1
        qc = pqc.bind({theta: np.pi / 4})
        sv = qc.statevector()
        expected = QuantumCircuit(2)
        expected.apply(Ry(np.pi / 4), 0).apply(Ry(np.pi / 4), 1)
        np.testing.assert_allclose(sv, expected.statevector(), atol=1e-10)

    def test_missing_parameter_raises(self):
        theta = Parameter("theta")
        pqc = ParameterizedCircuit(1)
        pqc.ry(theta, 0)
        with pytest.raises(ValueError, match="Missing parameter"):
            pqc.bind({})

    def test_bind_array(self):
        pv = ParameterVector("p", 3)
        pqc = ParameterizedCircuit(1)
        pqc.ry(pv[0], 0).rz(pv[1], 0).ry(pv[2], 0)
        values = [0.1, 0.2, 0.3]
        qc = pqc.bind_array(values, parameter_order=list(pv))
        sv = qc.statevector()
        expected = QuantumCircuit(1)
        expected.apply(Ry(0.1), 0).apply(Rz(0.2), 0).apply(Ry(0.3), 0)
        np.testing.assert_allclose(sv, expected.statevector(), atol=1e-10)

    def test_bind_array_wrong_length(self):
        theta = Parameter("theta")
        pqc = ParameterizedCircuit(1)
        pqc.ry(theta, 0)
        with pytest.raises(ValueError, match="Expected 1"):
            pqc.bind_array([1.0, 2.0])

    def test_bind_multiple_times(self):
        theta = Parameter("theta")
        pqc = ParameterizedCircuit(1)
        pqc.ry(theta, 0)
        qc1 = pqc.bind({theta: 0.0})
        qc2 = pqc.bind({theta: np.pi})
        assert abs(qc1.statevector()[0]) > 0.99
        assert abs(qc2.statevector()[1]) > 0.99

    def test_mixed_fixed_and_parameterized(self):
        theta = Parameter("theta")
        pqc = ParameterizedCircuit(2)
        pqc.apply(H, 0)
        pqc.ry(theta, 1)
        pqc.apply(CNOT, 0, 1)
        assert pqc.num_parameters == 1
        qc = pqc.bind({theta: np.pi / 3})
        assert len(qc.instructions) == 3

    def test_repr(self):
        pqc = ParameterizedCircuit(2)
        pqc.ry(Parameter("t"), 0)
        assert "qubits=2" in repr(pqc)
        assert "parameters=1" in repr(pqc)


class TestHardwareEfficientTemplate:
    def test_creation(self):
        pqc, params = hardware_efficient_template(2, depth=1)
        assert pqc.num_qubits == 2
        assert len(params) == 4
        assert pqc.num_parameters == 4

    def test_depth_2(self):
        pqc, params = hardware_efficient_template(3, depth=2)
        assert len(params) == 12
        assert pqc.num_parameters == 12

    def test_bind_and_run(self):
        pqc, params = hardware_efficient_template(2, depth=1)
        values = np.random.uniform(-np.pi, np.pi, size=len(params))
        qc = pqc.bind_array(values, parameter_order=params)
        sv = qc.statevector()
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    def test_matches_manual_ansatz(self):
        pqc, params = hardware_efficient_template(2, depth=1)
        values = [0.1, 0.2, 0.3, 0.4]
        qc = pqc.bind_array(values, parameter_order=params)
        manual = QuantumCircuit(2)
        manual.apply(Ry(0.1), 0).apply(Rz(0.2), 0)
        manual.apply(Ry(0.3), 1).apply(Rz(0.4), 1)
        manual.apply(CNOT, 0, 1)
        np.testing.assert_allclose(qc.statevector(), manual.statevector(), atol=1e-10)
