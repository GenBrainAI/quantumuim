"""Tests for cross-SDK circuit interoperability."""

from __future__ import annotations


import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import (
    CCX,
    CNOT,
    CZ,
    SWAP,
    CRx,
    CRy,
    CRz,
    H,
    Rx,
    Rxx,
    Ry,
    Rz,
    Rzz,
    S,
    T,
    X,
    Y,
    Z,
)


# ── Qiskit conversion tests ─────────────────────────────────────────────


class TestQiskitConversion:
    @pytest.fixture(autouse=True)
    def _skip_if_no_qiskit(self):
        pytest.importorskip("qiskit")

    def _import(self):
        from quantumium.interop.qiskit_convert import from_qiskit, to_qiskit
        return to_qiskit, from_qiskit

    def test_bell_state_roundtrip(self):
        to_qiskit, from_qiskit = self._import()
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)

        qiskit_qc = to_qiskit(qc)
        roundtrip = from_qiskit(qiskit_qc)

        sv_orig = qc.statevector()
        sv_rt = roundtrip.statevector()
        assert abs(np.vdot(sv_orig, sv_rt)) > 1 - 1e-8

    def test_single_qubit_gates(self):
        to_qiskit, from_qiskit = self._import()
        for gate, name in [(H, "H"), (X, "X"), (Y, "Y"), (Z, "Z"), (S, "S"), (T, "T")]:
            qc = QuantumCircuit(1)
            qc.apply(gate, 0)
            rt = from_qiskit(to_qiskit(qc))
            assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-8, f"Failed for {name}"

    def test_two_qubit_gates(self):
        to_qiskit, from_qiskit = self._import()
        for gate, name in [(CNOT, "CNOT"), (CZ, "CZ"), (SWAP, "SWAP")]:
            qc = QuantumCircuit(2)
            qc.apply(H, 0)
            qc.apply(gate, 0, 1)
            rt = from_qiskit(to_qiskit(qc))
            assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-8, f"Failed for {name}"

    def test_toffoli_gate(self):
        to_qiskit, from_qiskit = self._import()
        qc = QuantumCircuit(3)
        qc.apply(X, 0)
        qc.apply(X, 1)
        qc.apply(CCX, 0, 1, 2)
        rt = from_qiskit(to_qiskit(qc))
        assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-8

    def test_parametric_rx(self):
        to_qiskit, from_qiskit = self._import()
        qc = QuantumCircuit(1)
        qc.apply(Rx(0.7), 0)
        rt = from_qiskit(to_qiskit(qc))
        assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-8

    def test_parametric_ry(self):
        to_qiskit, from_qiskit = self._import()
        qc = QuantumCircuit(1)
        qc.apply(Ry(1.2), 0)
        rt = from_qiskit(to_qiskit(qc))
        assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-8

    def test_parametric_rz(self):
        to_qiskit, from_qiskit = self._import()
        qc = QuantumCircuit(1)
        qc.apply(Rz(2.1), 0)
        rt = from_qiskit(to_qiskit(qc))
        assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-8

    def test_controlled_rotation(self):
        to_qiskit, from_qiskit = self._import()
        for gate_fn, name in [(CRx, "CRx"), (CRy, "CRy"), (CRz, "CRz")]:
            qc = QuantumCircuit(2)
            qc.apply(H, 0)
            qc.apply(gate_fn(0.8), 0, 1)
            rt = from_qiskit(to_qiskit(qc))
            assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-8, f"Failed for {name}"

    def test_rxx_rzz(self):
        to_qiskit, from_qiskit = self._import()
        for gate_fn, name in [(Rxx, "Rxx"), (Rzz, "Rzz")]:
            qc = QuantumCircuit(2)
            qc.apply(H, 0)
            qc.apply(gate_fn(0.5), 0, 1)
            rt = from_qiskit(to_qiskit(qc))
            assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-8, f"Failed for {name}"

    def test_complex_circuit_roundtrip(self):
        to_qiskit, from_qiskit = self._import()
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(Ry(0.5), 2)
        qc.apply(CZ, 1, 2)
        qc.apply(Rx(1.0), 0)
        qc.apply(SWAP, 0, 2)
        rt = from_qiskit(to_qiskit(qc))
        assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-8

    def test_num_qubits_preserved(self):
        to_qiskit, from_qiskit = self._import()
        qc = QuantumCircuit(5)
        qc.apply(H, 3)
        qiskit_qc = to_qiskit(qc)
        assert qiskit_qc.num_qubits == 5
        rt = from_qiskit(qiskit_qc)
        assert rt.num_qubits == 5

    def test_unsupported_gate_raises(self):
        from quantumium.interop.qiskit_convert import to_qiskit
        from quantumium.core.gate import Gate
        qc = QuantumCircuit(1)
        qc.apply(Gate(np.eye(2, dtype=complex), "CustomGate"), 0)
        with pytest.raises(ValueError, match="Unsupported gate"):
            to_qiskit(qc)

    def test_qiskit_identity_and_barrier_skipped(self):
        from quantumium.interop.qiskit_convert import from_qiskit
        import qiskit
        qk = qiskit.QuantumCircuit(2)
        qk.id(0)
        qk.barrier()
        qk.h(0)
        qk.cx(0, 1)
        qc = from_qiskit(qk)
        assert len(qc.instructions) == 2

    def test_from_qiskit_unsupported_raises(self):
        from quantumium.interop.qiskit_convert import from_qiskit
        import qiskit
        qk = qiskit.QuantumCircuit(2)
        qk.iswap(0, 1)
        with pytest.raises(ValueError, match="Unsupported Qiskit gate"):
            from_qiskit(qk)


# ── Cirq conversion tests ───────────────────────────────────────────────


class TestCirqConversion:
    @pytest.fixture(autouse=True)
    def _skip_if_no_cirq(self):
        pytest.importorskip("cirq")

    def _import(self):
        from quantumium.interop.cirq_convert import from_cirq, to_cirq
        return to_cirq, from_cirq

    def test_bell_state_roundtrip(self):
        to_cirq, from_cirq = self._import()
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)

        cirq_circuit = to_cirq(qc)
        roundtrip = from_cirq(cirq_circuit)

        sv_orig = qc.statevector()
        sv_rt = roundtrip.statevector()
        assert abs(np.vdot(sv_orig, sv_rt)) > 1 - 1e-8

    def test_single_qubit_gates(self):
        to_cirq, from_cirq = self._import()
        for gate, name in [(H, "H"), (X, "X"), (Y, "Y"), (Z, "Z"), (S, "S"), (T, "T")]:
            qc = QuantumCircuit(1)
            qc.apply(gate, 0)
            rt = from_cirq(to_cirq(qc))
            assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-8, f"Failed for {name}"

    def test_two_qubit_gates(self):
        to_cirq, from_cirq = self._import()
        for gate, name in [(CNOT, "CNOT"), (CZ, "CZ"), (SWAP, "SWAP")]:
            qc = QuantumCircuit(2)
            qc.apply(H, 0)
            qc.apply(gate, 0, 1)
            rt = from_cirq(to_cirq(qc))
            assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-8, f"Failed for {name}"

    def test_toffoli_gate(self):
        to_cirq, from_cirq = self._import()
        qc = QuantumCircuit(3)
        qc.apply(X, 0)
        qc.apply(X, 1)
        qc.apply(CCX, 0, 1, 2)
        rt = from_cirq(to_cirq(qc))
        assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-8

    def test_parametric_rx(self):
        to_cirq, from_cirq = self._import()
        qc = QuantumCircuit(1)
        qc.apply(Rx(0.7), 0)
        rt = from_cirq(to_cirq(qc))
        assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-6

    def test_parametric_ry(self):
        to_cirq, from_cirq = self._import()
        qc = QuantumCircuit(1)
        qc.apply(Ry(1.2), 0)
        rt = from_cirq(to_cirq(qc))
        assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-6

    def test_parametric_rz(self):
        to_cirq, from_cirq = self._import()
        qc = QuantumCircuit(1)
        qc.apply(Rz(2.1), 0)
        rt = from_cirq(to_cirq(qc))
        assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-6

    def test_complex_circuit_roundtrip(self):
        to_cirq, from_cirq = self._import()
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(Ry(0.5), 2)
        qc.apply(CZ, 1, 2)
        qc.apply(SWAP, 0, 2)
        rt = from_cirq(to_cirq(qc))
        assert abs(np.vdot(qc.statevector(), rt.statevector())) > 1 - 1e-6

    def test_num_qubits_preserved(self):
        to_cirq, from_cirq = self._import()
        qc = QuantumCircuit(4)
        qc.apply(H, 2)
        cirq_qc = to_cirq(qc)
        rt = from_cirq(cirq_qc, num_qubits=4)
        assert rt.num_qubits == 4

    def test_num_qubits_inferred(self):
        to_cirq, from_cirq = self._import()
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(X, 2)
        cirq_qc = to_cirq(qc)
        rt = from_cirq(cirq_qc)
        assert rt.num_qubits == 3

    def test_unsupported_gate_raises(self):
        from quantumium.interop.cirq_convert import to_cirq
        from quantumium.core.gate import Gate
        qc = QuantumCircuit(1)
        qc.apply(Gate(np.eye(2, dtype=complex), "WeirdGate"), 0)
        with pytest.raises(ValueError, match="Unsupported gate"):
            to_cirq(qc)


# ── Import error handling tests ──────────────────────────────────────────


class TestInteropImportHandling:
    def test_interop_init_loads_without_crash(self):
        import quantumium.interop
        assert isinstance(quantumium.interop.__all__, list)

    def test_no_qiskit_gives_helpful_error(self):
        qiskit = pytest.importorskip("qiskit", reason="testing error only when qiskit absent")
        # If qiskit IS installed, just verify import works
        from quantumium.interop.qiskit_convert import to_qiskit
        assert callable(to_qiskit)

    def test_no_cirq_gives_helpful_error(self):
        cirq = pytest.importorskip("cirq", reason="testing error only when cirq absent")
        from quantumium.interop.cirq_convert import to_cirq
        assert callable(to_cirq)
