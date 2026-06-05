"""Tests for quantumium.stabilizer — Clifford tableau simulation."""

import math
import numpy as np
import pytest

from quantumium.stabilizer import (
    StabilizerTableau,
    hadamard,
    phase_gate,
    cnot,
    cz,
    pauli_x,
    pauli_y,
    pauli_z,
    measure,
    to_statevector,
)


# ============================================================
# Tableau basics
# ============================================================

class TestTableauBasics:
    def test_zero_state_stabilizers(self):
        t = StabilizerTableau(2)
        stabs = t.stabilizer_generators()
        assert stabs == ["+ZI", "+IZ"]

    def test_zero_state_destabilizers(self):
        t = StabilizerTableau(2)
        destabs = t.destabilizer_generators()
        assert destabs == ["+XI", "+IX"]

    def test_single_qubit(self):
        t = StabilizerTableau(1)
        assert t.stabilizer_generators() == ["+Z"]
        assert t.destabilizer_generators() == ["+X"]

    def test_copy_independent(self):
        t = StabilizerTableau(2)
        t2 = t.copy()
        hadamard(t2, 0)
        assert t.stabilizer_generators() == ["+ZI", "+IZ"]
        assert t2.stabilizer_generators() != ["+ZI", "+IZ"]

    def test_num_qubits(self):
        t = StabilizerTableau(5)
        assert t.num_qubits == 5


# ============================================================
# Clifford gates
# ============================================================

class TestGates:
    def test_hadamard_on_zero(self):
        t = StabilizerTableau(1)
        hadamard(t, 0)
        assert t.stabilizer_generators() == ["+X"]

    def test_double_hadamard(self):
        t = StabilizerTableau(1)
        hadamard(t, 0)
        hadamard(t, 0)
        assert t.stabilizer_generators() == ["+Z"]

    def test_phase_gate(self):
        t = StabilizerTableau(1)
        hadamard(t, 0)
        phase_gate(t, 0)
        # H|0> = |+>, S|+> = |+i> stabilized by Y
        assert t.stabilizer_generators() == ["+Y"]

    def test_pauli_x_flips_z(self):
        t = StabilizerTableau(1)
        pauli_x(t, 0)
        # X flips Z eigenvalue: |0> → |1>, stabilizer Z → -Z
        assert t.stabilizer_generators() == ["-Z"]

    def test_pauli_z_flips_x(self):
        t = StabilizerTableau(1)
        hadamard(t, 0)
        pauli_z(t, 0)
        # |+> → |->: stabilizer X → -X
        assert t.stabilizer_generators() == ["-X"]

    def test_pauli_y(self):
        t = StabilizerTableau(1)
        pauli_y(t, 0)
        # Y|0> = i|1>: stabilizer Z → -Z
        assert t.stabilizer_generators() == ["-Z"]

    def test_cnot_entangles(self):
        t = StabilizerTableau(2)
        hadamard(t, 0)
        cnot(t, 0, 1)
        stabs = t.stabilizer_generators()
        assert "+XX" in stabs
        assert "+ZZ" in stabs

    def test_cz_gate(self):
        t = StabilizerTableau(2)
        hadamard(t, 0)
        hadamard(t, 1)
        cz(t, 0, 1)
        # CZ on |++> produces a state with stabilizers
        stabs = t.stabilizer_generators()
        assert "+XZ" in stabs
        assert "+ZX" in stabs


# ============================================================
# Statevector conversion
# ============================================================

class TestStatevector:
    def test_zero_state(self):
        t = StabilizerTableau(2)
        sv = to_statevector(t)
        expected = np.array([1, 0, 0, 0], dtype=complex)
        np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-10)

    def test_plus_state(self):
        t = StabilizerTableau(1)
        hadamard(t, 0)
        sv = to_statevector(t)
        expected = np.array([1, 1], dtype=complex) / math.sqrt(2)
        np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-10)

    def test_bell_state(self):
        t = StabilizerTableau(2)
        hadamard(t, 0)
        cnot(t, 0, 1)
        sv = to_statevector(t)
        # Bell state: (|00> + |11>) / √2
        assert abs(sv[0]) == pytest.approx(1 / math.sqrt(2), abs=1e-10)
        assert abs(sv[3]) == pytest.approx(1 / math.sqrt(2), abs=1e-10)
        assert abs(sv[1]) == pytest.approx(0, abs=1e-10)
        assert abs(sv[2]) == pytest.approx(0, abs=1e-10)

    def test_one_state(self):
        t = StabilizerTableau(1)
        pauli_x(t, 0)
        sv = to_statevector(t)
        assert abs(sv[1]) == pytest.approx(1.0, abs=1e-10)

    def test_ghz_state(self):
        n = 3
        t = StabilizerTableau(n)
        hadamard(t, 0)
        for i in range(n - 1):
            cnot(t, i, i + 1)
        sv = to_statevector(t)
        # GHZ: (|000> + |111>) / √2
        assert abs(sv[0]) == pytest.approx(1 / math.sqrt(2), abs=1e-10)
        assert abs(sv[7]) == pytest.approx(1 / math.sqrt(2), abs=1e-10)
        for i in range(1, 7):
            assert abs(sv[i]) == pytest.approx(0, abs=1e-10)

    def test_normalization(self):
        t = StabilizerTableau(3)
        hadamard(t, 0)
        hadamard(t, 1)
        cnot(t, 0, 2)
        sv = to_statevector(t)
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)


# ============================================================
# Measurement
# ============================================================

class TestMeasurement:
    def test_deterministic_zero(self):
        t = StabilizerTableau(1)
        rng = np.random.default_rng(42)
        outcome = measure(t, 0, rng)
        assert outcome == 0

    def test_deterministic_one(self):
        t = StabilizerTableau(1)
        pauli_x(t, 0)
        rng = np.random.default_rng(42)
        outcome = measure(t, 0, rng)
        assert outcome == 1

    def test_superposition_random(self):
        outcomes = {0: 0, 1: 0}
        for seed in range(200):
            t = StabilizerTableau(1)
            hadamard(t, 0)
            rng = np.random.default_rng(seed)
            outcome = measure(t, 0, rng)
            outcomes[outcome] += 1
        # Should be roughly 50/50
        assert outcomes[0] > 60
        assert outcomes[1] > 60

    def test_post_measurement_deterministic(self):
        t = StabilizerTableau(1)
        hadamard(t, 0)
        rng = np.random.default_rng(42)
        first = measure(t, 0, rng)
        # After measurement, result should be deterministic
        second = measure(t, 0, rng)
        assert first == second

    def test_bell_measurement_correlation(self):
        for seed in range(100):
            t = StabilizerTableau(2)
            hadamard(t, 0)
            cnot(t, 0, 1)
            rng = np.random.default_rng(seed)
            r0 = measure(t, 0, rng)
            r1 = measure(t, 1, rng)
            assert r0 == r1  # Bell state: qubits always agree

    def test_measurement_collapses_entanglement(self):
        t = StabilizerTableau(2)
        hadamard(t, 0)
        cnot(t, 0, 1)
        rng = np.random.default_rng(42)
        measure(t, 0, rng)
        # After measuring qubit 0, qubit 1 should be deterministic
        sv = to_statevector(t)
        probs = np.abs(sv) ** 2
        # Only one basis state should have non-zero probability for qubit 1
        non_zero = np.sum(probs > 1e-10)
        assert non_zero == 1


# ============================================================
# Gate sequences and identities
# ============================================================

class TestGateIdentities:
    def test_s_squared_is_z(self):
        t1 = StabilizerTableau(1)
        hadamard(t1, 0)
        phase_gate(t1, 0)
        phase_gate(t1, 0)

        t2 = StabilizerTableau(1)
        hadamard(t2, 0)
        pauli_z(t2, 0)

        assert t1.stabilizer_generators() == t2.stabilizer_generators()

    def test_hzh_equals_x(self):
        t1 = StabilizerTableau(1)
        hadamard(t1, 0)
        pauli_z(t1, 0)
        hadamard(t1, 0)

        t2 = StabilizerTableau(1)
        pauli_x(t2, 0)

        assert t1.stabilizer_generators() == t2.stabilizer_generators()

    def test_three_cnots_swap(self):
        # SWAP = CNOT(0,1) CNOT(1,0) CNOT(0,1)
        t = StabilizerTableau(2)
        pauli_x(t, 0)  # |10>
        cnot(t, 0, 1)
        cnot(t, 1, 0)
        cnot(t, 0, 1)
        # Should now be |01>
        rng = np.random.default_rng(0)
        assert measure(t, 0, rng) == 0
        assert measure(t, 1, rng) == 1

    def test_cnot_cnot_identity(self):
        t = StabilizerTableau(2)
        hadamard(t, 0)
        cnot(t, 0, 1)
        cnot(t, 0, 1)
        # Double CNOT should undo
        stabs = t.stabilizer_generators()
        # Back to H|0>⊗|0>, stabilizers should be XI and IZ
        assert "+XI" in stabs
        assert "+IZ" in stabs

    def test_teleportation(self):
        # Quantum teleportation of |+> state
        for seed in range(50):
            rng = np.random.default_rng(seed)
            t = StabilizerTableau(3)

            hadamard(t, 0)          # Prepare |+> to teleport
            hadamard(t, 1)          # Bell pair on 1,2
            cnot(t, 1, 2)
            cnot(t, 0, 1)           # Bell measurement
            hadamard(t, 0)
            m0 = measure(t, 0, rng)
            m1 = measure(t, 1, rng)
            if m1:
                pauli_x(t, 2)
            if m0:
                pauli_z(t, 2)

            # Verify qubit 2 is in |+> via statevector
            sv = to_statevector(t)
            probs = np.abs(sv) ** 2
            # Qubit 2 is the least significant bit: sum probs where bit 2 is 0 vs 1
            p0_q2 = sum(probs[i] for i in range(8) if (i & 1) == 0)
            p1_q2 = sum(probs[i] for i in range(8) if (i & 1) == 1)
            assert p0_q2 == pytest.approx(p1_q2, abs=1e-8), \
                f"seed={seed}: qubit 2 not in |+> state, p0={p0_q2}, p1={p1_q2}"
