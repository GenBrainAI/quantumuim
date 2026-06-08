"""Tests for quantum information theory module."""

import numpy as np

from quantumium.quantum_info.channels import (
    QuantumChannel,
    kraus_to_choi,
    choi_to_kraus,
    kraus_to_superoperator,
    superoperator_to_kraus,
    kraus_to_ptm,
    compose_channels,
    tensor_channels,
    channel_fidelity,
)
from quantumium.quantum_info.distances import (
    trace_distance,
    bures_distance,
    hellinger_distance,
    relative_entropy,
)
from quantumium.quantum_info.process_tomography import (
    process_tomography,
    chi_matrix,
)
from quantumium.core.noise import bit_flip, phase_flip, depolarizing, amplitude_damping


# --- Helper matrices ---
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


# ============================================================
# Channel Representation Tests
# ============================================================


class TestKrausToChoi:
    def test_identity_channel(self):
        choi = kraus_to_choi([I2])
        # Choi of identity is |Ω><Ω| (unnormalized maximally entangled)
        omega = np.array([1, 0, 0, 1], dtype=complex)
        expected = np.outer(omega, omega.conj())
        np.testing.assert_allclose(choi, expected, atol=1e-10)

    def test_x_channel(self):
        choi = kraus_to_choi([X])
        # X flips: E(|i><j|) = |1-i><1-j|
        omega_x = np.array([0, 1, 1, 0], dtype=complex)
        expected = np.outer(omega_x, omega_x.conj())
        np.testing.assert_allclose(choi, expected, atol=1e-10)

    def test_choi_positive_semidefinite(self):
        """Choi of a CPTP map must be positive semidefinite."""
        for p in [0.0, 0.1, 0.5, 1.0]:
            channel = bit_flip(p)
            choi = kraus_to_choi(channel.kraus_operators)
            eigvals = np.linalg.eigvalsh(choi)
            assert all(v >= -1e-10 for v in eigvals)

    def test_choi_trace(self):
        """Tr_output(Choi) = d * I for trace-preserving channels."""
        channel = depolarizing(0.3)
        choi = kraus_to_choi(channel.kraus_operators)
        # Partial trace over output (second) system
        d = 2
        ptr = np.zeros((d, d), dtype=complex)
        for k in range(d):
            for l in range(d):
                for m in range(d):
                    ptr[k, l] += choi[k * d + m, l * d + m]
        np.testing.assert_allclose(ptr, I2, atol=1e-10)


class TestChoiToKraus:
    def test_roundtrip_identity(self):
        original = [I2]
        choi = kraus_to_choi(original)
        recovered = choi_to_kraus(choi)
        # Apply both to a test state and compare
        rho = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        out_orig = sum(K @ rho @ K.conj().T for K in original)
        out_recov = sum(K @ rho @ K.conj().T for K in recovered)
        np.testing.assert_allclose(out_recov, out_orig, atol=1e-10)

    def test_roundtrip_depolarizing(self):
        channel = depolarizing(0.2)
        choi = kraus_to_choi(channel.kraus_operators)
        recovered = choi_to_kraus(choi)
        rho = np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]], dtype=complex)
        out_orig = sum(K @ rho @ K.conj().T for K in channel.kraus_operators)
        out_recov = sum(K @ rho @ K.conj().T for K in recovered)
        np.testing.assert_allclose(out_recov, out_orig, atol=1e-10)

    def test_roundtrip_amplitude_damping(self):
        channel = amplitude_damping(0.5)
        choi = kraus_to_choi(channel.kraus_operators)
        recovered = choi_to_kraus(choi)
        rho = np.array([[0.3, 0.4j], [-0.4j, 0.7]], dtype=complex)
        out_orig = sum(K @ rho @ K.conj().T for K in channel.kraus_operators)
        out_recov = sum(K @ rho @ K.conj().T for K in recovered)
        np.testing.assert_allclose(out_recov, out_orig, atol=1e-10)


class TestSuperoperator:
    def test_identity(self):
        S = kraus_to_superoperator([I2])
        np.testing.assert_allclose(S, np.eye(4, dtype=complex), atol=1e-10)

    def test_action(self):
        """S vec(rho) should equal vec(E(rho))."""
        channel = phase_flip(0.3)
        S = kraus_to_superoperator(channel.kraus_operators)
        rho = np.array([[0.6, 0.4], [0.4, 0.4]], dtype=complex)
        vec_rho = rho.flatten(order="F")
        expected = sum(K @ rho @ K.conj().T for K in channel.kraus_operators)
        result_vec = S @ vec_rho
        result = result_vec.reshape(2, 2, order="F")
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_roundtrip(self):
        channel = bit_flip(0.15)
        S = kraus_to_superoperator(channel.kraus_operators)
        recovered = superoperator_to_kraus(S)
        rho = np.array([[0.5, 0.3], [0.3, 0.5]], dtype=complex)
        out_orig = sum(K @ rho @ K.conj().T for K in channel.kraus_operators)
        out_recov = sum(K @ rho @ K.conj().T for K in recovered)
        np.testing.assert_allclose(out_recov, out_orig, atol=1e-10)


class TestPTM:
    def test_identity_ptm(self):
        """PTM of identity channel is the identity matrix."""
        ptm = kraus_to_ptm([I2])
        np.testing.assert_allclose(ptm, np.eye(4), atol=1e-10)

    def test_ptm_is_real(self):
        """PTM must be real for any CPTP map."""
        channel = depolarizing(0.25)
        ptm = kraus_to_ptm(channel.kraus_operators)
        assert ptm.dtype in (np.float64, np.float32)

    def test_x_gate_ptm(self):
        """X gate: preserves I and X, flips Y and Z."""
        ptm = kraus_to_ptm([X])
        # X P_i X: I->I, X->X, Y->-Y, Z->-Z
        expected = np.diag([1.0, 1.0, -1.0, -1.0])
        np.testing.assert_allclose(ptm, expected, atol=1e-10)

    def test_depolarizing_ptm(self):
        """Depolarizing channel shrinks Bloch vector uniformly."""
        p = 0.2
        ptm = kraus_to_ptm(depolarizing(p).kraus_operators)
        # First row/column preserved (trace preservation + unitality)
        assert abs(ptm[0, 0] - 1.0) < 1e-10
        # Off-diagonal blocks should be 0 (symmetric channel)
        shrink = 1 - p
        for i in range(1, 4):
            assert abs(ptm[i, i] - shrink) < 1e-10


class TestComposeChannels:
    def test_identity_compose(self):
        """Composing with identity is a no-op."""
        channel = bit_flip(0.2)
        composed = compose_channels([I2], channel.kraus_operators)
        rho = np.array([[0.7, 0.2], [0.2, 0.3]], dtype=complex)
        out_orig = sum(K @ rho @ K.conj().T for K in channel.kraus_operators)
        out_comp = sum(K @ rho @ K.conj().T for K in composed)
        np.testing.assert_allclose(out_comp, out_orig, atol=1e-10)

    def test_compose_two_depolarizing(self):
        """Composing two depolarizing channels."""
        ch1 = depolarizing(0.1)
        ch2 = depolarizing(0.2)
        composed = compose_channels(ch1.kraus_operators, ch2.kraus_operators)
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        step1 = sum(K @ rho @ K.conj().T for K in ch1.kraus_operators)
        expected = sum(K @ step1 @ K.conj().T for K in ch2.kraus_operators)
        result = sum(K @ rho @ K.conj().T for K in composed)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestTensorChannels:
    def test_tensor_identity(self):
        """I ⊗ I gives 2-qubit identity."""
        tensored = tensor_channels([I2], [I2])
        assert len(tensored) == 1
        np.testing.assert_allclose(tensored[0], np.eye(4, dtype=complex), atol=1e-10)

    def test_tensor_action(self):
        """(A ⊗ B)(rho_AB) = A(rho_A) ⊗ B(rho_B) for product states."""
        ch_a = bit_flip(0.1)
        ch_b = phase_flip(0.2)
        tensored = tensor_channels(ch_a.kraus_operators, ch_b.kraus_operators)

        rho_a = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        rho_b = np.array([[0.5, 0.4], [0.4, 0.5]], dtype=complex)
        rho_ab = np.kron(rho_a, rho_b)

        out_a = sum(K @ rho_a @ K.conj().T for K in ch_a.kraus_operators)
        out_b = sum(K @ rho_b @ K.conj().T for K in ch_b.kraus_operators)
        expected = np.kron(out_a, out_b)

        result = sum(K @ rho_ab @ K.conj().T for K in tensored)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestChannelFidelity:
    def test_identity_fidelity(self):
        assert abs(channel_fidelity([I2]) - 1.0) < 1e-10

    def test_x_gate_fidelity(self):
        """X gate has process fidelity 0 with identity but F_avg = 1/3."""
        f = channel_fidelity([X])
        expected = (2 * 0 + 1) / 3  # F_proc(I, X) = 0
        assert abs(f - expected) < 1e-10

    def test_depolarizing_fidelity(self):
        """F_avg should decrease with noise."""
        f_low = channel_fidelity(depolarizing(0.01).kraus_operators)
        f_high = channel_fidelity(depolarizing(0.5).kraus_operators)
        assert f_low > f_high


class TestQuantumChannelClass:
    def test_from_kraus(self):
        ch = QuantumChannel.from_kraus(bit_flip(0.1).kraus_operators, name="BF")
        assert ch.dim == 2
        assert ch.name == "BF"
        assert ch.choi.shape == (4, 4)
        assert ch.superoperator.shape == (4, 4)
        assert ch.ptm.shape == (4, 4)

    def test_from_choi(self):
        choi = kraus_to_choi([I2])
        ch = QuantumChannel.from_choi(choi)
        assert ch.dim == 2
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = ch.apply(rho)
        np.testing.assert_allclose(result, rho, atol=1e-10)

    def test_from_unitary(self):
        ch = QuantumChannel.from_unitary(H_gate, name="H")
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        result = ch.apply(rho)
        expected = H_gate @ rho @ H_gate.conj().T
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_compose(self):
        ch_x = QuantumChannel.from_unitary(X)
        ch_h = QuantumChannel.from_unitary(H_gate)
        composed = ch_h.compose(ch_x)  # H(X(rho))
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        expected = H_gate @ X @ rho @ X.conj().T @ H_gate.conj().T
        result = composed.apply(rho)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_tensor(self):
        ch_x = QuantumChannel.from_unitary(X)
        ch_z = QuantumChannel.from_unitary(Z)
        tensored = ch_x.tensor(ch_z)
        assert tensored.dim == 4

    def test_is_unital(self):
        ch_dep = QuantumChannel.from_kraus(depolarizing(0.3).kraus_operators)
        assert ch_dep.is_unital()

        ch_ad = QuantumChannel.from_kraus(amplitude_damping(0.5).kraus_operators)
        assert not ch_ad.is_unital()

    def test_average_fidelity(self):
        ch = QuantumChannel.from_unitary(I2)
        assert abs(ch.average_fidelity() - 1.0) < 1e-10


# ============================================================
# Distance Measure Tests
# ============================================================


class TestTraceDistance:
    def test_identical_states(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        assert abs(trace_distance(rho, rho)) < 1e-10

    def test_orthogonal_pure_states(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        sigma = np.array([[0, 0], [0, 1]], dtype=complex)
        assert abs(trace_distance(rho, sigma) - 1.0) < 1e-10

    def test_symmetry(self):
        rho = np.array([[0.7, 0.2], [0.2, 0.3]], dtype=complex)
        sigma = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=complex)
        assert abs(trace_distance(rho, sigma) - trace_distance(sigma, rho)) < 1e-10

    def test_statevector_input(self):
        psi = np.array([1, 0], dtype=complex)
        phi = np.array([0, 1], dtype=complex)
        assert abs(trace_distance(psi, phi) - 1.0) < 1e-10

    def test_mixed_vs_pure(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        sigma = np.eye(2, dtype=complex) / 2
        assert abs(trace_distance(rho, sigma) - 0.5) < 1e-10

    def test_dimension_mismatch(self):
        try:
            trace_distance(np.eye(2, dtype=complex), np.eye(4, dtype=complex))
            assert False
        except ValueError:
            pass


class TestBuresDistance:
    def test_identical(self):
        rho = np.array([[0.6, 0.2], [0.2, 0.4]], dtype=complex)
        assert abs(bures_distance(rho, rho)) < 1e-10

    def test_orthogonal(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        sigma = np.array([[0, 0], [0, 1]], dtype=complex)
        assert abs(bures_distance(rho, sigma) - np.sqrt(2)) < 1e-10

    def test_range(self):
        rho = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
        sigma = np.array([[0.5, 0.2], [0.2, 0.5]], dtype=complex)
        d = bures_distance(rho, sigma)
        assert 0 <= d <= np.sqrt(2) + 1e-10


class TestHellingerDistance:
    def test_identical(self):
        rho = np.array([[0.5, 0.3], [0.3, 0.5]], dtype=complex)
        assert abs(hellinger_distance(rho, rho)) < 1e-6

    def test_orthogonal(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        sigma = np.array([[0, 0], [0, 1]], dtype=complex)
        assert abs(hellinger_distance(rho, sigma) - np.sqrt(2)) < 1e-10

    def test_pure_state_symmetry(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        sigma = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        d1 = hellinger_distance(rho, sigma)
        d2 = hellinger_distance(sigma, rho)
        assert abs(d1 - d2) < 1e-10


class TestRelativeEntropy:
    def test_identical_states(self):
        rho = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
        assert abs(relative_entropy(rho, rho)) < 1e-10

    def test_pure_vs_mixed(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        sigma = np.eye(2, dtype=complex) / 2
        # S(|0><0| || I/2) = -0 - log2(1/2) = 1
        assert abs(relative_entropy(rho, sigma) - 1.0) < 1e-10

    def test_orthogonal_support_gives_infinity(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        sigma = np.array([[0, 0], [0, 1]], dtype=complex)
        assert relative_entropy(rho, sigma) == float("inf")

    def test_non_negative(self):
        rho = np.array([[0.6, 0.2], [0.2, 0.4]], dtype=complex)
        sigma = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=complex)
        assert relative_entropy(rho, sigma) >= -1e-10


# ============================================================
# Process Tomography Tests
# ============================================================


class TestChiMatrix:
    def test_identity_chi(self):
        """Chi of identity: only chi[0,0] = 1 (identity Pauli component)."""
        chi = chi_matrix(I2)
        assert abs(chi[0, 0] - 1.0) < 1e-10
        assert abs(np.sum(np.abs(chi)) - 1.0) < 1e-10

    def test_x_gate_chi(self):
        """Chi of X gate: only chi[1,1] = 1 (X Pauli component)."""
        chi = chi_matrix(X)
        assert abs(chi[1, 1] - 1.0) < 1e-10
        assert abs(np.sum(np.abs(chi)) - 1.0) < 1e-10

    def test_hermitian(self):
        """Chi matrix must be Hermitian."""
        chi = chi_matrix(H_gate)
        np.testing.assert_allclose(chi, chi.conj().T, atol=1e-10)

    def test_positive_semidefinite(self):
        """Chi matrix of a CPTP map must be positive semidefinite."""
        chi = chi_matrix(H_gate)
        eigvals = np.linalg.eigvalsh(chi)
        assert all(v >= -1e-10 for v in eigvals)


class TestProcessTomography:
    def test_identity_process(self):
        result = process_tomography(I2, target_unitary=I2)
        assert result.num_qubits == 1
        assert result.input_states_used == 4
        assert abs(result.process_fidelity - 1.0) < 1e-10

    def test_x_gate_process(self):
        result = process_tomography(X, target_unitary=X)
        assert abs(result.process_fidelity - 1.0) < 1e-10

    def test_choi_positive(self):
        result = process_tomography(H_gate)
        eigvals = np.linalg.eigvalsh(result.choi_matrix)
        assert all(v >= -1e-10 for v in eigvals)

    def test_imperfect_fidelity(self):
        """Comparing X to identity should give low fidelity."""
        result = process_tomography(X, target_unitary=I2)
        assert result.process_fidelity < 0.01

    def test_close_unitaries(self):
        """Small rotation should have high fidelity with identity."""
        theta = 0.01
        Rz = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
        result = process_tomography(Rz, target_unitary=I2)
        assert result.process_fidelity > 0.99

    def test_no_target(self):
        result = process_tomography(X)
        assert result.process_fidelity is None
        assert result.chi_matrix.shape == (4, 4)
        assert result.choi_matrix.shape == (4, 4)
