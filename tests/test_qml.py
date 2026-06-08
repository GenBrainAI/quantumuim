"""Tests for quantum machine learning module."""

import numpy as np

from quantumium.qml.encoding import amplitude_encode, angle_encode, zz_feature_map
from quantumium.qml.classifier import variational_classifier, _predict_single
from quantumium.qml.kernel import quantum_kernel, quantum_kernel_matrix


class TestAmplitudeEncoding:
    def test_basic(self):
        """Encode [1, 0, 0, 0] -> |00>."""
        data = np.array([1.0, 0.0, 0.0, 0.0])
        qc = amplitude_encode(data)
        sv = qc.statevector()
        assert qc.num_qubits == 2
        # Should be close to |00>
        assert abs(sv[0]) > 0.99

    def test_uniform(self):
        """Encode uniform vector -> equal superposition."""
        data = np.array([1.0, 1.0, 1.0, 1.0])
        qc = amplitude_encode(data)
        sv = qc.statevector()
        expected = np.ones(4) / 2.0
        np.testing.assert_allclose(np.abs(sv), expected, atol=0.01)

    def test_normalized(self):
        """Encoded state should be normalized."""
        data = np.array([3.0, 4.0, 0.0, 0.0])
        qc = amplitude_encode(data)
        sv = qc.statevector()
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    def test_preserves_ratios(self):
        """Amplitude ratios should match data ratios."""
        data = np.array([3.0, 4.0])
        qc = amplitude_encode(data)
        sv = qc.statevector()
        ratio = abs(sv[0]) / abs(sv[1])
        assert abs(ratio - 0.75) < 0.01

    def test_invalid_length(self):
        try:
            amplitude_encode(np.array([1.0, 2.0, 3.0]))
            assert False
        except ValueError:
            pass


class TestAngleEncoding:
    def test_basic(self):
        """Angle encoding uses one qubit per feature."""
        data = np.array([0.5, 1.0, 1.5])
        qc = angle_encode(data)
        assert qc.num_qubits == 3

    def test_zero_angle(self):
        """Zero features should leave qubits in |0>."""
        data = np.array([0.0, 0.0])
        qc = angle_encode(data)
        sv = qc.statevector()
        assert abs(sv[0]) > 0.999

    def test_pi_rotation(self):
        """Ry(pi) should map |0> to |1>."""
        data = np.array([np.pi])
        qc = angle_encode(data)
        sv = qc.statevector()
        assert abs(sv[1]) > 0.999

    def test_normalized(self):
        data = np.array([0.3, 0.7, 1.2])
        qc = angle_encode(data)
        sv = qc.statevector()
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10


class TestZZFeatureMap:
    def test_basic(self):
        """ZZ feature map uses correct number of qubits."""
        data = np.array([0.5, 1.0])
        qc = zz_feature_map(data, reps=1)
        assert qc.num_qubits == 2

    def test_normalized(self):
        data = np.array([0.1, 0.2, 0.3])
        qc = zz_feature_map(data, reps=2)
        sv = qc.statevector()
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    def test_different_data_different_states(self):
        """Different data should produce different states."""
        sv1 = zz_feature_map(np.array([0.0, 0.0]), reps=1).statevector()
        sv2 = zz_feature_map(np.array([1.0, 1.0]), reps=1).statevector()
        fidelity = abs(np.dot(sv1.conj(), sv2)) ** 2
        assert fidelity < 0.99

    def test_same_data_same_state(self):
        """Same data should produce identical states."""
        data = np.array([0.5, 0.8])
        sv1 = zz_feature_map(data, reps=2).statevector()
        sv2 = zz_feature_map(data, reps=2).statevector()
        np.testing.assert_allclose(sv1, sv2, atol=1e-12)


class TestQuantumKernel:
    def test_self_kernel(self):
        """k(x, x) should be 1."""
        x = np.array([0.5, 1.0])
        k = quantum_kernel(x, x, feature_map="zz")
        assert abs(k - 1.0) < 1e-10

    def test_self_kernel_angle(self):
        x = np.array([0.3, 0.7, 1.1])
        k = quantum_kernel(x, x, feature_map="angle")
        assert abs(k - 1.0) < 1e-10

    def test_range(self):
        """Kernel values should be in [0, 1]."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([np.pi, np.pi])
        k = quantum_kernel(x1, x2, feature_map="angle")
        assert 0 <= k <= 1

    def test_symmetry(self):
        """k(x, y) = k(y, x)."""
        x = np.array([0.5, 1.0])
        y = np.array([1.5, 0.3])
        k1 = quantum_kernel(x, y)
        k2 = quantum_kernel(y, x)
        assert abs(k1 - k2) < 1e-10

    def test_dimension_mismatch(self):
        try:
            quantum_kernel(np.array([1.0]), np.array([1.0, 2.0]))
            assert False
        except ValueError:
            pass


class TestQuantumKernelMatrix:
    def test_symmetric(self):
        """K(X, X) should be symmetric."""
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        K = quantum_kernel_matrix(X, feature_map="angle")
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_diagonal_ones(self):
        """Diagonal of K(X, X) should be 1."""
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        K = quantum_kernel_matrix(X, feature_map="angle")
        np.testing.assert_allclose(np.diag(K), np.ones(2), atol=1e-10)

    def test_positive_semidefinite(self):
        """Kernel matrix should be positive semidefinite."""
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        K = quantum_kernel_matrix(X, feature_map="angle")
        eigenvalues = np.linalg.eigvalsh(K)
        assert all(e >= -1e-10 for e in eigenvalues)

    def test_asymmetric(self):
        """K(X, Y) with different X and Y."""
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        Y = np.array([[0.5, 0.6]])
        K = quantum_kernel_matrix(X, Y, feature_map="angle")
        assert K.shape == (2, 1)

    def test_zz_kernel_matrix(self):
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        K = quantum_kernel_matrix(X, feature_map="zz", reps=1)
        assert K.shape == (2, 2)
        np.testing.assert_allclose(np.diag(K), np.ones(2), atol=1e-10)


class TestVariationalClassifier:
    def test_basic_training(self):
        """VQC should train on a simple linearly separable dataset."""
        # Simple 2-feature dataset
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, np.pi],
                [np.pi, 0.0],
                [np.pi, np.pi],
            ]
        )
        y = np.array([1.0, -1.0, -1.0, 1.0])  # XOR-like

        result = variational_classifier(X, y, num_layers=2, epochs=30, learning_rate=0.2, seed=42)
        assert result.weights is not None
        assert len(result.loss_history) == 30
        assert result.loss_history[-1] <= result.loss_history[0]

    def test_result_fields(self):
        X = np.array([[0.0, 0.0], [np.pi, np.pi]])
        y = np.array([1.0, -1.0])
        result = variational_classifier(X, y, num_layers=1, epochs=5, seed=0)
        assert result.weights.shape == (4,)  # 1 layer * 2 qubits * 2 params
        assert isinstance(result.bias, float)
        assert 0 <= result.accuracy <= 1

    def test_prediction_range(self):
        """Raw predictions should be in [-1-bias, 1+bias] range."""
        weights = np.zeros(4)
        bias = 0.0
        x = np.array([0.5, 0.5])
        pred = _predict_single(x, weights, bias, num_layers=1)
        assert -2.0 <= pred <= 2.0

    def test_invalid_input(self):
        try:
            variational_classifier(np.array([1, 2, 3]), np.array([1, -1, 1]))
            assert False
        except ValueError:
            pass
