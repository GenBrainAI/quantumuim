"""Tests for matplotlib-based quantum state visualization."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, CNOT, CZ, SWAP, CCX, Rx, Ry, Rz
from quantumium.visualization import (
    plot_bloch_sphere,
    plot_bloch_multivector,
    plot_histogram,
    plot_state_city,
    plot_state_hinton,
    plot_circuit,
)


class TestPlotBlochSphere:
    def test_zero_state(self) -> None:
        sv = np.array([1, 0], dtype=complex)
        fig = plot_bloch_sphere(sv)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_one_state(self) -> None:
        sv = np.array([0, 1], dtype=complex)
        fig = plot_bloch_sphere(sv)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_plus_state(self) -> None:
        sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
        fig = plot_bloch_sphere(sv)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_density_matrix_input(self) -> None:
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        fig = plot_bloch_sphere(rho)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_invalid_shape_raises(self) -> None:
        sv = np.array([1, 0, 0, 0], dtype=complex)
        with pytest.raises(ValueError, match="Expected 1-qubit"):
            plot_bloch_sphere(sv)


class TestPlotBlochMultivector:
    def test_bell_state(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        fig = plot_bloch_multivector(qc.statevector())
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 2
        matplotlib.pyplot.close(fig)

    def test_single_qubit(self) -> None:
        sv = np.array([1, 0], dtype=complex)
        fig = plot_bloch_multivector(sv)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 1
        matplotlib.pyplot.close(fig)

    def test_three_qubit_ghz(self) -> None:
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(CNOT, 0, 2)
        fig = plot_bloch_multivector(qc.statevector())
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 3
        matplotlib.pyplot.close(fig)

    def test_invalid_length_raises(self) -> None:
        with pytest.raises(ValueError, match="not a power of 2"):
            plot_bloch_multivector(np.array([1, 0, 0], dtype=complex))


class TestPlotHistogram:
    def test_bell_counts(self) -> None:
        counts = {"00": 500, "11": 500}
        fig = plot_histogram(counts)
        assert isinstance(fig, Figure)
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == "Measurement Outcome"
        assert ax.get_ylabel() == "Probability"
        matplotlib.pyplot.close(fig)

    def test_custom_title(self) -> None:
        counts = {"0": 700, "1": 300}
        fig = plot_histogram(counts, title="My Measurement")
        ax = fig.get_axes()[0]
        assert ax.get_title() == "My Measurement"
        matplotlib.pyplot.close(fig)

    def test_uniform_superposition(self) -> None:
        counts = {format(i, "03b"): 125 for i in range(8)}
        fig = plot_histogram(counts)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_single_outcome(self) -> None:
        counts = {"1": 1000}
        fig = plot_histogram(counts)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_empty_counts(self) -> None:
        counts: dict[str, int] = {}
        fig = plot_histogram(counts)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)


class TestPlotStateCity:
    def test_statevector_input(self) -> None:
        sv = np.array([1, 0, 0, 0], dtype=complex)
        fig = plot_state_city(sv)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 2
        matplotlib.pyplot.close(fig)

    def test_density_matrix_input(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        dm = qc.density_matrix()
        fig = plot_state_city(dm)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_custom_title(self) -> None:
        sv = np.array([1, 0], dtype=complex)
        fig = plot_state_city(sv, title="Test City")
        assert fig._suptitle.get_text() == "Test City"  # type: ignore[union-attr]
        matplotlib.pyplot.close(fig)

    def test_three_qubit(self) -> None:
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(CNOT, 0, 2)
        fig = plot_state_city(qc.statevector())
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)


class TestPlotStateHinton:
    def test_statevector_input(self) -> None:
        sv = np.array([1, 0, 0, 0], dtype=complex)
        fig = plot_state_hinton(sv)
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) == 2
        matplotlib.pyplot.close(fig)

    def test_density_matrix_input(self) -> None:
        rho = np.eye(4, dtype=complex) / 4
        fig = plot_state_hinton(rho)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_bell_state(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        fig = plot_state_hinton(qc.statevector())
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_custom_title(self) -> None:
        sv = np.array([1, 0], dtype=complex)
        fig = plot_state_hinton(sv, title="Test Hinton")
        assert fig._suptitle.get_text() == "Test Hinton"  # type: ignore[union-attr]
        matplotlib.pyplot.close(fig)


class TestPlotCircuit:
    def test_bell_circuit(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        fig = plot_circuit(qc)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_single_qubit_gates(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        qc.apply(X, 0)
        qc.apply(Rx(0.5), 0)
        fig = plot_circuit(qc)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_multi_qubit_gates(self) -> None:
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        qc.apply(CZ, 1, 2)
        qc.apply(SWAP, 0, 2)
        qc.apply(CCX, 0, 1, 2)
        fig = plot_circuit(qc)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_parametric_gates(self) -> None:
        qc = QuantumCircuit(2)
        qc.apply(Rx(1.0), 0)
        qc.apply(Ry(0.5), 1)
        qc.apply(Rz(2.0), 0)
        fig = plot_circuit(qc)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_empty_circuit(self) -> None:
        qc = QuantumCircuit(2)
        fig = plot_circuit(qc)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)

    def test_custom_title(self) -> None:
        qc = QuantumCircuit(1)
        qc.apply(H, 0)
        fig = plot_circuit(qc, title="My Circuit")
        ax = fig.get_axes()[0]
        assert ax.get_title() == "My Circuit"
        matplotlib.pyplot.close(fig)

    def test_three_qubit_circuit(self) -> None:
        qc = QuantumCircuit(3)
        qc.apply(H, 0)
        qc.apply(H, 1)
        qc.apply(H, 2)
        qc.apply(CNOT, 0, 1)
        qc.apply(CNOT, 1, 2)
        fig = plot_circuit(qc)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)
