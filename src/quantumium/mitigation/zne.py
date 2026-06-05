"""Zero-Noise Extrapolation (ZNE) for error mitigation.

ZNE runs circuits at multiple noise levels and extrapolates to the
zero-noise limit. Noise is amplified by inserting identity-resolved
gate pairs (G G†) after each gate in the circuit.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit, Instruction
from ..core.gate import Gate


def _adjoint_gate(gate: Gate) -> Gate:
    return Gate(gate.matrix.conj().T, f"{gate.name}†")


def noise_scaled_circuits(
    circuit: QuantumCircuit,
    scale_factors: list[int],
) -> list[QuantumCircuit]:
    """Create circuits with amplified noise via unitary folding.

    For scale factor k, each gate G is replaced with G (G†G)^((k-1)/2).
    Scale factors must be odd positive integers.
    """
    results = []
    for k in scale_factors:
        if k < 1 or k % 2 == 0:
            raise ValueError(f"Scale factors must be odd positive integers, got {k}")
        qc = QuantumCircuit(circuit.num_qubits)
        for inst in circuit.instructions:
            qc.apply(inst.gate, *inst.target_qubits)
            adj = _adjoint_gate(inst.gate)
            for _ in range((k - 1) // 2):
                qc.apply(adj, *inst.target_qubits)
                qc.apply(inst.gate, *inst.target_qubits)
        results.append(qc)
    return results


def richardson_extrapolate(
    scale_factors: list[float],
    expectation_values: list[float],
) -> float:
    """Richardson extrapolation to the zero-noise limit.

    Fits a polynomial of degree len(scale_factors)-1 through the
    (scale_factor, expectation) points and evaluates at scale_factor=0.
    """
    if len(scale_factors) != len(expectation_values):
        raise ValueError("scale_factors and expectation_values must have same length")
    if len(scale_factors) < 2:
        raise ValueError("Need at least 2 data points for extrapolation")

    x = np.array(scale_factors, dtype=float)
    y = np.array(expectation_values, dtype=float)
    coeffs = np.polyfit(x, y, deg=len(x) - 1)
    return float(np.polyval(coeffs, 0.0))


def linear_extrapolate(
    scale_factors: list[float],
    expectation_values: list[float],
) -> float:
    """Linear extrapolation to zero noise using first two points."""
    if len(scale_factors) < 2 or len(expectation_values) < 2:
        raise ValueError("Need at least 2 data points")
    x = np.array(scale_factors[:2], dtype=float)
    y = np.array(expectation_values[:2], dtype=float)
    slope = (y[1] - y[0]) / (x[1] - x[0])
    return float(y[0] - slope * x[0])


def zne_expectation(
    circuit: QuantumCircuit,
    observable_fn: Callable[[NDArray[np.complexfloating]], float],
    scale_factors: list[int] | None = None,
    extrapolation: str = "richardson",
) -> float:
    """Full ZNE pipeline: scale noise, measure, extrapolate.

    Args:
        circuit: The base circuit.
        observable_fn: Function that takes a statevector and returns
            an expectation value.
        scale_factors: Odd positive integers for noise scaling.
            Defaults to [1, 3, 5].
        extrapolation: "richardson" or "linear".

    Returns:
        The zero-noise extrapolated expectation value.
    """
    if scale_factors is None:
        scale_factors = [1, 3, 5]

    scaled_circuits = noise_scaled_circuits(circuit, scale_factors)
    expectations = [observable_fn(qc.statevector()) for qc in scaled_circuits]

    float_factors = [float(s) for s in scale_factors]
    if extrapolation == "linear":
        return linear_extrapolate(float_factors, expectations)
    return richardson_extrapolate(float_factors, expectations)
