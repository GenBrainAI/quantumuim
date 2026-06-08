"""Convert circuits between Quantumium and Cirq."""

from __future__ import annotations

import math
import re

import numpy as np

try:
    import cirq
except ImportError as e:
    raise ImportError(
        "Cirq is required for circuit conversion. Install it with: pip install quantumium[cirq]"
    ) from e

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import (
    CCX,
    CNOT,
    CZ,
    SWAP,
    Gate,
    H,
    Rx,
    Ry,
    Rz,
    S,
    T,
    X,
    Y,
    Z,
)

_PARAM_GATE_PATTERN = re.compile(r"^(Rx|Ry|Rz|CRx|CRy|CRz|Rxx|Rzz)\((-?[\d.]+)\)$")


def to_cirq(circuit: QuantumCircuit) -> cirq.Circuit:
    """Convert a Quantumium circuit to a Cirq Circuit."""
    qubits = cirq.LineQubit.range(circuit.num_qubits)
    moments: list[cirq.Operation] = []

    for inst in circuit.instructions:
        name = inst.gate.name
        target = [qubits[i] for i in inst.target_qubits]

        if name == "H":
            moments.append(cirq.H(*target))
        elif name == "X":
            moments.append(cirq.X(*target))
        elif name == "Y":
            moments.append(cirq.Y(*target))
        elif name == "Z":
            moments.append(cirq.Z(*target))
        elif name == "S":
            moments.append(cirq.S(*target))
        elif name == "T":
            moments.append(cirq.T(*target))
        elif name == "CNOT":
            moments.append(cirq.CNOT(*target))
        elif name == "CZ":
            moments.append(cirq.CZ(*target))
        elif name == "SWAP":
            moments.append(cirq.SWAP(*target))
        elif name == "CCX":
            moments.append(cirq.CCX(*target))
        else:
            m = _PARAM_GATE_PATTERN.match(name)
            if m:
                gate_type, angle_str = m.group(1), m.group(2)
                angle = float(angle_str)
                if gate_type == "Rx":
                    moments.append(cirq.rx(angle)(*target))
                elif gate_type == "Ry":
                    moments.append(cirq.ry(angle)(*target))
                elif gate_type == "Rz":
                    moments.append(cirq.rz(angle)(*target))
                elif gate_type in ("CRx", "CRy", "CRz", "Rxx", "Rzz"):
                    gate_matrix = inst.gate.matrix
                    moments.append(cirq.MatrixGate(gate_matrix)(*target))
                else:
                    raise ValueError(f"Unsupported parametric gate: {gate_type}")
            else:
                raise ValueError(
                    f"Unsupported gate for Cirq conversion: {name}. "
                    f"Supported: H, X, Y, Z, S, T, CNOT, CZ, SWAP, CCX, "
                    f"Rx, Ry, Rz, CRx, CRy, CRz, Rxx, Rzz."
                )

    return cirq.Circuit(moments)


def from_cirq(
    cirq_circuit: cirq.Circuit,
    num_qubits: int | None = None,
) -> QuantumCircuit:
    """Convert a Cirq Circuit to a Quantumium circuit.

    Args:
        cirq_circuit: The Cirq circuit to convert.
        num_qubits: Explicit qubit count. If None, inferred from circuit.
    """
    all_qubits = sorted(cirq_circuit.all_qubits())
    qubit_map = {q: i for i, q in enumerate(all_qubits)}
    n_qubits = num_qubits if num_qubits is not None else len(all_qubits)
    qc = QuantumCircuit(n_qubits)

    for moment in cirq_circuit:
        for op in moment.operations:
            gate = op.gate
            op_qubits = [qubit_map[q] for q in op.qubits]

            if gate is None:
                continue

            if isinstance(gate, cirq.HPowGate) and gate.exponent == 1:
                qc.apply(H, *op_qubits)
            elif isinstance(gate, cirq.XPowGate) and gate.exponent == 1:
                qc.apply(X, *op_qubits)
            elif isinstance(gate, cirq.YPowGate) and gate.exponent == 1:
                qc.apply(Y, *op_qubits)
            elif isinstance(gate, cirq.ZPowGate) and gate.exponent == 1:
                qc.apply(Z, *op_qubits)
            elif isinstance(gate, cirq.ZPowGate) and gate.exponent == 0.5:
                qc.apply(S, *op_qubits)
            elif isinstance(gate, cirq.ZPowGate) and gate.exponent == 0.25:
                qc.apply(T, *op_qubits)
            elif isinstance(gate, cirq.CNotPowGate) and gate.exponent == 1:
                qc.apply(CNOT, *op_qubits)
            elif isinstance(gate, cirq.CZPowGate) and gate.exponent == 1:
                qc.apply(CZ, *op_qubits)
            elif isinstance(gate, cirq.SwapPowGate) and gate.exponent == 1:
                qc.apply(SWAP, *op_qubits)
            elif isinstance(gate, cirq.CCXPowGate) and gate.exponent == 1:
                qc.apply(CCX, *op_qubits)
            elif isinstance(gate, (cirq.Rx, type(cirq.rx(0)))):
                angle = float(gate._rads) if hasattr(gate, "_rads") else gate.exponent * math.pi
                qc.apply(Rx(angle), *op_qubits)
            elif isinstance(gate, (cirq.Ry, type(cirq.ry(0)))):
                angle = float(gate._rads) if hasattr(gate, "_rads") else gate.exponent * math.pi
                qc.apply(Ry(angle), *op_qubits)
            elif isinstance(gate, (cirq.Rz, type(cirq.rz(0)))):
                angle = float(gate._rads) if hasattr(gate, "_rads") else gate.exponent * math.pi
                qc.apply(Rz(angle), *op_qubits)
            elif isinstance(gate, cirq.MatrixGate):
                matrix = cirq.unitary(gate)
                qc.apply(
                    Gate(
                        np.array(matrix, dtype=complex),
                        gate.__class__.__name__,
                    ),
                    *op_qubits,
                )
            else:
                try:
                    matrix = cirq.unitary(gate)
                    qc.apply(
                        Gate(np.array(matrix, dtype=complex), str(gate)),
                        *op_qubits,
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Unsupported Cirq gate for conversion: {gate}.") from exc

    return qc
