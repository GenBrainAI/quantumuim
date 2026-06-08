"""Convert circuits between Quantumium and Qiskit."""

from __future__ import annotations

import math
import re

import numpy as np

try:
    import qiskit  # type: ignore[import-not-found]
    from qiskit.circuit import QuantumCircuit as QiskitCircuit  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(
        "Qiskit is required for circuit conversion. "
        "Install it with: pip install quantumium[qiskit]"
    ) from e

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import (
    CCX,
    CNOT,
    CZ,
    SWAP,
    CRx,
    CRy,
    CRz,
    Gate,
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

_FIXED_GATE_TO_QISKIT: dict[str, str] = {
    "H": "h",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "S": "s",
    "T": "t",
    "CNOT": "cx",
    "CZ": "cz",
    "SWAP": "swap",
    "CCX": "ccx",
}

_PARAM_GATE_PATTERN = re.compile(r"^(Rx|Ry|Rz|CRx|CRy|CRz|Rxx|Rzz)\((-?[\d.]+)\)$")


def to_qiskit(circuit: QuantumCircuit) -> QiskitCircuit:  # type: ignore[type-arg]
    """Convert a Quantumium circuit to a Qiskit QuantumCircuit."""
    qc = QiskitCircuit(circuit.num_qubits)

    for inst in circuit.instructions:
        name = inst.gate.name
        qubits = list(inst.target_qubits)

        if name in _FIXED_GATE_TO_QISKIT:
            getattr(qc, _FIXED_GATE_TO_QISKIT[name])(*qubits)
            continue

        m = _PARAM_GATE_PATTERN.match(name)
        if m:
            gate_type, angle_str = m.group(1), m.group(2)
            angle = float(angle_str)
            _qiskit_param_gate = {
                "Rx": "rx",
                "Ry": "ry",
                "Rz": "rz",
                "CRx": "crx",
                "CRy": "cry",
                "CRz": "crz",
                "Rxx": "rxx",
                "Rzz": "rzz",
            }
            getattr(qc, _qiskit_param_gate[gate_type])(angle, *qubits)
            continue

        raise ValueError(
            f"Unsupported gate for Qiskit conversion: {name}. "
            f"Supported: {', '.join(sorted(_FIXED_GATE_TO_QISKIT.keys()))} "
            f"and Rx, Ry, Rz, CRx, CRy, CRz, Rxx, Rzz."
        )

    return qc


_QISKIT_GATE_MAP: dict[str, Gate] = {
    "h": H,
    "x": X,
    "y": Y,
    "z": Z,
    "s": S,
    "t": T,
    "cx": CNOT,
    "cz": CZ,
    "swap": SWAP,
    "ccx": CCX,
}

_QISKIT_PARAM_GATE_MAP: dict[str, type] = {
    "rx": Rx,  # type: ignore[dict-item]
    "ry": Ry,  # type: ignore[dict-item]
    "rz": Rz,  # type: ignore[dict-item]
    "crx": CRx,  # type: ignore[dict-item]
    "cry": CRy,  # type: ignore[dict-item]
    "crz": CRz,  # type: ignore[dict-item]
    "rxx": Rxx,  # type: ignore[dict-item]
    "rzz": Rzz,  # type: ignore[dict-item]
}


def from_qiskit(qiskit_circuit: QiskitCircuit) -> QuantumCircuit:  # type: ignore[type-arg]
    """Convert a Qiskit QuantumCircuit to a Quantumium circuit."""
    qc = QuantumCircuit(qiskit_circuit.num_qubits)

    for instruction in qiskit_circuit.data:
        op = instruction.operation
        qubits = [qiskit_circuit.find_bit(q).index for q in instruction.qubits]
        name = op.name

        if name in _QISKIT_GATE_MAP:
            qc.apply(_QISKIT_GATE_MAP[name], *qubits)
        elif name in _QISKIT_PARAM_GATE_MAP:
            angle = float(op.params[0])
            gate = _QISKIT_PARAM_GATE_MAP[name](angle)
            qc.apply(gate, *qubits)
        elif name == "id":
            pass
        elif name == "measure":
            pass
        elif name == "barrier":
            pass
        else:
            raise ValueError(
                f"Unsupported Qiskit gate for conversion: {name}. "
                f"Supported: {', '.join(sorted(_QISKIT_GATE_MAP.keys()))} "
                f"and {', '.join(sorted(_QISKIT_PARAM_GATE_MAP.keys()))}."
            )

    return qc
