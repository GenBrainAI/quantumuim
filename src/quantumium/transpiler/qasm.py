"""OpenQASM 2.0 export and import for quantum circuits.

Converts between Quantumium circuits and the industry-standard
OpenQASM 2.0 text format for interoperability with other tools.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, I, X, Y, Z, H, S, T, CNOT, SWAP, Rx, Ry, Rz


GATE_TO_QASM: dict[str, str] = {
    "I": "id",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "H": "h",
    "S": "s",
    "T": "t",
    "CNOT": "cx",
    "SWAP": "swap",
}
"""Mapping from Quantumium gate names to their OpenQASM 2.0 equivalents."""

QASM_TO_GATE: dict[str, Gate] = {
    "id": I,
    "x": X,
    "y": Y,
    "z": Z,
    "h": H,
    "s": S,
    "t": T,
    "cx": CNOT,
    "swap": SWAP,
}
"""Mapping from OpenQASM 2.0 gate names to Quantumium Gate objects."""

ROTATION_QASM: dict[str, str] = {
    "Rx": "rx",
    "Ry": "ry",
    "Rz": "rz",
}
"""Mapping from Quantumium rotation gate prefixes to OpenQASM names."""


def _extract_rotation_angle(gate: Gate) -> tuple[str, float] | None:
    """Attempt to identify a rotation gate and extract its angle.

    First tries to parse the angle from the gate's name (e.g.,
    ``"Rx(1.5708)"``). If the name does not encode a rotation, falls back
    to numerical analysis of the unitary matrix to detect Rx, Ry, or Rz
    structure.

    Args:
        gate (Gate): The gate to inspect.

    Returns:
        tuple[str, float] | None: A ``(prefix, angle)`` pair such as
            ``("Rx", 1.5708)`` if the gate is a recognized rotation, or
            ``None`` if it cannot be identified as one.
    """
    name = gate.name
    for prefix in ("Rx", "Ry", "Rz"):
        if name.startswith(prefix + "("):
            try:
                angle = float(name[len(prefix) + 1 : -1])
                return prefix, angle
            except ValueError:
                pass

    m = gate.matrix
    if gate.num_qubits != 1:
        return None

    cos_half = m[0, 0]
    if np.isclose(cos_half.imag, 0) and np.isclose(m[1, 0].real, 0):
        sin_half_i = m[0, 1]
        if np.isclose(sin_half_i.real, 0) and not np.isclose(sin_half_i.imag, 0):
            theta = 2 * np.arctan2(-sin_half_i.imag, cos_half.real)
            return "Rx", float(theta)

    if np.isclose(cos_half.imag, 0) and np.isclose(m[0, 1].imag, 0):
        if np.isclose(m[0, 0].real, m[1, 1].real) and np.isclose(m[0, 1].real, -m[1, 0].real):
            theta = 2 * np.arctan2(-m[0, 1].real, m[0, 0].real)
            return "Ry", float(theta)

    if np.isclose(m[0, 1], 0) and np.isclose(m[1, 0], 0):
        phase0 = np.angle(m[0, 0])
        phase1 = np.angle(m[1, 1])
        theta = phase1 - phase0
        return "Rz", float(theta)

    return None


def to_qasm(circuit: QuantumCircuit) -> str:
    """Export a quantum circuit to OpenQASM 2.0 format.

    Args:
        circuit: The quantum circuit to export.

    Returns:
        OpenQASM 2.0 string.
    """
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{circuit.num_qubits}];",
        f"creg c[{circuit.num_qubits}];",
        "",
    ]

    for inst in circuit.instructions:
        gate = inst.gate
        qubits = inst.target_qubits
        qubit_args = ",".join(f"q[{q}]" for q in qubits)

        if gate.name in GATE_TO_QASM:
            qasm_name = GATE_TO_QASM[gate.name]
            lines.append(f"{qasm_name} {qubit_args};")
            continue

        rotation = _extract_rotation_angle(gate)
        if rotation is not None:
            prefix, angle = rotation
            qasm_name = ROTATION_QASM[prefix]
            lines.append(f"{qasm_name}({angle:.10f}) {qubit_args};")
            continue

        if gate.num_qubits == 1:
            a, b, c, d = gate.matrix.flat
            lines.append(f"// Custom gate: {gate.name}")
            lines.append(f"u3({_decompose_u3(gate.matrix)}) {qubit_args};")
        else:
            lines.append(f"// Unsupported multi-qubit gate: {gate.name}")

    return "\n".join(lines) + "\n"


def _decompose_u3(matrix: np.ndarray) -> str:
    """Decompose a single-qubit unitary into U3(theta, phi, lambda) parameters.

    Extracts the three Euler angles that parametrize the U3 gate from an
    arbitrary 2x2 unitary matrix.

    Args:
        matrix (np.ndarray): A 2x2 complex unitary matrix.

    Returns:
        str: Comma-separated string ``"theta,phi,lambda"`` with 10 decimal
            places, suitable for embedding in an OpenQASM ``u3(...)`` call.
    """
    a = matrix[0, 0]
    matrix[0, 1]
    c = matrix[1, 0]
    d = matrix[1, 1]

    theta = 2 * np.arccos(min(abs(a), 1.0))

    if np.isclose(abs(a), 0):
        phi = 0.0
        lam = np.angle(d) - np.angle(c)
    elif np.isclose(abs(c), 0):
        phi = 0.0
        lam = 0.0
    else:
        phi = np.angle(c) - np.angle(a)
        lam = np.angle(d) - np.angle(c) - phi + np.pi

    phi = float(phi) % (2 * np.pi)
    lam = float(lam) % (2 * np.pi)

    return f"{float(theta):.10f},{phi:.10f},{lam:.10f}"


def from_qasm(qasm_str: str) -> QuantumCircuit:
    """Parse an OpenQASM 2.0 string into a QuantumCircuit.

    Supports the standard gates defined in ``qelib1.inc`` (id, x, y, z, h,
    s, t, cx, swap), parametric rotation gates (rx, ry, rz), and the
    universal U3 gate. Lines beginning with ``//``, ``OPENQASM``, or
    ``include`` are silently skipped.

    Args:
        qasm_str (str): A complete OpenQASM 2.0 program string.

    Returns:
        QuantumCircuit: The reconstructed Quantumium circuit.

    Raises:
        ValueError: If the string contains no ``qreg`` declaration.
    """
    num_qubits = None
    instructions: list[tuple[str, list[float], list[int]]] = []

    for line in qasm_str.strip().split("\n"):
        line = line.strip()
        if (
            not line
            or line.startswith("//")
            or line.startswith("OPENQASM")
            or line.startswith("include")
        ):
            continue

        qreg_match = re.match(r"qreg\s+\w+\[(\d+)\];", line)
        if qreg_match:
            num_qubits = int(qreg_match.group(1))
            continue

        if line.startswith("creg"):
            continue

        gate_match = re.match(
            r"(\w+)(?:\(([^)]*)\))?\s+(.*);",
            line,
        )
        if not gate_match:
            continue

        gate_name = gate_match.group(1)
        params_str = gate_match.group(2)
        qubits_str = gate_match.group(3)

        params = []
        if params_str:
            params = [float(p.strip()) for p in params_str.split(",")]

        qubit_indices = []
        for qarg in qubits_str.split(","):
            qarg = qarg.strip()
            idx_match = re.match(r"\w+\[(\d+)\]", qarg)
            if idx_match:
                qubit_indices.append(int(idx_match.group(1)))

        instructions.append((gate_name, params, qubit_indices))

    if num_qubits is None:
        raise ValueError("No qreg declaration found in QASM")

    qc = QuantumCircuit(num_qubits)

    rotation_factories: dict[str, Callable[[float], Gate]] = {
        "rx": Rx,
        "ry": Ry,
        "rz": Rz,
    }

    for gate_name, params, qubits in instructions:
        if gate_name in QASM_TO_GATE:
            qc.apply(QASM_TO_GATE[gate_name], *qubits)
        elif gate_name in rotation_factories:
            qc.apply(rotation_factories[gate_name](params[0]), *qubits)
        elif gate_name == "u3" and len(params) == 3:
            theta, phi, lam = params
            matrix = np.array(
                [
                    [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                    [
                        np.exp(1j * phi) * np.sin(theta / 2),
                        np.exp(1j * (phi + lam)) * np.cos(theta / 2),
                    ],
                ],
                dtype=complex,
            )
            qc.apply(Gate(matrix, "U3"), *qubits)
        elif gate_name == "measure":
            pass

    return qc
