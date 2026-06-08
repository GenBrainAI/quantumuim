"""JSON serialization for quantum circuits.

Provides save/load to a compact JSON format for circuit persistence
and interchange between Quantumium tools.
"""

from __future__ import annotations

import json

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, I, X, Y, Z, H, S, T, CNOT, SWAP, Rx, Ry, Rz


NAMED_GATES: dict[str, Gate] = {
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z,
    "H": H,
    "S": S,
    "T": T,
    "CNOT": CNOT,
    "SWAP": SWAP,
}
"""Lookup table mapping gate name strings to pre-built Gate instances."""

ROTATION_FACTORIES: dict[str, callable] = {"Rx": Rx, "Ry": Ry, "Rz": Rz}
"""Mapping of rotation gate prefixes to their factory functions."""


def to_dict(circuit: QuantumCircuit) -> dict:
    """Serialize a quantum circuit to a plain dictionary.

    Named gates are stored by name alone; rotation gates include the angle;
    custom gates store both the real and imaginary parts of their unitary
    matrix.

    Args:
        circuit (QuantumCircuit): The circuit to serialize.

    Returns:
        dict: A dictionary in ``quantumium-v1`` format with keys
            ``"format"``, ``"num_qubits"``, and ``"instructions"``.
    """
    instructions = []
    for inst in circuit.instructions:
        entry: dict = {"qubits": list(inst.target_qubits)}

        if inst.gate.name in NAMED_GATES:
            entry["gate"] = inst.gate.name
        else:
            for prefix, factory in ROTATION_FACTORIES.items():
                if inst.gate.name.startswith(prefix + "("):
                    try:
                        angle = float(inst.gate.name[len(prefix) + 1 : -1])
                        entry["gate"] = prefix
                        entry["angle"] = angle
                        break
                    except ValueError:
                        pass
            else:
                if "gate" not in entry:
                    matrix = inst.gate.matrix
                    entry["gate"] = "custom"
                    entry["name"] = inst.gate.name
                    entry["matrix_real"] = matrix.real.tolist()
                    entry["matrix_imag"] = matrix.imag.tolist()

        instructions.append(entry)

    return {
        "format": "quantumium-v1",
        "num_qubits": circuit.num_qubits,
        "instructions": instructions,
    }


def from_dict(data: dict) -> QuantumCircuit:
    """Deserialize a quantum circuit from a dictionary.

    Accepts a dictionary previously produced by :func:`to_dict` and
    reconstructs the corresponding :class:`QuantumCircuit`.

    Args:
        data (dict): A dictionary with ``"format": "quantumium-v1"``
            and the expected ``"num_qubits"`` / ``"instructions"`` keys.

    Returns:
        QuantumCircuit: The reconstructed circuit.

    Raises:
        ValueError: If the format field is missing or unrecognized, or if
            an instruction references an unknown gate name.
    """
    if data.get("format") != "quantumium-v1":
        raise ValueError(f"Unknown format: {data.get('format')}")

    qc = QuantumCircuit(data["num_qubits"])

    for entry in data["instructions"]:
        qubits = tuple(entry["qubits"])
        gate_name = entry["gate"]

        if gate_name in NAMED_GATES:
            qc.apply(NAMED_GATES[gate_name], *qubits)
        elif gate_name in ROTATION_FACTORIES:
            qc.apply(ROTATION_FACTORIES[gate_name](entry["angle"]), *qubits)
        elif gate_name == "custom":
            real = np.array(entry["matrix_real"])
            imag = np.array(entry["matrix_imag"])
            matrix = real + 1j * imag
            qc.apply(Gate(matrix, entry.get("name", "custom")), *qubits)
        else:
            raise ValueError(f"Unknown gate: {gate_name}")

    return qc


def to_json(circuit: QuantumCircuit, indent: int = 2) -> str:
    """Serialize a quantum circuit to a JSON string.

    Args:
        circuit (QuantumCircuit): The circuit to serialize.
        indent (int): Number of spaces for JSON indentation. Defaults to 2.

    Returns:
        str: A pretty-printed JSON string in ``quantumium-v1`` format.
    """
    return json.dumps(to_dict(circuit), indent=indent)


def from_json(json_str: str) -> QuantumCircuit:
    """Deserialize a quantum circuit from a JSON string.

    Args:
        json_str (str): A JSON string previously produced by :func:`to_json`.

    Returns:
        QuantumCircuit: The reconstructed circuit.

    Raises:
        ValueError: If the JSON does not conform to ``quantumium-v1`` format.
    """
    return from_dict(json.loads(json_str))


def save(circuit: QuantumCircuit, filepath: str) -> None:
    """Save a quantum circuit to a JSON file on disk.

    Args:
        circuit (QuantumCircuit): The circuit to persist.
        filepath (str): Destination file path. The file is created or
            overwritten with indented JSON.
    """
    with open(filepath, "w") as f:
        json.dump(to_dict(circuit), f, indent=2)


def load(filepath: str) -> QuantumCircuit:
    """Load a quantum circuit from a JSON file on disk.

    Args:
        filepath (str): Path to a JSON file in ``quantumium-v1`` format.

    Returns:
        QuantumCircuit: The deserialized circuit.

    Raises:
        ValueError: If the file does not conform to ``quantumium-v1`` format.
        FileNotFoundError: If the file does not exist.
    """
    with open(filepath) as f:
        return from_dict(json.load(f))
