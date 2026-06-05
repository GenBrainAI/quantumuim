"""ASCII circuit diagram visualization.

Renders quantum circuits as text-based diagrams for terminal display.
"""

from __future__ import annotations

from ..core.circuit import QuantumCircuit, Instruction


def circuit_to_ascii(circuit: QuantumCircuit) -> str:
    """Render a quantum circuit as an ASCII text diagram for terminal display.

    Produces a human-readable diagram with horizontal wire lines for each
    qubit and gate symbols placed along the wires. Multi-qubit gates are
    connected by vertical lines.

    Example output::

        q0: --H--*--
                 |
        q1: -----X--

    Args:
        circuit (QuantumCircuit): The quantum circuit to render.

    Returns:
        str: Multi-line string containing the ASCII circuit diagram.
    """
    n = circuit.num_qubits
    instructions = circuit.instructions

    if not instructions:
        lines = [f"q{i}: ──" for i in range(n)]
        return "\n".join(lines)

    columns: list[list[str]] = []

    for inst in instructions:
        col = _render_instruction(inst, n)
        columns.append(col)

    wire_rows: list[list[str]] = [[] for _ in range(2 * n - 1)]

    for col in columns:
        max_width = max(len(cell) for cell in col)
        for row_idx, cell in enumerate(col):
            wire_rows[row_idx].append(cell.center(max_width))

    result_lines = []
    for row_idx in range(2 * n - 1):
        if row_idx % 2 == 0:
            qubit_idx = row_idx // 2
            prefix = f"q{qubit_idx}: ──"
            joined = "──".join(wire_rows[row_idx])
            result_lines.append(prefix + joined + "──")
        else:
            prefix = " " * len(f"q{row_idx // 2}: ──")
            joined = "  ".join(wire_rows[row_idx])
            result_lines.append(prefix + joined)

    return "\n".join(result_lines)


def _render_instruction(inst: Instruction, num_qubits: int) -> list[str]:
    """Render a single instruction into a column of cells (one per row including spacers)."""
    total_rows = 2 * num_qubits - 1
    col = [""] * total_rows

    qubits = inst.target_qubits
    name = inst.gate.name

    if len(qubits) == 1:
        q = qubits[0]
        row = q * 2
        label = _gate_symbol(name)
        col[row] = label

    elif len(qubits) == 2 and name in ("CNOT", "CX"):
        control, target = qubits
        ctrl_row = control * 2
        tgt_row = target * 2
        col[ctrl_row] = "●"
        col[tgt_row] = "⊕"
        _fill_vertical(col, ctrl_row, tgt_row)

    elif len(qubits) == 2 and name == "SWAP":
        q0, q1 = qubits
        r0 = q0 * 2
        r1 = q1 * 2
        col[r0] = "✕"
        col[r1] = "✕"
        _fill_vertical(col, r0, r1)

    elif len(qubits) == 2:
        q0, q1 = qubits
        r0 = q0 * 2
        r1 = q1 * 2
        col[r0] = name
        col[r1] = name
        _fill_vertical(col, r0, r1)

    else:
        for q in qubits:
            col[q * 2] = name

    for i in range(total_rows):
        if col[i] == "":
            if i % 2 == 0:
                col[i] = "─"
            else:
                col[i] = " "

    return col


def _gate_symbol(name: str) -> str:
    """Map gate names to compact display symbols."""
    symbols = {
        "H": "H",
        "X": "X",
        "Y": "Y",
        "Z": "Z",
        "S": "S",
        "T": "T",
        "Sdg": "S†",
        "I": "I",
    }
    if name in symbols:
        return symbols[name]
    if name.startswith("R"):
        return name
    return name


def _fill_vertical(col: list[str], row_a: int, row_b: int) -> None:
    """Fill vertical connector lines between two qubit rows."""
    lo, hi = min(row_a, row_b), max(row_a, row_b)
    for r in range(lo + 1, hi):
        if col[r] == "":
            col[r] = "│"
