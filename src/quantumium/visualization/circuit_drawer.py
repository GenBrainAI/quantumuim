"""Matplotlib-based circuit diagram renderer."""

from __future__ import annotations

import re

import matplotlib
import matplotlib.axes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from quantumium.core.circuit import QuantumCircuit


_GATE_SYMBOLS: dict[str, str] = {
    "H": "H",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "S": "S",
    "T": "T",
    "Sdg": "S†",
    "Tdg": "T†",
    "I": "I",
}

_PARAM_RE = re.compile(r"^(Rx|Ry|Rz|CRx|CRy|CRz|Rxx|Rzz)\((-?[\d.]+)\)$")


def plot_circuit(circuit: QuantumCircuit, title: str | None = None) -> Figure:
    """Render a quantum circuit as a matplotlib figure.

    Args:
        circuit: The quantum circuit to draw.
        title: Optional title for the figure.

    Returns:
        A matplotlib Figure.
    """
    n_qubits = circuit.num_qubits
    instructions = circuit.instructions
    n_cols = len(instructions) if instructions else 1

    fig_width = max(4, (n_cols + 2) * 0.8)
    fig_height = max(2, n_qubits * 0.7 + 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for q in range(n_qubits):
        y = n_qubits - 1 - q
        ax.plot([0, n_cols + 1], [y, y], color="black", linewidth=0.8, zorder=0)
        ax.text(
            -0.3,
            y,
            f"q{q}",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    for col_idx, inst in enumerate(instructions):
        x = col_idx + 1
        qubits = inst.target_qubits
        name = inst.gate.name

        if name in ("CNOT", "CX") and len(qubits) == 2:
            _draw_cnot(ax, x, qubits, n_qubits)
        elif name == "CZ" and len(qubits) == 2:
            _draw_cz(ax, x, qubits, n_qubits)
        elif name == "SWAP" and len(qubits) == 2:
            _draw_swap(ax, x, qubits, n_qubits)
        elif name == "CCX" and len(qubits) == 3:
            _draw_toffoli(ax, x, qubits, n_qubits)
        elif len(qubits) == 1:
            label = _gate_label(name)
            _draw_gate_box(ax, x, qubits[0], label, n_qubits)
        elif len(qubits) == 2:
            label = _gate_label(name)
            _draw_two_qubit_box(ax, x, qubits, label, n_qubits)
        else:
            for q in qubits:
                _draw_gate_box(ax, x, q, name[:4], n_qubits)

    ax.set_xlim(-0.8, n_cols + 1.5)
    ax.set_ylim(-0.7, n_qubits - 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig


def _gate_label(name: str) -> str:
    if name in _GATE_SYMBOLS:
        return _GATE_SYMBOLS[name]
    m = _PARAM_RE.match(name)
    if m:
        gate_type = m.group(1)
        angle = float(m.group(2))
        return f"{gate_type}\n{angle:.2f}"
    return name[:6]


def _draw_gate_box(
    ax: matplotlib.axes.Axes, x: float, qubit: int, label: str, n_qubits: int
) -> None:
    y = n_qubits - 1 - qubit
    box_size = 0.35
    rect = mpatches.FancyBboxPatch(
        (x - box_size, y - box_size),
        2 * box_size,
        2 * box_size,
        boxstyle="round,pad=0.05",
        facecolor="#D4E6F1",
        edgecolor="black",
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(rect)
    fontsize = 8 if "\n" in label or len(label) > 3 else 10
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, zorder=3)


def _draw_cnot(ax: matplotlib.axes.Axes, x: float, qubits: tuple[int, ...], n_qubits: int) -> None:
    ctrl, tgt = qubits
    y_ctrl = n_qubits - 1 - ctrl
    y_tgt = n_qubits - 1 - tgt
    ax.plot([x, x], [y_ctrl, y_tgt], color="black", linewidth=1.0, zorder=1)
    ax.plot(x, y_ctrl, "o", color="black", markersize=6, zorder=3)
    circle = mpatches.Circle((x, y_tgt), 0.2, fill=False, color="black", linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.plot([x, x], [y_tgt - 0.2, y_tgt + 0.2], color="black", linewidth=1.0, zorder=3)
    ax.plot([x - 0.2, x + 0.2], [y_tgt, y_tgt], color="black", linewidth=1.0, zorder=3)


def _draw_cz(ax: matplotlib.axes.Axes, x: float, qubits: tuple[int, ...], n_qubits: int) -> None:
    q0, q1 = qubits
    y0 = n_qubits - 1 - q0
    y1 = n_qubits - 1 - q1
    ax.plot([x, x], [y0, y1], color="black", linewidth=1.0, zorder=1)
    ax.plot(x, y0, "o", color="black", markersize=6, zorder=3)
    ax.plot(x, y1, "o", color="black", markersize=6, zorder=3)


def _draw_swap(ax: matplotlib.axes.Axes, x: float, qubits: tuple[int, ...], n_qubits: int) -> None:
    q0, q1 = qubits
    y0 = n_qubits - 1 - q0
    y1 = n_qubits - 1 - q1
    ax.plot([x, x], [y0, y1], color="black", linewidth=1.0, zorder=1)
    cross_size = 0.15
    for y in (y0, y1):
        ax.plot(
            [x - cross_size, x + cross_size],
            [y - cross_size, y + cross_size],
            color="black",
            linewidth=1.5,
            zorder=3,
        )
        ax.plot(
            [x - cross_size, x + cross_size],
            [y + cross_size, y - cross_size],
            color="black",
            linewidth=1.5,
            zorder=3,
        )


def _draw_toffoli(
    ax: matplotlib.axes.Axes, x: float, qubits: tuple[int, ...], n_qubits: int
) -> None:
    c0, c1, tgt = qubits
    y_c0 = n_qubits - 1 - c0
    y_c1 = n_qubits - 1 - c1
    y_tgt = n_qubits - 1 - tgt
    y_min = min(y_c0, y_c1, y_tgt)
    y_max = max(y_c0, y_c1, y_tgt)
    ax.plot([x, x], [y_min, y_max], color="black", linewidth=1.0, zorder=1)
    ax.plot(x, y_c0, "o", color="black", markersize=6, zorder=3)
    ax.plot(x, y_c1, "o", color="black", markersize=6, zorder=3)
    circle = mpatches.Circle((x, y_tgt), 0.2, fill=False, color="black", linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.plot([x, x], [y_tgt - 0.2, y_tgt + 0.2], color="black", linewidth=1.0, zorder=3)
    ax.plot([x - 0.2, x + 0.2], [y_tgt, y_tgt], color="black", linewidth=1.0, zorder=3)


def _draw_two_qubit_box(
    ax: matplotlib.axes.Axes, x: float, qubits: tuple[int, ...], label: str, n_qubits: int
) -> None:
    q0, q1 = qubits
    y0 = n_qubits - 1 - q0
    y1 = n_qubits - 1 - q1
    y_min = min(y0, y1)
    y_max = max(y0, y1)
    box_size = 0.35
    rect = mpatches.FancyBboxPatch(
        (x - box_size, y_min - box_size),
        2 * box_size,
        (y_max - y_min) + 2 * box_size,
        boxstyle="round,pad=0.05",
        facecolor="#FADBD8",
        edgecolor="black",
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(rect)
    y_mid = (y0 + y1) / 2
    fontsize = 7 if "\n" in label or len(label) > 4 else 9
    ax.text(x, y_mid, label, ha="center", va="center", fontsize=fontsize, zorder=3)
