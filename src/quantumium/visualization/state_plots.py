"""State visualization: city plot and Hinton diagram."""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _to_density_matrix(state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Convert a statevector or density matrix to density matrix form."""
    if state.ndim == 1:
        return np.outer(state, np.conj(state))
    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        return state
    raise ValueError(f"Expected statevector (1D) or density matrix (2D square), got {state.shape}")


def plot_state_city(state: np.ndarray[Any, Any], title: str | None = None) -> Figure:
    """Plot density matrix as a 3D bar chart (city plot).

    Shows real and imaginary parts side by side.

    Args:
        state: Statevector (1D) or density matrix (2D).
        title: Optional plot title.

    Returns:
        A matplotlib Figure.
    """
    rho = _to_density_matrix(state)
    dim = rho.shape[0]
    num_qubits = int(np.log2(dim))
    labels = [format(i, f"0{num_qubits}b") for i in range(dim)]

    fig = plt.figure(figsize=(14, 5))

    for idx, (part, part_name) in enumerate([(np.real(rho), "Real"), (np.imag(rho), "Imaginary")]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")

        x_pos, y_pos = np.meshgrid(range(dim), range(dim), indexing="ij")
        x_flat = x_pos.flatten()
        y_flat = y_pos.flatten()
        z_pos = np.zeros_like(x_flat)
        values = part.flatten()

        colors = np.where(values >= 0, "#4C72B0", "#DD8452")

        ax.bar3d(
            x_flat - 0.4,
            y_flat - 0.4,
            z_pos,
            0.8,
            0.8,
            values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.3,
        )

        ax.set_xticks(range(dim))
        ax.set_yticks(range(dim))
        ax.set_xticklabels(labels, fontsize=7, rotation=45)
        ax.set_yticklabels(labels, fontsize=7, rotation=-45)
        ax.set_zlabel("Value")
        ax.set_title(part_name)

    fig.suptitle(title or "State City Plot", fontsize=14)
    fig.tight_layout()
    return fig


def plot_state_hinton(state: np.ndarray[Any, Any], title: str | None = None) -> Figure:
    """Plot density matrix as a Hinton diagram.

    Square sizes represent magnitude, colors represent sign of real part.

    Args:
        state: Statevector (1D) or density matrix (2D).
        title: Optional plot title.

    Returns:
        A matplotlib Figure.
    """
    rho = _to_density_matrix(state)
    dim = rho.shape[0]
    num_qubits = int(np.log2(dim))
    labels = [format(i, f"0{num_qubits}b") for i in range(dim)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, part, part_name in [
        (axes[0], np.real(rho), "Real"),
        (axes[1], np.imag(rho), "Imaginary"),
    ]:
        max_val = np.max(np.abs(part)) or 1.0

        ax.set_aspect("equal")
        ax.set_xlim(-0.5, dim - 0.5)
        ax.set_ylim(-0.5, dim - 0.5)
        ax.invert_yaxis()

        for i in range(dim):
            for j in range(dim):
                val = part[i, j]
                size = float(np.sqrt(np.abs(val) / max_val) * 0.9)
                color = "#4C72B0" if val >= 0 else "#DD8452"

                if size > 0.01:
                    rect = mpatches.Rectangle(
                        (j - size / 2, i - size / 2),
                        size,
                        size,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    ax.add_patch(rect)

        ax.set_xticks(range(dim))
        ax.set_yticks(range(dim))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(part_name)
        ax.grid(True, alpha=0.2)

    fig.suptitle(title or "Hinton Diagram", fontsize=14)
    fig.tight_layout()
    return fig
