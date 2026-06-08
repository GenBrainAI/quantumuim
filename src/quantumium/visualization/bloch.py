"""Bloch sphere visualization for single- and multi-qubit states."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]  # noqa: F401


def _statevector_to_bloch(rho_2x2: np.ndarray) -> tuple[float, float, float]:
    """Convert a 2x2 density matrix to Bloch vector (x, y, z)."""
    x = float(2 * np.real(rho_2x2[0, 1]))
    y = float(2 * np.imag(rho_2x2[1, 0]))
    z = float(np.real(rho_2x2[0, 0] - rho_2x2[1, 1]))
    return x, y, z


def _partial_trace_single(rho: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    """Trace out all qubits except the specified one."""
    rho_tensor = rho.reshape([2] * (2 * num_qubits))
    keep = qubit
    trace_out = [i for i in range(num_qubits) if i != keep]
    for i in sorted(trace_out, reverse=True):
        rho_tensor = np.trace(rho_tensor, axis1=i, axis2=i + num_qubits)
        num_qubits -= 1
        keep = sum(1 for t in trace_out if t < qubit) if keep > 0 else 0
        keep = qubit - sum(1 for t in trace_out if t < qubit)
        trace_out = [t for t in trace_out if t != i]
    return rho_tensor.reshape(2, 2)


def _reduced_density_matrix(statevector: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    """Get the reduced density matrix for a single qubit."""
    rho = np.outer(statevector, np.conj(statevector))
    rho_tensor = rho.reshape([2] * (2 * num_qubits))
    axes_to_trace = [i for i in range(num_qubits) if i != qubit]
    for offset, ax in enumerate(sorted(axes_to_trace)):
        shifted_ax = ax - offset
        rho_tensor = np.trace(
            rho_tensor, axis1=shifted_ax, axis2=shifted_ax + (num_qubits - offset)
        )
    return rho_tensor.reshape(2, 2)


def _draw_bloch_frame(ax: Axes3D) -> None:
    """Draw the Bloch sphere wireframe and axis labels."""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        x_sphere,
        y_sphere,
        z_sphere,
        alpha=0.05,
        color="gray",
        linewidth=0,
    )
    ax.plot(np.cos(u), np.sin(u), 0, color="gray", alpha=0.3, linewidth=0.5)
    ax.plot(np.zeros_like(u), np.sin(u), np.cos(u), color="gray", alpha=0.3, linewidth=0.5)
    ax.plot(np.sin(u), np.zeros_like(u), np.cos(u), color="gray", alpha=0.3, linewidth=0.5)

    axis_len = 1.3
    ax.plot([-axis_len, axis_len], [0, 0], [0, 0], color="gray", alpha=0.3, linewidth=0.5)
    ax.plot([0, 0], [-axis_len, axis_len], [0, 0], color="gray", alpha=0.3, linewidth=0.5)
    ax.plot([0, 0], [0, 0], [-axis_len, axis_len], color="gray", alpha=0.3, linewidth=0.5)

    label_offset = 1.4
    ax.text(0, 0, label_offset, r"|0$\rangle$", ha="center", fontsize=10)
    ax.text(0, 0, -label_offset, r"|1$\rangle$", ha="center", fontsize=10)
    ax.text(label_offset, 0, 0, r"|+$\rangle$", ha="center", fontsize=10)
    ax.text(-label_offset, 0, 0, r"|-$\rangle$", ha="center", fontsize=10)
    ax.text(0, label_offset, 0, r"|i$\rangle$", ha="center", fontsize=10)
    ax.text(0, -label_offset, 0, r"|-i$\rangle$", ha="center", fontsize=10)

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_axis_off()


def plot_bloch_sphere(statevector: np.ndarray) -> Figure:
    """Plot a single-qubit state on the Bloch sphere.

    Args:
        statevector: A 2-element complex statevector or 2x2 density matrix.

    Returns:
        A matplotlib Figure.
    """
    if statevector.shape == (2,):
        rho = np.outer(statevector, np.conj(statevector))
    elif statevector.shape == (2, 2):
        rho = statevector
    else:
        raise ValueError(
            f"Expected 1-qubit statevector (2,) or density matrix (2,2), "
            f"got shape {statevector.shape}"
        )

    bx, by, bz = _statevector_to_bloch(rho)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_bloch_frame(ax)

    ax.quiver(0, 0, 0, bx, by, bz, color="red", arrow_length_ratio=0.1, linewidth=2)
    ax.scatter([bx], [by], [bz], color="red", s=40, zorder=5)

    ax.set_title("Bloch Sphere")
    fig.tight_layout()
    return fig


def plot_bloch_multivector(statevector: np.ndarray) -> Figure:
    """Plot one Bloch sphere per qubit using reduced density matrices.

    Args:
        statevector: An n-qubit statevector (length 2^n).

    Returns:
        A matplotlib Figure with one subplot per qubit.
    """
    n_elements = len(statevector)
    num_qubits = int(np.log2(n_elements))
    if 2**num_qubits != n_elements:
        raise ValueError(f"Statevector length {n_elements} is not a power of 2")

    fig = plt.figure(figsize=(5 * num_qubits, 5))
    for qubit in range(num_qubits):
        rho_q = _reduced_density_matrix(statevector, qubit, num_qubits)
        bx, by, bz = _statevector_to_bloch(rho_q)

        ax = fig.add_subplot(1, num_qubits, qubit + 1, projection="3d")
        _draw_bloch_frame(ax)
        ax.quiver(0, 0, 0, bx, by, bz, color="red", arrow_length_ratio=0.1, linewidth=2)
        ax.scatter([bx], [by], [bz], color="red", s=40, zorder=5)
        ax.set_title(f"Qubit {qubit}")

    fig.suptitle("Bloch Multivector", fontsize=14)
    fig.tight_layout()
    return fig
