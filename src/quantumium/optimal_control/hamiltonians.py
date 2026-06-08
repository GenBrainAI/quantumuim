"""Control Hamiltonians for quantum systems.

Defines drift and control Hamiltonians used in pulse-level optimization.
The total Hamiltonian is:

    H(t) = H_drift + Σ_k u_k(t) H_k

where H_drift is the always-on system Hamiltonian, H_k are the control
Hamiltonians, and u_k(t) are the time-dependent control amplitudes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ControlHamiltonian:
    """A quantum control system with drift and control terms.

    Attributes:
        drift: Drift (always-on) Hamiltonian H_0 (dim × dim).
        controls: List of control Hamiltonians [H_1, ..., H_m].
        dim: Hilbert space dimension.
        num_controls: Number of control channels.
    """

    drift: NDArray[np.complexfloating]
    controls: list[NDArray[np.complexfloating]]

    @property
    def dim(self) -> int:
        """Return the Hilbert space dimension of the control system.

        Returns:
            int: Dimension d of the drift Hamiltonian (d x d).
        """
        return self.drift.shape[0]  # type: ignore[no-any-return]

    @property
    def num_controls(self) -> int:
        """Return the number of control channels.

        Returns:
            int: Number of control Hamiltonians.
        """
        return len(self.controls)

    def total_hamiltonian(
        self,
        amplitudes: NDArray[np.floating],
    ) -> NDArray[np.complexfloating]:
        """Compute the total Hamiltonian for given control amplitudes.

        Evaluates H(t) = H_drift + sum_k u_k * H_k where u_k are
        the control amplitudes.

        Args:
            amplitudes (NDArray[np.floating]): Array of control amplitudes
                [u_0, u_1, ..., u_{m-1}], one per control channel.

        Returns:
            NDArray[np.complexfloating]: The total Hamiltonian matrix
                (dim x dim).
        """
        H = self.drift.copy()
        for k, u_k in enumerate(amplitudes):
            H = H + u_k * self.controls[k]
        return H  # type: ignore[no-any-return]


def drift_hamiltonian(
    frequency: float,
    num_qubits: int = 1,
) -> NDArray[np.complexfloating]:
    """Single-qubit drift Hamiltonian H_0 = ω/2 · Z.

    In the rotating frame at the qubit frequency, this becomes zero.
    Off-resonance driving gives a non-zero detuning term.

    Args:
        frequency: Qubit frequency ω (in angular units).
        num_qubits: Number of qubits.

    Returns:
        Drift Hamiltonian matrix.
    """
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    if num_qubits == 1:
        return frequency / 2 * Z  # type: ignore[return-value]

    H = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for q in range(num_qubits):
        term = np.array([[1]], dtype=complex)
        for k in range(num_qubits):
            term = np.kron(term, Z if k == q else I2)
        H += frequency / 2 * term

    return H


def control_hamiltonian_x(
    qubit: int = 0,
    num_qubits: int = 1,
) -> NDArray[np.complexfloating]:
    """Control Hamiltonian for X-drive: H_x = σ_x / 2.

    Args:
        qubit: Target qubit index.
        num_qubits: Total number of qubits.

    Returns:
        Control Hamiltonian matrix.
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    if num_qubits == 1:
        return X / 2  # type: ignore[return-value]

    term = np.array([[1]], dtype=complex)
    for k in range(num_qubits):
        term = np.kron(term, X / 2 if k == qubit else I2)
    return term


def control_hamiltonian_y(
    qubit: int = 0,
    num_qubits: int = 1,
) -> NDArray[np.complexfloating]:
    """Construct the Y-drive control Hamiltonian H_y = sigma_y / 2.

    For multi-qubit systems, the operator acts on the specified qubit
    with identity on all others.

    Args:
        qubit (int): Target qubit index (0-indexed). Defaults to 0.
        num_qubits (int): Total number of qubits. Defaults to 1.

    Returns:
        NDArray[np.complexfloating]: The Y-drive control Hamiltonian
            matrix of shape (2^num_qubits, 2^num_qubits).
    """
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    if num_qubits == 1:
        return Y / 2  # type: ignore[return-value]

    term = np.array([[1]], dtype=complex)
    for k in range(num_qubits):
        term = np.kron(term, Y / 2 if k == qubit else I2)
    return term


def transmon_hamiltonian(
    omega: float = 5.0,
    anharmonicity: float = -0.3,
    num_levels: int = 3,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Transmon qubit Hamiltonian with anharmonicity.

    H = Σ_n (n·ω + n(n-1)·α/2) |n⟩⟨n|

    Control via charge drive: H_ctrl = (a + a†) / 2

    Args:
        omega: Qubit frequency (GHz).
        anharmonicity: Anharmonicity α (GHz), typically negative.
        num_levels: Number of energy levels to include.

    Returns:
        (H_drift, H_x, H_y) — drift and X/Y control Hamiltonians.
    """
    dim = num_levels

    # Number operator diagonal
    H_drift = np.zeros((dim, dim), dtype=complex)
    for n in range(dim):
        H_drift[n, n] = n * omega + n * (n - 1) * anharmonicity / 2

    # Annihilation operator
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(n)

    a_dag = a.conj().T

    H_x = (a + a_dag) / 2
    H_y = 1j * (a_dag - a) / 2

    return H_drift, H_x, H_y
