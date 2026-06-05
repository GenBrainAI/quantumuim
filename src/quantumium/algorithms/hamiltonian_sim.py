"""Hamiltonian simulation via Trotter-Suzuki decomposition.

Simulates quantum system evolution e^{-iHt} by decomposing H into
a sum of terms H = Σ_k H_k and approximating the exponential as
a product of simpler exponentials.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate


def exact_evolution(hamiltonian: NDArray[np.complexfloating], time: float) -> NDArray[np.complexfloating]:
    """Compute exact time evolution operator U = e^{-iHt} via diagonalization."""
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    exp_diag = np.diag(np.exp(-1j * eigenvalues * time))
    return eigenvectors @ exp_diag @ eigenvectors.conj().T


def first_order_trotter(
    terms: list[NDArray[np.complexfloating]],
    time: float,
    steps: int,
) -> NDArray[np.complexfloating]:
    """First-order Trotter decomposition.

    e^{-i(H1+H2+...)t} ≈ (e^{-iH1 dt} e^{-iH2 dt} ...)^n

    Error: O(t²/n) for each step, O(t²/n) total.

    Args:
        terms: List of Hamiltonian term matrices (must all be same size).
        time: Total evolution time.
        steps: Number of Trotter steps (higher = more accurate).
    """
    dt = time / steps
    dim = terms[0].shape[0]
    result = np.eye(dim, dtype=complex)

    for _ in range(steps):
        for H_k in terms:
            step_unitary = exact_evolution(H_k, dt)
            result = step_unitary @ result

    return result


def second_order_trotter(
    terms: list[NDArray[np.complexfloating]],
    time: float,
    steps: int,
) -> NDArray[np.complexfloating]:
    """Second-order (symmetric) Trotter decomposition.

    e^{-iHt} ≈ (e^{-iH1 dt/2} e^{-iH2 dt/2} ... e^{-iH_n dt} ... e^{-iH2 dt/2} e^{-iH1 dt/2})^n

    Error: O(t³/n²) — significantly better than first-order.
    """
    dt = time / steps
    dim = terms[0].shape[0]
    result = np.eye(dim, dtype=complex)
    n_terms = len(terms)

    for _ in range(steps):
        # Forward half-steps
        for k in range(n_terms - 1):
            result = exact_evolution(terms[k], dt / 2) @ result
        # Full step for last term
        result = exact_evolution(terms[n_terms - 1], dt) @ result
        # Backward half-steps
        for k in range(n_terms - 2, -1, -1):
            result = exact_evolution(terms[k], dt / 2) @ result

    return result


def trotter_error(
    terms: list[NDArray[np.complexfloating]],
    time: float,
    steps: int,
    order: int = 1,
) -> float:
    """Compute the operator norm error of a Trotter approximation.

    Returns ‖U_exact - U_trotter‖ (spectral norm).
    """
    H_full = sum(terms)
    U_exact = exact_evolution(H_full, time)

    if order == 1:
        U_trotter = first_order_trotter(terms, time, steps)
    elif order == 2:
        U_trotter = second_order_trotter(terms, time, steps)
    else:
        raise ValueError(f"Unsupported Trotter order: {order}")

    return float(np.linalg.norm(U_exact - U_trotter, ord=2))


def simulate_evolution(
    num_qubits: int,
    hamiltonian_terms: list[NDArray[np.complexfloating]],
    time: float,
    steps: int = 10,
    order: int = 1,
) -> QuantumCircuit:
    """Build a quantum circuit that simulates Hamiltonian evolution.

    Creates a circuit implementing e^{-iHt} via Trotter decomposition,
    where H = Σ_k hamiltonian_terms[k].

    Args:
        num_qubits: Number of qubits.
        hamiltonian_terms: List of Hamiltonian term matrices.
        time: Evolution time.
        steps: Number of Trotter steps.
        order: Trotter order (1 or 2).
    """
    if order == 1:
        U = first_order_trotter(hamiltonian_terms, time, steps)
    elif order == 2:
        U = second_order_trotter(hamiltonian_terms, time, steps)
    else:
        raise ValueError(f"Unsupported order: {order}")

    gate = Gate(U, f"U_trotter(t={time:.2f},n={steps})")
    qc = QuantumCircuit(num_qubits)
    qc.apply(gate, *range(num_qubits))
    return qc


# Common Hamiltonian terms for spin systems

def pauli_x(qubit: int, num_qubits: int) -> NDArray[np.complexfloating]:
    """Single-qubit Pauli X operator on specified qubit."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    result = np.array([[1]], dtype=complex)
    for q in range(num_qubits):
        result = np.kron(result, X if q == qubit else I)
    return result


def pauli_z(qubit: int, num_qubits: int) -> NDArray[np.complexfloating]:
    """Single-qubit Pauli Z operator on specified qubit."""
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    result = np.array([[1]], dtype=complex)
    for q in range(num_qubits):
        result = np.kron(result, Z if q == qubit else I)
    return result


def zz_interaction(qubit_i: int, qubit_j: int, num_qubits: int) -> NDArray[np.complexfloating]:
    """Z_i Z_j interaction term."""
    return pauli_z(qubit_i, num_qubits) @ pauli_z(qubit_j, num_qubits)


def transverse_field_ising(num_qubits: int, J: float = 1.0, h: float = 0.5) -> list[NDArray[np.complexfloating]]:
    """Transverse-field Ising model Hamiltonian terms.

    H = -J Σ_{i} Z_i Z_{i+1} - h Σ_i X_i

    Returns separate terms suitable for Trotter decomposition.
    """
    terms = []
    for i in range(num_qubits - 1):
        terms.append(-J * zz_interaction(i, i + 1, num_qubits))
    for i in range(num_qubits):
        terms.append(-h * pauli_x(i, num_qubits))
    return terms
