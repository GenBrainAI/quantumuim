"""Variational Quantum Eigensolver (VQE) implementation.

VQE is a hybrid classical-quantum algorithm for finding the ground state
energy of a Hamiltonian. It uses a parameterized quantum circuit (ansatz)
and classical optimization to minimize ⟨ψ(θ)|H|ψ(θ)⟩.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import CNOT, Ry, Rz


@dataclass
class VQEResult:
    """Result container for a VQE optimization run.

    Attributes
    ----------
    optimal_energy : float
        The lowest energy found across all random starts.
    optimal_params : NDArray[np.floating]
        The parameter vector that achieved the optimal energy.
    history : list[float]
        Energy values at each iteration of the best optimization run.
    num_iterations : int
        Number of iterations completed in the best run.
    """

    optimal_energy: float
    optimal_params: NDArray[np.floating]
    history: list[float]
    num_iterations: int


def pauli_z_expectation(
    statevector: NDArray[np.complexfloating], qubit: int, num_qubits: int
) -> float:
    """Compute the expectation value <psi|Z_i|psi> for a single Pauli-Z operator.

    Parameters
    ----------
    statevector : NDArray[np.complexfloating]
        The quantum state vector of length 2^num_qubits.
    qubit : int
        Index of the qubit on which the Pauli-Z acts (0-indexed).
    num_qubits : int
        Total number of qubits in the system.

    Returns
    -------
    float
        The expectation value, in the range [-1, 1].
    """
    probs = np.abs(statevector) ** 2
    expectation = 0.0
    for basis_state in range(len(statevector)):
        bit = (basis_state >> (num_qubits - 1 - qubit)) & 1
        sign = 1 - 2 * bit  # +1 for |0⟩, -1 for |1⟩
        expectation += sign * probs[basis_state]
    return expectation


def pauli_zz_expectation(
    statevector: NDArray[np.complexfloating], qubit_i: int, qubit_j: int, num_qubits: int
) -> float:
    """Compute the expectation value <psi|Z_i Z_j|psi> for a two-qubit ZZ operator.

    Parameters
    ----------
    statevector : NDArray[np.complexfloating]
        The quantum state vector of length 2^num_qubits.
    qubit_i : int
        Index of the first qubit (0-indexed).
    qubit_j : int
        Index of the second qubit (0-indexed).
    num_qubits : int
        Total number of qubits in the system.

    Returns
    -------
    float
        The expectation value, in the range [-1, 1]. Positive when qubits
        tend to agree (both |0> or both |1>), negative otherwise.
    """
    probs = np.abs(statevector) ** 2
    expectation = 0.0
    for basis_state in range(len(statevector)):
        bit_i = (basis_state >> (num_qubits - 1 - qubit_i)) & 1
        bit_j = (basis_state >> (num_qubits - 1 - qubit_j)) & 1
        sign = 1 - 2 * (bit_i ^ bit_j)  # +1 if same, -1 if different
        expectation += sign * probs[basis_state]
    return expectation


def hamiltonian_expectation(
    statevector: NDArray[np.complexfloating],
    hamiltonian: NDArray[np.complexfloating],
) -> float:
    """Compute <psi|H|psi> for an arbitrary Hamiltonian matrix.

    Parameters
    ----------
    statevector : NDArray[np.complexfloating]
        The quantum state vector of length 2^n.
    hamiltonian : NDArray[np.complexfloating]
        The Hamiltonian as a 2^n x 2^n Hermitian matrix.

    Returns
    -------
    float
        The real part of the expectation value <psi|H|psi>.
    """
    return float(np.real(statevector.conj() @ hamiltonian @ statevector))


def hardware_efficient_ansatz(
    num_qubits: int,
    params: NDArray[np.floating],
    depth: int = 1,
) -> QuantumCircuit:
    """Build a hardware-efficient variational ansatz circuit.

    Each layer consists of single-qubit Ry and Rz rotations on every qubit
    followed by a linear chain of CNOT gates for entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    params : NDArray[np.floating]
        Parameter vector of length ``num_qubits * 2 * depth``. Parameters
        are consumed in order: for each layer, Ry(theta) and Rz(theta) for
        each qubit (2 parameters per qubit per layer).
    depth : int, optional
        Number of ansatz layers (default 1). More layers increase
        expressibility but also the parameter count.

    Returns
    -------
    QuantumCircuit
        A parameterized quantum circuit implementing the ansatz.

    Raises
    ------
    ValueError
        If the length of ``params`` does not equal ``num_qubits * 2 * depth``.
    """
    expected_params = num_qubits * 2 * depth
    if len(params) != expected_params:
        raise ValueError(f"Expected {expected_params} params, got {len(params)}")

    qc = QuantumCircuit(num_qubits)
    idx = 0

    for layer in range(depth):
        for q in range(num_qubits):
            qc.apply(Ry(params[idx]), q)
            qc.apply(Rz(params[idx + 1]), q)
            idx += 2
        for q in range(num_qubits - 1):
            qc.apply(CNOT, q, q + 1)

    return qc


def h2_hamiltonian() -> NDArray[np.complexfloating]:
    """Build a simplified H2 molecule Hamiltonian in the STO-3G basis (2 qubits).

    Constructs the qubit Hamiltonian after Jordan-Wigner transformation
    at an equilibrium bond length of approximately 0.735 Angstroms. The
    exact ground state energy is approximately -1.137 Hartrees.

    The Hamiltonian is:
    H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1

    Returns
    -------
    NDArray[np.complexfloating]
        A 4x4 Hermitian matrix representing the H2 Hamiltonian in the
        computational basis of 2 qubits.

    Example
    -------
    >>> H = h2_hamiltonian()
    >>> eigenvalues = np.linalg.eigvalsh(H)
    >>> ground_energy = eigenvalues[0]  # approximately -1.137
    """
    g0 = -0.8105
    g1 = +0.1721
    g2 = -0.2257
    g3 = +0.1709
    g4 = +0.0453
    g5 = +0.0453

    I = np.eye(4, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    Z0 = np.kron(Z, np.eye(2, dtype=complex))
    Z1 = np.kron(np.eye(2, dtype=complex), Z)
    Z0Z1 = np.kron(Z, Z)
    X0X1 = np.kron(X, X)
    Y0Y1 = np.kron(Y, Y)

    H = g0 * I + g1 * Z0 + g2 * Z1 + g3 * Z0Z1 + g4 * X0X1 + g5 * Y0Y1
    return H


def vqe(
    num_qubits: int,
    hamiltonian: NDArray[np.complexfloating],
    ansatz_depth: int = 2,
    max_iter: int = 200,
    learning_rate: float = 0.01,
    num_starts: int = 5,
) -> VQEResult:
    """Run VQE with a hardware-efficient ansatz to find the ground state energy.

    Uses the parameter-shift rule for analytic gradient computation and the
    Adam optimizer for parameter updates. Multiple random starting points
    are tried to mitigate local minima.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz and Hamiltonian.
    hamiltonian : NDArray[np.complexfloating]
        The Hamiltonian matrix of size 2^n x 2^n whose ground state
        energy is to be found.
    ansatz_depth : int, optional
        Number of ansatz layers (default 2). Each layer has
        ``num_qubits * 2`` parameters.
    max_iter : int, optional
        Maximum number of Adam optimization iterations per random start
        (default 200).
    learning_rate : float, optional
        Adam optimizer step size (default 0.01).
    num_starts : int, optional
        Number of independent random initial parameter sets to try
        (default 5). The best result across all starts is returned.

    Returns
    -------
    VQEResult
        Contains the optimal energy, optimal parameters, energy history
        of the best run, and the number of iterations.
    """
    num_params = num_qubits * 2 * ansatz_depth

    best_energy = np.inf
    best_params = None
    best_history: list[float] = []

    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(num_starts):
        params = np.random.uniform(-np.pi, np.pi, size=num_params)
        m = np.zeros(num_params)
        v = np.zeros(num_params)
        history: list[float] = []

        for t in range(1, max_iter + 1):
            qc = hardware_efficient_ansatz(num_qubits, params, ansatz_depth)
            sv = qc.statevector()
            energy = hamiltonian_expectation(sv, hamiltonian)
            history.append(energy)

            grad = np.zeros_like(params)
            shift = np.pi / 2
            for i in range(num_params):
                params_plus = params.copy()
                params_plus[i] += shift
                sv_plus = hardware_efficient_ansatz(
                    num_qubits, params_plus, ansatz_depth
                ).statevector()
                e_plus = hamiltonian_expectation(sv_plus, hamiltonian)

                params_minus = params.copy()
                params_minus[i] -= shift
                sv_minus = hardware_efficient_ansatz(
                    num_qubits, params_minus, ansatz_depth
                ).statevector()
                e_minus = hamiltonian_expectation(sv_minus, hamiltonian)

                grad[i] = (e_plus - e_minus) / 2

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            params -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        final_qc = hardware_efficient_ansatz(num_qubits, params, ansatz_depth)
        final_energy = hamiltonian_expectation(final_qc.statevector(), hamiltonian)

        if final_energy < best_energy:
            best_energy = final_energy
            best_params = params.copy()
            best_history = history

    return VQEResult(
        optimal_energy=best_energy,
        optimal_params=best_params,
        history=best_history,
        num_iterations=len(best_history),
    )
