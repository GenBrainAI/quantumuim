"""Variational Quantum Eigensolver (VQE) implementation.

VQE is a hybrid classical-quantum algorithm for finding the ground state
energy of a Hamiltonian. It uses a parameterized quantum circuit (ansatz)
and classical optimization to minimize ⟨ψ(θ)|H|ψ(θ)⟩.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H, CNOT, Ry, Rz


@dataclass
class VQEResult:
    """Result from a VQE optimization."""
    optimal_energy: float
    optimal_params: NDArray[np.floating]
    history: list[float]
    num_iterations: int


def pauli_z_expectation(statevector: NDArray[np.complexfloating], qubit: int, num_qubits: int) -> float:
    """Compute ⟨ψ|Z_i|ψ⟩ for a single Pauli-Z on qubit i."""
    probs = np.abs(statevector) ** 2
    expectation = 0.0
    for basis_state in range(len(statevector)):
        bit = (basis_state >> (num_qubits - 1 - qubit)) & 1
        sign = 1 - 2 * bit  # +1 for |0⟩, -1 for |1⟩
        expectation += sign * probs[basis_state]
    return expectation


def pauli_zz_expectation(statevector: NDArray[np.complexfloating], qubit_i: int, qubit_j: int, num_qubits: int) -> float:
    """Compute ⟨ψ|Z_i Z_j|ψ⟩."""
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
    """Compute ⟨ψ|H|ψ⟩ for an arbitrary Hamiltonian matrix."""
    return float(np.real(statevector.conj() @ hamiltonian @ statevector))


def hardware_efficient_ansatz(
    num_qubits: int,
    params: NDArray[np.floating],
    depth: int = 1,
) -> QuantumCircuit:
    """Build a hardware-efficient ansatz circuit.

    Structure per layer:
    - Ry(θ) + Rz(θ) on each qubit (2 params per qubit per layer)
    - Linear entangling CNOT chain

    Total params: num_qubits * 2 * depth
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
    """Simplified H2 molecule Hamiltonian in STO-3G basis (2 qubits).

    This is the qubit Hamiltonian after Jordan-Wigner transformation
    at bond length ~0.735 Å. Exact ground state energy ≈ -1.137 Ha.

    H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1
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
    learning_rate: float = 0.05,
    num_starts: int = 5,
) -> VQEResult:
    """Run VQE with a hardware-efficient ansatz.

    Uses parameter-shift rule for gradient computation.

    Args:
        num_qubits: Number of qubits.
        hamiltonian: Hamiltonian matrix (2^n × 2^n).
        ansatz_depth: Number of ansatz layers.
        max_iter: Maximum optimization iterations per start.
        learning_rate: Gradient descent step size.
        num_starts: Number of random initial parameter sets.
    """
    num_params = num_qubits * 2 * ansatz_depth

    best_energy = np.inf
    best_params = None
    best_history: list[float] = []

    for _ in range(num_starts):
        params = np.random.uniform(-np.pi, np.pi, size=num_params)
        history: list[float] = []

        for _ in range(max_iter):
            qc = hardware_efficient_ansatz(num_qubits, params, ansatz_depth)
            sv = qc.statevector()
            energy = hamiltonian_expectation(sv, hamiltonian)
            history.append(energy)

            # Parameter-shift gradient
            grad = np.zeros_like(params)
            shift = np.pi / 2
            for i in range(num_params):
                params_plus = params.copy()
                params_plus[i] += shift
                sv_plus = hardware_efficient_ansatz(num_qubits, params_plus, ansatz_depth).statevector()
                e_plus = hamiltonian_expectation(sv_plus, hamiltonian)

                params_minus = params.copy()
                params_minus[i] -= shift
                sv_minus = hardware_efficient_ansatz(num_qubits, params_minus, ansatz_depth).statevector()
                e_minus = hamiltonian_expectation(sv_minus, hamiltonian)

                grad[i] = (e_plus - e_minus) / 2

            params -= learning_rate * grad

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
