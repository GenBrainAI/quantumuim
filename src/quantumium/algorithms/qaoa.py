"""Quantum Approximate Optimization Algorithm (QAOA) implementation.

QAOA finds approximate solutions to combinatorial optimization problems
by alternating between a problem Hamiltonian and a mixer Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H


@dataclass
class QAOAResult:
    """Result from a QAOA optimization run."""
    optimal_params: NDArray[np.floating]
    optimal_value: float
    optimal_bitstring: str
    history: list[float]


def maxcut_cost_operator(num_qubits: int, edges: list[tuple[int, int]]) -> NDArray[np.complexfloating]:
    """Build the diagonal cost operator for MaxCut.

    C = Σ_{(i,j)∈E} (1 - Z_i Z_j) / 2

    Each edge contributes 1 if qubits i,j differ, 0 if they agree.
    """
    dim = 2**num_qubits
    cost = np.zeros(dim, dtype=float)
    for basis_state in range(dim):
        for i, j in edges:
            bit_i = (basis_state >> (num_qubits - 1 - i)) & 1
            bit_j = (basis_state >> (num_qubits - 1 - j)) & 1
            if bit_i != bit_j:
                cost[basis_state] += 1.0
    return cost


def _cost_unitary(cost_diag: NDArray, gamma: float) -> Gate:
    """e^{-i γ C} where C is diagonal."""
    phases = np.exp(-1j * gamma * cost_diag)
    matrix = np.diag(phases)
    return Gate(matrix, f"U_C(γ={gamma:.3f})")


def _mixer_unitary(num_qubits: int, beta: float) -> Gate:
    """e^{-i β B} where B = Σ X_i (transverse field mixer).

    Built by exponentiating the full mixer Hamiltonian.
    """
    dim = 2**num_qubits
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    mixer_h = np.zeros((dim, dim), dtype=complex)
    for q in range(num_qubits):
        term = np.array([[1]], dtype=complex)
        for k in range(num_qubits):
            term = np.kron(term, pauli_x if k == q else identity)
        mixer_h += term

    eigenvalues, eigenvectors = np.linalg.eigh(mixer_h)
    exp_diag = np.diag(np.exp(-1j * beta * eigenvalues))
    matrix = eigenvectors @ exp_diag @ eigenvectors.conj().T
    return Gate(matrix, f"U_B(β={beta:.3f})")


def qaoa_circuit(
    num_qubits: int,
    cost_diag: NDArray,
    gammas: NDArray[np.floating],
    betas: NDArray[np.floating],
) -> QuantumCircuit:
    """Build a QAOA circuit with p layers.

    Args:
        num_qubits: Number of qubits.
        cost_diag: Diagonal of the cost Hamiltonian (2^n entries).
        gammas: Cost layer parameters (length p).
        betas: Mixer layer parameters (length p).
    """
    p = len(gammas)
    assert len(betas) == p, "gammas and betas must have same length"

    qc = QuantumCircuit(num_qubits)
    qubits = tuple(range(num_qubits))

    for q in range(num_qubits):
        qc.apply(H, q)

    for layer in range(p):
        cost_gate = _cost_unitary(cost_diag, gammas[layer])
        mixer_gate = _mixer_unitary(num_qubits, betas[layer])
        qc.apply(cost_gate, *qubits)
        qc.apply(mixer_gate, *qubits)

    return qc


def qaoa_expectation(
    num_qubits: int,
    cost_diag: NDArray,
    gammas: NDArray[np.floating],
    betas: NDArray[np.floating],
) -> float:
    """Compute ⟨γ,β|C|γ,β⟩ — the expected cost value."""
    qc = qaoa_circuit(num_qubits, cost_diag, gammas, betas)
    sv = qc.statevector()
    probs = np.abs(sv) ** 2
    return float(np.dot(probs, cost_diag))


def qaoa_maxcut(
    num_qubits: int,
    edges: list[tuple[int, int]],
    p: int = 1,
    num_starts: int = 10,
    max_iter: int = 100,
    learning_rate: float = 0.1,
) -> QAOAResult:
    """Run QAOA for MaxCut with gradient-free optimization.

    Uses coordinate descent with finite differences for simplicity.
    For production, replace with scipy.optimize.minimize(method='COBYLA').

    Args:
        num_qubits: Number of nodes in the graph.
        edges: List of (i, j) edges.
        p: Number of QAOA layers (depth).
        num_starts: Number of random restarts.
        max_iter: Maximum iterations per restart.
        learning_rate: Step size for parameter updates.
    """
    cost_diag = maxcut_cost_operator(num_qubits, edges)

    best_value = -np.inf
    best_params = None
    best_history: list[float] = []

    for _ in range(num_starts):
        gammas = np.random.uniform(0, 2 * np.pi, size=p)
        betas = np.random.uniform(0, np.pi, size=p)
        params = np.concatenate([gammas, betas])
        history: list[float] = []

        for _ in range(max_iter):
            g = params[:p]
            b = params[p:]
            val = qaoa_expectation(num_qubits, cost_diag, g, b)
            history.append(val)

            grad = np.zeros_like(params)
            eps = 0.01
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += eps
                g_p, b_p = params_plus[:p], params_plus[p:]
                val_plus = qaoa_expectation(num_qubits, cost_diag, g_p, b_p)
                grad[i] = (val_plus - val) / eps

            params += learning_rate * grad

        g_final, b_final = params[:p], params[p:]
        final_val = qaoa_expectation(num_qubits, cost_diag, g_final, b_final)

        if final_val > best_value:
            best_value = final_val
            best_params = params.copy()
            best_history = history

    g_opt, b_opt = best_params[:p], best_params[p:]
    qc = qaoa_circuit(num_qubits, cost_diag, g_opt, b_opt)
    counts = qc.measure_all(shots=1024)
    optimal_bitstring = max(counts, key=lambda k: counts[k])

    return QAOAResult(
        optimal_params=best_params,
        optimal_value=best_value,
        optimal_bitstring=optimal_bitstring,
        history=best_history,
    )
