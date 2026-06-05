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
    """Result container for a QAOA optimization run.

    Attributes
    ----------
    optimal_params : NDArray[np.floating]
        Concatenation of the optimal gamma and beta parameters
        (length 2*p, where p is the number of QAOA layers).
    optimal_value : float
        The best expected cost function value found.
    optimal_bitstring : str
        The bitstring sampled most frequently from the optimal circuit,
        representing the approximate solution to the optimization problem.
    history : list[float]
        Expected cost value at each iteration of the best optimization run.
    """
    optimal_params: NDArray[np.floating]
    optimal_value: float
    optimal_bitstring: str
    history: list[float]


def maxcut_cost_operator(num_qubits: int, edges: list[tuple[int, int]]) -> NDArray[np.complexfloating]:
    """Build the diagonal cost operator for the MaxCut problem.

    The cost function is C = sum_{(i,j) in E} (1 - Z_i Z_j) / 2,
    where each edge contributes 1 if qubits i and j are in different
    states (a "cut" edge) and 0 if they agree.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (nodes in the graph).
    edges : list[tuple[int, int]]
        List of edges as ``(i, j)`` pairs, where i and j are qubit indices.

    Returns
    -------
    NDArray[np.complexfloating]
        A 1-D array of length 2^num_qubits containing the diagonal of the
        cost Hamiltonian. Entry ``cost[k]`` is the number of cut edges for
        computational basis state ``k``.
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
    """Build the cost unitary exp(-i * gamma * C) where C is diagonal.

    Parameters
    ----------
    cost_diag : NDArray
        Diagonal of the cost Hamiltonian (1-D array of length 2^n).
    gamma : float
        The cost layer parameter.

    Returns
    -------
    Gate
        A diagonal unitary gate implementing the cost evolution.
    """
    phases = np.exp(-1j * gamma * cost_diag)
    matrix = np.diag(phases)
    return Gate(matrix, f"U_C(γ={gamma:.3f})")


def _mixer_unitary(num_qubits: int, beta: float) -> Gate:
    """Build the mixer unitary exp(-i * beta * B) where B = sum_i X_i.

    The mixer Hamiltonian is the transverse field B = sum_i X_i. The
    exponential is computed via eigendecomposition of the full mixer matrix.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    beta : float
        The mixer layer parameter.

    Returns
    -------
    Gate
        A unitary gate implementing the mixer evolution.
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

    Constructs the circuit: initial Hadamard layer to create a uniform
    superposition, then p alternating applications of the cost unitary
    exp(-i*gamma_l*C) and the mixer unitary exp(-i*beta_l*B).

    Parameters
    ----------
    num_qubits : int
        Number of qubits (must match the dimension of ``cost_diag``).
    cost_diag : NDArray
        Diagonal of the cost Hamiltonian as a 1-D array of length 2^num_qubits.
    gammas : NDArray[np.floating]
        Cost layer parameters, one per QAOA layer (length p).
    betas : NDArray[np.floating]
        Mixer layer parameters, one per QAOA layer (length p). Must have
        the same length as ``gammas``.

    Returns
    -------
    QuantumCircuit
        A quantum circuit implementing the p-layer QAOA ansatz.
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
    """Compute the expected cost value <gamma,beta|C|gamma,beta>.

    Builds the QAOA circuit, computes the statevector, and evaluates the
    expectation of the diagonal cost operator.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    cost_diag : NDArray
        Diagonal of the cost Hamiltonian (length 2^num_qubits).
    gammas : NDArray[np.floating]
        Cost layer parameters (length p).
    betas : NDArray[np.floating]
        Mixer layer parameters (length p).

    Returns
    -------
    float
        The expected cost value for the given parameters.
    """
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
    """Run QAOA for the MaxCut problem with gradient-ascent optimization.

    Uses finite-difference gradient estimation with multiple random restarts
    to find the QAOA parameters that maximize the expected number of cut edges.

    Parameters
    ----------
    num_qubits : int
        Number of nodes in the graph.
    edges : list[tuple[int, int]]
        List of edges as ``(i, j)`` pairs, where i and j are node indices.
    p : int, optional
        Number of QAOA layers / depth (default 1). Higher values allow
        better approximation ratios but increase parameter count.
    num_starts : int, optional
        Number of random restarts for the optimizer (default 10).
    max_iter : int, optional
        Maximum number of gradient-ascent iterations per restart (default 100).
    learning_rate : float, optional
        Step size for parameter updates (default 0.1).

    Returns
    -------
    QAOAResult
        Contains the optimal parameters, expected cost value, most likely
        bitstring solution, and optimization history.

    Note
    ----
    For production use, consider replacing the finite-difference optimizer
    with ``scipy.optimize.minimize(method='COBYLA')`` for better convergence.
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
