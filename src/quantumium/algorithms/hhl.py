"""HHL algorithm — quantum linear systems solver (educational).

The Harrow-Hassidim-Lloyd algorithm solves Ax = b for a Hermitian
matrix A, producing a quantum state |x> proportional to A^{-1}|b>.

Steps:
1. Encode |b> into the state register
2. QPE to extract eigenvalues of A into a clock register
3. Controlled rotation: ancilla gets amplitude C/lambda
4. Inverse QPE to uncompute the clock register
5. Postselect on ancilla |1> to get |x>

Classical: O(N polylog(1/eps)) via conjugate gradient
Quantum: O(polylog(N) / eps) — exponential speedup in N
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H, SWAP


@dataclass
class HHLResult:
    """Result container for the HHL algorithm.

    Attributes
    ----------
    solution_state : NDArray[np.complexfloating]
        The normalized quantum state proportional to A^{-1}|b>, extracted
        via post-selection on the ancilla qubit.
    success_probability : float
        The probability of successful post-selection (measuring the ancilla
        in state |1>). Low values indicate that the algorithm may need
        more clock qubits or parameter tuning.
    classical_solution : NDArray[np.complexfloating]
        The normalized classical solution x = A^{-1}b for comparison.
    fidelity : float
        The state fidelity |<x_quantum|x_classical>|^2 between the quantum
        and classical solutions. Values close to 1.0 indicate high accuracy.
    """

    solution_state: NDArray[np.complexfloating]
    success_probability: float
    classical_solution: NDArray[np.complexfloating]
    fidelity: float


def _hermitian_exp(A: NDArray, t: float) -> NDArray:
    """Compute the matrix exponential exp(i*A*t) via eigendecomposition.

    Parameters
    ----------
    A : NDArray
        A Hermitian matrix.
    t : float
        The time/scaling parameter.

    Returns
    -------
    NDArray
        The unitary matrix exp(i*A*t).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    return eigenvectors @ np.diag(np.exp(1j * eigenvalues * t)) @ eigenvectors.conj().T


def _apply_qft(qc: QuantumCircuit, qubits: list[int]) -> None:
    """Apply the Quantum Fourier Transform to specified qubits in a circuit.

    Appends QFT gates (Hadamard, controlled-phase, SWAP) directly to the
    given circuit, modifying it in place.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to modify.
    qubits : list[int]
        Indices of the qubits to transform.
    """
    from ..circuits.templates import _controlled_phase

    n = len(qubits)
    for i in range(n):
        qc.apply(H, qubits[i])
        for j in range(i + 1, n):
            k = j - i + 1
            theta = 2 * np.pi / (2**k)
            cp = _controlled_phase(theta)
            qc.apply(cp, qubits[j], qubits[i])
    for i in range(n // 2):
        qc.apply(SWAP, qubits[i], qubits[n - 1 - i])


def _apply_inverse_qft(qc: QuantumCircuit, qubits: list[int]) -> None:
    """Apply the inverse Quantum Fourier Transform to specified qubits.

    Appends inverse QFT gates (SWAP, inverse controlled-phase, Hadamard)
    directly to the given circuit, modifying it in place.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to modify.
    qubits : list[int]
        Indices of the qubits to transform.
    """
    from ..circuits.templates import _controlled_phase

    n = len(qubits)
    for i in range(n // 2):
        qc.apply(SWAP, qubits[i], qubits[n - 1 - i])
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            k = j - i + 1
            theta = -2 * np.pi / (2**k)
            cp = _controlled_phase(theta)
            qc.apply(cp, qubits[j], qubits[i])
        qc.apply(H, qubits[i])


def _build_hhl_circuit(
    A: NDArray[np.complexfloating],
    b: NDArray[np.complexfloating],
    num_clock: int,
    t: float,
    C: float,
) -> tuple[QuantumCircuit, int]:
    """Build the full HHL circuit for solving Ax = b.

    Constructs the complete HHL circuit consisting of:
    1. State preparation: encodes |b> into the state register.
    2. QPE: extracts eigenvalues of A into the clock register.
    3. Eigenvalue inversion: controlled rotation on the ancilla with
       amplitude C/lambda for each eigenvalue lambda.
    4. Inverse QPE: uncomputes the clock register.

    Layout:
      - Qubits 0 to num_clock-1: clock register (for QPE)
      - Qubit num_clock: ancilla (for eigenvalue inversion)
      - Qubits num_clock+1 to end: state register (encodes |b>)

    Parameters
    ----------
    A : NDArray[np.complexfloating]
        The Hermitian matrix of the linear system.
    b : NDArray[np.complexfloating]
        The right-hand side vector.
    num_clock : int
        Number of clock qubits for QPE precision.
    t : float
        Evolution time parameter for the Hamiltonian simulation in QPE.
    C : float
        Scaling constant for eigenvalue inversion. Should be less than or
        equal to the smallest eigenvalue of A.

    Returns
    -------
    tuple[QuantumCircuit, int]
        A pair ``(circuit, total_qubits)`` where ``circuit`` is the
        constructed HHL circuit and ``total_qubits`` is the total number
        of qubits used.
    """
    n_state = int(np.log2(len(b)))
    total = num_clock + 1 + n_state
    ancilla = num_clock

    qc = QuantumCircuit(total)

    # Step 1: Prepare |b> in state register via unitary mapping |0> -> |b>
    b_norm = b / np.linalg.norm(b)
    dim = 2**n_state
    M = np.eye(dim, dtype=complex)
    M[:, 0] = b_norm
    Q, _ = np.linalg.qr(M)
    if np.real(np.dot(Q[:, 0].conj(), b_norm)) < 0:
        Q *= -1
    prep_matrix = Q

    state_qubits = list(range(num_clock + 1, total))
    rev_state = state_qubits[::-1]
    qc.apply(Gate(prep_matrix, "Prep_b"), *rev_state)

    # Step 2: QPE
    clock_qubits = list(range(num_clock))

    for q in clock_qubits:
        qc.apply(H, q)

    for q in clock_qubits:
        power = 2 ** (num_clock - 1 - q)
        U_power = _hermitian_exp(A, t * power)
        dim = U_power.shape[0]
        cu_matrix = np.eye(2 * dim, dtype=complex)
        cu_matrix[dim:, dim:] = U_power
        qc.apply(Gate(cu_matrix, f"CU^{power}"), q, *rev_state)

    _apply_inverse_qft(qc, clock_qubits)

    # Step 3: Controlled rotation — eigenvalue inversion
    # Build a single unitary that applies Ry(theta_k) on ancilla
    # conditioned on clock register encoding k.
    num_clock_states = 2**num_clock
    ctrl_dim = 2 ** (num_clock + 1)
    ctrl_matrix = np.eye(ctrl_dim, dtype=complex)

    for k in range(1, num_clock_states):
        lambda_k = 2 * np.pi * k / (num_clock_states * t)
        if abs(lambda_k) < 1e-12:
            continue
        ratio = min(C / abs(lambda_k), 1.0)
        theta = 2 * np.arcsin(ratio)

        ry = np.array(
            [
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dtype=complex,
        )

        # Gate matrix basis: |clock_val, ancilla_bit>
        # Index = clock_val * 2 + ancilla_bit
        idx0 = k * 2
        idx1 = k * 2 + 1
        ctrl_matrix[idx0, idx0] = ry[0, 0]
        ctrl_matrix[idx0, idx1] = ry[0, 1]
        ctrl_matrix[idx1, idx0] = ry[1, 0]
        ctrl_matrix[idx1, idx1] = ry[1, 1]

    # _expand_gate maps first qubit arg to MSB of gate index.
    # Clock qubit 0 is MSB of the clock register in QPE output.
    qc.apply(Gate(ctrl_matrix, "EigInv"), *clock_qubits, ancilla)

    # Step 4: Inverse QPE (undo QPE)
    _apply_qft(qc, clock_qubits)

    for q in range(num_clock - 1, -1, -1):
        power = 2 ** (num_clock - 1 - q)
        U_inv = _hermitian_exp(A, -t * power)
        dim = U_inv.shape[0]
        cu_matrix = np.eye(2 * dim, dtype=complex)
        cu_matrix[dim:, dim:] = U_inv
        qc.apply(Gate(cu_matrix, f"CU^-{power}"), q, *rev_state)

    for q in clock_qubits:
        qc.apply(H, q)

    return qc, total


def hhl_solve(
    A: NDArray[np.complexfloating],
    b: NDArray[np.complexfloating],
    num_clock: int = 2,
    t: float | None = None,
) -> HHLResult:
    """Solve the linear system Ax = b using the HHL algorithm.

    Implements the Harrow-Hassidim-Lloyd algorithm, which produces a quantum
    state |x> proportional to A^{-1}|b>. The solution is extracted by
    post-selecting on the ancilla qubit in state |1>.

    Parameters
    ----------
    A : NDArray[np.complexfloating]
        A Hermitian matrix with positive eigenvalues. Must be square and
        its dimension must be a power of 2.
    b : NDArray[np.complexfloating]
        The right-hand side vector. Length must match the dimension of A.
    num_clock : int, optional
        Number of clock qubits for QPE precision (default 2). More clock
        qubits improve eigenvalue resolution but increase circuit size.
    t : float or None, optional
        Evolution time for Hamiltonian simulation within QPE. If None
        (default), auto-selected as ``2*pi / 2^num_clock`` so that integer
        eigenvalues map to distinct clock register states.

    Returns
    -------
    HHLResult
        Contains the normalized solution state, post-selection success
        probability, the classical reference solution, and the fidelity
        between quantum and classical solutions.

    Raises
    ------
    ValueError
        If A is not square, not Hermitian, if b has incompatible length,
        or if A has no positive eigenvalues.

    Example
    -------
    >>> import numpy as np
    >>> A = np.array([[2, 0], [0, 3]], dtype=complex)
    >>> b = np.array([1, 0], dtype=complex)
    >>> result = hhl_solve(A, b, num_clock=3)
    >>> result.fidelity > 0.9
    True
    """
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if not np.allclose(A, A.conj().T, atol=1e-10):
        raise ValueError("A must be Hermitian")
    if len(b) != n:
        raise ValueError(f"b length {len(b)} doesn't match A dimension {n}")

    eigenvalues = np.linalg.eigvalsh(A)
    pos_eigs = [e for e in eigenvalues if e > 1e-10]
    if not pos_eigs:
        raise ValueError("A must have positive eigenvalues")
    lambda_min = min(pos_eigs)

    if t is None:
        # Choose t so eigenvalues map to clock states in [1, 2^num_clock - 1].
        # t = 2*pi / 2^num_clock makes k = lambda for integer eigenvalues.
        t = 2 * np.pi / (2**num_clock)

    C = 0.5 * lambda_min

    qc, total = _build_hhl_circuit(A, b, num_clock, t, C)
    sv = qc.statevector()

    # Postselect on clock=0, ancilla=1
    # Qubit 0 is LSB of statevector index. Layout:
    #   qubits 0..nc-1 = clock, qubit nc = ancilla, qubits nc+1..end = state
    # sv_idx = sum_i q_i * 2^i
    # For clock=0, ancilla=1: sv_idx = 2^nc + state_idx * 2^(nc+1)
    solution = np.zeros(n, dtype=complex)
    for state_idx in range(n):
        idx = (state_idx << (num_clock + 1)) | (1 << num_clock)
        if idx < len(sv):
            solution[state_idx] = sv[idx]

    success_prob = np.linalg.norm(solution) ** 2

    if success_prob > 1e-15:
        solution_normalized = solution / np.linalg.norm(solution)
    else:
        solution_normalized = solution

    x_classical = np.linalg.solve(A, b)
    x_classical_norm = x_classical / np.linalg.norm(x_classical)

    fidelity = abs(np.dot(solution_normalized.conj(), x_classical_norm)) ** 2

    return HHLResult(
        solution_state=solution_normalized,
        success_probability=success_prob,
        classical_solution=x_classical_norm,
        fidelity=fidelity,
    )
