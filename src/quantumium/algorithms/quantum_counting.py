"""Quantum counting and amplitude estimation algorithms.

Quantum counting estimates the number of marked items in an unstructured
search space by applying QPE to the Grover operator. Amplitude estimation
generalizes this to estimate the probability amplitude of a "good" subspace.

Algorithms:
1. Quantum Counting: QPE on G = D·O to estimate theta, then M = N sin^2(theta)
2. Canonical Amplitude Estimation: QPE-based estimation of sin^2(theta)
3. Iterative Amplitude Estimation: Suzuki et al. (2020), no QPE needed

References:
    Brassard et al., "Quantum Amplitude Amplification and Estimation"
    (Contemporary Mathematics 305, 53-74, 2002)
    Suzuki et al., "Amplitude estimation without phase estimation"
    (Quantum Inf. Process. 19, 75, 2020)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate, H
from .grover import grover_oracle, grover_diffuser


def _bit_reverse(val: int, n: int) -> int:
    """Reverse the bit-order of an n-bit integer.

    Parameters
    ----------
    val : int
        The integer whose bits are to be reversed.
    n : int
        The number of bits to consider.

    Returns
    -------
    int
        The bit-reversed integer.
    """
    result = 0
    for i in range(n):
        if val & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _build_grover_operator(
    num_qubits: int,
    marked_states: list[int],
) -> NDArray[np.complexfloating]:
    """Build the Grover iterate G = D * O as a full matrix.

    The Grover operator is the product of the diffusion operator D and the
    oracle O. Its eigenvalues encode the fraction of marked states.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the search register.
    marked_states : list[int]
        Indices of the marked (target) states.

    Returns
    -------
    NDArray[np.complexfloating]
        The 2^num_qubits x 2^num_qubits Grover operator matrix.
    """
    oracle = grover_oracle(num_qubits, marked_states)
    diffuser = grover_diffuser(num_qubits)
    return diffuser.matrix @ oracle.matrix


def _controlled_gate(
    U: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Build a controlled-U gate matrix.

    Constructs the block-diagonal matrix [I, 0; 0, U] where the first
    qubit is the control.

    Parameters
    ----------
    U : NDArray[np.complexfloating]
        The unitary matrix to be controlled (dimension d x d).

    Returns
    -------
    NDArray[np.complexfloating]
        A 2d x 2d controlled-U matrix.
    """
    n = U.shape[0]
    dim = 2 * n
    CU = np.eye(dim, dtype=complex)
    CU[n:, n:] = U
    return CU


def _controlled_phase(theta: float) -> Gate:
    """Build a two-qubit controlled phase gate.

    Applies exp(i*theta) to the |11> state and acts as identity on all
    other basis states.

    Parameters
    ----------
    theta : float
        The phase angle in radians.

    Returns
    -------
    Gate
        A two-qubit controlled phase gate.
    """
    return Gate(np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * theta)],
    ], dtype=complex), f"CP({theta:.3f})")


def _apply_inverse_qft_inplace(
    qc: QuantumCircuit,
    qubits: list[int],
) -> None:
    """Apply the inverse QFT to specified qubits within an existing circuit.

    Modifies the circuit in place by appending SWAP gates, inverse
    controlled-phase gates, and Hadamard gates.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to modify.
    qubits : list[int]
        Indices of the qubits to which the inverse QFT is applied.
    """
    from ..core.gate import SWAP
    n = len(qubits)

    for i in range(n // 2):
        qc.apply(SWAP, qubits[i], qubits[n - 1 - i])

    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            k = j - i + 1
            theta = -2 * math.pi / (2**k)
            cp = _controlled_phase(theta)
            qc.apply(cp, qubits[j], qubits[i])
        qc.apply(H, qubits[i])


def _extract_phase_from_counting_register(
    sv: NDArray[np.complexfloating],
    num_counting_qubits: int,
    total_qubits: int,
) -> float:
    """Extract the estimated phase from a QPE statevector.

    Marginalizes over non-counting qubits to compute the probability
    distribution on the counting register, finds the most likely outcome,
    and converts it to a phase in [0, 1).

    Parameters
    ----------
    sv : NDArray[np.complexfloating]
        The full statevector of the QPE circuit (length 2^total_qubits).
    num_counting_qubits : int
        Number of counting (evaluation) qubits.
    total_qubits : int
        Total number of qubits in the circuit.

    Returns
    -------
    float
        The estimated phase phi in [0, 1), where the QPE eigenvalue is
        exp(2*pi*i*phi).
    """
    probs = np.abs(sv) ** 2
    num_counting_states = 2**num_counting_qubits

    counting_probs = np.zeros(num_counting_states)
    for idx in range(len(probs)):
        counting_idx = idx % num_counting_states
        counting_probs[counting_idx] += probs[idx]

    best_idx = int(np.argmax(counting_probs))
    best_reversed = _bit_reverse(best_idx, num_counting_qubits)

    # Phase φ = y / 2^n, where eigenvalue = exp(2πiφ)
    return best_reversed / (2**num_counting_qubits)


# ============================================================
# Quantum Counting
# ============================================================

@dataclass
class QuantumCountingResult:
    """Result container for quantum counting.

    Attributes
    ----------
    estimated_count : float
        The estimated number of marked items M = N * sin^2(theta).
    theta : float
        The estimated Grover angle theta in radians.
    total_states : int
        The total search space size N = 2^num_search_qubits.
    num_counting_qubits : int
        Number of counting qubits used for QPE precision.
    confidence_interval : tuple[float, float]
        A ``(lower, upper)`` confidence interval for the estimated count,
        based on the QPE resolution.
    """
    estimated_count: float
    theta: float
    total_states: int
    num_counting_qubits: int
    confidence_interval: tuple[float, float]


def quantum_counting(
    num_search_qubits: int,
    marked_states: list[int],
    num_counting_qubits: int = 4,
) -> QuantumCountingResult:
    """Estimate the number of marked items using quantum counting.

    Applies quantum phase estimation (QPE) to the Grover operator G = D*O
    to estimate the angle theta such that the eigenvalues of G are
    exp(+/-2i*theta). The number of marked items is then M = N * sin^2(theta).

    Parameters
    ----------
    num_search_qubits : int
        Number of qubits in the search register. The total search space
        has N = 2^num_search_qubits states.
    marked_states : list[int]
        List of integer indices identifying the marked (target) states.
    num_counting_qubits : int, optional
        Number of counting qubits for QPE precision (default 4). More
        counting qubits improve the accuracy of the count estimate.

    Returns
    -------
    QuantumCountingResult
        Contains the estimated count, Grover angle, and a confidence
        interval for the estimate.

    Raises
    ------
    ValueError
        If ``marked_states`` is empty or contains all states.

    Example
    -------
    >>> result = quantum_counting(3, [0, 1, 2, 3], num_counting_qubits=4)
    >>> abs(result.estimated_count - 4) < 1.0  # N=8, M=4
    True
    """
    N = 2**num_search_qubits
    M_true = len(marked_states)

    if M_true == 0:
        raise ValueError("Must mark at least one state")
    if M_true >= N:
        raise ValueError("Cannot mark all states")

    G = _build_grover_operator(num_search_qubits, marked_states)

    total_qubits = num_counting_qubits + num_search_qubits
    qc = QuantumCircuit(total_qubits)

    # Prepare uniform superposition on search register
    for q in range(num_counting_qubits, total_qubits):
        qc.apply(H, q)

    # Apply H to counting qubits
    for q in range(num_counting_qubits):
        qc.apply(H, q)

    # Controlled-G^(2^k) applications
    for k in range(num_counting_qubits):
        power = 2 ** (num_counting_qubits - 1 - k)
        G_pow = np.linalg.matrix_power(G, power)
        CG = _controlled_gate(G_pow)
        target_qubits = list(range(num_counting_qubits, total_qubits))
        qc.apply(Gate(CG, f"CG^{power}"), k, *target_qubits)

    # Inverse QFT on counting register
    _apply_inverse_qft_inplace(qc, list(range(num_counting_qubits)))

    # Extract phase: eigenvalue = exp(2πiφ), φ = y/2^n
    sv = qc.statevector()
    phi = _extract_phase_from_counting_register(
        sv, num_counting_qubits, total_qubits
    )

    # Grover eigenvalue exp(2πiφ) = exp(2iθ), so θ = πφ
    theta_est = math.pi * phi
    M_est = N * math.sin(theta_est) ** 2

    # Handle the complementary solution: QPE can measure either
    # φ or 1-φ (corresponding to exp(±2iθ))
    M_alt = N * math.sin(math.pi * (1 - phi)) ** 2
    # Both give the same M since sin(πφ) = sin(π(1-φ))
    # But pick the one closer to an integer
    if abs(M_alt - round(M_alt)) < abs(M_est - round(M_est)):
        M_est = M_alt
        theta_est = math.pi * (1 - phi)

    # Confidence interval
    delta_phi = 1.0 / (2**num_counting_qubits)
    delta_theta = math.pi * delta_phi
    theta_lo = max(0, theta_est - delta_theta)
    theta_hi = min(math.pi / 2, theta_est + delta_theta)
    M_lo = N * math.sin(theta_lo) ** 2
    M_hi = N * math.sin(theta_hi) ** 2
    if M_lo > M_hi:
        M_lo, M_hi = M_hi, M_lo

    return QuantumCountingResult(
        estimated_count=M_est,
        theta=theta_est,
        total_states=N,
        num_counting_qubits=num_counting_qubits,
        confidence_interval=(M_lo, M_hi),
    )


# ============================================================
# Canonical Amplitude Estimation
# ============================================================

@dataclass
class AmplitudeEstimationResult:
    """Result container for canonical amplitude estimation.

    Attributes
    ----------
    estimated_amplitude : float
        The estimated amplitude a = sin(theta) of the "good" subspace.
    estimated_probability : float
        The estimated probability a^2 = sin^2(theta).
    theta : float
        The estimated Grover angle theta in radians.
    num_evaluation_qubits : int
        Number of QPE evaluation qubits used.
    num_grover_iterations : int
        Total number of Grover operator applications (2^num_evaluation_qubits - 1).
    """
    estimated_amplitude: float
    estimated_probability: float
    theta: float
    num_evaluation_qubits: int
    num_grover_iterations: int


def amplitude_estimation(
    oracle_circuit: QuantumCircuit,
    state_prep_circuit: QuantumCircuit | None = None,
    good_states: list[int] | None = None,
    num_evaluation_qubits: int = 4,
) -> AmplitudeEstimationResult:
    """Canonical (QPE-based) amplitude estimation.

    Estimates the amplitude a = sin(theta) of the "good" subspace after
    applying state preparation A: |0> -> a|good> + b|bad>. Uses QPE on
    the Grover operator Q = A * S_0 * A^dagger * S_good, where S_good
    is the oracle marking good states.

    Parameters
    ----------
    oracle_circuit : QuantumCircuit
        A circuit implementing the oracle (marks good states by flipping
        their phase). Must operate on the same number of qubits as the
        state preparation circuit.
    state_prep_circuit : QuantumCircuit or None, optional
        A circuit implementing the state preparation operator A. If None,
        a uniform superposition (Hadamard on all qubits) is used.
    good_states : list[int] or None, optional
        Indices of good states. Currently unused (the oracle handles
        marking), but included for API consistency.
    num_evaluation_qubits : int, optional
        Number of QPE evaluation qubits (default 4). More qubits improve
        precision but increase circuit size exponentially.

    Returns
    -------
    AmplitudeEstimationResult
        Contains the estimated amplitude, probability, Grover angle, and
        resource counts.
    """
    n = oracle_circuit.num_qubits

    # Build oracle matrix
    O_matrix = _circuit_to_unitary(oracle_circuit)

    # Build state prep: H^⊗n
    dim = 2**n
    if state_prep_circuit is not None:
        A_matrix = _circuit_to_unitary(state_prep_circuit)
    else:
        A_matrix = np.eye(dim, dtype=complex)
        for q in range(n):
            H_full = np.eye(1, dtype=complex)
            for i in range(n):
                if i == n - 1 - q:
                    H_full = np.kron(H_full, H.matrix)
                else:
                    H_full = np.kron(H_full, np.eye(2, dtype=complex))
            A_matrix = H_full @ A_matrix

    # Grover operator: Q = A · (2|0><0| - I) · A^dag · Oracle
    S_0 = -np.eye(dim, dtype=complex)
    S_0[0, 0] = 1
    Q = A_matrix @ S_0 @ A_matrix.conj().T @ O_matrix

    # QPE on Q
    total_qubits = num_evaluation_qubits + n
    qc = QuantumCircuit(total_qubits)

    # Prepare state A|0> on target register
    target_qubits = list(range(num_evaluation_qubits, total_qubits))
    if state_prep_circuit is not None:
        for inst in state_prep_circuit.instructions:
            shifted = [q + num_evaluation_qubits for q in inst.target_qubits]
            qc.apply(inst.gate, *shifted)
    else:
        for q in target_qubits:
            qc.apply(H, q)

    # H on evaluation qubits
    for q in range(num_evaluation_qubits):
        qc.apply(H, q)

    # Controlled-Q^(2^k)
    for k in range(num_evaluation_qubits):
        power = 2 ** (num_evaluation_qubits - 1 - k)
        Q_pow = np.linalg.matrix_power(Q, power)
        CQ = _controlled_gate(Q_pow)
        qc.apply(Gate(CQ, f"CQ^{power}"), k, *target_qubits)

    # Inverse QFT
    _apply_inverse_qft_inplace(qc, list(range(num_evaluation_qubits)))

    # Extract phase
    sv = qc.statevector()
    phi = _extract_phase_from_counting_register(
        sv, num_evaluation_qubits, total_qubits
    )

    # θ = πφ, amplitude = sin(θ)
    theta = math.pi * phi
    amplitude = abs(math.sin(theta))
    probability = amplitude**2

    return AmplitudeEstimationResult(
        estimated_amplitude=amplitude,
        estimated_probability=probability,
        theta=theta,
        num_evaluation_qubits=num_evaluation_qubits,
        num_grover_iterations=2**num_evaluation_qubits - 1,
    )


# ============================================================
# Iterative Amplitude Estimation
# ============================================================

@dataclass
class IterativeAEResult:
    """Result container for iterative amplitude estimation.

    Attributes
    ----------
    estimated_amplitude : float
        The estimated amplitude a = sin(theta) of the marked subspace.
    estimated_probability : float
        The estimated probability a^2 = sin^2(theta).
    theta : float
        The estimated Grover angle theta in radians.
    num_rounds : int
        Number of refinement rounds performed.
    total_oracle_calls : int
        Total number of Grover operator applications across all rounds.
    """
    estimated_amplitude: float
    estimated_probability: float
    theta: float
    num_rounds: int
    total_oracle_calls: int


def iterative_amplitude_estimation(
    num_search_qubits: int,
    marked_states: list[int],
    num_rounds: int = 5,
    confidence: float = 0.05,
) -> IterativeAEResult:
    """Iterative amplitude estimation without QPE.

    Implements the method of Suzuki et al. (2020), which iteratively narrows
    the estimate of the Grover angle theta by measuring the success probability
    of Grover iterations with exponentially increasing depth (2^r iterations
    in round r). No phase estimation circuit is needed.

    Parameters
    ----------
    num_search_qubits : int
        Number of qubits in the search register (N = 2^n states).
    marked_states : list[int]
        Indices of the marked (target) states.
    num_rounds : int, optional
        Number of refinement rounds (default 5). Each round doubles the
        Grover depth, so the total oracle calls scale as O(2^num_rounds).
    confidence : float, optional
        Confidence parameter (default 0.05). Reserved for API consistency;
        not directly used in the current implementation.

    Returns
    -------
    IterativeAEResult
        Contains the estimated amplitude, probability, Grover angle,
        number of rounds, and total oracle calls.

    Raises
    ------
    ValueError
        If ``marked_states`` is empty or contains all states.
    """
    N = 2**num_search_qubits
    M_true = len(marked_states)

    if M_true == 0:
        raise ValueError("Must have at least one marked state")
    if M_true >= N:
        raise ValueError("Cannot mark all states")

    G = _build_grover_operator(num_search_qubits, marked_states)

    theta_lo = 0.0
    theta_hi = math.pi / 2
    total_calls = 0

    for r in range(num_rounds):
        k = 2**r

        qc = QuantumCircuit(num_search_qubits)
        for q in range(num_search_qubits):
            qc.apply(H, q)

        G_pow = np.linalg.matrix_power(G, k)
        g = Gate(G_pow, f"G^{k}")
        qc.apply(g, *range(num_search_qubits))

        sv = qc.statevector()

        p_k = sum(
            abs(sv[_bit_reverse(s, num_search_qubits)]) ** 2
            for s in marked_states
        )

        # p_k = sin^2((2k+1) * theta)
        angle = math.asin(min(1.0, max(0.0, math.sqrt(p_k))))

        possible_thetas = []
        for m_val in range(2 * k + 2):
            t1 = (angle + m_val * math.pi) / (2 * k + 1)
            t2 = ((math.pi - angle) + m_val * math.pi) / (2 * k + 1)
            if 0 <= t1 <= math.pi / 2:
                possible_thetas.append(t1)
            if 0 <= t2 <= math.pi / 2:
                possible_thetas.append(t2)

        theta_mid = (theta_lo + theta_hi) / 2
        if possible_thetas:
            best_theta = min(possible_thetas, key=lambda t: abs(t - theta_mid))
            delta = math.pi / (4 * (2 * k + 1))
            theta_lo = max(0, best_theta - delta)
            theta_hi = min(math.pi / 2, best_theta + delta)

        total_calls += k

    theta_est = (theta_lo + theta_hi) / 2
    amplitude = math.sin(theta_est)
    probability = amplitude**2

    return IterativeAEResult(
        estimated_amplitude=amplitude,
        estimated_probability=probability,
        theta=theta_est,
        num_rounds=num_rounds,
        total_oracle_calls=total_calls,
    )


def _circuit_to_unitary(qc: QuantumCircuit) -> NDArray[np.complexfloating]:
    """Extract the full unitary matrix representation of a quantum circuit.

    Multiplies out all gate matrices (expanded to the full Hilbert space)
    in instruction order to produce the overall circuit unitary.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit to convert.

    Returns
    -------
    NDArray[np.complexfloating]
        A 2^n x 2^n unitary matrix, where n = qc.num_qubits.
    """
    n = qc.num_qubits
    dim = 2**n
    U = np.eye(dim, dtype=complex)

    for inst in qc.instructions:
        gate_matrix = inst.gate.matrix
        target = inst.target_qubits

        if gate_matrix.shape[0] == dim:
            full = gate_matrix
        else:
            full = qc._expand_gate(gate_matrix, target)
        U = full @ U

    return U
