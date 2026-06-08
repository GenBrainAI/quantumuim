"""Hamiltonian simulation via Trotter-Suzuki decomposition.

Simulates quantum system evolution e^{-iHt} by decomposing H into
a sum of terms H = Σ_k H_k and approximating the exponential as
a product of simpler exponentials.
"""

from __future__ import annotations


import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import Gate


def exact_evolution(
    hamiltonian: NDArray[np.complexfloating], time: float
) -> NDArray[np.complexfloating]:
    """Compute the exact time evolution operator U = exp(-iHt) via eigendecomposition.

    Parameters
    ----------
    hamiltonian : NDArray[np.complexfloating]
        A Hermitian matrix representing the Hamiltonian.
    time : float
        Evolution time t.

    Returns
    -------
    NDArray[np.complexfloating]
        The unitary matrix U = exp(-i * H * t).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    exp_diag = np.diag(np.exp(-1j * eigenvalues * time))
    return eigenvectors @ exp_diag @ eigenvectors.conj().T  # type: ignore[no-any-return]


def first_order_trotter(
    terms: list[NDArray[np.complexfloating]],
    time: float,
    steps: int,
) -> NDArray[np.complexfloating]:
    """First-order Trotter (Lie-Trotter) decomposition.

    Approximates exp(-i*(H1+H2+...)*t) as the product
    (exp(-i*H1*dt) * exp(-i*H2*dt) * ...)^n, where dt = t/n.

    The approximation error scales as O(t^2 / n).

    Parameters
    ----------
    terms : list[NDArray[np.complexfloating]]
        List of Hamiltonian term matrices. All must have the same dimensions.
    time : float
        Total evolution time.
    steps : int
        Number of Trotter steps n. More steps yield higher accuracy.

    Returns
    -------
    NDArray[np.complexfloating]
        The approximate unitary evolution operator.
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
    """Second-order (symmetric) Trotter-Suzuki decomposition.

    Approximates exp(-i*H*t) using the symmetric (Strang) splitting:
    (exp(-i*H1*dt/2) * ... * exp(-i*H_{n-1}*dt/2) * exp(-i*H_n*dt) *
     exp(-i*H_{n-1}*dt/2) * ... * exp(-i*H1*dt/2))^steps

    The approximation error scales as O(t^3 / n^2), significantly better
    than the first-order decomposition.

    Parameters
    ----------
    terms : list[NDArray[np.complexfloating]]
        List of Hamiltonian term matrices. All must have the same dimensions.
    time : float
        Total evolution time.
    steps : int
        Number of Trotter steps. More steps yield higher accuracy.

    Returns
    -------
    NDArray[np.complexfloating]
        The approximate unitary evolution operator.
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
    """Compute the spectral norm error of a Trotter approximation.

    Calculates ||U_exact - U_trotter||_2 where U_exact = exp(-i*H*t) is the
    exact evolution and U_trotter is the Trotter approximation.

    Parameters
    ----------
    terms : list[NDArray[np.complexfloating]]
        List of Hamiltonian term matrices.
    time : float
        Total evolution time.
    steps : int
        Number of Trotter steps.
    order : int, optional
        Trotter order: 1 for first-order, 2 for second-order (default 1).

    Returns
    -------
    float
        The spectral (operator) norm of the difference between the exact
        and Trotter evolution operators.

    Raises
    ------
    ValueError
        If ``order`` is not 1 or 2.
    """
    H_full = sum(terms)
    U_exact = exact_evolution(H_full, time)  # type: ignore[arg-type]

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
    """Build a quantum circuit that simulates Hamiltonian time evolution.

    Creates a circuit implementing U = exp(-i*H*t) via Trotter decomposition,
    where H = sum_k hamiltonian_terms[k]. The full Trotter-approximated
    unitary is packaged as a single multi-qubit gate.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the system.
    hamiltonian_terms : list[NDArray[np.complexfloating]]
        List of Hamiltonian term matrices, each of size 2^num_qubits x 2^num_qubits.
    time : float
        Total evolution time.
    steps : int, optional
        Number of Trotter steps (default 10). More steps improve accuracy.
    order : int, optional
        Trotter decomposition order: 1 for first-order (Lie-Trotter),
        2 for second-order (Suzuki-Trotter). Default is 1.

    Returns
    -------
    QuantumCircuit
        A quantum circuit applying the Trotter-approximated evolution
        operator to all qubits.

    Raises
    ------
    ValueError
        If ``order`` is not 1 or 2.
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
    """Build the full-system Pauli-X operator acting on a single qubit.

    Constructs I (x) ... (x) X (x) ... (x) I where X is placed at the
    specified qubit position and I is the 2x2 identity on all other qubits.

    Parameters
    ----------
    qubit : int
        Index of the qubit on which X acts (0-indexed).
    num_qubits : int
        Total number of qubits in the system.

    Returns
    -------
    NDArray[np.complexfloating]
        A 2^num_qubits x 2^num_qubits matrix.
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    result = np.array([[1]], dtype=complex)
    for q in range(num_qubits):
        result = np.kron(result, X if q == qubit else I)
    return result


def pauli_z(qubit: int, num_qubits: int) -> NDArray[np.complexfloating]:
    """Build the full-system Pauli-Z operator acting on a single qubit.

    Constructs I (x) ... (x) Z (x) ... (x) I where Z is placed at the
    specified qubit position.

    Parameters
    ----------
    qubit : int
        Index of the qubit on which Z acts (0-indexed).
    num_qubits : int
        Total number of qubits in the system.

    Returns
    -------
    NDArray[np.complexfloating]
        A 2^num_qubits x 2^num_qubits diagonal matrix.
    """
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    result = np.array([[1]], dtype=complex)
    for q in range(num_qubits):
        result = np.kron(result, Z if q == qubit else I)
    return result


def zz_interaction(qubit_i: int, qubit_j: int, num_qubits: int) -> NDArray[np.complexfloating]:
    """Build the Z_i Z_j two-qubit interaction term.

    Constructs the tensor product of Pauli-Z on qubits i and j with
    identity on all other qubits.

    Parameters
    ----------
    qubit_i : int
        Index of the first qubit (0-indexed).
    qubit_j : int
        Index of the second qubit (0-indexed).
    num_qubits : int
        Total number of qubits in the system.

    Returns
    -------
    NDArray[np.complexfloating]
        A 2^num_qubits x 2^num_qubits diagonal matrix representing Z_i Z_j.
    """
    return pauli_z(qubit_i, num_qubits) @ pauli_z(qubit_j, num_qubits)


def transverse_field_ising(
    num_qubits: int, J: float = 1.0, h: float = 0.5
) -> list[NDArray[np.complexfloating]]:
    """Build the transverse-field Ising model Hamiltonian as separate terms.

    The Hamiltonian is H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i,
    representing nearest-neighbor Ising interactions with a transverse
    magnetic field. Open boundary conditions are used.

    The terms are returned separately (one matrix per Z_i Z_{i+1} term and
    one per X_i term) to enable Trotter decomposition.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (spin sites) in the chain.
    J : float, optional
        Ising coupling strength (default 1.0).
    h : float, optional
        Transverse field strength (default 0.5).

    Returns
    -------
    list[NDArray[np.complexfloating]]
        A list of Hamiltonian term matrices. The first ``num_qubits - 1``
        entries are the -J * Z_i Z_{i+1} terms, followed by ``num_qubits``
        entries for the -h * X_i terms.
    """
    terms = []
    for i in range(num_qubits - 1):
        terms.append(-J * zz_interaction(i, i + 1, num_qubits))
    for i in range(num_qubits):
        terms.append(-h * pauli_x(i, num_qubits))
    return terms
