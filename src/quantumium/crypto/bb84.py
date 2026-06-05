"""BB84 quantum key distribution protocol.

Simulates the Bennett-Brassard 1984 protocol for quantum key distribution:

1. Alice prepares qubits in random bases (Z or X) with random bit values
2. Alice sends qubits to Bob through a quantum channel
3. Bob measures each qubit in a randomly chosen basis (Z or X)
4. Alice and Bob publicly compare bases (not bit values)
5. They keep only bits where bases matched -> sifted key
6. They sacrifice some sifted bits for eavesdropper detection (QBER check)
7. Remaining bits form the shared secret key

With an eavesdropper (Eve) performing intercept-resend:
- Eve measures in a random basis, re-prepares, and forwards
- This introduces ~25% QBER on the sifted key
- Alice and Bob detect this via the error check
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.circuit import QuantumCircuit
from ..core.gate import H, X


@dataclass
class BB84Result:
    """Result from a BB84 quantum key distribution simulation.

    Attributes:
        alice_key: Alice's portion of the final shared key (list of 0/1).
        bob_key: Bob's portion of the final shared key (list of 0/1).
            In the absence of eavesdropping, ``alice_key == bob_key``.
        sifted_length: Number of bits remaining after basis sifting
            (before the error-check sacrifice).
        qber: Quantum Bit Error Rate estimated from the sacrificed check
            bits. Without eavesdropping this should be ~0; with
            intercept-resend it is ~0.25.
        final_key_length: Length of the usable key after removing the
            check bits.
        eavesdropper_detected: ``True`` if the estimated QBER exceeds
            :attr:`detection_threshold`.
        num_qubits_sent: Total number of qubits Alice transmitted.
        detection_threshold: QBER threshold used for eavesdropper
            detection (default 0.11).
    """
    alice_key: list[int]
    bob_key: list[int]
    sifted_length: int
    qber: float
    final_key_length: int
    eavesdropper_detected: bool
    num_qubits_sent: int
    detection_threshold: float


def _prepare_qubit(bit: int, basis: int) -> NDArray[np.complexfloating]:
    """Prepare a single qubit in the given basis and bit value.

    Args:
        bit: Bit value to encode (0 or 1).
        basis: Measurement basis selection.
            ``0`` (Z-basis): prepares |0> or |1>.
            ``1`` (X-basis): prepares |+> or |->.

    Returns:
        NDArray[np.complexfloating]: The 2-element statevector of the
            prepared qubit.
    """
    qc = QuantumCircuit(1)
    if bit:
        qc.apply(X, 0)
    if basis:
        qc.apply(H, 0)
    return qc.statevector()


def _measure_qubit(
    state: NDArray[np.complexfloating],
    basis: int,
    rng: np.random.Generator,
) -> int:
    """Measure a qubit in the given basis.

    For the X-basis, a Hadamard rotation is applied before the
    computational-basis (Z) measurement.

    Args:
        state: The 2-element statevector of the qubit.
        basis: Measurement basis. ``0`` for Z-basis, ``1`` for X-basis.
        rng: NumPy random generator used for the probabilistic
            measurement outcome.

    Returns:
        int: Measurement result, either ``0`` or ``1``.
    """
    if basis == 1:
        # Rotate to Z basis: apply H
        h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        state = h @ state

    prob_0 = float(abs(state[0]) ** 2)
    return 0 if rng.random() < prob_0 else 1


def _eve_intercept_resend(
    state: NDArray[np.complexfloating],
    rng: np.random.Generator,
) -> NDArray[np.complexfloating]:
    """Simulate Eve's intercept-resend attack on a single qubit.

    Eve chooses a random basis, measures the qubit (collapsing its state),
    and re-prepares a new qubit in the measured state/basis. When Eve's
    basis differs from Alice's, this introduces errors detectable via
    the QBER check.

    Args:
        state: The 2-element statevector of the qubit being transmitted.
        rng: NumPy random generator for Eve's basis choice and
            measurement outcome.

    Returns:
        NDArray[np.complexfloating]: The re-prepared qubit statevector.
    """
    eve_basis = int(rng.integers(0, 2))
    eve_bit = _measure_qubit(state, eve_basis, rng)
    return _prepare_qubit(eve_bit, eve_basis)


def bb84_protocol(
    num_qubits: int = 256,
    eavesdropper: bool = False,
    detection_threshold: float = 0.11,
    check_fraction: float = 0.25,
    seed: int | None = None,
) -> BB84Result:
    """Simulate the BB84 quantum key distribution protocol.

    Args:
        num_qubits: Number of qubits Alice sends to Bob.
        eavesdropper: If True, Eve intercepts and resends all qubits.
        detection_threshold: QBER threshold above which Eve is detected.
            Theoretical intercept-resend QBER is 25%; 11% is a safe cutoff.
        check_fraction: Fraction of sifted bits used for error checking.
        seed: Random seed for reproducibility.

    Returns:
        BB84Result with the shared key and eavesdropping detection results.
    """
    if num_qubits < 4:
        raise ValueError("Need at least 4 qubits for meaningful key exchange")
    if not 0 < check_fraction < 1:
        raise ValueError("check_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)

    # Step 1: Alice chooses random bits and bases
    alice_bits = rng.integers(0, 2, size=num_qubits).tolist()
    alice_bases = rng.integers(0, 2, size=num_qubits).tolist()

    # Step 2: Alice prepares and sends qubits
    states = [_prepare_qubit(alice_bits[i], alice_bases[i]) for i in range(num_qubits)]

    # Optional: Eve intercepts
    if eavesdropper:
        states = [_eve_intercept_resend(s, rng) for s in states]

    # Step 3: Bob chooses random bases and measures
    bob_bases = rng.integers(0, 2, size=num_qubits).tolist()
    bob_bits = [_measure_qubit(states[i], bob_bases[i], rng) for i in range(num_qubits)]

    # Step 4: Sifting — keep only matching-basis bits
    sifted_alice = []
    sifted_bob = []
    for i in range(num_qubits):
        if alice_bases[i] == bob_bases[i]:
            sifted_alice.append(alice_bits[i])
            sifted_bob.append(bob_bits[i])

    sifted_length = len(sifted_alice)

    # Step 5: Error checking — sacrifice some bits to estimate QBER
    num_check = max(1, int(sifted_length * check_fraction))
    num_key = sifted_length - num_check

    if sifted_length == 0:
        return BB84Result(
            alice_key=[],
            bob_key=[],
            sifted_length=0,
            qber=0.0,
            final_key_length=0,
            eavesdropper_detected=False,
            num_qubits_sent=num_qubits,
            detection_threshold=detection_threshold,
        )

    # Use last num_check bits for checking, first num_key bits for key
    check_errors = sum(
        sifted_alice[num_key + i] != sifted_bob[num_key + i]
        for i in range(num_check)
    )
    qber = check_errors / num_check if num_check > 0 else 0.0

    alice_key = sifted_alice[:num_key]
    bob_key = sifted_bob[:num_key]

    return BB84Result(
        alice_key=alice_key,
        bob_key=bob_key,
        sifted_length=sifted_length,
        qber=qber,
        final_key_length=num_key,
        eavesdropper_detected=qber > detection_threshold,
        num_qubits_sent=num_qubits,
        detection_threshold=detection_threshold,
    )
