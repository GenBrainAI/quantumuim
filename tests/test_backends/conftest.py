"""Shared fixtures and markers for backend validation tests."""

from __future__ import annotations

import numpy as np
import pytest

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, CNOT
from quantumium.backends import LocalSimulatorBackend


# ---------------------------------------------------------------------------
# Skip-condition helpers
# ---------------------------------------------------------------------------


def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


requires_qiskit = pytest.mark.skipif(
    not _can_import("qiskit"),
    reason="qiskit not installed",
)
requires_cirq = pytest.mark.skipif(
    not _can_import("cirq"),
    reason="cirq not installed",
)
requires_braket = pytest.mark.skipif(
    not _can_import("braket"),
    reason="amazon-braket-sdk not installed",
)

hardware = pytest.mark.skipif(
    True, reason="hardware tests disabled in CI (set QUANTUM_HW=1 to enable)"
)


# ---------------------------------------------------------------------------
# Standard test circuits
# ---------------------------------------------------------------------------


@pytest.fixture
def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.apply(H, 0)
    qc.apply(CNOT, 0, 1)
    return qc


@pytest.fixture
def ghz3_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.apply(H, 0)
    qc.apply(CNOT, 0, 1)
    qc.apply(CNOT, 0, 2)
    return qc


@pytest.fixture
def ghz5_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(5)
    qc.apply(H, 0)
    for i in range(4):
        qc.apply(CNOT, 0, i + 1)
    return qc


@pytest.fixture
def x_gate_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.apply(X, 0)
    return qc


@pytest.fixture
def qft_roundtrip_circuit() -> QuantumCircuit:
    """QFT followed by inverse QFT — should return to |000>."""
    n = 3
    qc = QuantumCircuit(n)
    # QFT
    for i in range(n):
        qc.apply(H, i)
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            from quantumium.core.gate import controlled, Rz as _Rz

            crz = controlled(_Rz(angle), name=f"CRz({angle:.4f})")
            qc.apply(crz, j, i)
    # Inverse QFT
    for i in reversed(range(n)):
        for j in reversed(range(i + 1, n)):
            angle = -np.pi / (2 ** (j - i))
            crz = controlled(_Rz(angle), name=f"CRz({angle:.4f})")
            qc.apply(crz, j, i)
        qc.apply(H, i)
    return qc


@pytest.fixture
def grover3_circuit() -> QuantumCircuit:
    """3-qubit Grover search for marked state |101> (index 5)."""
    from quantumium.algorithms.grover import grover_oracle, grover_diffuser

    n = 3
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.apply(H, i)

    oracle = grover_oracle(n, [5])
    diffuser = grover_diffuser(n)
    qc.apply(oracle, *range(n))
    qc.apply(diffuser, *range(n))

    return qc


# ---------------------------------------------------------------------------
# Statistical validation helpers
# ---------------------------------------------------------------------------


def _regularized_gamma_series(a: float, x: float) -> float:
    """Lower regularized incomplete gamma via series expansion."""
    import math

    if x == 0:
        return 0.0
    ap = a
    s = 1.0 / a
    d = s
    for _ in range(200):
        ap += 1
        d *= x / ap
        s += d
        if abs(d) < abs(s) * 1e-12:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _regularized_gamma_cf(a: float, x: float) -> float:
    """Upper regularized incomplete gamma via continued fraction."""
    import math

    b = x + 1 - a
    c, d = 1e30, 1.0 / b
    h = d
    for i in range(1, 201):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _chi2_cdf(x: float, k: int) -> float:
    """Chi-squared CDF without scipy."""
    if x <= 0:
        return 0.0
    a = k / 2.0
    z = x / 2.0
    if z < a + 1:
        return _regularized_gamma_series(a, z)
    return 1.0 - _regularized_gamma_cf(a, z)


def chi_squared_test(
    observed: dict[str, int],
    expected: dict[str, float],
    shots: int,
) -> float:
    """Return the p-value from a chi-squared goodness-of-fit test.

    Args:
        observed: Measured bitstring counts.
        expected: Expected probability distribution.
        shots: Total number of shots.

    Returns:
        p-value (float). Values > 0.01 indicate no significant deviation.
    """
    all_keys = sorted(set(observed.keys()) | set(expected.keys()))
    obs = np.array([observed.get(k, 0) for k in all_keys], dtype=float)
    exp = np.array([expected.get(k, 0.0) * shots for k in all_keys], dtype=float)

    mask = exp > 0
    if mask.sum() < 2:
        return 1.0

    chi2 = float(np.sum((obs[mask] - exp[mask]) ** 2 / exp[mask]))
    dof = int(mask.sum()) - 1
    p_value = 1.0 - _chi2_cdf(chi2, dof)
    return p_value


def hellinger_fidelity(p: dict[str, float], q: dict[str, float]) -> float:
    """Compute Hellinger fidelity between two probability distributions.

    Returns a value in [0, 1] where 1 means identical distributions.
    """
    all_keys = sorted(set(p.keys()) | set(q.keys()))
    fid = sum(np.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in all_keys)
    return float(fid) ** 2


@pytest.fixture
def local_backend() -> LocalSimulatorBackend:
    return LocalSimulatorBackend()
