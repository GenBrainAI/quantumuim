#!/usr/bin/env python3
"""Hardware validation demo — compare quantum backends.

Demonstrates how to validate backend correctness by running identical
circuits on the local simulator and comparing results using statistical
tests (chi-squared, Hellinger fidelity).

Usage:
    python examples/hardware_validation_demo.py
"""

from __future__ import annotations

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, X, CNOT, CCX
from quantumium.core.noise import NoiseModel, depolarizing
from quantumium.backends import LocalSimulatorBackend


def _reg_gamma_series(a: float, x: float) -> float:
    import math
    if x == 0:
        return 0.0
    ap, s = a, 1.0 / a
    d = s
    for _ in range(200):
        ap += 1; d *= x / ap; s += d
        if abs(d) < abs(s) * 1e-12:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _reg_gamma_cf(a: float, x: float) -> float:
    import math
    b = x + 1 - a; c, d = 1e30, 1.0 / b; h = d
    for i in range(1, 201):
        an = -i * (i - a); b += 2
        d = an * d + b
        if abs(d) < 1e-30: d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi_squared_pvalue(observed: dict[str, int], expected: dict[str, float], shots: int) -> float:
    """Chi-squared goodness-of-fit p-value (no scipy dependency)."""
    all_keys = sorted(set(observed) | set(expected))
    obs = np.array([observed.get(k, 0) for k in all_keys], dtype=float)
    exp = np.array([expected.get(k, 0.0) * shots for k in all_keys], dtype=float)
    mask = exp > 0
    if mask.sum() < 2:
        return 1.0
    chi2 = float(np.sum((obs[mask] - exp[mask]) ** 2 / exp[mask]))
    dof = int(mask.sum()) - 1
    a, z = dof / 2.0, chi2 / 2.0
    if z < a + 1:
        cdf = _reg_gamma_series(a, z)
    else:
        cdf = 1.0 - _reg_gamma_cf(a, z)
    return 1.0 - cdf


def hellinger_fidelity(p: dict[str, float], q: dict[str, float]) -> float:
    """Hellinger fidelity between two distributions."""
    all_keys = sorted(set(p) | set(q))
    return float(sum(np.sqrt(p.get(k, 0) * q.get(k, 0)) for k in all_keys)) ** 2


def main():
    backend = LocalSimulatorBackend()
    shots = 4096

    print("=" * 60)
    print("Quantumium Hardware Validation Demo")
    print("=" * 60)
    print(f"\nBackend: {backend.name}")
    print(f"Shots per circuit: {shots}\n")

    # ── 1. Bell State Validation ──────────────────────────────
    print("─" * 40)
    print("1. Bell State (2 qubits)")
    print("─" * 40)

    qc = QuantumCircuit(2)
    qc.apply(H, 0)
    qc.apply(CNOT, 0, 1)
    result = backend.run(qc, shots=shots)

    expected_bell = {"00": 0.5, "11": 0.5}
    p_value = chi_squared_pvalue(result.counts, expected_bell, shots)
    fid = hellinger_fidelity(result.probabilities, expected_bell)

    print(f"  Counts: {result.counts}")
    print(f"  Chi-squared p-value: {p_value:.4f} {'PASS' if p_value > 0.01 else 'FAIL'}")
    print(f"  Hellinger fidelity:  {fid:.6f}")
    print()

    # ── 2. GHZ State Validation ───────────────────────────────
    print("─" * 40)
    print("2. GHZ State (3 qubits)")
    print("─" * 40)

    qc = QuantumCircuit(3)
    qc.apply(H, 0)
    qc.apply(CNOT, 0, 1)
    qc.apply(CNOT, 0, 2)
    result = backend.run(qc, shots=shots)

    expected_ghz = {"000": 0.5, "111": 0.5}
    p_value = chi_squared_pvalue(result.counts, expected_ghz, shots)
    fid = hellinger_fidelity(result.probabilities, expected_ghz)

    print(f"  Counts: {result.counts}")
    print(f"  Chi-squared p-value: {p_value:.4f} {'PASS' if p_value > 0.01 else 'FAIL'}")
    print(f"  Hellinger fidelity:  {fid:.6f}")
    print()

    # ── 3. Grover's Search Validation ─────────────────────────
    print("─" * 40)
    print("3. Grover's Search (3 qubits, target=|101>)")
    print("─" * 40)

    from quantumium.algorithms.grover import grover_oracle, grover_diffuser

    n = 3
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.apply(H, i)
    oracle = grover_oracle(n, [5])  # |101> = 5
    diffuser = grover_diffuser(n)
    qc.apply(oracle, *range(n))
    qc.apply(diffuser, *range(n))

    result = backend.run(qc, shots=shots)
    p_marked = result.probabilities.get("101", 0)

    print(f"  Most frequent: {result.most_frequent}")
    print(f"  P(|101>): {p_marked:.4f} (expected ~0.78)")
    print(f"  Above random (0.125): {'PASS' if p_marked > 0.125 else 'FAIL'}")
    print()

    # ── 4. QFT Round-Trip ─────────────────────────────────────
    print("─" * 40)
    print("4. QFT + Inverse QFT (3 qubits)")
    print("─" * 40)

    from quantumium.core.gate import controlled, Rz

    n = 3
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.apply(H, i)
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            crz = controlled(Rz(angle))
            qc.apply(crz, j, i)
    for i in reversed(range(n)):
        for j in reversed(range(i + 1, n)):
            angle = -np.pi / (2 ** (j - i))
            crz = controlled(Rz(angle))
            qc.apply(crz, j, i)
        qc.apply(H, i)

    result = backend.run(qc, shots=shots)
    p_zero = result.probabilities.get("000", 0)

    print(f"  P(|000>): {p_zero:.4f} (expected ~1.0)")
    print(f"  Identity test: {'PASS' if p_zero > 0.95 else 'FAIL'}")
    print()

    # ── 5. Noise Model Calibration ────────────────────────────
    print("─" * 40)
    print("5. Noise Model Calibration")
    print("─" * 40)

    qc = QuantumCircuit(2)
    qc.apply(H, 0)
    qc.apply(CNOT, 0, 1)

    rho_ideal = qc.density_matrix()

    print("  Depolarizing noise sweep:")
    for p in [0.0, 0.01, 0.05, 0.10, 0.20]:
        noise = NoiseModel(default_single_qubit_noise=depolarizing(p))
        rho_noisy = qc.density_matrix(noise_model=noise)
        fid = float(np.real(np.trace(rho_ideal @ rho_noisy)))
        bar = "█" * int(fid * 40) + "░" * (40 - int(fid * 40))
        print(f"  p={p:.2f}: {bar} F={fid:.4f}")
    print()

    # ── 6. Backend Availability Check ─────────────────────────
    print("─" * 40)
    print("6. Backend Availability")
    print("─" * 40)

    from quantumium.backends.qiskit_backend import QiskitBackend
    from quantumium.backends.cirq_backend import CirqBackend
    from quantumium.backends.braket_backend import BraketBackend

    backends = [
        ("LocalSimulator", LocalSimulatorBackend()),
        ("Qiskit (IBM)", QiskitBackend()),
        ("Cirq (Google)", CirqBackend()),
        ("Braket (Amazon)", BraketBackend()),
    ]

    for name, be in backends:
        status = "AVAILABLE" if be.is_available() else "not installed"
        print(f"  {name:20s}: {status}")
    print()

    # ── 7. Cross-Backend Comparison ───────────────────────────
    print("─" * 40)
    print("7. Cross-Backend Comparison (Bell state)")
    print("─" * 40)

    ref_result = LocalSimulatorBackend().run(
        QuantumCircuit(2).apply(H, 0).apply(CNOT, 0, 1),
        shots=shots,
    )

    for name, be in backends:
        if not be.is_available():
            print(f"  {name:20s}: skipped (not installed)")
            continue
        qc = QuantumCircuit(2)
        qc.apply(H, 0)
        qc.apply(CNOT, 0, 1)
        r = be.run(qc, shots=shots)
        fid = hellinger_fidelity(r.probabilities, ref_result.probabilities)
        print(f"  {name:20s}: fidelity vs local = {fid:.6f}")

    print()
    print("=" * 60)
    print("Validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
