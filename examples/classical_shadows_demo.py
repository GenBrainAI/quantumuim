"""Classical Shadows Demo

Demonstrates the classical shadow protocol for efficient quantum
state property estimation (Huang, Kueng, Preskill 2020).
"""

import numpy as np
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT, X, Rz
from quantumium.shadows.shadows import (
    ClassicalShadow,
    shadow_norm_pauli,
    shadow_norm_observable,
    required_snapshots,
    pauli_decomposition,
    PAULI_X, PAULI_Y, PAULI_Z, PAULI_I,
)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ── 1. Shadow Construction ─────────────────────────────────────────────

section("1. Classical Shadow Construction")

state_0 = np.array([1, 0], dtype=complex)
shadow = ClassicalShadow.from_state(state_0, num_snapshots=5000, seed=42)
print(f"State: |0>")
print(f"Shadow: {shadow.num_qubits} qubit, {shadow.num_snapshots} snapshots")
print(f"\nFirst 5 snapshots:")
for s in shadow.snapshots[:5]:
    print(f"  Basis: {s.bases}, Outcome: {s.outcomes}")


# ── 2. Pauli Expectation Values ────────────────────────────────────────

section("2. Pauli Expectation Values")

print("State: |0>")
for pauli in ['X', 'Y', 'Z']:
    result = shadow.estimate_pauli(pauli)
    exact = {'X': 0.0, 'Y': 0.0, 'Z': 1.0}[pauli]
    print(f"  <{pauli}> = {result.value:+.3f}  (exact: {exact:+.1f})")

plus_state = np.array([1, 1], dtype=complex) / np.sqrt(2)
shadow_plus = ClassicalShadow.from_state(plus_state, 5000, seed=42)
print(f"\nState: |+>")
for pauli in ['X', 'Y', 'Z']:
    result = shadow_plus.estimate_pauli(pauli)
    exact = {'X': 1.0, 'Y': 0.0, 'Z': 0.0}[pauli]
    print(f"  <{pauli}> = {result.value:+.3f}  (exact: {exact:+.1f})")


# ── 3. Bell State Properties ──────────────────────────────────────────

section("3. Bell State — Multi-Qubit Correlations")

qc = QuantumCircuit(2)
qc.apply(H, 0)
qc.apply(CNOT, 0, 1)
bell = qc.statevector()

shadow_bell = ClassicalShadow.from_state(bell, 5000, seed=42)
print("State: Bell = (|00> + |11>)/sqrt(2)\n")

correlations = [
    ('ZZ', +1.0), ('XX', +1.0), ('YY', -1.0),
    ('ZI', 0.0), ('IZ', 0.0), ('XY', 0.0),
]
print(f"{'Observable':<12} {'Shadow':>8} {'Exact':>8}")
print("-" * 30)
for pauli, exact in correlations:
    result = shadow_bell.estimate_pauli(pauli)
    print(f"{pauli:<12} {result.value:>+8.3f} {exact:>+8.1f}")


# ── 4. Fidelity Estimation ────────────────────────────────────────────

section("4. Fidelity Estimation")

fid_bell = shadow_bell.estimate_fidelity(bell)
print(f"Fidelity with Bell state: {fid_bell.value:.3f} (expected 1.0)")

zero_2q = np.array([1, 0, 0, 0], dtype=complex)
fid_zero = shadow_bell.estimate_fidelity(zero_2q)
print(f"Fidelity with |00>:       {fid_zero.value:.3f} (expected 0.5)")


# ── 5. Shadow Norm & Sample Complexity ────────────────────────────────

section("5. Shadow Norm & Sample Complexity")

print("Shadow norms for random Pauli ensemble:")
print(f"  ||Z||_shadow     = {shadow_norm_pauli('Z'):.0f}   (1-local)")
print(f"  ||ZZ||_shadow    = {shadow_norm_pauli('ZZ'):.0f}   (2-local)")
print(f"  ||ZZZ||_shadow   = {shadow_norm_pauli('ZZZ'):.0f}  (3-local)")
print(f"  ||ZZZZ||_shadow  = {shadow_norm_pauli('ZZZZ'):.0f}  (4-local)")

print(f"\nRequired snapshots (delta=0.01):")
for eps in [0.1, 0.01]:
    T1 = required_snapshots(shadow_norm_pauli('Z'), epsilon=eps)
    T2 = required_snapshots(shadow_norm_pauli('ZZ'), epsilon=eps)
    print(f"  eps={eps}: 1-local needs {T1:>7d},  2-local needs {T2:>7d}")

print(f"\nScaling for M observables (eps=0.1, 1-local):")
for M in [1, 10, 100, 1000]:
    T = required_snapshots(3.0, epsilon=0.1, num_observables=M)
    print(f"  M={M:>4d}: T = {T:>7d}  (O(log M) scaling)")


# ── 6. Pauli Decomposition ────────────────────────────────────────────

section("6. Pauli Decomposition of Observables")

proj_0 = (PAULI_I + PAULI_Z) / 2
decomp = pauli_decomposition(proj_0)
print("|0><0| = ", end="")
terms = [f"{c:+.1f}*{p}" for p, c in sorted(decomp.items())]
print(" ".join(terms))

obs = 0.7 * PAULI_Z + 0.3 * PAULI_X
decomp2 = pauli_decomposition(obs)
print("0.7*Z + 0.3*X = ", end="")
terms2 = [f"{c:+.1f}*{p}" for p, c in sorted(decomp2.items())]
print(" ".join(terms2))


# ── 7. Convergence with Snapshot Count ────────────────────────────────

section("7. Convergence Study")

state = np.array([1, 0], dtype=complex)
print(f"{'Snapshots':>10} {'<Z> estimate':>14} {'|error|':>10}")
print("-" * 38)
for T in [100, 500, 1000, 2000, 5000, 10000]:
    shadow_t = ClassicalShadow.from_state(state, T, seed=42)
    result = shadow_t.estimate_pauli('Z')
    error = abs(result.value - 1.0)
    print(f"{T:>10d} {result.value:>14.4f} {error:>10.4f}")


print(f"\n{'='*60}")
print(f"  Demo complete — Classical Shadows working correctly")
print(f"{'='*60}")
