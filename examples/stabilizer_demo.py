"""Stabilizer simulation demo — efficient Clifford circuit simulation.

Demonstrates:
  1. Building stabilizer states with Clifford gates
  2. Inspecting stabilizer generators
  3. Quantum teleportation via stabilizer formalism
  4. Comparing with statevector simulation
"""
from __future__ import annotations

import numpy as np

from quantumium.stabilizer import (
    StabilizerTableau,
    hadamard,
    phase_gate,
    cnot,
    cz,
    pauli_x,
    pauli_z,
    measure,
    to_statevector,
)


def bell_state_demo():
    print("=" * 60)
    print("  Stabilizer Simulation — Bell State")
    print("=" * 60)

    tab = StabilizerTableau(2)
    print(f"\nInitial state |00>:")
    print(f"  Stabilizers:    {tab.stabilizer_generators()}")
    print(f"  Destabilizers:  {tab.destabilizer_generators()}")

    hadamard(tab, 0)
    print(f"\nAfter H on qubit 0 (|+0> state):")
    print(f"  Stabilizers:    {tab.stabilizer_generators()}")

    cnot(tab, 0, 1)
    print(f"\nAfter CNOT(0,1) — Bell state (|00> + |11>)/√2:")
    print(f"  Stabilizers:    {tab.stabilizer_generators()}")

    sv = to_statevector(tab)
    print(f"  Statevector:    {np.round(sv, 4)}")
    print(f"  Expected:       [0.7071, 0, 0, 0.7071]")

    # Measure — should get correlated results
    rng = np.random.default_rng(42)
    results = {"00": 0, "01": 0, "10": 0, "11": 0}
    for _ in range(100):
        t = StabilizerTableau(2)
        hadamard(t, 0)
        cnot(t, 0, 1)
        r0 = measure(t, 0, rng)
        r1 = measure(t, 1, rng)
        results[f"{r0}{r1}"] += 1

    print(f"\n  100 measurements: {results}")
    print(f"  (Only 00 and 11 should appear — perfect correlations)")


def ghz_demo():
    print("\n" + "=" * 60)
    print("  Stabilizer Simulation — 5-qubit GHZ State")
    print("=" * 60)

    n = 5
    tab = StabilizerTableau(n)
    hadamard(tab, 0)
    for i in range(1, n):
        cnot(tab, 0, i)

    print(f"\n  GHZ state |00000> + |11111>)/√2:")
    print(f"  Stabilizers:")
    for gen in tab.stabilizer_generators():
        print(f"    {gen}")

    sv = to_statevector(tab)
    print(f"\n  Non-zero amplitudes:")
    for i, amp in enumerate(sv):
        if abs(amp) > 1e-10:
            print(f"    |{i:05b}> : {amp:.4f}")


def teleportation_demo():
    print("\n" + "=" * 60)
    print("  Stabilizer Simulation — Quantum Teleportation")
    print("=" * 60)

    rng = np.random.default_rng(123)

    # Teleport |+> state (H|0>) from qubit 0 to qubit 2
    # Qubits: 0=Alice's data, 1=Alice's EPR, 2=Bob's EPR
    tab = StabilizerTableau(3)

    # Prepare |+> on qubit 0
    hadamard(tab, 0)
    print(f"\n  Step 1: Prepare |+> on qubit 0")
    print(f"    Stabilizers: {tab.stabilizer_generators()}")

    # Create EPR pair between qubits 1 and 2
    hadamard(tab, 1)
    cnot(tab, 1, 2)
    print(f"  Step 2: EPR pair on qubits 1-2")
    print(f"    Stabilizers: {tab.stabilizer_generators()}")

    # Bell measurement on qubits 0-1
    cnot(tab, 0, 1)
    hadamard(tab, 0)
    m0 = measure(tab, 0, rng)
    m1 = measure(tab, 1, rng)
    print(f"  Step 3: Bell measurement → {m0}{m1}")

    # Corrections on qubit 2
    if m1 == 1:
        pauli_x(tab, 2)
    if m0 == 1:
        pauli_z(tab, 2)
    print(f"  Step 4: Applied corrections")

    # Verify qubit 2 is now |+>
    sv = to_statevector(tab)
    print(f"    Stabilizers: {tab.stabilizer_generators()}")

    # Measure qubit 2 many times to check it's in |+>
    results = {0: 0, 1: 0}
    for _ in range(200):
        t = StabilizerTableau(3)
        hadamard(t, 0)
        hadamard(t, 1)
        cnot(t, 1, 2)
        cnot(t, 0, 1)
        hadamard(t, 0)
        r0 = measure(t, 0, rng)
        r1 = measure(t, 1, rng)
        if r1 == 1:
            pauli_x(t, 2)
        if r0 == 1:
            pauli_z(t, 2)
        # Measure in X basis to check |+>
        hadamard(t, 2)
        r2 = measure(t, 2, rng)
        results[r2] += 1

    print(f"\n  Teleported state measured in X basis (200 shots): {results}")
    print(f"  (|+> in X basis → always 0)")


def cz_demo():
    print("\n" + "=" * 60)
    print("  Stabilizer Simulation — CZ Gate and Phase Kickback")
    print("=" * 60)

    tab = StabilizerTableau(2)
    # Prepare |+> on both qubits
    hadamard(tab, 0)
    hadamard(tab, 1)
    print(f"\n  State |++>:")
    print(f"    Stabilizers: {tab.stabilizer_generators()}")

    # Apply CZ
    cz(tab, 0, 1)
    print(f"  After CZ:")
    print(f"    Stabilizers: {tab.stabilizer_generators()}")

    sv = to_statevector(tab)
    print(f"    Statevector: {np.round(sv, 4)}")
    print(f"    (This is a cluster state / graph state)")


if __name__ == "__main__":
    bell_state_demo()
    ghz_demo()
    teleportation_demo()
    cz_demo()
    print("\nDone!")
