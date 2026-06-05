"""Tensor Network (MPS) Simulation Demo.

Demonstrates:
1. Matrix Product State construction
2. Gate application and truncation
3. Entanglement entropy measurement
4. Comparison with exact statevector simulation
"""

import math
import numpy as np

from quantumium.tensor_networks import (
    MPS,
    mps_zero_state,
    mps_product_state,
    mps_from_statevector,
    apply_single_qubit_gate,
    apply_two_qubit_gate,
    entanglement_entropy,
    inner_product,
)


X = np.array([[0, 1], [1, 0]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)


def demo_mps_basics():
    print("=" * 60)
    print("  MPS Basics — State Construction")
    print("=" * 60)

    n = 4
    mps = mps_zero_state(n)
    sv = mps.to_statevector()
    print(f"\n{n}-qubit |0000> state:")
    print(f"  Statevector length: {len(sv)}")
    print(f"  |0000> amplitude:   {sv[0]:.4f}")
    print(f"  Norm:               {mps.norm():.6f}")

    ket0, ket1 = np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)
    states_1010 = [ket1, ket0, ket1, ket0]
    mps_prod = mps_product_state(states_1010)
    sv_prod = mps_prod.to_statevector()
    idx = 0b1010
    print(f"\n  |1010> product state → amplitude at index {idx}: {sv_prod[idx]:.4f}")


def demo_gate_application():
    print("\n" + "=" * 60)
    print("  Gate Application on MPS")
    print("=" * 60)

    n = 4
    mps = mps_zero_state(n)

    mps = apply_single_qubit_gate(mps, H, 0)
    print(f"\n  After H on qubit 0:")
    sv = mps.to_statevector()
    print(f"    |0000> = {sv[0]:.4f},  |1000> = {sv[8]:.4f}")

    mps = apply_two_qubit_gate(mps, CNOT, 0, 1)
    sv = mps.to_statevector()
    print(f"\n  After CNOT(0,1) — Bell-like state on qubits 0,1:")
    for i, amp in enumerate(sv):
        if abs(amp) > 0.01:
            label = format(i, f"0{n}b")
            print(f"    |{label}> = {amp:.4f}")


def demo_entanglement():
    print("\n" + "=" * 60)
    print("  Entanglement Entropy")
    print("=" * 60)

    n = 4
    mps = mps_zero_state(n)
    S_product = entanglement_entropy(mps, 2)
    print(f"\n  Product state |0000>: S(qubit 0,1 | 2,3) = {S_product:.6f}")

    mps_bell = mps_zero_state(n)
    mps_bell = apply_single_qubit_gate(mps_bell, H, 0)
    mps_bell = apply_two_qubit_gate(mps_bell, CNOT, 0, 1)
    S_bell = entanglement_entropy(mps_bell, 1)
    print(f"  Bell pair on 0,1:     S(qubit 0 | 1,2,3) = {S_bell:.6f} bits  (expect 1.0)")

    mps_ghz = mps_zero_state(n)
    mps_ghz = apply_single_qubit_gate(mps_ghz, H, 0)
    for i in range(n - 1):
        mps_ghz = apply_two_qubit_gate(mps_ghz, CNOT, i, i + 1)
    S_ghz = entanglement_entropy(mps_ghz, n // 2)
    print(f"  GHZ state |0000>+|1111>: S(half|half) = {S_ghz:.6f} bits  (expect 1.0)")


def demo_truncation():
    print("\n" + "=" * 60)
    print("  Bond Dimension Truncation")
    print("=" * 60)

    n = 6
    mps_exact = mps_zero_state(n)
    mps_exact = apply_single_qubit_gate(mps_exact, H, 0)
    for i in range(n - 1):
        mps_exact = apply_two_qubit_gate(mps_exact, CNOT, i, i + 1)

    sv_exact = mps_exact.to_statevector()
    print(f"\n  {n}-qubit GHZ state (exact MPS):")
    print(f"    Max bond dim: {max(t.shape[0] for t in mps_exact.tensors[1:])}")

    mps_trunc = mps_zero_state(n, max_bond_dim=2)
    mps_trunc = apply_single_qubit_gate(mps_trunc, H, 0)
    for i in range(n - 1):
        mps_trunc = apply_two_qubit_gate(mps_trunc, CNOT, i, i + 1)

    sv_trunc = mps_trunc.to_statevector()
    overlap = abs(np.dot(sv_exact.conj(), sv_trunc))
    print(f"    Truncated (chi=2): overlap with exact = {overlap:.6f}")


def demo_inner_product():
    print("\n" + "=" * 60)
    print("  Inner Product of MPS States")
    print("=" * 60)

    n = 4
    mps_zero = mps_zero_state(n)
    mps_plus = mps_zero_state(n)
    for i in range(n):
        mps_plus = apply_single_qubit_gate(mps_plus, H, i)

    ip = inner_product(mps_zero, mps_plus)
    expected = 1.0 / 2**(n / 2)
    print(f"\n  <0000|++++> = {ip:.6f}  (expected {expected:.6f})")

    ip_self = inner_product(mps_zero, mps_zero)
    print(f"  <0000|0000> = {ip_self:.6f}  (expected 1.0)")


if __name__ == "__main__":
    demo_mps_basics()
    demo_gate_application()
    demo_entanglement()
    demo_truncation()
    demo_inner_product()
    print("\n" + "=" * 60)
    print("  All tensor network demos completed!")
    print("=" * 60)
