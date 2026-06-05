"""Quantum Signal Processing & QSVT Demo.

Demonstrates:
1. QSP polynomial evaluation (single-qubit)
2. Block encoding of matrices
3. QSVT matrix transformations
4. Applications: matrix inversion, Hamiltonian simulation
"""

import numpy as np

from quantumium.qsp import (
    qsp_sequence,
    evaluate_qsp_polynomial,
    qsp_circuit,
    block_encoding,
    qsvt_transform,
    chebyshev_angles,
    qsvt_matrix_inversion,
    qsvt_amplitude_amplification,
)


def demo_qsp_basics():
    print("=" * 60)
    print("  Quantum Signal Processing — Basics")
    print("=" * 60)

    # Simple QSP with zero angles: P(a) = a^d
    angles = [0.0, 0.0, 0.0]  # 3 angles → degree 2
    grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    p_vals = evaluate_qsp_polynomial(angles, grid)

    print("\n  Zero-angle QSP (degree 2):")
    print(f"  {'a':>6s} | {'Re(P(a))':>10s} | {'|P(a)|':>10s}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}")
    for a, p in zip(grid, p_vals):
        print(f"  {a:6.2f} | {np.real(p):10.6f} | {abs(p):10.6f}")

    # Verify QSP unitarity property
    print("\n  Unitarity check: |P(a)|² + |Q(a)|² = 1")
    for a in [0.3, 0.7]:
        U = qsp_sequence(angles, a)
        P, Q = U[0, 0], U[0, 1]
        print(f"    a={a}: |P|²={abs(P)**2:.8f}  |Q|²={abs(Q)**2:.8f}  "
              f"sum={abs(P)**2 + abs(Q)**2:.8f}")


def demo_chebyshev():
    print("\n" + "=" * 60)
    print("  Chebyshev Polynomial via QSP Angles")
    print("=" * 60)

    for d in [3, 5]:
        angles = chebyshev_angles(d)
        grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        p_vals = evaluate_qsp_polynomial(angles, grid)

        print(f"\n  T_{d}(a) via QSP ({d+1} angles):")
        print(f"  {'a':>6s} | {'Re(P(a))':>10s} | {'T_d(a) exact':>12s}")
        print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*12}")
        for a, p in zip(grid, p_vals):
            exact = np.cos(d * np.arccos(np.clip(a, -1, 1)))
            print(f"  {a:6.2f} | {np.real(p):10.6f} | {exact:12.6f}")


def demo_block_encoding():
    print("\n" + "=" * 60)
    print("  Block Encoding")
    print("=" * 60)

    A = np.array([[0.5, 0.2], [0.1, 0.3]], dtype=complex)
    gate, alpha = block_encoding(A)

    print(f"\n  Matrix A:")
    for row in A:
        print(f"    [{row[0].real:6.3f}  {row[1].real:6.3f}]")

    print(f"\n  Block-encoding norm α = {alpha:.4f}")
    print(f"  Unitary size: {gate.matrix.shape[0]}×{gate.matrix.shape[1]}")

    U = gate.matrix
    extracted = U[:2, :2] * alpha
    print(f"\n  Extracted (upper-left × α):")
    for row in extracted:
        print(f"    [{row[0].real:6.3f}  {row[1].real:6.3f}]")

    err = np.linalg.norm(extracted - A)
    print(f"\n  Reconstruction error: {err:.2e}")

    UUd = U @ U.conj().T
    print(f"  Unitarity check: ‖UU†-I‖ = {np.linalg.norm(UUd - np.eye(4)):.2e}")


def demo_qsvt_transform():
    print("\n" + "=" * 60)
    print("  QSVT — Matrix Transformation")
    print("=" * 60)

    A = np.diag([0.8, 0.3]).astype(complex)
    print(f"\n  Input: A = diag({A[0,0].real:.1f}, {A[1,1].real:.1f})")

    angles = chebyshev_angles(3)
    result = qsvt_transform(A, angles)

    print(f"  QSVT with T_3 (degree {result.degree}):")
    T = result.transformed_matrix
    print(f"  Transformed diagonal: [{T[0,0].real:.6f}, {T[1,1].real:.6f}]")

    sv_A = [0.8, 0.3]
    alpha = result.block_encoding_norm
    expected = [np.cos(3 * np.arccos(s / alpha)) for s in sv_A]
    print(f"  Expected T_3(σ/α):    [{expected[0]:.6f}, {expected[1]:.6f}]")


def demo_applications():
    print("\n" + "=" * 60)
    print("  QSVT Applications")
    print("=" * 60)

    # Matrix inversion
    print("\n  --- Matrix Inversion ---")
    A = np.diag([0.8, 0.5]).astype(complex)
    result = qsvt_matrix_inversion(A, kappa_cutoff=0.1, degree=11)
    T = result.transformed_matrix
    print(f"  A = diag(0.8, 0.5)")
    print(f"  QSVT inverse approx diagonal: [{T[0,0].real:.6f}, {T[1,1].real:.6f}]")
    print(f"  Exact 1/σ (normalized):        [{1/0.8:.6f}, {1/0.5:.6f}]")

    # Amplitude amplification
    print("\n  --- Amplitude Amplification ---")
    A = np.diag([0.9, 0.1]).astype(complex)
    result = qsvt_amplitude_amplification(A, num_iterations=2)
    T = result.transformed_matrix
    print(f"  A = diag(0.9, 0.1)")
    print(f"  After amplification: [{abs(T[0,0]):.6f}, {abs(T[1,1]):.6f}]")
    print(f"  (Small singular value 0.1 is amplified via T_5)")


if __name__ == "__main__":
    demo_qsp_basics()
    demo_chebyshev()
    demo_block_encoding()
    demo_qsvt_transform()
    demo_applications()
    print("\nDone!")
