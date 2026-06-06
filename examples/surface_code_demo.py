"""Surface code error correction demo.

Demonstrates:
1. Building a rotated planar surface code
2. Injecting single-qubit errors
3. Syndrome extraction and MWPM decoding
4. Verifying that all weight-1 errors are corrected
5. Logical error rate vs physical error rate (threshold behavior)
"""

from quantumium.surface_code import (
    SurfaceCode,
    inject_error,
    measure_syndrome,
    mwpm_decode,
    logical_x,
    logical_z,
    estimate_threshold,
)
from quantumium.surface_code.logical import has_logical_error


def demo_single_error_correction():
    """Correct every single-qubit X error on a distance-3 code."""
    print("=" * 60)
    print("Surface Code Demo — Single-Qubit Error Correction (d=3)")
    print("=" * 60)

    code = SurfaceCode(distance=3)
    print(f"\nDistance: {code.distance}")
    print(f"Data qubits: {code.n_data}")
    print(f"X stabilizers: {code.n_x_stabilizers}")
    print(f"Z stabilizers: {code.n_z_stabilizers}")

    successes = 0
    for q in range(code.n_data):
        code.reset()
        inject_error(code, "X", q)
        x_syn, z_syn = measure_syndrome(code)
        x_corr, z_corr = mwpm_decode(code, x_syn, z_syn)
        if not has_logical_error(code, x_corr, z_corr):
            successes += 1
        else:
            print(f"  FAILED on qubit {q}")

    print(f"\nX errors corrected: {successes}/{code.n_data}")

    successes_z = 0
    for q in range(code.n_data):
        code.reset()
        inject_error(code, "Z", q)
        x_syn, z_syn = measure_syndrome(code)
        x_corr, z_corr = mwpm_decode(code, x_syn, z_syn)
        if not has_logical_error(code, x_corr, z_corr):
            successes_z += 1

    print(f"Z errors corrected: {successes_z}/{code.n_data}")


def demo_threshold():
    """Estimate the error correction threshold."""
    print("\n" + "=" * 60)
    print("Surface Code Demo — Threshold Estimation")
    print("=" * 60)

    distances = [3, 5]
    error_rates = [0.001, 0.005, 0.01, 0.02]
    num_trials = 200

    print(f"\nDistances: {distances}")
    print(f"Error rates: {error_rates}")
    print(f"Trials per point: {num_trials}")

    result = estimate_threshold(
        distances=distances,
        error_rates=error_rates,
        num_trials=num_trials,
        seed=42,
    )

    print(f"\n{'p_phys':>10} | ", end="")
    for d in distances:
        print(f"d={d:>2} P_L", end="  | ")
    print()
    print("-" * (12 + 12 * len(distances)))

    for i, p in enumerate(error_rates):
        print(f"{p:>10.4f} | ", end="")
        for d in distances:
            pl = result.logical_error_rates[d][i]
            print(f"{pl:>8.4f}", end="  | ")
        print()

    print("\nBelow threshold: larger d → lower logical error rate")
    print("Above threshold: larger d → higher logical error rate")


if __name__ == "__main__":
    demo_single_error_correction()
    demo_threshold()
    print("\nDone!")
