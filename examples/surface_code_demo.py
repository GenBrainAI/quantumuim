"""Surface code error correction demo.

Demonstrates the rotated planar surface code with MWPM decoder:
1. Build a d=3 lattice and show its structure
2. Inject single-qubit errors and decode them
3. Run Monte Carlo threshold estimation for d=3,5
"""

from quantumium.surface_code import (
    SurfaceCodeLattice,
    apply_error,
    extract_syndrome,
    MWPMDecoder,
    logical_x,
    logical_z,
)
from quantumium.surface_code.logical import has_logical_error
from quantumium.surface_code.threshold import logical_error_rate


def main():
    print("=" * 60)
    print("Surface Code Error Correction Demo")
    print("=" * 60)

    # 1. Lattice structure
    print("\n1. Lattice Structure (d=3)")
    print("-" * 40)
    lattice = SurfaceCodeLattice(3)
    print(f"   Code distance: {lattice.d}")
    print(f"   Data qubits: {lattice.num_data}")
    print(f"   X stabilizers: {lattice.num_x_stabilizers}")
    print(f"   Z stabilizers: {lattice.num_z_stabilizers}")
    print(f"   Logical X support: {[lattice.data_qubits[i] for i, v in enumerate(logical_x(lattice)) if v]}")
    print(f"   Logical Z support: {[lattice.data_qubits[i] for i, v in enumerate(logical_z(lattice)) if v]}")

    # 2. Single-qubit error correction
    print("\n2. Single-Qubit Error Correction (d=3)")
    print("-" * 40)
    decoder = MWPMDecoder(lattice)

    successes = 0
    for q in range(lattice.num_data):
        error = apply_error(lattice, "X", q)
        syndrome = extract_syndrome(lattice, error, "X")
        correction = decoder.decode_x_errors(syndrome)
        is_logical = has_logical_error(error, correction, lattice, "X")
        status = "FAIL (logical error)" if is_logical else "OK"
        if not is_logical:
            successes += 1
        print(f"   X error on qubit {q} {lattice.data_qubits[q]}: syndrome={syndrome.tolist()}, {status}")

    print(f"\n   Corrected {successes}/{lattice.num_data} single-qubit X errors")

    # Also test Z errors
    z_successes = 0
    for q in range(lattice.num_data):
        error = apply_error(lattice, "Z", q)
        syndrome = extract_syndrome(lattice, error, "Z")
        correction = decoder.decode_z_errors(syndrome)
        is_logical = has_logical_error(error, correction, lattice, "Z")
        if not is_logical:
            z_successes += 1

    print(f"   Corrected {z_successes}/{lattice.num_data} single-qubit Z errors")

    # 3. Multi-distance comparison
    print("\n3. Logical Error Rate vs Physical Error Rate")
    print("-" * 40)
    p_values = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]
    for d in [3, 5]:
        print(f"\n   d={d}:")
        for p in p_values:
            rate = logical_error_rate(d, p, num_trials=200, seed=42)
            bar = "#" * int(rate * 50)
            print(f"     p={p:.2f}: logical_err={rate:.3f} {bar}")

    print("\n" + "=" * 60)
    print("Surface code demo complete.")
    print("Key insight: logical error rate should decrease with")
    print("increasing code distance below the threshold (~10%).")
    print("=" * 60)


if __name__ == "__main__":
    main()
