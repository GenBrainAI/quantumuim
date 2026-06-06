"""ZX-calculus optimization engine demo.

Demonstrates:
1. Circuit → ZX-diagram → circuit round-trip
2. Spider fusion merging adjacent phase gates
3. T-count reduction through CNOT controls
4. Full optimization pipeline via zx_optimize()
"""

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, T, S, Z, CNOT, CZ, Rz
from quantumium.zx import ZXDiagram, zx_optimize


def main():
    print("=" * 60)
    print("ZX-Calculus Optimization Engine Demo")
    print("=" * 60)

    # 1. Round-trip conversion
    print("\n1. Circuit → ZX → Circuit Round-Trip")
    print("-" * 40)
    qc = QuantumCircuit(2)
    qc.apply(H, 0).apply(T, 0).apply(CNOT, 0, 1).apply(S, 1)
    diagram = ZXDiagram.from_circuit(qc)
    print(f"   Original: {len(qc.instructions)} gates")
    print(f"   ZX diagram: {len(diagram.spider_ids())} spiders")
    roundtrip = diagram.to_circuit()
    print(f"   Round-trip: {len(roundtrip.instructions)} gates")
    d1 = ZXDiagram.from_circuit(qc)
    d2 = ZXDiagram.from_circuit(roundtrip)
    print(f"   Equivalent: {d1.is_equivalent(d2)}")

    # 2. Spider fusion
    print("\n2. Spider Fusion (merging adjacent phase gates)")
    print("-" * 40)
    qc2 = QuantumCircuit(1)
    qc2.apply(T, 0).apply(T, 0).apply(T, 0).apply(T, 0)
    d = ZXDiagram.from_circuit(qc2)
    print(f"   Before: {len(d.spider_ids())} spiders, T-count={d.t_count()}")
    d.simplify()
    print(f"   After:  {len(d.spider_ids())} spiders, T-count={d.t_count()}")
    result = d.to_circuit()
    print(f"   4×T merged to: {result.instructions[0].gate.name if result.instructions else 'identity'}")
    print(f"   Equivalent: {ZXDiagram.from_circuit(qc2).is_equivalent(d)}")

    # 3. T-count reduction through CNOT
    print("\n3. T-Count Reduction (phase gates through CNOT controls)")
    print("-" * 40)
    circuits = [
        ("T-CNOT-T",       _build_t_cnot_t()),
        ("T-CNOT-T†",      _build_t_cnot_tdg()),
        ("T-CNOT-T-CNOT-T", _build_multi_t()),
    ]
    for label, circ in circuits:
        d_orig = ZXDiagram.from_circuit(circ)
        t_before = d_orig.t_count()
        optimized = zx_optimize(circ)
        d_opt = ZXDiagram.from_circuit(optimized)
        t_after = d_opt.t_count()
        equiv = ZXDiagram.from_circuit(circ).is_equivalent(d_opt)
        print(f"   {label:20s}: T={t_before} → {t_after}  "
              f"gates={len(circ.instructions)} → {len(optimized.instructions)}  "
              f"equiv={equiv}")

    # 4. Color change (H-T-H = Rx)
    print("\n4. Color Change Rule (H·Z(α)·H = X(α))")
    print("-" * 40)
    qc4 = QuantumCircuit(1)
    qc4.apply(H, 0).apply(T, 0).apply(H, 0)
    d4 = ZXDiagram.from_circuit(qc4)
    d4.simplify()
    s = d4.spider_ids()[0]
    print(f"   H·T·H simplified to: {d4.color(s)}-spider "
          f"(phase={d4.phase(s):.4f} = π/4)")
    r4 = d4.to_circuit()
    print(f"   Extracted gate: {r4.instructions[0].gate.name}")

    # 5. Full pipeline
    print("\n5. Full Optimization Pipeline")
    print("-" * 40)
    qc5 = QuantumCircuit(3)
    qc5.apply(H, 0)
    qc5.apply(T, 0).apply(CNOT, 0, 1).apply(T, 0)
    qc5.apply(S, 1).apply(CNOT, 1, 2)
    qc5.apply(T, 2).apply(T, 2)
    qc5.apply(H, 2)
    print(f"   Original: {len(qc5.instructions)} gates, "
          f"T-count={ZXDiagram.from_circuit(qc5).t_count()}")
    opt5 = zx_optimize(qc5)
    print(f"   Optimized: {len(opt5.instructions)} gates, "
          f"T-count={ZXDiagram.from_circuit(opt5).t_count()}")
    print(f"   Equivalent: {ZXDiagram.from_circuit(qc5).is_equivalent(ZXDiagram.from_circuit(opt5))}")

    print("\n" + "=" * 60)
    print("ZX-calculus demo complete.")
    print("Key insight: ZX-calculus merges phase gates through")
    print("CNOT controls, reducing T-count for fault-tolerant circuits.")
    print("=" * 60)


def _build_t_cnot_t():
    qc = QuantumCircuit(2)
    qc.apply(T, 0).apply(CNOT, 0, 1).apply(T, 0)
    return qc


def _build_t_cnot_tdg():
    qc = QuantumCircuit(2)
    qc.apply(T, 0).apply(CNOT, 0, 1).apply(Rz(-3.141592653589793 / 4), 0)
    return qc


def _build_multi_t():
    qc = QuantumCircuit(2)
    qc.apply(T, 0).apply(CNOT, 0, 1).apply(T, 0)
    qc.apply(CNOT, 0, 1).apply(T, 0)
    return qc


if __name__ == "__main__":
    main()
