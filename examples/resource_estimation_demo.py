"""Resource Estimation Demo — fault-tolerant quantum computing cost analysis.

Demonstrates:
1. Circuit analysis (T-count, T-depth, Clifford count)
2. Surface code distance selection
3. Magic state distillation factory cost
4. Full physical resource estimation
5. Rotation gate decomposition costs
6. Scaling with circuit size and error targets
7. Comparison across hardware assumptions
"""

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, T, S, X, CNOT, CCX, Rz
from quantumium.resource_estimation import (
    CircuitAnalyzer,
    ResourceEstimator,
    MagicStateFactory,
)


# ── 1. Circuit Analysis ──────────────────────────────────────────────

print("=" * 60)
print("1. Circuit Analysis — T-gate Counting")
print("=" * 60)

analyzer = CircuitAnalyzer()

qc_bell = QuantumCircuit(2)
qc_bell.apply(H, 0)
qc_bell.apply(CNOT, 0, 1)
res_bell = analyzer.analyze(qc_bell)
print(f"  Bell state: T={res_bell.n_t_gates}, Clifford={res_bell.n_clifford}, "
      f"depth={res_bell.circuit_depth}")

qc_toffoli = QuantumCircuit(3)
qc_toffoli.apply(CCX, 0, 1, 2)
res_tof = analyzer.analyze(qc_toffoli)
print(f"  Toffoli:    T={res_tof.n_t_gates} (expected 7), depth={res_tof.circuit_depth}")

qc_t_heavy = QuantumCircuit(3)
for i in range(3):
    qc_t_heavy.apply(H, i)
for _ in range(5):
    for i in range(3):
        qc_t_heavy.apply(T, i)
res_heavy = analyzer.analyze(qc_t_heavy)
print(f"  T-heavy:    T={res_heavy.n_t_gates}, T-depth={res_heavy.t_depth}")


# ── 2. Surface Code Distance ─────────────────────────────────────────

print("\n" + "=" * 60)
print("2. Surface Code Distance Selection")
print("=" * 60)

est = ResourceEstimator(physical_error_rate=1e-3)
for target in [1e-3, 1e-6, 1e-10, 1e-15]:
    d = est.code_distance(target, n_logical_ops=1000)
    qubits_per = 2 * d * d - 1
    print(f"  p_target={target:.0e}, n_ops=1000 -> d={d}, "
          f"qubits/logical={qubits_per}")


# ── 3. Magic State Factory ───────────────────────────────────────────

print("\n" + "=" * 60)
print("3. Magic State Distillation")
print("=" * 60)

for d in [5, 7, 13]:
    factory = MagicStateFactory(code_distance=d)
    print(f"  d={d:2d}: {factory.qubits_per_factory:,} qubits/factory, "
          f"error={factory.output_error_rate:.2e}, "
          f"{factory.cycles_per_t_state} cycles/T-state")

print("\n  Two-round distillation (d=7):")
f2 = MagicStateFactory(code_distance=7, n_distillation_rounds=2)
print(f"    Output error: {f2.output_error_rate:.2e}")
print(f"    Qubits:       {f2.qubits_per_factory:,}")


# ── 4. Full Resource Estimation ──────────────────────────────────────

print("\n" + "=" * 60)
print("4. Full Physical Resource Estimation")
print("=" * 60)

qc = QuantumCircuit(5)
for i in range(5):
    qc.apply(H, i)
for i in range(4):
    qc.apply(CNOT, i, i + 1)
    qc.apply(T, i)
qc.apply(T, 4)

estimator = ResourceEstimator(physical_error_rate=1e-3)
result = estimator.estimate(qc, target_error=1e-6)
print(result.summary())


# ── 5. Rotation Gate Costs ───────────────────────────────────────────

print("\n" + "=" * 60)
print("5. Rotation Gate Decomposition Costs")
print("=" * 60)

qc_rot = QuantumCircuit(2)
qc_rot.apply(H, 0)
qc_rot.apply(Rz(0.123), 0)
qc_rot.apply(CNOT, 0, 1)
qc_rot.apply(Rz(0.456), 1)
qc_rot.apply(Rz(0.789), 0)

for prec in [1e-3, 1e-6, 1e-10]:
    analyzer_p = CircuitAnalyzer(rotation_precision=prec)
    res_p = analyzer_p.analyze(qc_rot)
    print(f"  precision={prec:.0e}: rotation_T_cost={res_p.rotation_t_cost}, "
          f"total_T_budget={res_p.total_t_budget}")


# ── 6. Scaling with Circuit Size ─────────────────────────────────────

print("\n" + "=" * 60)
print("6. Scaling: Physical Qubits vs. Logical Qubits")
print("=" * 60)

estimator = ResourceEstimator(physical_error_rate=1e-3)
for n in [2, 5, 10, 20, 50]:
    qc_n = QuantumCircuit(n)
    for i in range(n):
        qc_n.apply(H, i)
    for i in range(n - 1):
        qc_n.apply(CNOT, i, i + 1)
    result_n = estimator.estimate(qc_n, target_error=1e-6)
    print(f"  n={n:3d} logical -> {result_n.total_qubits:>10,} physical "
          f"(d={result_n.code_distance})")


# ── 7. Hardware Comparison ───────────────────────────────────────────

print("\n" + "=" * 60)
print("7. Hardware Comparison: Error Rate Impact")
print("=" * 60)

qc_bench = QuantumCircuit(4)
for i in range(4):
    qc_bench.apply(H, i)
    qc_bench.apply(T, i)
for i in range(3):
    qc_bench.apply(CNOT, i, i + 1)

for p_phys, label in [(1e-2, "Near-threshold"), (1e-3, "Current"), (1e-4, "Improved")]:
    est_hw = ResourceEstimator(physical_error_rate=p_phys)
    r = est_hw.estimate(qc_bench, target_error=1e-8)
    print(f"  {label:15s} (p={p_phys:.0e}): d={r.code_distance:2d}, "
          f"{r.total_qubits:>8,} qubits, {r.runtime_seconds:.4f}s")

print()
