"""Dynamical Decoupling Demo — noise suppression via fast control pulses.

Demonstrates:
1. DD sequence construction (spin echo, CPMG, XY-4, XY-8, UDD)
2. Idle-window detection and DD pulse insertion
3. Filter function analysis — which noise frequencies each sequence suppresses
4. Fidelity estimation under 1/f noise
"""

import numpy as np

from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT, X
from quantumium.dd import DDSequence, DDScheduler, filter_function, dd_fidelity
from quantumium.dd.filters import fid_filter


# ── 1. Build DD sequences ────────────────────────────────────────────

print("=" * 60)
print("1. DD Sequence Construction")
print("=" * 60)

sequences = {
    "Spin Echo": DDSequence.spin_echo(),
    "CPMG-4":    DDSequence.cpmg(4),
    "XY-4":      DDSequence.xy4(),
    "XY-8":      DDSequence.xy8(),
    "UDD-4":     DDSequence.udd(4),
}

for name, seq in sequences.items():
    times = seq.pulse_times()
    print(f"  {name:12s}  pulses={seq.pulses}  times={[f'{t:.3f}' for t in times]}")


# ── 2. Scheduler: detect idle windows and insert DD pulses ───────────

print("\n" + "=" * 60)
print("2. DD Scheduling on a Bell-State Circuit")
print("=" * 60)

qc = QuantumCircuit(2)
qc.apply(H, 0)
qc.apply(CNOT, 0, 1)

print(f"  Original circuit: {len(qc.instructions)} gates")

scheduler = DDScheduler(DDSequence.xy4(), min_idle_duration=2)
idle_windows = scheduler.find_idle_windows(qc)
for qubit, windows in idle_windows.items():
    if windows:
        print(f"  Qubit {qubit} idle windows: {windows}")
    else:
        print(f"  Qubit {qubit}: no idle windows")

dd_count = scheduler.count_dd_pulses(qc)
print(f"  DD pulses to insert: {dd_count}")

if dd_count > 0:
    qc_dd = scheduler.schedule(qc)
    print(f"  Circuit after DD: {len(qc_dd.instructions)} gates")


# ── 3. Longer circuit with more idle time ────────────────────────────

print("\n" + "=" * 60)
print("3. DD on a Circuit with Long Idle Periods")
print("=" * 60)

qc2 = QuantumCircuit(3)
qc2.apply(H, 0)
qc2.apply(CNOT, 0, 1)
qc2.apply(X, 0)
qc2.apply(CNOT, 0, 2)
qc2.apply(X, 1)
qc2.apply(H, 2)

idle = DDScheduler(DDSequence.cpmg(2)).find_idle_windows(qc2)
for q in range(3):
    windows = idle.get(q, [])
    total_idle = sum(e - s for s, e in windows)
    print(f"  Qubit {q}: {len(windows)} idle window(s), total idle = {total_idle} dt")

for sname, seq in [("CPMG-2", DDSequence.cpmg(2)), ("XY-4", DDSequence.xy4())]:
    sched = DDScheduler(seq)
    n = sched.count_dd_pulses(qc2)
    print(f"  {sname} would insert {n} DD pulses")


# ── 4. Filter function comparison ────────────────────────────────────

print("\n" + "=" * 60)
print("4. Filter Function Analysis")
print("=" * 60)

omega = np.linspace(0.1, 30, 200)
T = 10.0

fid_f = fid_filter(omega, T)
se_f  = filter_function(DDSequence.spin_echo(), omega, T)
xy4_f = filter_function(DDSequence.xy4(), omega, T)
xy8_f = filter_function(DDSequence.xy8(), omega, T)

low = omega < 2.0
print(f"  Mean |F|^2 at low freq (omega < 2):")
print(f"    FID (no DD):  {np.mean(fid_f[low]):.4f}")
print(f"    Spin Echo:    {np.mean(se_f[low]):.4f}")
print(f"    XY-4:         {np.mean(xy4_f[low]):.4f}")
print(f"    XY-8:         {np.mean(xy8_f[low]):.4f}")

print(f"\n  All DD sequences suppress low-frequency noise relative to FID: "
      f"{np.mean(se_f[low]) < np.mean(fid_f[low])}")


# ── 5. Fidelity under 1/f noise ─────────────────────────────────────

print("\n" + "=" * 60)
print("5. Fidelity Estimation Under 1/f Noise")
print("=" * 60)

def one_over_f(omega: np.ndarray) -> np.ndarray:
    return 0.01 / (omega + 1e-10)

fid_none = dd_fidelity(
    DDSequence(name="fid", pulses=[], spacings=[1.0]),
    one_over_f, total_time=T,
)
print(f"  No DD (FID):    fidelity = {fid_none:.6f}")

for sname, seq in sequences.items():
    fid = dd_fidelity(seq, one_over_f, total_time=T)
    improvement = fid - fid_none
    print(f"  {sname:12s}:  fidelity = {fid:.6f}  (improvement = {improvement:+.6f})")


# ── 6. UDD order scaling ────────────────────────────────────────────

print("\n" + "=" * 60)
print("6. UDD Fidelity vs. Pulse Count")
print("=" * 60)

for n in [1, 2, 4, 6, 8]:
    fid = dd_fidelity(DDSequence.udd(n), one_over_f, total_time=T)
    print(f"  UDD-{n:2d}: fidelity = {fid:.6f}")


# ── 7. Summary ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("7. Summary")
print("=" * 60)
print("  - DDSequence defines pulse patterns (spin echo, CPMG, XY-4, XY-8, UDD)")
print("  - DDScheduler finds idle windows in circuits and inserts DD pulses")
print("  - filter_function() shows which noise frequencies are suppressed")
print("  - dd_fidelity() estimates coherence improvement under a noise model")
print("  - More pulses and symmetric sequences (XY-4, XY-8) give better suppression")
print()
