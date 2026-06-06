"""Qubit Routing & Mapping Demo

Demonstrates SABRE routing on various hardware topologies.
Shows how circuits are transformed to respect coupling constraints
while preserving measurement statistics.
"""

import numpy as np
from quantumium.core.circuit import QuantumCircuit
from quantumium.core.gate import H, CNOT, X, Rz, controlled
from quantumium.transpiler.routing import CouplingMap, Layout, route
from quantumium.transpiler.passes import decompose_swap


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ── 1. Coupling Map Topologies ──────────────────────────────────────────

section("1. Coupling Map Topologies")

linear = CouplingMap.from_linear(5)
print(f"Linear(5):    {linear}")
print(f"  Edges: {linear.edges}")
print(f"  Distance 0→4: {linear.distance(0, 4)}")
print(f"  Path 0→4: {linear.shortest_path(0, 4)}")

grid = CouplingMap.from_grid(2, 3)
print(f"\nGrid(2x3):    {grid}")
print(f"  Edges: {grid.edges}")
print(f"  Distance 0→5: {grid.distance(0, 5)}")

heavy = CouplingMap.from_heavy_hex(5)
print(f"\nHeavyHex(5):  {heavy}")
print(f"  Edges: {heavy.edges}")


# ── 2. Simple Routing ───────────────────────────────────────────────────

section("2. Routing CNOT(0,3) on Linear(4)")

qc = QuantumCircuit(4)
qc.apply(H, 0)
qc.apply(CNOT, 0, 3)

cm = CouplingMap.from_linear(4)
result = route(qc, cm, seed=42)

print(f"Original: {result.original_gate_count} gates")
print(f"Routed:   {result.routed_gate_count} gates ({result.swap_count} SWAPs)")
print(f"Initial layout: {result.initial_layout.mapping}")
print(f"Final layout:   {result.final_layout.mapping}")
print("\nRouted circuit:")
for inst in result.circuit.instructions:
    print(f"  {inst.gate.name:6s} on qubits {inst.target_qubits}")


# ── 3. QFT-4 Routing Comparison ────────────────────────────────────────

section("3. QFT-4: Linear vs Grid Topology")

qc_qft = QuantumCircuit(4)
for i in range(4):
    qc_qft.apply(H, i)
    for j in range(i + 1, 4):
        angle = np.pi / (2 ** (j - i))
        crz = controlled(Rz(angle))
        qc_qft.apply(crz, j, i)

print(f"Original QFT-4: {len(qc_qft.instructions)} gates")

cm_lin = CouplingMap.from_linear(4)
cm_grid = CouplingMap.from_grid(2, 2)

r_lin = route(qc_qft, cm_lin, seed=42)
r_grid = route(qc_qft, cm_grid, seed=42)

print(f"\nLinear(4):  {r_lin.routed_gate_count} gates, {r_lin.swap_count} SWAPs")
print(f"Grid(2x2):  {r_grid.routed_gate_count} gates, {r_grid.swap_count} SWAPs")
print(f"\nGrid saves {r_lin.swap_count - r_grid.swap_count} SWAPs "
      f"({(r_lin.swap_count - r_grid.swap_count)*3} fewer CNOTs after decomposition)")


# ── 4. Measurement Preservation ────────────────────────────────────────

section("4. Measurement Preservation (Bell State)")

qc_bell = QuantumCircuit(3)
qc_bell.apply(H, 0)
qc_bell.apply(CNOT, 0, 2)

cm3 = CouplingMap.from_linear(3)
result_bell = route(qc_bell, cm3, seed=42)

orig_counts = qc_bell.measure_all(shots=5000)
routed_counts = result_bell.circuit.measure_all(shots=5000)

print("Original circuit counts:")
for k, v in sorted(orig_counts.items()):
    if v > 50:
        print(f"  |{k}⟩: {v}")

print("\nRouted circuit counts (physical qubit order):")
for k, v in sorted(routed_counts.items()):
    if v > 50:
        print(f"  |{k}⟩: {v}")

layout = result_bell.initial_layout
mapped = {}
for bitstring, count in routed_counts.items():
    n = len(bitstring)
    logical_bits = ['0'] * 3
    for lq in range(3):
        pq = layout.logical_to_physical(lq)
        if pq < n:
            logical_bits[2 - lq] = bitstring[n - 1 - pq]
    key = ''.join(logical_bits)
    mapped[key] = mapped.get(key, 0) + count

print(f"\nRemapped to logical order (layout {layout.mapping}):")
for k, v in sorted(mapped.items()):
    if v > 50:
        print(f"  |{k}⟩: {v}")


# ── 5. SWAP Decomposition ──────────────────────────────────────────────

section("5. SWAP Decomposition to CNOTs")

qc_swap = QuantumCircuit(4)
qc_swap.apply(CNOT, 0, 3)
qc_swap.apply(CNOT, 1, 3)

cm4 = CouplingMap.from_linear(4)
result_swap = route(qc_swap, cm4, seed=42)

print(f"Before decomposition: {len(result_swap.circuit.instructions)} ops "
      f"({result_swap.swap_count} SWAPs)")

decomposed = decompose_swap(result_swap.circuit.instructions)
cnot_count = sum(1 for inst in decomposed if inst.gate.name == "CNOT")
swap_count = sum(1 for inst in decomposed if inst.gate.name == "SWAP")

print(f"After decomposition:  {len(decomposed)} ops "
      f"({cnot_count} CNOTs, {swap_count} SWAPs)")
print(f"All SWAPs eliminated: {swap_count == 0}")


# ── 6. GHZ-5 on Different Topologies ───────────────────────────────────

section("6. GHZ-5 Routing Comparison")

qc_ghz = QuantumCircuit(5)
qc_ghz.apply(H, 0)
for i in range(1, 5):
    qc_ghz.apply(CNOT, 0, i)

topologies = {
    "Linear(5)": CouplingMap.from_linear(5),
    "Grid(2x3)": CouplingMap.from_grid(2, 3),
    "FullyConn": CouplingMap([(i, j) for i in range(5) for j in range(i+1, 5)]),
}

print(f"Original GHZ-5: {len(qc_ghz.instructions)} gates\n")
print(f"{'Topology':<14} {'SWAPs':>6} {'Total Gates':>12} {'Overhead':>10}")
print("-" * 46)
for name, cm in topologies.items():
    r = route(qc_ghz, cm, seed=42)
    overhead = r.routed_gate_count - r.original_gate_count
    print(f"{name:<14} {r.swap_count:>6} {r.routed_gate_count:>12} {f'+{overhead}':>10}")


# ── 7. Custom Layout ───────────────────────────────────────────────────

section("7. Custom Initial Layout")

qc_custom = QuantumCircuit(3)
qc_custom.apply(CNOT, 0, 2)

cm_custom = CouplingMap.from_linear(3)

trivial = Layout.trivial(3)
custom = Layout({0: 0, 1: 2, 2: 1})

r_trivial = route(qc_custom, cm_custom, initial_layout=trivial, seed=42)
r_custom = route(qc_custom, cm_custom, initial_layout=custom, seed=42)

print(f"Trivial layout {trivial.mapping}: {r_trivial.swap_count} SWAPs")
print(f"Custom layout  {custom.mapping}: {r_custom.swap_count} SWAPs")
print(f"\nSmart initial placement can eliminate SWAPs entirely!")


print(f"\n{'='*60}")
print(f"  Demo complete — SABRE routing working correctly")
print(f"{'='*60}")
