"""Qubit routing and mapping for hardware-constrained backends.

Implements topology-aware SWAP insertion so circuits can execute on
hardware with limited qubit connectivity. The SABRE algorithm
(Li et al., 2019) uses bidirectional heuristic search with lookahead.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..core.circuit import QuantumCircuit, Instruction
from ..core.gate import SWAP, CNOT, Gate


class CouplingMap:
    """Physical qubit connectivity graph.

    Stores an undirected graph of physical qubit connections and
    provides BFS-based shortest path and distance queries.
    """

    def __init__(self, edges: list[tuple[int, int]]) -> None:
        self._adj: dict[int, set[int]] = {}
        for a, b in edges:
            self._adj.setdefault(a, set()).add(b)
            self._adj.setdefault(b, set()).add(a)
        self._num_qubits = max(max(a, b) for a, b in edges) + 1 if edges else 0
        self._dist: dict[tuple[int, int], int] | None = None

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def edges(self) -> list[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        result = []
        for a, neighbors in sorted(self._adj.items()):
            for b in sorted(neighbors):
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    result.append(key)
        return result

    def neighbors(self, qubit: int) -> set[int]:
        return set(self._adj.get(qubit, set()))

    def is_connected(self, q1: int, q2: int) -> bool:
        return q2 in self._adj.get(q1, set())

    def _build_distance_matrix(self) -> None:
        self._dist = {}
        for src in self._adj:
            visited = {src: 0}
            queue = deque([src])
            while queue:
                node = queue.popleft()
                for neighbor in self._adj.get(node, set()):
                    if neighbor not in visited:
                        visited[neighbor] = visited[node] + 1
                        queue.append(neighbor)
            for dst, d in visited.items():
                self._dist[(src, dst)] = d

    def distance(self, q1: int, q2: int) -> int:
        if self._dist is None:
            self._build_distance_matrix()
        return self._dist.get((q1, q2), float('inf'))

    def shortest_path(self, q1: int, q2: int) -> list[int]:
        if q1 == q2:
            return [q1]
        visited = {q1: None}
        queue = deque([q1])
        while queue:
            node = queue.popleft()
            for neighbor in self._adj.get(node, set()):
                if neighbor not in visited:
                    visited[neighbor] = node
                    if neighbor == q2:
                        path = []
                        cur = q2
                        while cur is not None:
                            path.append(cur)
                            cur = visited[cur]
                        return path[::-1]
                    queue.append(neighbor)
        return []

    @classmethod
    def from_linear(cls, n: int) -> CouplingMap:
        edges = [(i, i + 1) for i in range(n - 1)]
        return cls(edges)

    @classmethod
    def from_grid(cls, rows: int, cols: int) -> CouplingMap:
        edges = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if c + 1 < cols:
                    edges.append((idx, idx + 1))
                if r + 1 < rows:
                    edges.append((idx, idx + cols))
        return cls(edges)

    @classmethod
    def from_heavy_hex(cls, n: int) -> CouplingMap:
        edges = []
        for i in range(n - 1):
            edges.append((i, i + 1))
        bridge_start = n
        for i in range(0, n - 1, 2):
            edges.append((i, bridge_start))
            edges.append((bridge_start, i + 1))
            bridge_start += 1
        return cls(edges)

    def __repr__(self) -> str:
        return f"CouplingMap({self.num_qubits} qubits, {len(self.edges)} edges)"


class Layout:
    """Bidirectional mapping between logical and physical qubits."""

    def __init__(self, logical_to_physical: dict[int, int]) -> None:
        self._l2p = dict(logical_to_physical)
        self._p2l = {p: l for l, p in self._l2p.items()}

    @classmethod
    def trivial(cls, n: int) -> Layout:
        return cls({i: i for i in range(n)})

    def logical_to_physical(self, logical: int) -> int:
        return self._l2p[logical]

    def physical_to_logical(self, physical: int) -> int:
        return self._p2l[physical]

    def swap(self, p1: int, p2: int) -> None:
        l1 = self._p2l.get(p1)
        l2 = self._p2l.get(p2)
        if l1 is not None:
            self._l2p[l1] = p2
        if l2 is not None:
            self._l2p[l2] = p1
        self._p2l[p1] = l2
        self._p2l[p2] = l1

    def copy(self) -> Layout:
        return Layout(dict(self._l2p))

    @property
    def mapping(self) -> dict[int, int]:
        return dict(self._l2p)

    def __repr__(self) -> str:
        return f"Layout({self._l2p})"


@dataclass
class RoutingResult:
    """Result of routing a circuit onto a coupling map."""
    circuit: QuantumCircuit
    initial_layout: Layout
    final_layout: Layout
    swap_count: int
    original_gate_count: int
    routed_gate_count: int


class SABRERouter:
    """SABRE routing algorithm (Li et al., 2019).

    Uses forward and backward passes with lookahead scoring to
    minimize SWAP insertions when mapping a circuit onto hardware
    with limited connectivity.
    """

    def __init__(self, seed: int | None = None, max_iterations: int = 3):
        self._seed = seed
        self._max_iterations = max_iterations

    def route(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
        initial_layout: Layout | None = None,
    ) -> RoutingResult:
        if initial_layout is None:
            initial_layout = Layout.trivial(circuit.num_qubits)

        instructions = circuit.instructions

        single_q_only = all(inst.gate.num_qubits == 1 for inst in instructions)
        if single_q_only:
            routed = QuantumCircuit(coupling_map.num_qubits)
            layout = initial_layout.copy()
            for inst in instructions:
                (lq,) = inst.target_qubits
                pq = layout.logical_to_physical(lq)
                routed.apply(inst.gate, pq)
            return RoutingResult(
                circuit=routed,
                initial_layout=initial_layout.copy(),
                final_layout=layout,
                swap_count=0,
                original_gate_count=len(instructions),
                routed_gate_count=len(instructions),
            )

        rng = np.random.RandomState(self._seed)
        best_result = None
        best_initial = None

        for iteration in range(self._max_iterations):
            if iteration % 2 == 0:
                dag = list(instructions)
            else:
                dag = list(reversed(instructions))

            start_layout = initial_layout.copy() if iteration == 0 else best_layout.copy()
            routed_ops, swap_count, final_layout = self._route_pass(
                dag, coupling_map, start_layout, rng
            )

            if iteration % 2 == 1:
                routed_ops = list(reversed(routed_ops))
                start_layout, final_layout = final_layout, start_layout

            if best_result is None or swap_count < best_result[1]:
                best_result = (routed_ops, swap_count, final_layout)
                best_initial = start_layout.copy()
                best_layout = final_layout.copy()

        routed_ops, swap_count, final_layout = best_result

        routed = QuantumCircuit(coupling_map.num_qubits)
        for gate, phys_qubits in routed_ops:
            routed.apply(gate, *phys_qubits)

        return RoutingResult(
            circuit=routed,
            initial_layout=best_initial,
            final_layout=final_layout,
            swap_count=swap_count,
            original_gate_count=len(instructions),
            routed_gate_count=len(routed_ops),
        )

    def _route_pass(
        self,
        instructions: list[Instruction],
        coupling_map: CouplingMap,
        layout: Layout,
        rng: np.random.RandomState,
    ) -> tuple[list[tuple[Gate, tuple[int, ...]]], int, Layout]:
        routed: list[tuple[Gate, tuple[int, ...]]] = []
        swap_count = 0
        remaining = list(range(len(instructions)))

        qubit_deps: dict[int, int] = {}
        resolved = [False] * len(instructions)

        while remaining:
            front_layer = self._get_front_layer(instructions, remaining, resolved, qubit_deps)

            executed = []
            for idx in front_layer:
                inst = instructions[idx]
                if inst.gate.num_qubits == 1:
                    (lq,) = inst.target_qubits
                    pq = layout.logical_to_physical(lq)
                    routed.append((inst.gate, (pq,)))
                    executed.append(idx)
                elif inst.gate.num_qubits == 2:
                    lq0, lq1 = inst.target_qubits
                    pq0 = layout.logical_to_physical(lq0)
                    pq1 = layout.logical_to_physical(lq1)
                    if coupling_map.is_connected(pq0, pq1):
                        routed.append((inst.gate, (pq0, pq1)))
                        executed.append(idx)

            if executed:
                for idx in executed:
                    resolved[idx] = True
                    remaining.remove(idx)
                    inst = instructions[idx]
                    for q in inst.target_qubits:
                        if qubit_deps.get(q) == idx:
                            del qubit_deps[q]
                continue

            unresolved_2q = [
                idx for idx in front_layer
                if instructions[idx].gate.num_qubits == 2
            ]

            if not unresolved_2q:
                for idx in front_layer:
                    inst = instructions[idx]
                    if inst.gate.num_qubits == 1:
                        (lq,) = inst.target_qubits
                        pq = layout.logical_to_physical(lq)
                        routed.append((inst.gate, (pq,)))
                        resolved[idx] = True
                        remaining.remove(idx)
                        if qubit_deps.get(lq) == idx:
                            del qubit_deps[lq]
                continue

            extended_set = self._get_extended_set(
                instructions, remaining, resolved, qubit_deps, front_layer, depth=20
            )

            best_swap = None
            best_score = float('inf')

            candidate_phys = set()
            for idx in unresolved_2q:
                inst = instructions[idx]
                lq0, lq1 = inst.target_qubits
                pq0 = layout.logical_to_physical(lq0)
                pq1 = layout.logical_to_physical(lq1)
                candidate_phys.update(coupling_map.neighbors(pq0))
                candidate_phys.update(coupling_map.neighbors(pq1))
                candidate_phys.add(pq0)
                candidate_phys.add(pq1)

            swap_candidates = []
            for p in candidate_phys:
                for neighbor in coupling_map.neighbors(p):
                    key = (min(p, neighbor), max(p, neighbor))
                    swap_candidates.append(key)
            swap_candidates = list(set(swap_candidates))

            for p1, p2 in swap_candidates:
                score = self._score_swap(
                    p1, p2, layout, coupling_map, instructions,
                    unresolved_2q, extended_set
                )
                if score < best_score:
                    best_score = score
                    best_swap = (p1, p2)
                elif score == best_score and best_swap is not None:
                    if rng.random() < 0.5:
                        best_swap = (p1, p2)

            if best_swap is None:
                break

            p1, p2 = best_swap
            routed.append((SWAP, (p1, p2)))
            layout.swap(p1, p2)
            swap_count += 1

        return routed, swap_count, layout

    def _get_front_layer(
        self,
        instructions: list[Instruction],
        remaining: list[int],
        resolved: list[bool],
        qubit_deps: dict[int, int],
    ) -> list[int]:
        front = []
        for idx in remaining:
            inst = instructions[idx]
            can_execute = True
            for q in inst.target_qubits:
                dep = qubit_deps.get(q)
                if dep is not None and dep != idx:
                    can_execute = False
                    break
            if can_execute:
                front.append(idx)
                for q in inst.target_qubits:
                    if q not in qubit_deps:
                        qubit_deps[q] = idx
        return front

    def _get_extended_set(
        self,
        instructions: list[Instruction],
        remaining: list[int],
        resolved: list[bool],
        qubit_deps: dict[int, int],
        front_layer: list[int],
        depth: int = 20,
    ) -> list[int]:
        extended = []
        front_set = set(front_layer)
        temp_resolved = dict(enumerate(resolved))
        temp_deps = dict(qubit_deps)

        for idx in front_layer:
            temp_resolved[idx] = True
            inst = instructions[idx]
            for q in inst.target_qubits:
                if temp_deps.get(q) == idx:
                    del temp_deps[q]

        count = 0
        temp_remaining = [i for i in remaining if i not in front_set]

        while temp_remaining and count < depth:
            next_front = []
            for idx in temp_remaining:
                inst = instructions[idx]
                can_execute = True
                for q in inst.target_qubits:
                    dep = temp_deps.get(q)
                    if dep is not None and dep != idx:
                        can_execute = False
                        break
                if can_execute:
                    next_front.append(idx)
                    for q in inst.target_qubits:
                        if q not in temp_deps:
                            temp_deps[q] = idx

            if not next_front:
                break

            for idx in next_front:
                if instructions[idx].gate.num_qubits == 2:
                    extended.append(idx)
                    count += 1
                temp_resolved[idx] = True
                inst = instructions[idx]
                for q in inst.target_qubits:
                    if temp_deps.get(q) == idx:
                        del temp_deps[q]
                temp_remaining.remove(idx)

        return extended

    def _score_swap(
        self,
        p1: int,
        p2: int,
        layout: Layout,
        coupling_map: CouplingMap,
        instructions: list[Instruction],
        front_layer_indices: list[int],
        extended_set_indices: list[int],
    ) -> float:
        temp_layout = layout.copy()
        temp_layout.swap(p1, p2)

        W_FRONT = 1.0
        W_EXTENDED = 0.5

        score = 0.0
        for idx in front_layer_indices:
            inst = instructions[idx]
            lq0, lq1 = inst.target_qubits
            pq0_before = layout.logical_to_physical(lq0)
            pq1_before = layout.logical_to_physical(lq1)
            pq0_after = temp_layout.logical_to_physical(lq0)
            pq1_after = temp_layout.logical_to_physical(lq1)
            dist_before = coupling_map.distance(pq0_before, pq1_before)
            dist_after = coupling_map.distance(pq0_after, pq1_after)
            score += W_FRONT * (dist_after - dist_before)

        for idx in extended_set_indices:
            inst = instructions[idx]
            lq0, lq1 = inst.target_qubits
            pq0_before = layout.logical_to_physical(lq0)
            pq1_before = layout.logical_to_physical(lq1)
            pq0_after = temp_layout.logical_to_physical(lq0)
            pq1_after = temp_layout.logical_to_physical(lq1)
            dist_before = coupling_map.distance(pq0_before, pq1_before)
            dist_after = coupling_map.distance(pq0_after, pq1_after)
            score += W_EXTENDED * (dist_after - dist_before)

        return score


def route(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    initial_layout: Layout | None = None,
    seed: int | None = None,
) -> RoutingResult:
    """Route a circuit onto a coupling map using SABRE.

    Args:
        circuit: The logical circuit to route.
        coupling_map: Physical qubit connectivity.
        initial_layout: Starting logical-to-physical mapping.
            Defaults to trivial (identity) layout.
        seed: Random seed for tie-breaking.

    Returns:
        RoutingResult with the routed circuit and metadata.
    """
    router = SABRERouter(seed=seed)
    return router.route(circuit, coupling_map, initial_layout)
