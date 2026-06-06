"""ZX-diagram graph data structure."""

from __future__ import annotations

import math

REGULAR = 0
HADAMARD = 1


def _norm(phase: float) -> float:
    TWO_PI = 2 * math.pi
    p = phase % TWO_PI
    for k in range(9):
        target = k * math.pi / 4
        if abs(p - target) < 1e-10:
            return 0.0 if k == 8 else target
    return p


def _phase_eq(a: float, b: float) -> bool:
    return abs(_norm(a) - _norm(b)) < 1e-10


class ZXDiagram:
    """Graph-based ZX-diagram for quantum circuit optimization.

    Nodes are spiders (Z/X) or boundary nodes (B).
    Edges are regular or Hadamard.
    Qubit wires track the linear ordering per qubit for extraction.
    """

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self._next_id = 0
        self._color: dict[int, str] = {}
        self._phase: dict[int, float] = {}
        self._adj: dict[int, dict[int, int]] = {}
        self._qubit: dict[int, int] = {}
        self.inputs: list[int] = []
        self.outputs: list[int] = []
        self._wires: list[list[int]] = []

        for i in range(num_qubits):
            inp = self._add_node("B", 0.0)
            self.inputs.append(inp)
            self._qubit[inp] = i
        for i in range(num_qubits):
            out = self._add_node("B", 0.0)
            self.outputs.append(out)
            self._qubit[out] = i
        for i in range(num_qubits):
            self._wires.append([self.inputs[i], self.outputs[i]])
            self.add_edge(self.inputs[i], self.outputs[i], REGULAR)

    # ---- node ops ----

    def _add_node(self, color: str, phase: float = 0.0) -> int:
        nid = self._next_id
        self._next_id += 1
        self._color[nid] = color
        self._phase[nid] = _norm(phase)
        self._adj[nid] = {}
        return nid

    def add_spider(self, color: str, phase: float = 0.0) -> int:
        return self._add_node(color, phase)

    def remove_node(self, nid: int) -> None:
        for nb in list(self._adj.get(nid, {})):
            self._adj[nb].pop(nid, None)
        self._adj.pop(nid, None)
        self._color.pop(nid, None)
        self._phase.pop(nid, None)
        self._qubit.pop(nid, None)

    # ---- edge ops ----

    def add_edge(self, n1: int, n2: int, edge_type: int = REGULAR) -> None:
        if n2 in self._adj.get(n1, {}):
            existing = self._adj[n1][n2]
            if existing == edge_type:
                del self._adj[n1][n2]
                del self._adj[n2][n1]
                return
        self._adj[n1][n2] = edge_type
        self._adj[n2][n1] = edge_type

    def remove_edge(self, n1: int, n2: int) -> None:
        self._adj.get(n1, {}).pop(n2, None)
        self._adj.get(n2, {}).pop(n1, None)

    def has_edge(self, n1: int, n2: int) -> bool:
        return n2 in self._adj.get(n1, {})

    def edge_type(self, n1: int, n2: int) -> int:
        return self._adj[n1][n2]

    # ---- queries ----

    def neighbors(self, nid: int) -> list[int]:
        return list(self._adj.get(nid, {}).keys())

    def degree(self, nid: int) -> int:
        return len(self._adj.get(nid, {}))

    def color(self, nid: int) -> str:
        return self._color[nid]

    def phase(self, nid: int) -> float:
        return self._phase[nid]

    def set_phase(self, nid: int, phase: float) -> None:
        self._phase[nid] = _norm(phase)

    def set_color(self, nid: int, color: str) -> None:
        self._color[nid] = color

    def qubit_of(self, nid: int) -> int:
        return self._qubit.get(nid, -1)

    def wire(self, qubit: int) -> list[int]:
        return list(self._wires[qubit])

    def spider_ids(self) -> list[int]:
        return [n for n in self._color if self._color[n] != "B"]

    def t_count(self) -> int:
        count = 0
        pi4 = math.pi / 4
        for nid in self.spider_ids():
            p = self._phase[nid]
            if _phase_eq(p, pi4) or _phase_eq(p, 7 * pi4):
                count += 1
        return count

    # ---- wire management ----

    def wire_insert(self, qubit: int, spider_id: int, edge_in_type: int = REGULAR) -> None:
        """Insert spider at end of qubit wire (just before output boundary)."""
        wire = self._wires[qubit]
        pos = len(wire) - 1
        prev_node = wire[pos - 1]
        out_node = wire[pos]
        prev_et = self._adj.get(prev_node, {}).get(out_node, REGULAR)
        self.remove_edge(prev_node, out_node)
        self.add_edge(prev_node, spider_id, edge_in_type)
        self.add_edge(spider_id, out_node, REGULAR)
        wire.insert(pos, spider_id)
        self._qubit[spider_id] = qubit

    def wire_remove(self, qubit: int, spider_id: int) -> None:
        """Remove spider from qubit wire (does NOT touch graph edges)."""
        self._wires[qubit].remove(spider_id)

    # ---- structural ----

    def copy(self) -> ZXDiagram:
        d = ZXDiagram.__new__(ZXDiagram)
        d.num_qubits = self.num_qubits
        d._next_id = self._next_id
        d._color = dict(self._color)
        d._phase = dict(self._phase)
        d._adj = {n: dict(adj) for n, adj in self._adj.items()}
        d._qubit = dict(self._qubit)
        d.inputs = list(self.inputs)
        d.outputs = list(self.outputs)
        d._wires = [list(w) for w in self._wires]
        return d

    # ---- high-level API ----

    @classmethod
    def from_circuit(cls, circuit) -> ZXDiagram:
        from .convert import circuit_to_zx
        return circuit_to_zx(circuit)

    def to_circuit(self):
        from .convert import zx_to_circuit
        return zx_to_circuit(self)

    def simplify(self) -> ZXDiagram:
        from .rules import full_simplify
        full_simplify(self)
        return self

    def is_equivalent(self, other: ZXDiagram) -> bool:
        from .convert import _circuit_unitary
        import numpy as np
        u1 = _circuit_unitary(self.to_circuit())
        u2 = _circuit_unitary(other.to_circuit())
        if u1.shape != u2.shape:
            return False
        product = u1.conj().T @ u2
        diag = np.diag(product)
        if not np.allclose(np.abs(diag), 1.0, atol=1e-6):
            return False
        return bool(np.allclose(diag, diag[0], atol=1e-6))
