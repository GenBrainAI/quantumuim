"""ZX-calculus rewrite rules and simplification pipeline."""

from __future__ import annotations

import math

from .diagram import HADAMARD, REGULAR, ZXDiagram, _norm, _phase_eq


def spider_fusion(diag: ZXDiagram) -> bool:
    """Fuse adjacent same-color spiders connected by a regular edge on the same wire."""
    changed = False
    for q in range(diag.num_qubits):
        fused = True
        while fused:
            fused = False
            wire = diag._wires[q]
            for i in range(1, len(wire) - 2):
                s1 = wire[i]
                s2 = wire[i + 1]
                if diag.color(s1) == "B" or diag.color(s2) == "B":
                    continue
                if diag.color(s1) != diag.color(s2):
                    continue
                if not diag.has_edge(s1, s2):
                    continue
                if diag.edge_type(s1, s2) != REGULAR:
                    continue
                _fuse_pair(diag, q, i, s1, s2)
                fused = True
                changed = True
                break
    return changed


def _fuse_pair(diag: ZXDiagram, qubit: int, pos: int, keep: int, remove: int) -> None:
    new_phase = diag.phase(keep) + diag.phase(remove)
    diag.set_phase(keep, new_phase)
    diag.remove_edge(keep, remove)
    for nb in list(diag.neighbors(remove)):
        et = diag.edge_type(remove, nb)
        diag.remove_edge(remove, nb)
        if nb != keep:
            diag.add_edge(keep, nb, et)
    diag.remove_node(remove)
    diag._wires[qubit].remove(remove)


def identity_removal(diag: ZXDiagram) -> bool:
    """Remove phase-0 spiders with degree 2 and matching edge types (both regular or both H)."""
    changed = False
    for q in range(diag.num_qubits):
        removed = True
        while removed:
            removed = False
            wire = diag._wires[q]
            for i in range(1, len(wire) - 1):
                node = wire[i]
                if diag.color(node) == "B":
                    continue
                if abs(diag.phase(node)) > 1e-10:
                    continue
                if diag.degree(node) != 2:
                    continue
                nbs = diag.neighbors(node)
                if len(nbs) != 2:
                    continue
                et0 = diag.edge_type(node, nbs[0])
                et1 = diag.edge_type(node, nbs[1])
                if et0 != et1:
                    continue
                prev_node = wire[i - 1]
                next_node = wire[i + 1]
                diag.remove_edge(node, nbs[0])
                diag.remove_edge(node, nbs[1])
                diag.remove_node(node)
                wire.pop(i)
                diag.add_edge(prev_node, next_node, REGULAR)
                removed = True
                changed = True
                break
    return changed


def color_change(diag: ZXDiagram) -> bool:
    """Convert spiders where ALL edges are Hadamard to the opposite color with regular edges."""
    changed = False
    for nid in list(diag.spider_ids()):
        if nid not in diag._color:
            continue
        nbs = diag.neighbors(nid)
        if not nbs:
            continue
        all_h = all(diag.edge_type(nid, nb) == HADAMARD for nb in nbs)
        if not all_h:
            continue
        new_color = "X" if diag.color(nid) == "Z" else "Z"
        diag.set_color(nid, new_color)
        for nb in nbs:
            diag.remove_edge(nid, nb)
            diag.add_edge(nid, nb, REGULAR)
        changed = True
    return changed


def hadamard_cancel(diag: ZXDiagram) -> bool:
    """Cancel pairs of Hadamard edges between same-color degree-2 phaseless spiders."""
    changed = False
    for q in range(diag.num_qubits):
        wire = diag._wires[q]
        cancelled = True
        while cancelled:
            cancelled = False
            wire = diag._wires[q]
            for i in range(1, len(wire) - 2):
                s1 = wire[i]
                s2 = wire[i + 1]
                if diag.color(s1) == "B" or diag.color(s2) == "B":
                    continue
                if diag.degree(s1) != 2 or diag.degree(s2) != 2:
                    continue
                if abs(diag.phase(s1)) > 1e-10 or abs(diag.phase(s2)) > 1e-10:
                    continue
                if not diag.has_edge(s1, s2):
                    continue
                et = diag.edge_type(s1, s2)
                if et != HADAMARD:
                    continue
                prev_node = wire[i - 1]
                next_node = wire[i + 2]
                ep = diag.edge_type(prev_node, s1) if diag.has_edge(prev_node, s1) else REGULAR
                en = diag.edge_type(s2, next_node) if diag.has_edge(s2, next_node) else REGULAR
                for nd in [s1, s2]:
                    for nb in list(diag.neighbors(nd)):
                        diag.remove_edge(nd, nb)
                    diag.remove_node(nd)
                wire.pop(i)
                wire.pop(i)
                new_et = ep ^ en ^ HADAMARD
                diag.add_edge(prev_node, next_node, new_et)
                cancelled = True
                changed = True
                break
    return changed


def full_simplify(diag: ZXDiagram) -> None:
    """Apply all simplification rules until fixed point."""
    for _ in range(50):
        progress = False
        progress |= spider_fusion(diag)
        progress |= identity_removal(diag)
        progress |= color_change(diag)
        progress |= spider_fusion(diag)
        progress |= identity_removal(diag)
        if not progress:
            break
