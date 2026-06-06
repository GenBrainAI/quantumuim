"""Circuit ↔ ZX-diagram conversion."""

from __future__ import annotations

import math

import numpy as np

from .diagram import HADAMARD, REGULAR, ZXDiagram, _norm, _phase_eq


# ---------------------------------------------------------------------------
# Circuit → ZX
# ---------------------------------------------------------------------------

def circuit_to_zx(circuit) -> ZXDiagram:
    n = circuit.num_qubits
    diag = ZXDiagram(n)
    current = list(diag.inputs)
    pending_h = [False] * n

    for inst in circuit.instructions:
        _process_gate(diag, inst.gate, inst.target_qubits, current, pending_h)

    for q in range(n):
        if pending_h[q]:
            diag.remove_edge(current[q], diag.outputs[q])
            diag.add_edge(current[q], diag.outputs[q], HADAMARD)

    return diag


def _wire_add(diag: ZXDiagram, qubit: int, spider_id: int,
              current: list[int], pending_h: list[bool]) -> None:
    cur = current[qubit]
    out = diag.outputs[qubit]
    edge_in = HADAMARD if pending_h[qubit] else REGULAR
    diag.remove_edge(cur, out)
    diag.add_edge(cur, spider_id, edge_in)
    diag.add_edge(spider_id, out, REGULAR)
    wire = diag._wires[qubit]
    pos = wire.index(out)
    wire.insert(pos, spider_id)
    diag._qubit[spider_id] = qubit
    current[qubit] = spider_id
    pending_h[qubit] = False


def _process_gate(diag, gate, qubits, current, pending_h):
    name = gate.name
    nq = gate.num_qubits

    if nq == 1:
        q = qubits[0]
        if name == "H":
            pending_h[q] = not pending_h[q]
            return
        if name == "I":
            return
        phase_z = _detect_z_phase(gate)
        if phase_z is not None:
            if abs(_norm(phase_z)) > 1e-12:
                s = diag.add_spider("Z", phase_z)
                _wire_add(diag, q, s, current, pending_h)
            return
        phase_x = _detect_x_phase(gate)
        if phase_x is not None:
            if abs(_norm(phase_x)) > 1e-12:
                s = diag.add_spider("X", phase_x)
                _wire_add(diag, q, s, current, pending_h)
            return
        if name == "Y":
            s1 = diag.add_spider("Z", math.pi)
            _wire_add(diag, q, s1, current, pending_h)
            s2 = diag.add_spider("X", math.pi)
            _wire_add(diag, q, s2, current, pending_h)
            return
        _decompose_single_qubit(diag, gate, q, current, pending_h)
        return

    if nq == 2:
        q0, q1 = qubits
        if name == "CNOT":
            zs = diag.add_spider("Z", 0.0)
            _wire_add(diag, q0, zs, current, pending_h)
            xs = diag.add_spider("X", 0.0)
            _wire_add(diag, q1, xs, current, pending_h)
            diag.add_edge(zs, xs, REGULAR)
            return
        if name == "CZ":
            zs0 = diag.add_spider("Z", 0.0)
            _wire_add(diag, q0, zs0, current, pending_h)
            zs1 = diag.add_spider("Z", 0.0)
            _wire_add(diag, q1, zs1, current, pending_h)
            diag.add_edge(zs0, zs1, HADAMARD)
            return
        if name == "SWAP":
            for _ in range(3):
                ctrl, tgt = (q0, q1) if _ % 2 == 0 else (q1, q0)
                zs = diag.add_spider("Z", 0.0)
                _wire_add(diag, ctrl, zs, current, pending_h)
                xs = diag.add_spider("X", 0.0)
                _wire_add(diag, tgt, xs, current, pending_h)
                diag.add_edge(zs, xs, REGULAR)
            return
        _decompose_two_qubit(diag, gate, qubits, current, pending_h)
        return

    raise ValueError(
        f"Gate {name} ({nq} qubits) not supported. Decompose to 1- and 2-qubit gates first."
    )


def _detect_z_phase(gate) -> float | None:
    name = gate.name
    _map = {"Z": math.pi, "S": math.pi / 2, "T": math.pi / 4}
    if name in _map:
        return _map[name]
    m = gate._matrix
    if abs(m[0, 1]) < 1e-10 and abs(m[1, 0]) < 1e-10:
        return float(np.angle(m[1, 1]) - np.angle(m[0, 0]))
    return None


def _detect_x_phase(gate) -> float | None:
    name = gate.name
    if name == "X":
        return math.pi
    H_mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    conj = H_mat @ gate._matrix @ H_mat
    if abs(conj[0, 1]) < 1e-10 and abs(conj[1, 0]) < 1e-10:
        return float(np.angle(conj[1, 1]) - np.angle(conj[0, 0]))
    return None


def _decompose_single_qubit(diag, gate, qubit, current, pending_h):
    m = gate._matrix
    a = m[0, 0]
    cos_half = abs(a)
    if cos_half > 1 - 1e-12:
        total = float(np.angle(m[1, 1]) - np.angle(m[0, 0]))
        if abs(_norm(total)) > 1e-12:
            s = diag.add_spider("Z", total)
            _wire_add(diag, qubit, s, current, pending_h)
        return
    beta = 2 * np.arccos(np.clip(cos_half, 0, 1))
    alpha_plus_gamma = float(np.angle(m[1, 1]) - np.angle(m[0, 0]))
    b = m[0, 1]
    c = m[1, 0]
    if abs(b) > 1e-12:
        alpha_minus_gamma = float(np.angle(c) - np.angle(-b))
    else:
        alpha_minus_gamma = 0.0
    alpha = (alpha_plus_gamma + alpha_minus_gamma) / 2
    gamma = (alpha_plus_gamma - alpha_minus_gamma) / 2
    za = alpha - math.pi / 2
    xb = beta
    zc = gamma + math.pi / 2
    if abs(_norm(zc)) > 1e-12:
        s = diag.add_spider("Z", zc)
        _wire_add(diag, qubit, s, current, pending_h)
    if abs(_norm(xb)) > 1e-12:
        s = diag.add_spider("X", xb)
        _wire_add(diag, qubit, s, current, pending_h)
    if abs(_norm(za)) > 1e-12:
        s = diag.add_spider("Z", za)
        _wire_add(diag, qubit, s, current, pending_h)


def _decompose_two_qubit(diag, gate, qubits, current, pending_h):
    q0, q1 = qubits
    m = gate._matrix
    from quantumium.core.gate import CNOT as CNOT_GATE, CZ as CZ_GATE
    if np.allclose(m, CZ_GATE._matrix, atol=1e-10):
        zs0 = diag.add_spider("Z", 0.0)
        _wire_add(diag, q0, zs0, current, pending_h)
        zs1 = diag.add_spider("Z", 0.0)
        _wire_add(diag, q1, zs1, current, pending_h)
        diag.add_edge(zs0, zs1, HADAMARD)
        return
    if np.allclose(m, CNOT_GATE._matrix, atol=1e-10):
        zs = diag.add_spider("Z", 0.0)
        _wire_add(diag, q0, zs, current, pending_h)
        xs = diag.add_spider("X", 0.0)
        _wire_add(diag, q1, xs, current, pending_h)
        diag.add_edge(zs, xs, REGULAR)
        return
    pending_h[q0] = not pending_h[q0]
    pending_h[q1] = not pending_h[q1]
    zs0 = diag.add_spider("Z", 0.0)
    _wire_add(diag, q0, zs0, current, pending_h)
    zs1 = diag.add_spider("Z", 0.0)
    _wire_add(diag, q1, zs1, current, pending_h)
    diag.add_edge(zs0, zs1, HADAMARD)
    pending_h[q0] = not pending_h[q0]
    pending_h[q1] = not pending_h[q1]


# ---------------------------------------------------------------------------
# ZX → Circuit
# ---------------------------------------------------------------------------

def zx_to_circuit(diag: ZXDiagram):
    from quantumium.core.circuit import QuantumCircuit
    from quantumium.core.gate import H as H_GATE, CNOT as CNOT_GATE, CZ as CZ_GATE

    n = diag.num_qubits
    qc = QuantumCircuit(n)
    wires = [diag.wire(q) for q in range(n)]
    times = _compute_spider_times(diag, wires)
    emitted: set[tuple[int, int]] = set()
    events: list[tuple] = []

    for q in range(n):
        wire = wires[q]
        for i in range(1, len(wire)):
            node = wire[i]
            prev = wire[i - 1]
            t = times.get(node, i)
            et_in = diag.edge_type(prev, node) if diag.has_edge(prev, node) else REGULAR

            if node in diag.outputs:
                if et_in == HADAMARD:
                    events.append((t - 0.3, "H", q))
                continue

            if et_in == HADAMARD:
                events.append((t - 0.3, "H", q))

            phase = diag.phase(node)
            color = diag.color(node)
            if abs(phase) > 1e-10:
                events.append((t, "phase", color, phase, q))

            for nb in diag.neighbors(node):
                if nb in wire:
                    continue
                if nb in diag.inputs or nb in diag.outputs:
                    continue
                pair_key = (min(node, nb), max(node, nb))
                if pair_key in emitted:
                    continue
                nb_qubit = diag.qubit_of(nb)
                if nb_qubit < 0:
                    continue
                gate_t = max(t, times.get(nb, t)) - 0.1
                et_cross = diag.edge_type(node, nb)
                cnb = diag.color(nb)
                if color == "Z" and cnb == "X" and et_cross == REGULAR:
                    events.append((gate_t, "CNOT", q, nb_qubit))
                elif color == "X" and cnb == "Z" and et_cross == REGULAR:
                    events.append((gate_t, "CNOT", nb_qubit, q))
                elif et_cross == HADAMARD:
                    events.append((gate_t, "CZ", min(q, nb_qubit), max(q, nb_qubit)))
                else:
                    events.append((gate_t, "CZ", min(q, nb_qubit), max(q, nb_qubit)))
                emitted.add(pair_key)

    _TYPE_ORDER = {"H": 0, "phase": 1, "CNOT": 2, "CZ": 2}
    events.sort(key=lambda e: (e[0], _TYPE_ORDER.get(e[1], 3)))

    for ev in events:
        kind = ev[1]
        if kind == "H":
            qc.apply(H_GATE, ev[2])
        elif kind == "phase":
            g = _phase_to_gate(ev[2], ev[3])
            if g is not None:
                qc.apply(g, ev[4])
        elif kind == "CNOT":
            qc.apply(CNOT_GATE, ev[2], ev[3])
        elif kind == "CZ":
            qc.apply(CZ_GATE, ev[2], ev[3])

    return qc


def _compute_spider_times(diag: ZXDiagram, wires: list[list[int]]) -> dict[int, float]:
    times: dict[int, float] = {}
    n = len(wires)
    for q in range(n):
        for i, node in enumerate(wires[q]):
            times[node] = float(i)

    for _ in range(20):
        changed = False
        for q in range(n):
            wire = wires[q]
            for i in range(1, len(wire)):
                node = wire[i]
                new_t = times[wire[i - 1]] + 1
                for j in range(1, i):
                    prev_s = wire[j]
                    for nb in diag.neighbors(prev_s):
                        if nb in wire or nb in diag.inputs or nb in diag.outputs:
                            continue
                        gate_t = max(times.get(prev_s, 0), times.get(nb, 0))
                        new_t = max(new_t, gate_t + 1)
                if times[node] < new_t:
                    times[node] = new_t
                    changed = True
        if not changed:
            break

    return times


def _phase_to_gate(color: str, phase: float):
    from quantumium.core.gate import X, Z, S, T, Rz, Rx
    p = _norm(phase)
    if p < 1e-10:
        return None
    if color == "Z":
        if _phase_eq(p, math.pi):
            return Z
        if _phase_eq(p, math.pi / 2):
            return S
        if _phase_eq(p, math.pi / 4):
            return T
        return Rz(phase)
    if _phase_eq(p, math.pi):
        return X
    return Rx(phase)


# ---------------------------------------------------------------------------
# One-liner optimiser
# ---------------------------------------------------------------------------

def zx_optimize(circuit):
    diag = circuit_to_zx(circuit)
    diag.simplify()
    return zx_to_circuit(diag)


# ---------------------------------------------------------------------------
# Unitary helper (for equivalence checks on small circuits)
# ---------------------------------------------------------------------------

def _circuit_unitary(circuit):
    n = circuit.num_qubits
    dim = 2 ** n
    U = np.eye(dim, dtype=complex)
    for inst in circuit.instructions:
        G = circuit._expand_gate(inst.gate._matrix, inst.target_qubits, n)
        U = G @ U
    return U
