"""Minimum-weight perfect matching (MWPM) decoder for surface codes."""

from __future__ import annotations

from collections import deque

import numpy as np

from quantumium.surface_code.lattice import SurfaceCode


def _build_adjacency(stabilizers: list[list[int]]) -> dict[int, list[int]]:
    """Build stabilizer adjacency graph: two stabilizers are adjacent if they share a data qubit."""
    qubit_to_stabs: dict[int, list[int]] = {}
    for idx, stab in enumerate(stabilizers):
        for q in stab:
            qubit_to_stabs.setdefault(q, []).append(idx)
    adj: dict[int, list[int]] = {i: [] for i in range(len(stabilizers))}
    for stab_list in qubit_to_stabs.values():
        for i in range(len(stab_list)):
            for j in range(i + 1, len(stab_list)):
                a, b = stab_list[i], stab_list[j]
                if b not in adj[a]:
                    adj[a].append(b)
                    adj[b].append(a)
    return adj


def _find_boundary_stabs(
    stabilizers: list[list[int]],
    d: int,
    use_columns: bool,
) -> set[int]:
    """Find stabilizers adjacent to the rough boundary.

    For X stabilizers: rough boundaries are left/right (columns 0, d-1).
    For Z stabilizers: rough boundaries are top/bottom (rows 0, d-1).
    """
    boundary_stabs = set()
    for idx, stab in enumerate(stabilizers):
        for q in stab:
            if use_columns:
                col = q % d
                if col == 0 or col == d - 1:
                    boundary_stabs.add(idx)
                    break
            else:
                row = q // d
                if row == 0 or row == d - 1:
                    boundary_stabs.add(idx)
                    break
    return boundary_stabs


def _bfs_distances(
    adj: dict[int, list[int]],
    sources: list[int],
    n_nodes: int,
) -> dict[int, int]:
    """BFS from multiple sources, return distance to each reachable node."""
    dist: dict[int, int] = {}
    queue = deque()  # type: ignore[var-annotated]
    for s in sources:
        dist[s] = 0
        queue.append(s)
    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _shared_qubit(stab_a: list[int], stab_b: list[int]) -> int | None:
    """Return a qubit shared by both stabilizers, or None."""
    for q in stab_a:
        if q in stab_b:
            return q
    return None


def _correction_path(
    stab_a_idx: int,
    stab_b_idx: int | None,
    stabilizers: list[list[int]],
    adj: dict[int, list[int]],
    boundary_stabs: set[int],
    d: int,
    use_columns: bool,
) -> list[int]:
    """Find the correction qubits for a matched pair.

    If stab_b_idx is None, match stab_a to the boundary.
    """
    if stab_b_idx is not None:
        # Match two defects: find shortest path and pick one shared qubit per step.
        path = _bfs_path(adj, stab_a_idx, stab_b_idx)
        if path is None:
            return []
        correction = []
        for i in range(len(path) - 1):
            sq = _shared_qubit(stabilizers[path[i]], stabilizers[path[i + 1]])
            if sq is not None:
                correction.append(sq)
        return correction

    # Match to boundary: find path to nearest boundary-adjacent stabilizer.
    stab_a = stabilizers[stab_a_idx]
    if stab_a_idx in boundary_stabs:
        # Already at boundary — pick the qubit on the boundary edge.
        if use_columns:
            boundary_qs = [q for q in stab_a if q % d == 0 or q % d == d - 1]
        else:
            boundary_qs = [q for q in stab_a if q // d == 0 or q // d == d - 1]
        return [boundary_qs[0]] if boundary_qs else [stab_a[0]]

    # BFS to a boundary stabilizer.
    path = _bfs_path_to_set(adj, stab_a_idx, boundary_stabs)
    if path is None:
        return []
    correction = []
    for i in range(len(path) - 1):
        sq = _shared_qubit(stabilizers[path[i]], stabilizers[path[i + 1]])
        if sq is not None:
            correction.append(sq)
    # Also add the boundary exit qubit from the last stabilizer.
    last_stab = stabilizers[path[-1]]
    if use_columns:
        boundary_qs = [q for q in last_stab if q % d == 0 or q % d == d - 1]
    else:
        boundary_qs = [q for q in last_stab if q // d == 0 or q // d == d - 1]
    if boundary_qs:
        correction.append(boundary_qs[0])
    return correction


def _bfs_path(adj: dict[int, list[int]], start: int, end: int) -> list[int] | None:
    """BFS shortest path from start to end."""
    if start == end:
        return [start]
    visited = {start}
    parent: dict[int, int] = {}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                if v == end:
                    path = [v]
                    while path[-1] != start:
                        path.append(parent[path[-1]])
                    return list(reversed(path))
                queue.append(v)
    return None


def _bfs_path_to_set(
    adj: dict[int, list[int]],
    start: int,
    targets: set[int],
) -> list[int] | None:
    """BFS shortest path from start to any node in targets."""
    if start in targets:
        return [start]
    visited = {start}
    parent: dict[int, int] = {}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                if v in targets:
                    path = [v]
                    while path[-1] != start:
                        path.append(parent[path[-1]])
                    return list(reversed(path))
                queue.append(v)
    return None


def _greedy_matching(
    defects: list[int],
    adj: dict[int, list[int]],
    boundary_stabs: set[int],
) -> list[tuple[int, int | None]]:
    """Greedy minimum-weight matching of defects."""
    if not defects:
        return []

    n = len(defects)

    # Compute pairwise distances via BFS.
    pair_dist: dict[tuple[int, int], int] = {}
    bndry_dist: dict[int, int] = {}

    for i, d_i in enumerate(defects):
        dists = _bfs_distances(adj, [d_i], len(adj))
        for j in range(i + 1, n):
            d_j = defects[j]
            if d_j in dists:
                pair_dist[(i, j)] = dists[d_j]

        # Boundary distance.
        min_bd = float("inf")
        for bs in boundary_stabs:
            if bs in dists:
                bd = dists[bs] + 1  # +1 for the boundary exit qubit
                if bd < min_bd:
                    min_bd = bd
        if d_i in boundary_stabs:
            min_bd = 1  # already at boundary, just 1 exit qubit
        bndry_dist[i] = int(min_bd) if min_bd < float("inf") else 999

    # Build edge list.
    edges: list[tuple[int, int, int | None]] = []
    for i in range(n):
        edges.append((bndry_dist[i], i, None))
        for j in range(i + 1, n):
            if (i, j) in pair_dist:
                edges.append((pair_dist[(i, j)], i, j))
    edges.sort(key=lambda e: (e[0], e[2] is None, e[2] if e[2] is not None else 0))

    matched = [False] * n
    pairs: list[tuple[int, int | None]] = []
    for dist, i, j in edges:  # type: ignore[assignment]
        if matched[i]:
            continue
        if j is None:
            matched[i] = True
            pairs.append((defects[i], None))
        elif not matched[j]:
            matched[i] = True
            matched[j] = True
            pairs.append((defects[i], defects[j]))

    return pairs


def mwpm_decode(
    code: SurfaceCode,
    x_syndrome: np.ndarray,
    z_syndrome: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode syndromes and return (x_correction, z_correction) vectors."""
    d = code.distance

    # --- Z errors → X syndrome → correct with Z ---
    x_adj = _build_adjacency(code.x_stabilizers)
    x_boundary = _find_boundary_stabs(code.x_stabilizers, d, use_columns=True)
    x_defects = list(np.where(x_syndrome)[0])

    z_correction = np.zeros(code.n_data, dtype=int)
    pairs = _greedy_matching(x_defects, x_adj, x_boundary)
    for idx_a, idx_b in pairs:
        for q in _correction_path(
            idx_a, idx_b, code.x_stabilizers, x_adj, x_boundary, d, use_columns=True
        ):
            z_correction[q] ^= 1

    # --- X errors → Z syndrome → correct with X ---
    z_adj = _build_adjacency(code.z_stabilizers)
    z_boundary = _find_boundary_stabs(code.z_stabilizers, d, use_columns=False)
    z_defects = list(np.where(z_syndrome)[0])

    x_correction = np.zeros(code.n_data, dtype=int)
    pairs = _greedy_matching(z_defects, z_adj, z_boundary)
    for idx_a, idx_b in pairs:
        for q in _correction_path(
            idx_a, idx_b, code.z_stabilizers, z_adj, z_boundary, d, use_columns=False
        ):
            x_correction[q] ^= 1

    return x_correction, z_correction
