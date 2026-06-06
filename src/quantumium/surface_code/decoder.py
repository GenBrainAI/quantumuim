"""Minimum-weight perfect matching (MWPM) decoder for the surface code.

Uses a self-contained greedy matching implementation — no external
dependencies beyond numpy. For each syndrome, the decoder finds pairs
of defects (syndrome bits = 1) with minimum total weight matching,
where weight is the graph distance between defect positions on the
stabilizer adjacency graph. Unpaired defects are matched to the nearest
boundary.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from .lattice import SurfaceCodeLattice


class MWPMDecoder:
    """Minimum-weight perfect matching decoder for surface codes.

    Decodes X-error syndromes (detected by Z stabilizers) and Z-error
    syndromes (detected by X stabilizers) independently.
    """

    def __init__(self, lattice: SurfaceCodeLattice) -> None:
        self.lattice = lattice
        self._z_adj, self._z_shared = self._build_adjacency(lattice.z_stabilizers)
        self._x_adj, self._x_shared = self._build_adjacency(lattice.x_stabilizers)
        self._z_exclusive = self._find_exclusive_qubits(lattice.z_stabilizers)
        self._x_exclusive = self._find_exclusive_qubits(lattice.x_stabilizers)
        self._z_boundary_dists = self._bfs_boundary_distances(
            self._z_adj, self._z_exclusive
        )
        self._x_boundary_dists = self._bfs_boundary_distances(
            self._x_adj, self._x_exclusive
        )

    @staticmethod
    def _build_adjacency(
        stabilizers: list[list[int]],
    ) -> tuple[list[list[int]], dict[tuple[int, int], int]]:
        n = len(stabilizers)
        adj: list[list[int]] = [[] for _ in range(n)]
        shared: dict[tuple[int, int], int] = {}
        for i in range(n):
            si = set(stabilizers[i])
            for j in range(i + 1, n):
                common = si & set(stabilizers[j])
                if common:
                    adj[i].append(j)
                    adj[j].append(i)
                    q = min(common)
                    shared[(i, j)] = q
                    shared[(j, i)] = q
        return adj, shared

    @staticmethod
    def _find_exclusive_qubits(
        stabilizers: list[list[int]],
    ) -> list[int | None]:
        qubit_count: dict[int, int] = {}
        for stab in stabilizers:
            for q in stab:
                qubit_count[q] = qubit_count.get(q, 0) + 1
        result: list[int | None] = []
        for stab in stabilizers:
            excl = None
            for q in stab:
                if qubit_count[q] == 1:
                    excl = q
                    break
            result.append(excl)
        return result

    @staticmethod
    def _bfs_boundary_distances(
        adj: list[list[int]], exclusive: list[int | None]
    ) -> list[float]:
        n = len(adj)
        dist = [float("inf")] * n
        queue: deque[int] = deque()
        for i in range(n):
            if exclusive[i] is not None:
                dist[i] = 1.0
                queue.append(i)
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def _greedy_match(
        self,
        defect_indices: list[int],
        adj: list[list[int]],
        boundary_dists: list[float],
    ) -> list[tuple[int, int | None]]:
        if not defect_indices:
            return []

        n = len(defect_indices)
        matched = [False] * n
        pairs: list[tuple[int, int | None]] = []

        edges: list[tuple[float, int, int | None]] = []
        for i in range(n):
            di = defect_indices[i]
            edges.append((boundary_dists[di], i, None))
            for j in range(i + 1, n):
                dj = defect_indices[j]
                dist = self._graph_distance(di, dj, adj)
                edges.append((dist, i, j))

        edges.sort(key=lambda e: (e[0], e[2] is None))

        for dist, i, j in edges:
            if matched[i]:
                continue
            if j is None:
                matched[i] = True
                pairs.append((defect_indices[i], None))
            elif not matched[j]:
                matched[i] = True
                matched[j] = True
                pairs.append((defect_indices[i], defect_indices[j]))

        return pairs

    @staticmethod
    def _graph_distance(s1: int, s2: int, adj: list[list[int]]) -> float:
        if s1 == s2:
            return 0
        visited = {s1}
        queue: deque[tuple[int, int]] = deque([(s1, 0)])
        while queue:
            u, d = queue.popleft()
            for v in adj[u]:
                if v == s2:
                    return d + 1
                if v not in visited:
                    visited.add(v)
                    queue.append((v, d + 1))
        return float("inf")

    def decode_x_errors(self, syndrome: NDArray[np.int8]) -> NDArray[np.int8]:
        return self._decode(
            syndrome,
            self.lattice.z_stabilizers,
            self._z_adj,
            self._z_shared,
            self._z_exclusive,
            self._z_boundary_dists,
        )

    def decode_z_errors(self, syndrome: NDArray[np.int8]) -> NDArray[np.int8]:
        return self._decode(
            syndrome,
            self.lattice.x_stabilizers,
            self._x_adj,
            self._x_shared,
            self._x_exclusive,
            self._x_boundary_dists,
        )

    def _decode(
        self,
        syndrome: NDArray[np.int8],
        stabilizers: list[list[int]],
        adj: list[list[int]],
        shared: dict[tuple[int, int], int],
        exclusive: list[int | None],
        boundary_dists: list[float],
    ) -> NDArray[np.int8]:
        defect_indices = [i for i, s in enumerate(syndrome) if s == 1]
        correction = np.zeros(self.lattice.num_data, dtype=np.int8)
        if not defect_indices:
            return correction

        pairs = self._greedy_match(defect_indices, adj, boundary_dists)

        for d1, d2 in pairs:
            if d2 is None:
                path = self._path_to_boundary(d1, adj, shared, exclusive)
            else:
                path = self._path_between(d1, d2, adj, shared)
            for q in path:
                correction[q] ^= 1

        return correction

    def _path_between(
        self,
        s1: int,
        s2: int,
        adj: list[list[int]],
        shared: dict[tuple[int, int], int],
    ) -> list[int]:
        if s1 == s2:
            return []
        if (s1, s2) in shared:
            return [shared[(s1, s2)]]

        prev: dict[int, int] = {s1: -1}
        queue: deque[int] = deque([s1])
        while queue:
            u = queue.popleft()
            if u == s2:
                break
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    queue.append(v)

        path_qubits: list[int] = []
        cur = s2
        while prev.get(cur, -1) != -1:
            p = prev[cur]
            path_qubits.append(shared[(p, cur)])
            cur = p
        path_qubits.reverse()
        return path_qubits

    def _path_to_boundary(
        self,
        s: int,
        adj: list[list[int]],
        shared: dict[tuple[int, int], int],
        exclusive: list[int | None],
    ) -> list[int]:
        if exclusive[s] is not None:
            return [exclusive[s]]

        prev: dict[int, int] = {s: -1}
        queue: deque[int] = deque([s])
        target = -1
        while queue:
            u = queue.popleft()
            if u != s and exclusive[u] is not None:
                target = u
                break
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    queue.append(v)

        if target == -1:
            return []

        chain: list[int] = []
        cur = target
        while prev.get(cur, -1) != -1:
            p = prev[cur]
            chain.append(shared[(p, cur)])
            cur = p
        chain.reverse()
        chain.append(exclusive[target])
        return chain
