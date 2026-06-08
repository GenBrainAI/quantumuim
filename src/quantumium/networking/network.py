"""QuantumNetwork — graph-based quantum network topology."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantumChannel:
    node_a: str
    node_b: str
    distance_km: float
    loss_db_per_km: float = 0.2
    depolarizing_rate: float = 0.0

    @property
    def transmission_probability(self) -> float:
        loss_db = self.loss_db_per_km * self.distance_km
        return 10 ** (-loss_db / 10)

    @property
    def channel_fidelity(self) -> float:
        if self.depolarizing_rate <= 0:
            p = self.transmission_probability
            return 1.0 - (1.0 - p) * 0.75
        p_depol = 1.0 - self.depolarizing_rate * self.distance_km
        p_depol = max(0.0, min(1.0, p_depol))
        return 0.25 + 0.75 * p_depol


@dataclass
class NetworkNode:
    name: str
    is_repeater: bool = False
    memory_coherence_time: float = 1.0


class QuantumNetwork:
    def __init__(self) -> None:
        self._nodes: dict[str, NetworkNode] = {}
        self._channels: dict[tuple[str, str], QuantumChannel] = {}
        self._adjacency: dict[str, list[str]] = {}

    def add_node(
        self, name: str, is_repeater: bool = False, memory_coherence_time: float = 1.0
    ) -> None:
        self._nodes[name] = NetworkNode(name, is_repeater, memory_coherence_time)
        self._adjacency.setdefault(name, [])

    def add_link(
        self,
        node_a: str,
        node_b: str,
        distance_km: float = 10.0,
        loss_db_per_km: float = 0.2,
        depolarizing_rate: float = 0.0,
    ) -> None:
        if node_a not in self._nodes:
            raise ValueError(f"Unknown node: {node_a}")
        if node_b not in self._nodes:
            raise ValueError(f"Unknown node: {node_b}")
        key = (min(node_a, node_b), max(node_a, node_b))
        ch = QuantumChannel(node_a, node_b, distance_km, loss_db_per_km, depolarizing_rate)
        self._channels[key] = ch
        if node_b not in self._adjacency[node_a]:
            self._adjacency[node_a].append(node_b)
        if node_a not in self._adjacency[node_b]:
            self._adjacency[node_b].append(node_a)

    def get_channel(self, node_a: str, node_b: str) -> QuantumChannel:
        key = (min(node_a, node_b), max(node_a, node_b))
        if key not in self._channels:
            raise ValueError(f"No channel between {node_a} and {node_b}")
        return self._channels[key]

    def neighbors(self, node: str) -> list[str]:
        return list(self._adjacency.get(node, []))

    def nodes(self) -> list[str]:
        return list(self._nodes.keys())

    def get_node(self, name: str) -> NetworkNode:
        return self._nodes[name]

    def find_path(self, source: str, target: str) -> list[str]:
        if source not in self._nodes or target not in self._nodes:
            raise ValueError("Unknown node(s)")
        visited: set[str] = set()
        queue: list[list[str]] = [[source]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == target:
                return path
            if node in visited:
                continue
            visited.add(node)
            for nb in self._adjacency.get(node, []):
                if nb not in visited:
                    queue.append(path + [nb])
        raise ValueError(f"No path from {source} to {target}")

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_channels(self) -> int:
        return len(self._channels)

    @classmethod
    def star(
        cls, hub: str, spokes: list[str], distance_km: float = 10.0, loss_db_per_km: float = 0.2
    ) -> QuantumNetwork:
        net = cls()
        net.add_node(hub, is_repeater=True)
        for s in spokes:
            net.add_node(s)
            net.add_link(hub, s, distance_km, loss_db_per_km)
        return net

    @classmethod
    def chain(
        cls, node_names: list[str], segment_distance_km: float = 50.0, loss_db_per_km: float = 0.2
    ) -> QuantumNetwork:
        net = cls()
        for i, name in enumerate(node_names):
            is_rep = 0 < i < len(node_names) - 1
            net.add_node(name, is_repeater=is_rep)
        for i in range(len(node_names) - 1):
            net.add_link(node_names[i], node_names[i + 1], segment_distance_km, loss_db_per_km)
        return net

    @classmethod
    def mesh(
        cls, rows: int, cols: int, distance_km: float = 10.0, loss_db_per_km: float = 0.2
    ) -> QuantumNetwork:
        net = cls()
        names: list[list[str]] = []
        for r in range(rows):
            row = []
            for c in range(cols):
                name = f"N{r}_{c}"
                is_corner = (r in (0, rows - 1)) and (c in (0, cols - 1))
                net.add_node(name, is_repeater=not is_corner)
                row.append(name)
            names.append(row)
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    net.add_link(names[r][c], names[r][c + 1], distance_km, loss_db_per_km)
                if r + 1 < rows:
                    net.add_link(names[r][c], names[r + 1][c], distance_km, loss_db_per_km)
        return net
