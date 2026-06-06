"""High-level one-liner API for entanglement distribution."""

from __future__ import annotations

from .entanglement import EntangledPair, generate_entanglement, entanglement_swap
from .network import QuantumNetwork
from .purification import purify_bbpssw


def distribute_entanglement(
    network: QuantumNetwork,
    source: str,
    target: str,
    purification_rounds: int = 0,
) -> EntangledPair:
    path = network.find_path(source, target)

    if len(path) < 2:
        raise ValueError("Source and target must be different nodes")

    if len(path) == 2:
        pair = generate_entanglement(network, path[0], path[1])
    else:
        segments: list[EntangledPair] = []
        for i in range(len(path) - 1):
            segments.append(generate_entanglement(network, path[i], path[i + 1]))
        pair = segments[0]
        for seg in segments[1:]:
            pair = entanglement_swap(pair, seg)

    for _ in range(purification_rounds):
        path2 = network.find_path(source, target)
        if len(path2) == 2:
            sacrifice = generate_entanglement(network, path2[0], path2[1])
        else:
            segs2: list[EntangledPair] = []
            for i in range(len(path2) - 1):
                segs2.append(generate_entanglement(network, path2[i], path2[i + 1]))
            sacrifice = segs2[0]
            for seg in segs2[1:]:
                sacrifice = entanglement_swap(sacrifice, seg)
        pair = purify_bbpssw(pair, sacrifice)

    return pair
