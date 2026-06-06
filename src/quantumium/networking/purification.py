"""BBPSSW entanglement purification protocol."""

from __future__ import annotations

from dataclasses import dataclass

from .entanglement import EntangledPair


def purify_bbpssw(pair1: EntangledPair, pair2: EntangledPair) -> EntangledPair:
    f1 = pair1.fidelity
    f2 = pair2.fidelity
    numerator = f1 * f2 + (1 - f1) * (1 - f2) / 9
    denominator = (f1 * f2
                   + f1 * (1 - f2) / 3
                   + (1 - f1) * f2 / 3
                   + 5 * (1 - f1) * (1 - f2) / 9)
    f_new = numerator / denominator
    return EntangledPair(pair1.node_a, pair1.node_b, f_new)


def purify_rounds(pair: EntangledPair, num_rounds: int) -> EntangledPair:
    if num_rounds < 2:
        return pair
    current = pair
    for _ in range(num_rounds - 1):
        sacrifice = EntangledPair(current.node_a, current.node_b, current.fidelity)
        current = purify_bbpssw(current, sacrifice)
    return current
