"""Quantum walks on graphs.

Discrete-time and continuous-time quantum walks with various graph topologies
and coin operators.
"""

from .discrete import (
    DiscreteWalkResult,
    discrete_walk,
    grover_walk,
    hadamard_walk_line,
)
from .continuous import (
    ContinuousWalkResult,
    continuous_walk,
)
from .graphs import (
    line_graph,
    cycle_graph,
    complete_graph,
    hypercube_graph,
    grid_graph,
)

__all__ = [
    "DiscreteWalkResult",
    "discrete_walk",
    "grover_walk",
    "hadamard_walk_line",
    "ContinuousWalkResult",
    "continuous_walk",
    "line_graph",
    "cycle_graph",
    "complete_graph",
    "hypercube_graph",
    "grid_graph",
]
