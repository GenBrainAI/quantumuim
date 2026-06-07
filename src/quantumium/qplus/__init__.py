"""Q+ Algorithm — beyond-qubit-count quantum framework.

Three-layer strategy for solving problems larger than available qubit hardware:
1. Algorithmic compression (semiclassical QFT, modular arithmetic optimization)
2. Qubit recycling (measure-reset-reuse with classical feedforward)
3. Circuit cutting + multi-QPU distribution
"""

from .compress import SemiclassicalQFT, ModularArithmeticOptimizer
from .recycle import QubitRecycler, FeedForwardController
from .cut import CircuitCutter, CutRecombiner
from .distribute import QPUOrchestrator
from .shor import ShorQPlus, QplusResult

__all__ = [
    "SemiclassicalQFT",
    "ModularArithmeticOptimizer",
    "QubitRecycler",
    "FeedForwardController",
    "CircuitCutter",
    "CutRecombiner",
    "QPUOrchestrator",
    "ShorQPlus",
    "QplusResult",
]
