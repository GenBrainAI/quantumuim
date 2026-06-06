"""Surface code error correction with minimum-weight perfect matching decoder.

This package implements the rotated planar surface code, a leading candidate
for fault-tolerant quantum computing. It provides lattice construction,
syndrome extraction, MWPM decoding, logical operators, and threshold estimation.

Modules:
    lattice: Rotated planar surface code lattice for distance d=3,5,7,...
    syndrome: Stabilizer measurement and syndrome extraction.
    decoder: Minimum-weight perfect matching (MWPM) decoder.
    logical: Logical qubit operators (X_L, Z_L).
    threshold: Monte Carlo threshold estimation.
"""

from .lattice import SurfaceCodeLattice
from .syndrome import extract_syndrome, apply_error
from .decoder import MWPMDecoder
from .logical import logical_x, logical_z, apply_logical_x, apply_logical_z
from .threshold import estimate_threshold

__all__ = [
    "SurfaceCodeLattice",
    "extract_syndrome",
    "apply_error",
    "MWPMDecoder",
    "logical_x",
    "logical_z",
    "apply_logical_x",
    "apply_logical_z",
    "estimate_threshold",
]
