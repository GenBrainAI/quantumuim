"""Circuit transpilation and optimization passes.

This package provides a composable pipeline of optimization passes that
reduce gate count, cancel inverse pairs, merge single-qubit unitaries,
and decompose multi-qubit gates for hardware-constrained backends.

Submodules:
    passes: Individual optimization pass functions and the ``transpile()``
        pipeline.
    qasm: OpenQASM 2.0 import and export routines.
    serialization: JSON-based circuit serialization and file I/O.
"""

from .passes import transpile, count_gates, circuit_depth
from .qasm import to_qasm, from_qasm
from .serialization import to_dict, from_dict, to_json, from_json, save, load

__all__ = [
    "transpile",
    "count_gates",
    "circuit_depth",
    "to_qasm",
    "from_qasm",
    "to_dict",
    "from_dict",
    "to_json",
    "from_json",
    "save",
    "load",
]
