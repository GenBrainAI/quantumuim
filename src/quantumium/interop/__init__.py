"""Cross-SDK interoperability — convert circuits between Quantumium and other frameworks.

Optional converters are loaded lazily to avoid hard dependencies on Qiskit or Cirq.
Install the relevant backend extra to enable conversion:

    pip install quantumium[qiskit]   # Qiskit conversion
    pip install quantumium[cirq]     # Cirq conversion
"""

from __future__ import annotations

__all__: list[str] = []

try:
    from .qiskit_convert import from_qiskit, to_qiskit

    __all__ += ["to_qiskit", "from_qiskit"]
except ImportError:
    pass

try:
    from .cirq_convert import from_cirq, to_cirq

    __all__ += ["to_cirq", "from_cirq"]
except ImportError:
    pass
