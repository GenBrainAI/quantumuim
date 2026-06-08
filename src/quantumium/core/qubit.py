"""Qubit representations and state management."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Qubit:
    """A single qubit with state vector representation."""

    def __init__(self, alpha: complex = 1.0, beta: complex = 0.0) -> None:
        state = np.array([alpha, beta], dtype=complex)
        norm = np.linalg.norm(state)
        if norm < 1e-10:
            raise ValueError("Qubit state cannot be the zero vector")
        self._state: NDArray[np.complexfloating] = state / norm

    @classmethod
    def zero(cls) -> Qubit:
        """Create the |0⟩ state."""
        return cls(1.0, 0.0)

    @classmethod
    def one(cls) -> Qubit:
        """Create the |1⟩ state."""
        return cls(0.0, 1.0)

    @classmethod
    def plus(cls) -> Qubit:
        """Create the |+⟩ = (|0⟩ + |1⟩)/√2 state."""
        return cls(1.0 / np.sqrt(2), 1.0 / np.sqrt(2))

    @classmethod
    def minus(cls) -> Qubit:
        """Create the |−⟩ = (|0⟩ − |1⟩)/√2 state."""
        return cls(1.0 / np.sqrt(2), -1.0 / np.sqrt(2))

    @property
    def state(self) -> NDArray[np.complexfloating]:
        return self._state.copy()

    @property
    def probabilities(self) -> tuple[float, float]:
        return float(np.abs(self._state[0]) ** 2), float(np.abs(self._state[1]) ** 2)

    def measure(self) -> int:
        """Measure the qubit, collapsing it to |0⟩ or |1⟩. Returns 0 or 1."""
        p0, _ = self.probabilities
        result = 0 if np.random.random() < p0 else 1
        self._state = (
            np.array([1, 0], dtype=complex) if result == 0 else np.array([0, 1], dtype=complex)
        )
        return result

    def __repr__(self) -> str:
        return f"Qubit(α={self._state[0]:.4f}, β={self._state[1]:.4f})"
