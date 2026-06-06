"""Dynamical decoupling pulse sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DDSequence:
    """A dynamical decoupling pulse sequence.

    Attributes:
        name: Sequence identifier (e.g., "xy4", "cpmg-4").
        pulses: Gate names for each pulse (e.g., ["X", "Y", "X", "Y"]).
        spacings: Normalized time fractions between pulses.
            Length = len(pulses) + 1 (includes leading and trailing idle).
            Must sum to 1.0.
    """

    name: str
    pulses: list[str]
    spacings: list[float]

    def __post_init__(self) -> None:
        if len(self.spacings) != len(self.pulses) + 1:
            raise ValueError(
                f"spacings length ({len(self.spacings)}) must be "
                f"pulses length + 1 ({len(self.pulses) + 1})"
            )
        total = sum(self.spacings)
        if abs(total - 1.0) > 1e-10:
            raise ValueError(f"spacings must sum to 1.0, got {total}")

    @property
    def num_pulses(self) -> int:
        return len(self.pulses)

    @staticmethod
    def spin_echo() -> DDSequence:
        """Hahn spin echo: single X pulse at midpoint."""
        return DDSequence(
            name="spin_echo",
            pulses=["X"],
            spacings=[0.5, 0.5],
        )

    @staticmethod
    def cpmg(n_pulses: int) -> DDSequence:
        """CPMG sequence with n equally spaced Y pulses."""
        if n_pulses < 1:
            raise ValueError("CPMG requires at least 1 pulse")
        half = 1.0 / (2 * n_pulses)
        full = 1.0 / n_pulses
        spacings = [half] + [full] * (n_pulses - 1) + [half]
        return DDSequence(
            name=f"cpmg-{n_pulses}",
            pulses=["Y"] * n_pulses,
            spacings=spacings,
        )

    @staticmethod
    def xy4() -> DDSequence:
        """XY-4 sequence: X-Y-X-Y with symmetric spacing."""
        return DDSequence(
            name="xy4",
            pulses=["X", "Y", "X", "Y"],
            spacings=[0.125, 0.25, 0.25, 0.25, 0.125],
        )

    @staticmethod
    def xy8() -> DDSequence:
        """XY-8 sequence: XY-4 followed by its time-reverse."""
        pulses = ["X", "Y", "X", "Y", "Y", "X", "Y", "X"]
        n = len(pulses)
        half = 1.0 / (2 * n)
        full = 1.0 / n
        spacings = [half] + [full] * (n - 1) + [half]
        return DDSequence(
            name="xy8",
            pulses=pulses,
            spacings=spacings,
        )

    @staticmethod
    def udd(n_pulses: int) -> DDSequence:
        """Uhrig Dynamical Decoupling with non-uniform spacing.

        Pulse times: t_j = sin²(πj / (2N+2)) for j = 1..N (normalized to [0,1]).
        """
        if n_pulses < 1:
            raise ValueError("UDD requires at least 1 pulse")
        times = [
            math.sin(math.pi * j / (2 * n_pulses + 2)) ** 2
            for j in range(1, n_pulses + 1)
        ]
        spacings = [times[0]]
        for i in range(1, len(times)):
            spacings.append(times[i] - times[i - 1])
        spacings.append(1.0 - times[-1])
        return DDSequence(
            name=f"udd-{n_pulses}",
            pulses=["X"] * n_pulses,
            spacings=spacings,
        )

    def pulse_times(self) -> list[float]:
        """Return normalized pulse application times in [0, 1]."""
        times = []
        t = 0.0
        for i, spacing in enumerate(self.spacings[:-1]):
            t += spacing
            if i < len(self.pulses):
                times.append(t)
        return times
