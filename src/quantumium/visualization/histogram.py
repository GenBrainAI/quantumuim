"""Measurement result histogram visualization."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_histogram(
    counts: dict[str, int | float],
    title: str | None = None,
) -> Figure:
    """Plot measurement results as a bar chart with probability labels.

    Args:
        counts: Mapping of bitstrings to counts (int) or probabilities (float).
        title: Optional plot title.

    Returns:
        A matplotlib Figure.
    """
    labels = sorted(counts.keys())
    values = [counts[k] for k in labels]
    total = sum(values)

    if total == 0:
        probabilities = [0.0] * len(values)
    else:
        probabilities = [v / total for v in values]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4))

    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(labels)))  # type: ignore[attr-defined]
    bars = ax.bar(labels, probabilities, color=colors, edgecolor="black", linewidth=0.5)

    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{prob:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Measurement Outcome")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, max(probabilities) * 1.15 if probabilities else 1.0)
    ax.set_title(title or "Measurement Results")

    if len(labels) > 8:
        plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    return fig
