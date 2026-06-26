"""Shared plotting conventions for the benchmarking diagnostics.

One place to enforce the project-wide rules (feedback 2026-06-26): a grid on every axes unless
there is a good reason not to, and a single ``save_fig`` that tight-lays-out, writes at a
consistent DPI and closes the figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless: these run in studies/CI with no display

if TYPE_CHECKING:
    from pathlib import Path

    import matplotlib.pyplot as plt

_DPI = 150
_GRID_ALPHA = 0.3


def apply_grid(ax: plt.Axes) -> None:
    """Turn on a light grid (the project default for every axes)."""
    ax.grid(visible=True, alpha=_GRID_ALPHA)


def save_fig(fig: plt.Figure, path: Path) -> None:
    """Write ``fig`` to ``path`` at the standard DPI (tight bbox) and close it.

    Uses ``bbox_inches="tight"`` rather than ``tight_layout`` so figures with colorbars/imshow
    (which ``tight_layout`` warns about — and tests treat warnings as errors) lay out cleanly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.close(fig)
