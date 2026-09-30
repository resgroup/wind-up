"""Shared plotting conventions for the benchmarking diagnostics.

One place to enforce the project-wide rules (feedback 2026-06-26): a grid on every axes unless
there is a good reason not to, and a single ``save_fig`` that tight-lays-out, writes at a
consistent DPI and closes the figure.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import TYPE_CHECKING

import matplotlib as mpl

# Headless by default: these run in studies/CI with no display. Mirror wind_up/__init__.py — respect
# an explicit MPLBACKEND (checked by key presence), leave the backend alone if pyplot is already
# imported (e.g. an interactive notebook), and never let a late use() failure break import.
if "MPLBACKEND" not in os.environ and "matplotlib.pyplot" not in sys.modules:
    with contextlib.suppress(ImportError):
        mpl.use("Agg")

if TYPE_CHECKING:
    from pathlib import Path

    import matplotlib.pyplot as plt

_DPI = 150
_GRID_ALPHA = 0.3

# Every colour is used solid before any dash, and dashes run from the most solid-like to the least:
# 10 colours x 4 dashes = 40 distinguishable series.
_COLOURS = 10
_DASHES = ("-", "--", "-.", ":")


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


def series_style(index: int) -> tuple[str, str]:
    """Return the ``(colour, dash)`` pair for series ``index``: solid for the first 10, distinct for 40."""
    return f"C{index % _COLOURS}", _DASHES[(index // _COLOURS) % len(_DASHES)]
