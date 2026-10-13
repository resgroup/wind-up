"""Shared helpers for time-axis diagnostic plots: the upgrade boundary and segment shading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pandas as pd

    from benchmarking.diagnostics.context import DiagnosticContext

BASELINE_COLOR = "#59a89c"
UPGRADED_COLOR = "#e08214"


def upgrade_start(ctx: DiagnosticContext) -> pd.Timestamp | None:
    """Return the first upgraded timestamp (prepost changeover / toggle origin), or None."""
    upgraded = ctx.upgraded_ts
    return ctx.index[int(np.argmax(upgraded))] if upgraded.any() else None


def shade_segments(ax: plt.Axes, ctx: DiagnosticContext) -> None:
    """Shade the baseline and upgraded spans on a time axis and mark the upgrade start."""
    start = upgrade_start(ctx)
    first, last = ctx.index.min(), ctx.index.max()
    if start is not None:
        ax.axvspan(first, start, color=BASELINE_COLOR, alpha=0.12)
        ax.axvspan(start, last, color=UPGRADED_COLOR, alpha=0.12)
        ax.axvline(start, color="k", linestyle="--", linewidth=1.2)
    else:
        ax.axvspan(first, last, color=BASELINE_COLOR, alpha=0.12)
