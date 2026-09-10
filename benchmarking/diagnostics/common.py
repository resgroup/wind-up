"""The single entry point a method calls to emit the shared cross-method diagnostics.

:func:`write_common_diagnostics` runs every shared plot for a :class:`DiagnosticContext`. Each
plot is independent and guarded: one failing (e.g. a degenerate segment) logs and is skipped
rather than killing the rest of an unattended inspection run. Plots that need an absent signal
return ``None`` and are simply not written.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from benchmarking.diagnostics.coverage import (
    plot_excluded_fraction,
    plot_filter_coverage,
    plot_input_coverage,
    plot_input_timeline,
)
from benchmarking.diagnostics.curves import (
    plot_curves_by_upgrade,
    plot_ops_curves,
    plot_ops_curves_excluded,
    plot_ops_curves_kept,
    plot_power_factor,
    plot_reactive_vs_active,
)
from benchmarking.diagnostics.histograms import plot_condition_histograms
from benchmarking.diagnostics.northing import plot_northing_error

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from benchmarking.diagnostics.context import DiagnosticContext

logger = logging.getLogger(__name__)

# Every shared plot, in a sensible reading order. Each takes the context and returns a path
# (or None when it has nothing to draw for this source).
_PLOTS: tuple[Callable[[DiagnosticContext], Path | None], ...] = (
    plot_input_timeline,
    plot_input_coverage,
    plot_filter_coverage,
    plot_excluded_fraction,
    plot_condition_histograms,
    plot_ops_curves,
    plot_ops_curves_kept,
    plot_ops_curves_excluded,
    plot_curves_by_upgrade,
    plot_reactive_vs_active,
    plot_power_factor,
    plot_northing_error,
)


def write_common_diagnostics(ctx: DiagnosticContext) -> list[Path]:
    """Write every shared diagnostic plot for ``ctx``; return the paths actually written."""
    n = len(ctx.index)
    excluded_n = None if ctx.excluded_ts is None else len(ctx.excluded_ts)
    if len(ctx.treated_ts) != n or len(ctx.used_ts) != n or excluded_n not in (None, n):
        logger.error(
            "diagnostic masks misaligned with the index (index=%d, treated=%d, used=%d, excluded=%s);"
            " skipping diagnostics",
            n,
            len(ctx.treated_ts),
            len(ctx.used_ts),
            excluded_n,
        )
        return []
    ctx.plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for plot in _PLOTS:
        try:
            path = plot(ctx)
        except Exception:
            logger.exception("diagnostic plot %s failed for %s", plot.__name__, ctx.test_wtg)
            continue
        if path is not None:
            written.append(path)
    return written
