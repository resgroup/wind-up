"""The single entry point a method calls to emit the shared cross-method diagnostics.

:func:`write_common_diagnostics` runs every shared plot for a :class:`DiagnosticContext`. Each
plot is independent and guarded: one failing (e.g. a degenerate segment) logs and is skipped
rather than killing the rest of an unattended inspection run. Plots that need an absent signal
return ``None`` and are simply not written.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from benchmarking.diagnostics import stages
from benchmarking.diagnostics.coverage import (
    plot_excluded_fraction,
    plot_filter_coverage,
    plot_input_coverage,
    plot_input_timeline,
)
from benchmarking.diagnostics.curves import (
    plot_curves_by_upgrade,
    plot_ops_curves_excluded,
    plot_power_factor,
    plot_reactive_vs_active,
)
from benchmarking.diagnostics.histograms import plot_condition_histograms
from benchmarking.diagnostics.northing import plot_northed_error, plot_northing_error
from benchmarking.diagnostics.operating_states import plot_run_operating_states, plot_run_uplift_validity
from benchmarking.diagnostics.ops_relationships import plot_run_ops_relationships

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from benchmarking.diagnostics.context import DiagnosticContext

logger = logging.getLogger(__name__)

# Every shared plot with the stage folder it writes to, in a sensible reading order. Each takes the
# context and returns a path, a list of paths for a plot drawn once per turbine, or None when it has
# nothing to draw.
_PLOTS: tuple[tuple[str, Callable[[DiagnosticContext], Path | list[Path] | None]], ...] = (
    (stages.CHANGES, plot_input_timeline),
    (stages.CHANGES, plot_input_coverage),
    (stages.VALID_RECORDS, plot_filter_coverage),
    (stages.VALID_RECORDS, plot_excluded_fraction),
    (stages.VALID_RECORDS, plot_condition_histograms),
    (stages.VALID_RECORDS, plot_run_uplift_validity),
    (stages.VALID_RECORDS, plot_ops_curves_excluded),
    (stages.VALID_RECORDS, plot_curves_by_upgrade),
    (stages.CHANGES, plot_reactive_vs_active),
    (stages.CHANGES, plot_power_factor),
    (stages.CHANGES, plot_run_ops_relationships),
    (stages.OPERATING_STATES, plot_run_operating_states),
    (stages.NORTHING, plot_northing_error),
    (stages.NORTHING, plot_northed_error),
)


def write_common_diagnostics(ctx: DiagnosticContext, *, skip_stages: frozenset[str] = frozenset()) -> list[Path]:
    """Write every shared diagnostic plot for ``ctx``, except those of ``skip_stages``; return the paths written."""
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
    for stage, plot in _PLOTS:
        if stage in skip_stages:
            continue
        try:
            path = plot(ctx)
        except Exception:
            logger.exception("diagnostic plot %s failed for %s", plot.__name__, ctx.test_wtg)
            continue
        if isinstance(path, list):
            written.extend(path)
        elif path is not None:
            written.append(path)
    return written
