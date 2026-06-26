"""Data-coverage diagnostics: where, in time, each turbine (and ERA5) actually has data.

Three views (feedback 2026-06-26): a v0-style **window timeline** (data-present periods per
turbine + ERA5, with baseline/upgraded bands), a **raw coverage** line-plot (% present over time,
before any filtering), and a **filter coverage** line-plot (test turbine, raw vs used, so the
effect of the row filter is visible). Heatmaps were dropped — they were hard to read.

Wind speed / power columns are referred to by their original source-native names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from benchmarking.diagnostics import stages
from benchmarking.diagnostics.context import ERA5_WS_COL
from benchmarking.diagnostics.style import apply_grid, save_fig
from benchmarking.diagnostics.timeaxis import BASELINE_COLOR, UPGRADED_COLOR, shade_segments

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.diagnostics.context import DiagnosticContext

_COVERAGE_BUCKET = "7D"  # weekly buckets keep a multi-year window legible


def _present_runs(index: pd.DatetimeIndex, present: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous (start, width) spans in matplotlib date units where ``present`` is True."""
    runs: list[tuple[float, float]] = []
    if not present.any():
        return runs
    x = mdates.date2num(index.to_pydatetime())
    edges = np.flatnonzero(np.diff(np.concatenate([[0], present.astype(int), [0]])))
    for start_i, end_i in zip(edges[::2], edges[1::2], strict=True):
        x0 = x[start_i]
        x1 = x[min(end_i, len(x) - 1)]
        runs.append((x0, max(x1 - x0, 1e-6)))
    return runs


def plot_input_timeline(ctx: DiagnosticContext) -> Path:
    """Window timeline: data-present periods per turbine (+ ERA5), with baseline/upgraded bands."""
    rows: list[tuple[str, np.ndarray]] = []
    for turbine in [ctx.test_wtg, *ctx.references()]:
        label = f"{turbine}{' (test)' if turbine == ctx.test_wtg else ''}: {ctx.columns.active_power}"
        rows.append((label, ctx.turbine_series(turbine, ctx.columns.active_power).notna().to_numpy()))
    if ctx.era5_df is not None and ERA5_WS_COL in ctx.era5_df.columns:
        rows.append((f"ERA5: {ERA5_WS_COL}", ctx.era5_df[ERA5_WS_COL].reindex(ctx.index).notna().to_numpy()))

    fig, ax = plt.subplots(figsize=(13, 1.2 + 0.5 * len(rows)))
    shade_segments(ax, ctx)
    for y, (label, present) in enumerate(rows):  # noqa: B007 - y is the row position
        ax.broken_barh(_present_runs(ctx.index, present), (y + 0.6, 0.8), facecolors="C0")
    ax.set_yticks(np.arange(len(rows)) + 1.0)
    ax.set_yticklabels([label for label, _ in rows])
    ax.set_ylim(0.5, len(rows) + 0.5)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlabel("date")
    ax.set_title(f"{ctx.test_wtg}: input data timeline (bars = data present)")
    legend = [
        Patch(color=BASELINE_COLOR, alpha=0.4, label="baseline"),
        Patch(color=UPGRADED_COLOR, alpha=0.4, label="upgraded"),
    ]
    ax.legend(handles=legend, loc="lower right")
    apply_grid(ax)
    path = ctx.stage_dir(stages.INPUTS) / "input_data_timeline.png"
    save_fig(fig, path)
    return path


def _weekly_coverage(present: pd.Series, *, timebase: pd.Timedelta) -> pd.Series:
    """Weekly fraction (%) of expected timebase slots for which ``present`` is True."""
    expected = pd.Timedelta(_COVERAGE_BUCKET) / timebase
    return 100.0 * present.astype(float).resample(_COVERAGE_BUCKET).sum() / expected


def plot_input_coverage(ctx: DiagnosticContext) -> Path:
    """Weekly %-coverage line per turbine's power (+ ERA5 wind speed), before any filtering."""
    fig, ax = plt.subplots(figsize=(12, 6))
    shade_segments(ax, ctx)
    for turbine in [ctx.test_wtg, *ctx.references()]:
        present = pd.Series(ctx.turbine_series(turbine, ctx.columns.active_power).notna().to_numpy(), index=ctx.index)
        weekly = _weekly_coverage(present, timebase=ctx.timebase)
        label = f"{turbine}{' (test)' if turbine == ctx.test_wtg else ''}"
        ax.plot(weekly.index.to_numpy(), weekly.to_numpy(), linewidth=1.0, label=label)
    if ctx.era5_df is not None and ERA5_WS_COL in ctx.era5_df.columns:
        era5_present = pd.Series(ctx.era5_df[ERA5_WS_COL].reindex(ctx.index).notna().to_numpy(), index=ctx.index)
        weekly = _weekly_coverage(era5_present, timebase=ctx.timebase)
        ax.plot(weekly.index.to_numpy(), weekly.to_numpy(), linewidth=1.0, linestyle="--", label="ERA5")
    ax.set_ylim(0, 105)
    ax.set_xlabel("date")
    ax.set_ylabel(f"weekly {ctx.columns.active_power} coverage [%]")
    ax.set_title(f"{ctx.test_wtg}: input data coverage (before filtering)")
    apply_grid(ax)
    ax.legend(ncol=2, fontsize="small")
    path = ctx.stage_dir(stages.INPUTS) / "input_data_coverage.png"
    save_fig(fig, path)
    return path


def plot_filter_coverage(ctx: DiagnosticContext) -> Path:
    """Weekly %-coverage of the test turbine: raw present vs used (after the row filter)."""
    raw = pd.Series(ctx.test_series(ctx.columns.active_power).notna().to_numpy(), index=ctx.index)
    used = pd.Series(np.asarray(ctx.used_ts, dtype=bool), index=ctx.index)
    fig, ax = plt.subplots(figsize=(12, 6))
    shade_segments(ax, ctx)
    ax.plot(
        _weekly_coverage(raw, timebase=ctx.timebase).index.to_numpy(),
        _weekly_coverage(raw, timebase=ctx.timebase).to_numpy(),
        linewidth=1.2,
        color="C0",
        label=f"{ctx.columns.active_power} present (raw)",
    )
    ax.plot(
        _weekly_coverage(used, timebase=ctx.timebase).index.to_numpy(),
        _weekly_coverage(used, timebase=ctx.timebase).to_numpy(),
        linewidth=1.2,
        color="C3",
        label="used (after filter)",
    )
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("date")
    ax.set_ylabel("weekly coverage [%]")
    ax.set_title(f"{ctx.test_wtg}: data coverage before vs after the row filter")
    apply_grid(ax)
    ax.legend()
    path = ctx.stage_dir(stages.FILTER) / "filter_coverage.png"
    save_fig(fig, path)
    return path
