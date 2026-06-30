"""Baseline-vs-upgraded condition histograms (feedback 2026-06-26, item 22).

The prepost failure is an overlap/positivity problem: the baseline and upgraded segments do not
share a weather distribution. These overlaid histograms make that concrete — wind speed,
direction, temperature, turbulence, and the time-of-day / month structure — so a reviewer can
see exactly where the two segments differ.

Hour-of-day and month are derived from the timestamp index and are **diagnostics only**: they
are deliberately not model features (the R-learner uses shuffled K-fold cross-fitting, which
assumes no timestamp features — design-note §3/§4). Weather conditions use reference-derived
(treatment-invariant) signals so the comparison is honest for the upgraded segment too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from benchmarking.diagnostics import stages
from benchmarking.diagnostics.context import ERA5_WD_COL
from benchmarking.diagnostics.style import apply_grid, save_fig

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.diagnostics.context import DiagnosticContext

_MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Real Open-Meteo wind-direction column, preferred over the neutral ``era5_wd`` alias for labels.
_ERA5_RAW_WD = "wind_direction_100m"


def _era5_wd_column(ctx: DiagnosticContext) -> str | None:
    """Return the ERA5 wind-direction column to plot — the real Open-Meteo name if present, else the alias."""
    if ctx.era5_df is None:
        return None
    for col in (_ERA5_RAW_WD, ERA5_WD_COL):
        if col in ctx.era5_df.columns:
            return col
    return None


def _conditions(ctx: DiagnosticContext) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    """Return the (label, per-timestamp values, bins) conditions available for this run.

    ``bins`` is an explicit edge array for the discrete time features (hour, month) and ``None``
    for continuous ones (their edges are derived robustly at plot time so low-wind TI outliers do
    not flatten the histogram).
    """
    cols = ctx.columns
    index = ctx.index
    items: list[tuple[str, np.ndarray, np.ndarray | None]] = [
        (f"{cols.wind_speed} (ref mean) [m/s]", ctx.reference_mean(cols.wind_speed).to_numpy(dtype=float), None),
    ]

    wd_col = _era5_wd_column(ctx)
    if wd_col is not None and ctx.era5_df is not None:
        items.append((f"{wd_col} [deg]", ctx.era5_df[wd_col].reindex(index).to_numpy(dtype=float), None))
    elif ctx.has_column(cols.nacelle_position):
        label = f"{cols.nacelle_position} (ref mean) [deg]"
        items.append((label, ctx.reference_mean(cols.nacelle_position).to_numpy(dtype=float), None))

    if ctx.has_column(cols.wind_speed_sd):
        ws = ctx.reference_mean(cols.wind_speed).to_numpy(dtype=float)
        ws_sd = ctx.reference_mean(cols.wind_speed_sd).to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ti = np.where(ws > 0, ws_sd / ws, np.nan)
        items.append((f"TI = {cols.wind_speed_sd}/{cols.wind_speed} (ref mean, dimensionless)", ti, None))

    if ctx.has_column(cols.ambient_temp):
        items.append(
            (f"{cols.ambient_temp} (ref mean)", ctx.reference_mean(cols.ambient_temp).to_numpy(dtype=float), None)
        )

    items.append(("hour of day", index.hour.to_numpy(dtype=float), np.arange(-0.5, 24.5, 1.0)))
    items.append(("month", index.month.to_numpy(dtype=float), np.arange(0.5, 13.5, 1.0)))
    return items


def _robust_bins(values: np.ndarray, *, bins: int = 30) -> np.ndarray | int:
    """Bin edges over the 1st-99th percentile of the finite values, so outliers don't dominate."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return bins
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        return bins
    return np.linspace(lo, hi, bins + 1)


def plot_condition_histograms(ctx: DiagnosticContext) -> Path:
    """Overlaid baseline-vs-upgraded histograms for each available condition."""
    conditions = _conditions(ctx)
    baseline = ctx.baseline_ts
    upgraded = ctx.upgraded_ts
    ncols = 3
    nrows = int(np.ceil(len(conditions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    flat = axes.flatten()
    for ax, (label, values, explicit_bins) in zip(flat, conditions, strict=False):
        bins = explicit_bins if explicit_bins is not None else _robust_bins(values)
        for seg_label, seg, color in (("baseline", baseline, "C0"), ("upgraded", upgraded, "C1")):
            seg_vals = values[seg & np.isfinite(values)]
            if seg_vals.size:
                # filled with alpha so a coincident distribution (e.g. flat hour-of-day) shows both.
                ax.hist(
                    seg_vals, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=color, label=seg_label
                )
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        _label_time_axis(ax, label)
        apply_grid(ax)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    for ax in flat[len(conditions) :]:
        ax.set_visible(False)
    fig.suptitle(f"{ctx.test_wtg}: condition distributions on the USED (post-filter) data, baseline vs upgraded")
    path = ctx.stage_dir(stages.UPLIFT_INPUTS) / "condition_histograms.png"
    save_fig(fig, path)
    return path


def _label_time_axis(ax: plt.Axes, label: str) -> None:
    """Nicely tick the discrete hour-of-day and month axes."""
    if label == "hour of day":
        ax.set_xticks(range(0, 24, 3))
    elif label == "month":
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(_MONTH_LABELS, rotation=45, fontsize="small")
