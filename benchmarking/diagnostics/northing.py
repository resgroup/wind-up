"""Northing-error timeline (feedback 2026-06-26, item 15).

Reference nacelle positions may enter a method as raw features with **no** northing correction;
an offset or jumps in a turbine's yaw zero distorts the direction signal the model sees. This
plots, per turbine, the **monthly circular mean** of (nacelle position - ERA5 wind direction) over
time, so a drift or step in the offset stands out. Only rows where the turbine is generating
(≥ 5% of its rated power) are used, because a parked turbine often points away from the wind.

Drawn twice: from the raw nacelle position in ``1_inputs``, and from the north-calibrated column
the shared northing step writes in ``3_feature_eng`` -- the signal the model is actually given, so
a correction that did not take is visible as a residual offset or step.

Requires a nacelle-position column and aligned ERA5 direction; returns ``None`` otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.diagnostics import stages
from benchmarking.diagnostics.context import ERA5_WD_COL
from benchmarking.diagnostics.style import apply_grid, save_fig, series_style
from benchmarking.diagnostics.timeaxis import shade_segments

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.diagnostics.context import DiagnosticContext

_GENERATING_FRAC = 0.05  # keep rows above 5% of (proxy) rated power
_RATED_PERCENTILE = 99  # robust proxy for rated power


def _wrap180(deg: pd.Series) -> pd.Series:
    """Wrap an angle (degrees) to the [-180, 180) range."""
    return (deg + 180.0) % 360.0 - 180.0


def _monthly_circular_mean(error_deg: pd.Series) -> pd.Series:
    """Per-month circular mean (degrees) of an angle series, NaN-skipping."""
    rad = np.deg2rad(error_deg)
    sin = np.sin(rad).resample("MS").mean()
    cos = np.cos(rad).resample("MS").mean()
    return pd.Series(np.rad2deg(np.arctan2(sin, cos)), index=sin.index)


def plot_northing_error(ctx: DiagnosticContext) -> Path | None:
    """Per-turbine monthly circular-mean of (nacelle position - ERA5 direction) over time."""
    return _northing_error_figure(
        ctx,
        nacelle_col=ctx.columns.nacelle_position,
        stage=stages.INPUTS,
        filename="northing_error.png",
        described="no corrections",
    )


def plot_northed_error(ctx: DiagnosticContext) -> Path | None:
    """Draw the same timeline from the north-calibrated column the shared northing step wrote.

    What the model actually sees. A residual offset or a step surviving here is a correction that
    did not take, which the raw version cannot distinguish from one that was never applied.
    """
    if ctx.columns.nacelle_position is None:
        return None
    return _northing_error_figure(
        ctx,
        nacelle_col=ctx.columns.northed("nacelle_position"),
        stage=stages.FEATURE_ENG,
        filename="northed_error.png",
        described="after northing correction",
    )


def _northing_error_figure(
    ctx: DiagnosticContext, *, nacelle_col: str | None, stage: str, filename: str, described: str
) -> Path | None:
    """Draw the northing-error timeline from ``nacelle_col``; None when the inputs are not there."""
    if not ctx.has_column(nacelle_col) or ctx.era5_df is None or ERA5_WD_COL not in ctx.era5_df.columns:
        return None
    era5_wd = ctx.era5_df[ERA5_WD_COL].reindex(ctx.index)
    fig, ax = plt.subplots(figsize=(12, 6))
    shade_segments(ax, ctx)
    for position, turbine in enumerate([ctx.test_wtg, *ctx.references()]):
        nacelle = ctx.turbine_series(turbine, nacelle_col)
        power = ctx.turbine_series(turbine, ctx.columns.active_power)
        rated = np.nanpercentile(power.to_numpy(dtype=float), _RATED_PERCENTILE) if power.notna().any() else np.nan
        generating = power >= _GENERATING_FRAC * rated if np.isfinite(rated) else power.notna()
        error = _wrap180(nacelle - era5_wd).where(generating)
        monthly = _monthly_circular_mean(error)
        colour, dash = series_style(position)
        label = f"{turbine}{' (test)' if turbine == ctx.test_wtg else ''}"
        ax.plot(
            monthly.index.to_numpy(),
            monthly.to_numpy(),
            linewidth=1.0,
            marker=".",
            markersize=3,
            label=label,
            color=colour,
            linestyle=dash,
        )
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_xlabel("date")
    ax.set_ylabel(f"{nacelle_col} - {ERA5_WD_COL} [deg] (monthly circular mean)")
    ax.set_title(f"{ctx.test_wtg}: northing error over time (generating rows only, {described})")
    apply_grid(ax)
    ax.legend(ncol=2, fontsize="small")
    path = ctx.stage_dir(stage) / filename
    save_fig(fig, path)
    return path
