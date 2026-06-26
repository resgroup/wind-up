"""Northing-error timeline (feedback 2026-06-26, item 15).

The R-learner gets reference nacelle positions as raw features with **no** northing correction;
an offset or jumps in a turbine's yaw zero distorts the direction signal the model sees. This
plots, per turbine, the **monthly circular mean** of (nacelle position - ERA5 wind direction) over
time, so a drift or step in the offset stands out. Only rows where the turbine is generating
(≥ 5% of its rated power) are used, because a parked turbine often points away from the wind.

Requires a nacelle-position column and aligned ERA5 direction; returns ``None`` otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.diagnostics import stages
from benchmarking.diagnostics.context import ERA5_WD_COL
from benchmarking.diagnostics.style import apply_grid, save_fig
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
    if (
        not ctx.has_column(ctx.columns.nacelle_position)
        or ctx.era5_df is None
        or ERA5_WD_COL not in ctx.era5_df.columns
    ):
        return None
    era5_wd = ctx.era5_df[ERA5_WD_COL].reindex(ctx.index)
    fig, ax = plt.subplots(figsize=(12, 6))
    shade_segments(ax, ctx)
    for turbine in [ctx.test_wtg, *ctx.references()]:
        nacelle = ctx.turbine_series(turbine, ctx.columns.nacelle_position)
        power = ctx.turbine_series(turbine, ctx.columns.active_power)
        rated = np.nanpercentile(power.to_numpy(dtype=float), _RATED_PERCENTILE) if power.notna().any() else np.nan
        generating = power >= _GENERATING_FRAC * rated if np.isfinite(rated) else power.notna()
        error = _wrap180(nacelle - era5_wd).where(generating)
        monthly = _monthly_circular_mean(error)
        label = f"{turbine}{' (test)' if turbine == ctx.test_wtg else ''}"
        ax.plot(monthly.index.to_numpy(), monthly.to_numpy(), linewidth=1.0, marker=".", markersize=3, label=label)
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_xlabel("date")
    ax.set_ylabel(f"{ctx.columns.nacelle_position} - {ERA5_WD_COL} [deg] (monthly circular mean)")
    ax.set_title(f"{ctx.test_wtg}: northing error over time (generating rows only, no corrections)")
    apply_grid(ax)
    ax.legend(ncol=2, fontsize="small")
    path = ctx.stage_dir(stages.INPUTS) / "northing_error.png"
    save_fig(fig, path)
    return path
