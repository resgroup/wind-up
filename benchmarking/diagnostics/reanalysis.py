"""Reanalysis plots (method step 3): the time-shift check and ERA5 against the site wind speed.

* :func:`write_reanalysis_outputs` -- ``shift_check.png``, ``wind_speed_scatter.png`` and ``reanalysis.csv``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.diagnostics.style import apply_grid, save_fig
from benchmarking.harness.reanalysis import ERA5_WS_RAW, SHIFT_RAISE, SHIFT_WARN, minutes

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.harness.reanalysis import ReanalysisResult

_HOUR = pd.Timedelta(hours=1)
_MAX_SCATTER_POINTS = 50_000


def write_reanalysis_outputs(
    result: ReanalysisResult, *, era5_hourly_df: pd.DataFrame, cell_selection: str, out_dir: Path
) -> None:
    """Write the step 3 plots and summary CSV into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_shift_check(result, path=out_dir / "shift_check.png")
    plot_wind_speed_scatter(result, path=out_dir / "wind_speed_scatter.png")
    summary(result, era5_hourly_df=era5_hourly_df, cell_selection=cell_selection).to_csv(
        out_dir / "reanalysis.csv", index=False
    )


def plot_shift_check(result: ReanalysisResult, *, path: Path) -> None:
    """Plot the correlation against shift, the best shift marked and the warn and raise bands shaded."""
    sweep = result.check.sweep
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axvspan(-SHIFT_RAISE / _HOUR, SHIFT_RAISE / _HOUR, color="C1", alpha=0.12, label="warn")
    ax.axvspan(
        -SHIFT_WARN / _HOUR,
        SHIFT_WARN / _HOUR,
        color="C2",
        alpha=0.18,
        label="pass",
    )
    ax.plot(sweep["shift"] / _HOUR, sweep["corr"], color="C0")
    if np.isfinite(result.best_corr):
        ax.axvline(
            result.best_shift / _HOUR,
            color="C3",
            linestyle="--",
            label=f"best shift {minutes(result.best_shift)} (corr {result.best_corr:.3f})",
        )
    ax.set_xlabel("shift of ERA5 [h] (positive: the site lags ERA5)")
    ax.set_ylabel("correlation with the site wind speed")
    warn_min = SHIFT_WARN // pd.Timedelta(minutes=1)
    raise_min = SHIFT_RAISE // pd.Timedelta(minutes=1)
    ax.set_title(f"ERA5 time-shift check (warn beyond ±{warn_min} min, raise at ±{raise_min} min or more)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    apply_grid(ax)
    save_fig(fig, path)


def plot_wind_speed_scatter(result: ReanalysisResult, *, path: Path) -> None:
    """Scatter ERA5 100 m wind speed against the site wind speed, unshifted."""
    fig, ax = plt.subplots(figsize=(6, 6))
    pair = pd.DataFrame({"era5": result.aligned[ERA5_WS_RAW], "site": result.site_ws}).dropna()
    if len(pair) > _MAX_SCATTER_POINTS:
        pair = pair.sample(_MAX_SCATTER_POINTS, random_state=0)
    ax.scatter(pair["site"], pair["era5"], s=2, alpha=0.2, color="C0")
    top = float(np.nanmax([pair["site"].max(), pair["era5"].max(), 1.0])) if len(pair) else 1.0
    ax.plot([0, top], [0, top], color="k", linewidth=0.8)
    ax.set_xlabel("mean nacelle wind speed of the unchanged turbines [m/s]")
    ax.set_ylabel("ERA5 100 m wind speed [m/s]")
    ax.set_title(f"ERA5 against the site, corr {result.check.zero_corr:.3f}")
    apply_grid(ax)
    save_fig(fig, path)


def summary(result: ReanalysisResult, *, era5_hourly_df: pd.DataFrame, cell_selection: str) -> pd.DataFrame:
    """Return the one-row step 3 summary."""
    populated = era5_hourly_df.index[era5_hourly_df.notna().any(axis=1)]
    index = result.aligned.index
    timebase = pd.Timedelta(pd.Series(index).diff().median()) if len(index) > 1 else pd.Timedelta(0)
    uncovered_hours = sum(((b - a) + timebase) / _HOUR for a, b in result.uncovered)
    return pd.DataFrame(
        [
            {
                "era5_first_hour": populated.min(),
                "era5_last_hour": populated.max(),
                "scada_first_record": index.min(),
                "scada_last_record": index.max(),
                "uncovered_hours": uncovered_hours,
                "best_shift_minutes": result.best_shift / pd.Timedelta(minutes=1),
                "best_corr": result.best_corr,
                "zero_shift_corr": result.check.zero_corr,
                "cell_selection": cell_selection,
            }
        ]
    )
