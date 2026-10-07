"""Step 1 input-data plots: every turbine and every record provided, before anything is chosen.

The analyst reviews these to confirm the data is fit for purpose and to define exclusions and the
operating-state labels. They are drawn before the test turbine's span or references are chosen, so
nothing is cut: no exclusion, works window or span is applied. The per-run plots in a run's
``1_inputs`` folder show the same things for the test turbine and its references over the span.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from benchmarking.diagnostics.context import infer_timebase
from benchmarking.diagnostics.coverage import weekly_coverage
from benchmarking.diagnostics.curves import monthly_power_factor
from benchmarking.diagnostics.ops_relationships import plot_ops_relationships
from benchmarking.diagnostics.style import apply_grid, save_fig, series_style

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)


def write_input_data_plots(scada_df: pd.DataFrame, *, columns: ColumnSchema, out_dir: Path) -> list[Path]:
    """Write the farm-wide input-data plots for every turbine in ``scada_df`` and return their paths.

    One operating-relationships figure per turbine, and the power factor and data coverage of every
    turbine over the whole record. A plot whose signal is absent is skipped.

    :param scada_df: long-format source-native SCADA, indexed by timestamp
    :param columns: the schema ``scada_df`` is keyed by
    :param out_dir: the folder written to
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    timebase = infer_timebase(index)
    turbines = sorted(str(t) for t in pd.unique(scada_df[columns.turbine]))
    by_turbine = {t: scada_df[scada_df[columns.turbine].astype(str) == t] for t in turbines}

    written: list[Path] = []
    for turbine, rows in by_turbine.items():
        path = plot_ops_relationships(rows, turbine=turbine, columns=columns, timebase=timebase, out_dir=out_dir)
        if path is not None:
            written.append(path)
    written.append(_plot_coverage(by_turbine, columns=columns, index=index, timebase=timebase, out_dir=out_dir))
    if columns.reactive_power is not None and columns.reactive_power in scada_df.columns:
        written.append(_plot_power_factor(by_turbine, columns=columns, index=index, out_dir=out_dir))
    logger.info("Wrote %d input-data plots for %d turbines to %s", len(written), len(turbines), out_dir)
    return written


def _aligned(rows: pd.DataFrame, col: str, *, index: pd.DatetimeIndex) -> pd.Series:
    """One turbine's ``col`` on the farm's whole index, NaN where it has no record."""
    series = pd.Series(rows[col].to_numpy(dtype=float), index=pd.DatetimeIndex(rows.index))
    return series[~series.index.duplicated()].reindex(index)


def _period(index: pd.DatetimeIndex) -> str:
    return f"{index.min():%Y-%m-%d} to {index.max():%Y-%m-%d}"


def _plot_coverage(
    by_turbine: dict[str, pd.DataFrame],
    *,
    columns: ColumnSchema,
    index: pd.DatetimeIndex,
    timebase: pd.Timedelta,
    out_dir: Path,
) -> Path:
    """Weekly coverage of each turbine's active power over the whole record."""
    fig, ax = plt.subplots(figsize=(14, 6))
    for position, (turbine, rows) in enumerate(by_turbine.items()):
        present = _aligned(rows, columns.active_power, index=index).notna()
        weekly = weekly_coverage(present, timebase=timebase)
        colour, dash = series_style(position)
        ax.plot(weekly.index.to_numpy(), weekly.to_numpy(), linewidth=1.0, label=turbine, color=colour, linestyle=dash)
    ax.set_ylim(0, 105)
    ax.set_xlabel("date")
    ax.set_ylabel(f"weekly {columns.active_power} coverage [%]")
    ax.set_title(f"input data coverage, every turbine, {_period(index)} (all data provided)")
    apply_grid(ax)
    ax.legend(ncol=2, fontsize="small", loc="lower left")
    path = out_dir / "input_data_coverage.png"
    save_fig(fig, path)
    return path


def _plot_power_factor(
    by_turbine: dict[str, pd.DataFrame], *, columns: ColumnSchema, index: pd.DatetimeIndex, out_dir: Path
) -> Path:
    """Monthly active-power-weighted power factor of each turbine over the whole record."""
    fig, ax = plt.subplots(figsize=(14, 6))
    for position, (turbine, rows) in enumerate(by_turbine.items()):
        monthly = monthly_power_factor(
            _aligned(rows, columns.active_power, index=index),
            _aligned(rows, columns.reactive_power, index=index),  # type: ignore[arg-type]  # checked by the caller
        )
        colour, dash = series_style(position)
        ax.plot(
            monthly.index.to_numpy(),
            monthly.to_numpy(),
            linewidth=1.0,
            marker=".",
            markersize=3,
            label=turbine,
            color=colour,
            linestyle=dash,
        )
    ax.set_xlabel("date")
    ax.set_ylabel("power factor (active-power-weighted monthly mean)")
    ax.set_title(f"power factor over time, every turbine, {_period(index)} (all data provided)")
    apply_grid(ax)
    ax.legend(ncol=2, fontsize="small", loc="lower left")
    path = out_dir / "power_factor.png"
    save_fig(fig, path)
    return path
