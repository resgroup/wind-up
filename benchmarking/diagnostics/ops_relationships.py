"""Operating relationships over time: how one turbine signal relates to another, month by month.

Step 1 of the method looks for undeclared changes in the relationship between power, wind speed,
pitch angle and rotor speed. Each relationship is summarised by binning the y signal by the x signal
and taking the bin means per calendar month, with the bin edges fixed over the whole record so the
months can be compared. A change shows as a step in one or more bins' monthly series.

Only producing, fully available records are binned, so downtime and start-up do not mix into the
means. The turbine's own wind speed is used: this is inspection, not a model input.

* :func:`monthly_binned_means` -- the tidy frame, one row per month and bin.
* :func:`plot_ops_relationships` -- one figure per turbine: each relationship as a time series per
  bin and as a curve per month.
* :func:`plot_run_ops_relationships` -- the figures for a run's test turbine and power references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

from benchmarking.diagnostics import stages
from benchmarking.diagnostics.style import apply_grid, save_fig

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.diagnostics.context import DiagnosticContext
    from benchmarking.synthetic import ColumnSchema

# (x role, y role) on the ColumnSchema, in reading order.
PAIRS: tuple[tuple[str, str], ...] = (
    ("wind_speed", "active_power"),
    ("active_power", "gen_rpm"),
    ("active_power", "pitch"),
    ("wind_speed", "gen_rpm"),
    ("wind_speed", "pitch"),
)

N_BINS = 10
MIN_HOURS = 6.0

_COLUMNS = ["month", "bin", "x_lo", "x_hi", "x_mean", "y_mean", "hours"]
# Single-hue sequential ramp, light to dark; the lightest end is skipped so every line stays visible.
_CMAP = plt.get_cmap("Blues")
_CMAP_FLOOR = 0.3


def operating_mask(rows: pd.DataFrame, *, columns: ColumnSchema, timebase: pd.Timedelta) -> pd.Series:
    """Return True for producing (power > 0), fully available records; a missing counter is not available."""
    available = rows[columns.availability] >= timebase.total_seconds()
    return ((rows[columns.active_power] > 0) & available).fillna(value=False).astype(bool)


def monthly_binned_means(
    x: pd.Series,
    y: pd.Series,
    *,
    timebase: pd.Timedelta,
    n_bins: int = N_BINS,
    min_hours: float = MIN_HOURS,
) -> pd.DataFrame:
    """Return the mean of ``x`` and ``y`` per calendar month and quantile bin of ``x``.

    The bin edges are the quantiles of all of ``x``, so they do not move from month to month;
    duplicate edges, from a signal that sits at one value for a large share of the record, are merged.
    A month-bin cell with less than ``min_hours`` of data is dropped. Records missing either signal
    are ignored.

    :param x: the binning signal, indexed by timestamp
    :param y: the binned signal, on the same index
    :param timebase: the records' period, to express the data behind each mean in hours
    :param n_bins: how many quantile bins of ``x``
    :param min_hours: the least data a month-bin cell needs to be kept
    :return: one row per kept cell, columns ``month`` (start of the UTC calendar month), ``bin``,
        ``x_lo``/``x_hi`` (the bin's edges), ``x_mean``, ``y_mean`` and ``hours``
    """
    frame = pd.DataFrame({"x": x.to_numpy(dtype=float), "y": y.to_numpy(dtype=float)}, index=x.index).dropna()
    if frame.empty:
        return pd.DataFrame(columns=_COLUMNS)
    edges = np.unique(np.quantile(frame["x"].to_numpy(), np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:  # noqa: PLR2004 - a constant signal has one value and so one bin
        edges = np.array([edges[0], edges[0]])
        frame["bin"] = 0
    else:
        frame["bin"] = pd.cut(frame["x"], edges, labels=False, include_lowest=True).astype(int)
    index = pd.DatetimeIndex(frame.index)
    naive = index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index
    frame["month"] = naive.to_period("M").to_timestamp().tz_localize("UTC")
    grouped = frame.groupby(["month", "bin"])
    result = grouped.agg(x_mean=("x", "mean"), y_mean=("y", "mean"), n=("x", "size")).reset_index()
    result["hours"] = result["n"] * (timebase / pd.Timedelta(hours=1))
    result["x_lo"] = edges[result["bin"].to_numpy()]
    result["x_hi"] = edges[np.minimum(result["bin"].to_numpy() + 1, len(edges) - 1)]
    return result.loc[result["hours"] >= min_hours, _COLUMNS].reset_index(drop=True)


def _colour(fraction: float) -> tuple[float, float, float, float]:
    """Return the ramp's colour ``fraction`` of the way from its lightest used step to its darkest."""
    return _CMAP(_CMAP_FLOOR + (1 - _CMAP_FLOOR) * fraction)


def _ramp(n: int) -> list[tuple[float, float, float, float]]:
    """``n`` colours along the ramp, lightest first."""
    return [_colour(i / max(n - 1, 1)) for i in range(n)]


def _draw_by_bin(ax: plt.Axes, cells: pd.DataFrame, *, x_label: str, y_label: str) -> None:
    """Bin-mean ``y`` over months, one line per bin, coloured by bin; a gap where a month has no data."""
    months = pd.date_range(cells["month"].min(), cells["month"].max(), freq="MS")
    wide = cells.pivot_table(index="month", columns="bin", values="y_mean").reindex(months)
    bins = list(wide.columns)
    colours = _ramp(len(bins))
    for colour, b in zip(colours, bins, strict=True):
        ax.plot(wide.index.to_numpy(), wide[b].to_numpy(), color=colour, linewidth=1.5, marker=".", markersize=3)
    edges = cells.drop_duplicates("bin").sort_values("bin")
    boundaries = [*edges["x_lo"], float(edges["x_hi"].iloc[-1])]
    if len(set(boundaries)) == len(boundaries):
        mappable = ScalarMappable(norm=BoundaryNorm(boundaries, len(colours)), cmap=_listed(colours))
        bar = ax.figure.colorbar(mappable, ax=ax, pad=0.01)
        bar.set_label(f"{x_label} bin", fontsize="small")
        bar.ax.tick_params(labelsize="x-small")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_ylabel(y_label)
    apply_grid(ax)


def _draw_by_month(ax: plt.Axes, cells: pd.DataFrame, *, x_label: str, y_label: str) -> None:
    """Bin-mean ``y`` against bin-mean ``x``, one line per month, coloured by date."""
    months = sorted(cells["month"].unique())
    first, last = mdates.date2num(months[0]), mdates.date2num(months[-1])
    norm = Normalize(vmin=first, vmax=max(last, first + 1))
    for month in months:
        cell = cells[cells["month"] == month].sort_values("bin")
        colour = _colour(float(norm(mdates.date2num(month))))
        ax.plot(
            cell["x_mean"].to_numpy(), cell["y_mean"].to_numpy(), color=colour, linewidth=1.2, marker=".", markersize=3
        )
    mappable = ScalarMappable(norm=norm, cmap=_listed(_ramp(256)))
    bar = ax.figure.colorbar(mappable, ax=ax, pad=0.01)
    bar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    bar.ax.tick_params(labelsize="x-small")
    bar.set_label("month", fontsize="small")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    apply_grid(ax)


def _listed(colours: list[tuple[float, float, float, float]]) -> ListedColormap:
    """Return a colormap of exactly ``colours``."""
    return ListedColormap(colours)


def plot_ops_relationships(
    rows: pd.DataFrame,
    *,
    turbine: str,
    columns: ColumnSchema,
    timebase: pd.Timedelta,
    out_dir: Path,
    title: str | None = None,
    n_bins: int = N_BINS,
    min_hours: float = MIN_HOURS,
) -> Path | None:
    """Write ``ops_relationships_<turbine>.png``: each relationship of :data:`PAIRS` by bin and by month.

    A relationship whose signals are unset or absent is left out. Nothing is written, and ``None``
    returned, when no relationship has producing, fully available data.

    :param rows: one turbine's records, indexed by timestamp
    :param turbine: the turbine's name, for the file and title
    :param columns: the schema ``rows`` is keyed by
    :param timebase: the records' period
    :param out_dir: the folder written to
    :param title: the figure title; defaults to the turbine and the period covered
    """
    kept = rows[operating_mask(rows, columns=columns, timebase=timebase).to_numpy()]
    panels: list[tuple[str, str, pd.DataFrame]] = []
    for x_role, y_role in PAIRS:
        x_col, y_col = getattr(columns, x_role), getattr(columns, y_role)
        if x_col is None or y_col is None or x_col not in kept.columns or y_col not in kept.columns:
            continue
        cells = monthly_binned_means(kept[x_col], kept[y_col], timebase=timebase, n_bins=n_bins, min_hours=min_hours)
        if not cells.empty:
            panels.append((x_col, y_col, cells))
    if not panels:
        return None

    fig, axes = plt.subplots(len(panels), 2, figsize=(16, 3.4 * len(panels)), squeeze=False, layout="constrained")
    for (x_col, y_col, cells), (left, right) in zip(panels, axes, strict=True):
        _draw_by_bin(left, cells, x_label=x_col, y_label=y_col)
        left.set_title(f"{y_col} by {x_col} bin, monthly mean", fontsize="medium")
        _draw_by_month(right, cells, x_label=x_col, y_label=y_col)
        right.set_title(f"{y_col} vs {x_col}, one line per month", fontsize="medium")
    index = pd.DatetimeIndex(rows.index)
    period = f"{index.min():%Y-%m-%d} to {index.max():%Y-%m-%d}" if len(index) else ""
    fig.suptitle(
        title or f"{turbine}: operating relationships over time, {period} (power > 0, fully available)",
        fontsize="large",
    )
    path = out_dir / f"ops_relationships_{turbine}.png"
    save_fig(fig, path)
    return path


def plot_run_ops_relationships(ctx: DiagnosticContext) -> list[Path]:
    """Write the figures for a run's test turbine and power references, over the span the run sees."""
    references = ctx.power_references if ctx.power_references is not None else ctx.references()
    written: list[Path] = []
    for turbine in [ctx.test_wtg, *sorted(r for r in references if r != ctx.test_wtg)]:
        rows = ctx.scada_df[ctx.scada_df[ctx.turbine_col] == turbine]
        role = "test turbine" if turbine == ctx.test_wtg else "power reference"
        index = pd.DatetimeIndex(rows.index)
        period = f"{index.min():%Y-%m-%d} to {index.max():%Y-%m-%d}" if len(index) else ""
        path = plot_ops_relationships(
            rows,
            turbine=turbine,
            columns=ctx.columns,
            timebase=ctx.timebase,
            out_dir=ctx.stage_dir(stages.INPUTS),
            title=f"{turbine} ({role}): operating relationships over the span, {period} (power > 0, fully available)",
        )
        if path is not None:
            written.append(path)
    return written
