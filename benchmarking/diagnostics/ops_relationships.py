"""Operating relationships over time: how one turbine signal relates to another, month by month.

Step 1 of the method looks for undeclared changes in the relationship between power, wind speed,
pitch angle and rotor speed. Each relationship is summarised by binning the y signal by the x signal
and taking the bin means per calendar month, with the bin edges fixed over the whole record so the
months can be compared. The bins are deciles of x; against wind speed the top decile is split at
the 95th and 97.5th percentiles. A change shows as a step in one or more bins' monthly series.

Only producing, fully available records are binned, so downtime and start-up do not mix into the
means. The turbine's own wind speed is used: this is inspection, not a model input.

* :func:`monthly_binned_means` -- the tidy frame, one row per month and bin.
* :func:`plot_ops_relationships` -- one figure per turbine: each relationship as a time series per
  bin and as a curve per month, both coloured by date.
* :func:`plot_run_ops_relationships` -- the figures for a run's test turbine and power references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize

from benchmarking.diagnostics import stages
from benchmarking.diagnostics.style import apply_grid, save_fig
from benchmarking.diagnostics.timeaxis import upgrade_start

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from benchmarking.diagnostics.context import DiagnosticContext
    from benchmarking.synthetic import ColumnSchema

# (x role, y role) on the ColumnSchema, in reading order: each row a signal against wind speed, then
# against power.
PAIRS: tuple[tuple[str, str], ...] = (
    ("wind_speed", "active_power"),
    ("active_power", "reactive_power"),
    ("wind_speed", "gen_rpm"),
    ("active_power", "gen_rpm"),
    ("wind_speed", "pitch"),
    ("active_power", "pitch"),
)

N_BINS = 10
# Quantiles added to the deciles' edges when binning by wind speed.
WIND_SPEED_EXTRA_QUANTILES: tuple[float, ...] = (0.95, 0.975)
MIN_HOURS = 6.0

_COLUMNS = ["month", "bin", "x_lo", "x_hi", "x_mean", "y_mean", "hours"]
# Blue through purple to red, with no pale step, so every line stays visible.
_CMAP = LinearSegmentedColormap.from_list("blue_red", ["#2166ac", "#7b3294", "#d7191c"])
_CMAP_FLOOR = 0.0


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
    extra_quantiles: Sequence[float] = (),
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
    :param extra_quantiles: quantiles of ``x`` added as edges, splitting the bins they fall in
    :param min_hours: the least data a month-bin cell needs to be kept
    :return: one row per kept cell, columns ``month`` (start of the UTC calendar month), ``bin``,
        ``x_lo``/``x_hi`` (the bin's edges), ``x_mean``, ``y_mean`` and ``hours``
    """
    frame = pd.DataFrame({"x": x.to_numpy(dtype=float), "y": y.to_numpy(dtype=float)}, index=x.index).dropna()
    if frame.empty:
        return pd.DataFrame(columns=_COLUMNS)
    levels = np.unique([*np.linspace(0, 1, n_bins + 1), *extra_quantiles])
    edges = np.unique(np.quantile(frame["x"].to_numpy(), levels))
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


def _draw_by_bin(
    ax: plt.Axes, cells: pd.DataFrame, *, x_label: str, y_label: str, changeovers: Sequence[pd.Timestamp]
) -> None:
    """Draw bin-mean ``y`` over months, one line per bin labelled with its range, coloured by date.

    A month with no data leaves a gap.
    """
    months = pd.date_range(cells["month"].min(), cells["month"].max(), freq="MS")
    wide = cells.pivot_table(index="month", columns="bin", values="y_mean").reindex(months)
    norm = _month_norm(cells)
    when = mdates.date2num(months.to_pydatetime())
    colours = np.array([_colour(float(norm(t))) for t in when])
    edges = cells.drop_duplicates("bin").set_index("bin")
    levels: dict[str, float] = {}
    for b in wide.columns:
        y = wide[b].to_numpy(dtype=float)
        joined = np.isfinite(y[:-1]) & np.isfinite(y[1:])
        segments = np.stack([np.column_stack([when[:-1], y[:-1]]), np.column_stack([when[1:], y[1:]])], axis=1)
        ax.add_collection(LineCollection(segments[joined], colors=colours[1:][joined], linewidths=1.5))
        finite = np.isfinite(y)
        ax.scatter(when[finite], y[finite], c=colours[finite], s=6, zorder=3)
        if finite.any():
            levels[f"{_short(edges.loc[b, 'x_lo'])}-{_short(edges.loc[b, 'x_hi'])}"] = float(np.median(y[finite]))
    ax.set_xlim(when[0] - 15, when[-1] + 15)
    ax.autoscale(axis="y")
    _label_lines(ax, levels)
    for changeover in changeovers:
        ax.axvline(changeover, color="k", linestyle="--", linewidth=1.2)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} by {x_label} bin (labelled at right), monthly mean", fontsize="medium")
    apply_grid(ax)


def _short(value: float) -> str:
    """Return a bin edge rounded to suit its size."""
    size = abs(value)
    return f"{value:.0f}" if size >= 100 else f"{value:.1f}" if size >= 10 else f"{value:.2f}"  # noqa: PLR2004


def _label_lines(ax: plt.Axes, levels: dict[str, float], *, gap: float = 0.06) -> None:
    """Write each label just right of the axes at its level, pushed apart so none overlap."""
    if not levels:
        return
    low, high = ax.get_ylim()
    order = sorted(levels, key=levels.__getitem__)
    spots = [(levels[name] - low) / (high - low) for name in order]
    for i in range(1, len(spots)):
        spots[i] = max(spots[i], spots[i - 1] + gap)
    overshoot = spots[-1] - 1.0
    if overshoot > 0:
        spots = [spot - overshoot for spot in spots]
        for i in range(len(spots) - 2, -1, -1):
            spots[i] = min(spots[i], spots[i + 1] - gap)
    for name, spot in zip(order, spots, strict=True):
        ax.annotate(
            name,
            (1.0, spot),
            xycoords="axes fraction",
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize="xx-small",
            color="0.3",
            annotation_clip=False,
        )


def _month_norm(cells: pd.DataFrame) -> Normalize:
    """Return the date scale every panel of a relationship is coloured on."""
    months = sorted(cells["month"].unique())
    first, last = mdates.date2num(months[0]), mdates.date2num(months[-1])
    return Normalize(vmin=first, vmax=max(last, first + 1))


def _draw_by_month(
    ax: plt.Axes, cells: pd.DataFrame, *, x_label: str, y_label: str, changeovers: Sequence[pd.Timestamp]
) -> None:
    """Bin-mean ``y`` against bin-mean ``x``, one line per month, coloured by date; changeovers on the colour bar."""
    months = sorted(cells["month"].unique())
    norm = _month_norm(cells)
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
    for changeover in changeovers:
        bar.ax.axhline(mdates.date2num(changeover), color="k", linestyle="--", linewidth=1.2)
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
    changeovers: Sequence[pd.Timestamp] = (),
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
    :param changeovers: dates drawn as dashed lines on the time axes and colour bars
    """
    kept = rows[operating_mask(rows, columns=columns, timebase=timebase).to_numpy()]
    panels: list[tuple[str, str, pd.DataFrame]] = []
    for x_role, y_role in PAIRS:
        x_col, y_col = getattr(columns, x_role), getattr(columns, y_role)
        if x_col is None or y_col is None or x_col not in kept.columns or y_col not in kept.columns:
            continue
        cells = monthly_binned_means(
            kept[x_col],
            kept[y_col],
            timebase=timebase,
            n_bins=n_bins,
            extra_quantiles=WIND_SPEED_EXTRA_QUANTILES if x_role == "wind_speed" else (),
            min_hours=min_hours,
        )
        if not cells.empty:
            panels.append((x_col, y_col, cells))
    if not panels:
        return None

    fig, axes = plt.subplots(len(panels), 2, figsize=(16, 3.4 * len(panels)), squeeze=False, layout="constrained")
    for (x_col, y_col, cells), (left, right) in zip(panels, axes, strict=True):
        _draw_by_bin(left, cells, x_label=x_col, y_label=y_col, changeovers=changeovers)
        _draw_by_month(right, cells, x_label=x_col, y_label=y_col, changeovers=changeovers)
        right.set_title(f"{y_col} vs {x_col}, one line per month", fontsize="medium")
    index = pd.DatetimeIndex(rows.index)
    period = f"{index.min():%Y-%m-%d} to {index.max():%Y-%m-%d}" if len(index) else ""
    title = title or f"{turbine}: operating relationships over time, {period} (power > 0, fully available)"
    if changeovers:
        title += "; dashed: changeover"
    fig.suptitle(title, fontsize="large")
    path = out_dir / f"ops_relationships_{turbine}.png"
    save_fig(fig, path)
    return path


def plot_run_ops_relationships(ctx: DiagnosticContext) -> list[Path]:
    """Write the figures for a run's test turbine and power references, over the span the run sees."""
    references = ctx.plotted_references()
    start = upgrade_start(ctx) if ctx.mode == "prepost" else None
    changeovers = [start] if start is not None else []
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
            out_dir=ctx.stage_dir(stages.CHANGES),
            title=f"{turbine} ({role}): operating relationships over the span, {period} (power > 0, fully available)",
            changeovers=changeovers,
        )
        if path is not None:
            written.append(path)
    return written
