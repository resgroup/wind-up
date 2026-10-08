"""Operating-state plots: the labelled records, for the analyst to confirm the labels.

Power is drawn here to check the labels, never to make them.

* :func:`plot_operating_states` -- one turbine's operating relationships as scatters coloured by state.
* :func:`plot_monthly_state_hours` -- stacked monthly hours per state, one panel per turbine.
* :func:`write_operating_state_plots` -- both, plus the hours CSV, for every turbine.
* :func:`plot_run_operating_states` -- the scatters for a run's test turbine and power references.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from benchmarking.diagnostics import stages
from benchmarking.diagnostics.ops_relationships import PAIRS
from benchmarking.diagnostics.style import apply_grid, save_fig
from benchmarking.harness.operating_state import (
    FULL_DOWNTIME,
    GENERIC_STATES,
    MISSING,
    NORMAL_OPERATION,
    PARTIAL_DOWNTIME,
    STATE_COL,
    state_hours,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from benchmarking.diagnostics.context import DiagnosticContext
    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)

STATE_HOURS_CSV = "operating_state_hours.csv"

_GENERIC_COLOURS = {NORMAL_OPERATION: "C0", PARTIAL_DOWNTIME: "C1", FULL_DOWNTIME: "C3", MISSING: "C7"}
# Site labels take these in name order.
_SITE_COLOURS = ("C2", "C4", "C5", "C6", "C8", "C9")
_MARKER_SIZE = 1.5
_HOURS_COLUMNS = 4


def state_order(states: Iterable[str]) -> list[str]:
    """Return the drawing order: normal operation, the site labels by name, then the downtime and missing states."""
    present = set(states)
    site = sorted(present - set(GENERIC_STATES))
    tail = [s for s in (PARTIAL_DOWNTIME, FULL_DOWNTIME, MISSING) if s in present]
    return [*([NORMAL_OPERATION] if NORMAL_OPERATION in present else []), *site, *tail]


def state_colours(states: Iterable[str]) -> dict[str, str]:
    """Return each state's colour; generic states have fixed colours, site labels take theirs in name order."""
    site = sorted(set(states) - set(GENERIC_STATES))
    colours = {s: _SITE_COLOURS[i % len(_SITE_COLOURS)] for i, s in enumerate(site)}
    return {**_GENERIC_COLOURS, **colours}


def plot_operating_states(
    rows: pd.DataFrame,
    *,
    turbine: str,
    columns: ColumnSchema,
    timebase: pd.Timedelta,
    out_dir: Path,
    title: str | None = None,
) -> Path | None:
    """Write ``operating_states_<turbine>.png``: each relationship of the step 1 pairs, coloured by state.

    A pair whose signals are unset or absent is left out. Nothing is written, and ``None`` returned,
    when ``rows`` carries no states or no pair can be drawn.

    :param rows: one turbine's labelled records, indexed by timestamp
    :param turbine: the turbine's name, for the file and title
    :param columns: the schema ``rows`` is keyed by
    :param timebase: the records' period, to give each state's hours
    :param out_dir: the folder written to
    :param title: the figure title; defaults to the turbine and the period covered
    """
    if STATE_COL not in rows.columns or rows.empty:
        return None
    pairs = [
        (getattr(columns, x), getattr(columns, y))
        for x, y in PAIRS
        if getattr(columns, x) in rows.columns and getattr(columns, y) in rows.columns
    ]
    if not pairs:
        return None
    states = rows[STATE_COL].astype(str)
    order = state_order(states.unique())
    colours = state_colours(order)
    hours = states.value_counts() * (timebase / pd.Timedelta(hours=1))

    n_cols = 2
    n_rows = math.ceil(len(pairs) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows), squeeze=False, layout="constrained")
    for ax, (x_col, y_col) in zip(axes.flat, pairs, strict=False):
        for state in order:
            chosen = (states == state).to_numpy()
            ax.plot(
                rows[x_col].to_numpy(dtype=float)[chosen],
                rows[y_col].to_numpy(dtype=float)[chosen],
                linestyle="none",
                marker=".",
                markersize=_MARKER_SIZE,
                color=colours[state],
                rasterized=True,
            )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}", fontsize="medium")
        apply_grid(ax)
    for ax in axes.flat[len(pairs) :]:
        ax.set_visible(False)
    handles = [
        Line2D([], [], linestyle="none", marker="o", color=colours[s], label=f"{s} ({hours[s]:,.0f} h)") for s in order
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 4), title="operating state")
    index = pd.DatetimeIndex(rows.index)
    fig.suptitle(
        title or f"{turbine}: operating states, {index.min():%Y-%m-%d} to {index.max():%Y-%m-%d}", fontsize="large"
    )
    path = out_dir / f"operating_states_{turbine}.png"
    save_fig(fig, path)
    return path


def plot_monthly_state_hours(
    labelled: pd.DataFrame, *, columns: ColumnSchema, timebase: pd.Timedelta, out_dir: Path
) -> Path | None:
    """Write ``operating_state_hours.png``: each turbine's hours per state per calendar month, stacked.

    The states other than normal operation sit at the bottom, so curtailment and downtime periods
    are visible over time. ``None`` when ``labelled`` carries no states.
    """
    if STATE_COL not in labelled.columns or labelled.empty:
        return None
    index = pd.DatetimeIndex(labelled.index)
    naive = index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index
    frame = pd.DataFrame(
        {
            "turbine": labelled[columns.turbine].astype(str).to_numpy(),
            "state": labelled[STATE_COL].astype(str).to_numpy(),
            "month": naive.to_period("M").to_timestamp(),
        }
    )
    frame["hours"] = timebase / pd.Timedelta(hours=1)
    order = state_order(frame["state"].unique())
    stack = [*order[1:], *order[:1]] if order and order[0] == NORMAL_OPERATION else order
    colours = state_colours(order)
    turbines = sorted(frame["turbine"].unique())
    months = pd.date_range(frame["month"].min(), frame["month"].max(), freq="MS")

    n_rows = math.ceil(len(turbines) / _HOURS_COLUMNS)
    fig, axes = plt.subplots(
        n_rows,
        _HOURS_COLUMNS,
        figsize=(4.5 * _HOURS_COLUMNS, 2.6 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    width = 25.0  # days
    for ax, turbine in zip(axes.flat, turbines, strict=False):
        wide = (
            frame[frame["turbine"] == turbine]
            .pivot_table(index="month", columns="state", values="hours", aggfunc="sum")
            .reindex(index=months, columns=stack)
            .fillna(0.0)
        )
        bottom = np.zeros(len(months))
        for state in stack:
            values = wide[state].to_numpy()
            ax.bar(months, values, width=width, bottom=bottom, color=colours[state], align="edge", linewidth=0)
            bottom += values
        ax.set_title(turbine, fontsize="medium")
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        apply_grid(ax)
    for ax in axes.flat[len(turbines) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("hours per month")
    handles = [Line2D([], [], linestyle="none", marker="s", color=colours[s], label=s) for s in order]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 6), title="operating state")
    fig.suptitle(f"hours per operating state per month, {naive.min():%Y-%m-%d} to {naive.max():%Y-%m-%d}")
    path = out_dir / "operating_state_hours.png"
    save_fig(fig, path)
    return path


def write_operating_state_plots(
    labelled: pd.DataFrame, *, columns: ColumnSchema, timebase: pd.Timedelta, out_dir: Path
) -> list[Path]:
    """Write every turbine's operating-state scatters, the hours CSV and the monthly hours plot.

    :param labelled: long-format SCADA with the operating-state columns, indexed by timestamp
    :param columns: the schema ``labelled`` is keyed by
    :param timebase: the records' period
    :param out_dir: the folder written to
    :return: the files written; none when ``labelled`` carries no states
    """
    if STATE_COL not in labelled.columns:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    turbine_names = labelled[columns.turbine].astype(str)
    written: list[Path] = []
    for turbine in sorted(turbine_names.unique()):
        rows = labelled[(turbine_names == turbine).to_numpy()]
        path = plot_operating_states(rows, turbine=turbine, columns=columns, timebase=timebase, out_dir=out_dir)
        if path is not None:
            written.append(path)
    csv = out_dir / STATE_HOURS_CSV
    state_hours(labelled, columns=columns, timebase=timebase).to_csv(csv, index=False)
    written.append(csv)
    path = plot_monthly_state_hours(labelled, columns=columns, timebase=timebase, out_dir=out_dir)
    if path is not None:
        written.append(path)
    logger.info("Wrote %d operating-state files to %s", len(written), out_dir)
    return written


def plot_run_operating_states(ctx: DiagnosticContext) -> list[Path]:
    """Write the scatters for a run's test turbine and power references over the span the run sees."""
    if STATE_COL not in ctx.scada_df.columns:
        return []
    references = ctx.power_references if ctx.power_references is not None else ctx.references()
    out_dir = ctx.stage_dir(stages.OPERATING_STATES)
    written: list[Path] = []
    for turbine in [ctx.test_wtg, *sorted(r for r in references if r != ctx.test_wtg)]:
        rows = ctx.scada_df[ctx.scada_df[ctx.turbine_col] == turbine]
        if rows.empty:
            continue
        role = "test turbine" if turbine == ctx.test_wtg else "power reference"
        index = pd.DatetimeIndex(rows.index)
        period = f"{index.min():%Y-%m-%d} to {index.max():%Y-%m-%d}"
        path = plot_operating_states(
            rows,
            turbine=turbine,
            columns=ctx.columns,
            timebase=ctx.timebase,
            out_dir=out_dir,
            title=f"{turbine} ({role}): operating states over the span, {period}",
        )
        if path is not None:
            written.append(path)
    return written
