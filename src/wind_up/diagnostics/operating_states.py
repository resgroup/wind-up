"""Operating-state plots: the labelled records, for the analyst to confirm the labels.

Power is drawn here to check the labels, never to make them.

* :func:`plot_operating_states` -- one turbine's operating relationships as scatters coloured by state.
* :func:`plot_monthly_state_hours` -- stacked monthly hours per state, one panel per turbine.
* :func:`write_operating_state_plots` -- both, plus the hours CSV, for every turbine.
* :func:`plot_run_operating_states` -- the scatters for a run's test turbine and power references.
* :func:`write_validity_plots` -- the scatters of each turbine's records split by their validity for
  northing (:data:`NORTHING_VIEWS`) or waking (:data:`WAKING_VIEWS`), still coloured by state.
* :func:`plot_run_uplift_validity` -- a run's test turbine's and power references' records used for uplift by
  step 11, and those not used, coloured by why.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from wind_up.diagnostics import stages
from wind_up.diagnostics.ops_relationships import ALL_TURBINES, PAIRS
from wind_up.diagnostics.style import apply_grid, save_fig
from wind_up.operating_state import (
    FULL_DOWNTIME,
    GENERIC_STATES,
    MISSING,
    NORMAL_OPERATION,
    NOT_WAKING,
    PART_WAKING,
    PARTIAL_DOWNTIME,
    STATE_COL,
    VALID_NORTHING_COL,
    WAKING,
    WAKING_STATE_COL,
    state_hours,
)
from wind_up.record_selection import CATEGORIES

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from wind_up.diagnostics.context import DiagnosticContext
    from wind_up.schema import ColumnSchema

logger = logging.getLogger(__name__)

STATE_HOURS_CSV = "operating_state_hours.csv"

_GENERIC_COLOURS = {NORMAL_OPERATION: "C0", PARTIAL_DOWNTIME: "C1", FULL_DOWNTIME: "C3", MISSING: "C7"}
# Site labels take these in name order.
_SITE_COLOURS = ("C2", "C4", "C5", "C6", "C8", "C9")
_MARKER_SIZE = 1.5
_HOURS_COLUMNS = 4
_LIMIT_MARGIN = 0.03


@dataclass(frozen=True)
class ValidityView:
    """The records whose ``column`` equals ``value``, written as ``<name>_<turbine>.png`` and titled ``label``."""

    name: str
    label: str
    column: str
    value: object


NORTHING_VIEWS = (
    ValidityView("northing_used", "used for northing", VALID_NORTHING_COL, value=True),
    ValidityView("northing_not_used", "not used for northing", VALID_NORTHING_COL, value=False),
)
WAKING_VIEWS = (
    ValidityView("waking", "considered waking", WAKING_STATE_COL, WAKING),
    ValidityView("part_waking", "considered part waking", WAKING_STATE_COL, PART_WAKING),
    ValidityView("not_waking", "considered not waking", WAKING_STATE_COL, NOT_WAKING),
)


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
    name: str = "operating_states",
    subset: np.ndarray | None = None,
    colour_by: np.ndarray | None = None,
    colour_order: Sequence[str] = (),
    legend_title: str = "operating state",
) -> Path | None:
    """Write ``<name>_<turbine>.png``: each relationship of the step 1 pairs, coloured by state.

    The pairs sit two to a row, against wind speed on the left and against power on the right; a pair
    whose signals are unset, absent or empty shows why in its slot. Nothing is written, and ``None``
    returned, when ``rows`` carries no states or no pair can be drawn.

    :param rows: one turbine's labelled records, indexed by timestamp
    :param turbine: the turbine's name, for the file and title
    :param columns: the schema ``rows`` is keyed by
    :param timebase: the records' period, to give each state's hours
    :param out_dir: the folder written to
    :param title: the figure title; defaults to the turbine and the period covered
    :param name: the file's stem
    :param subset: which of ``rows`` to draw; the axes still span all of ``rows`` so subsets compare
    :param colour_by: a category per row to colour by instead of the operating state, in ``colour_order``
    :param colour_order: the categories ``colour_by`` may hold, in legend order
    :param legend_title: the legend's title
    """
    if STATE_COL not in rows.columns or rows.empty:
        return None
    slots = [(getattr(columns, x), getattr(columns, y)) for x, y in PAIRS]
    missing = [_missing_reason(rows, roles=roles, columns=columns) for roles in PAIRS]
    if all(missing):
        return None
    if colour_by is None:
        states = rows[STATE_COL].astype(str)
        order = state_order(states.unique())
        colours = state_colours(order)
    else:
        states = pd.Series(np.asarray(colour_by, dtype=object), index=rows.index).astype(str)
        order = [c for c in colour_order if c in set(states)]
        colours = {c: f"C{i % 10}" for i, c in enumerate(colour_order)}
    drawn = np.ones(len(rows), dtype=bool) if subset is None else np.asarray(subset, dtype=bool)
    hours = states[drawn].value_counts() * (timebase / pd.Timedelta(hours=1))

    n_cols = 2
    n_rows = math.ceil(len(slots) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows), squeeze=False, layout="constrained")
    for ax, (x_col, y_col), reason in zip(axes.flat, slots, missing, strict=True):
        if reason is not None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, reason, ha="center", va="center", wrap=True, color="0.4", transform=ax.transAxes)
            continue
        for state in order:
            chosen = (states == state).to_numpy() & drawn
            ax.plot(
                rows[x_col].to_numpy(dtype=float)[chosen],
                rows[y_col].to_numpy(dtype=float)[chosen],
                linestyle="none",
                marker=".",
                markersize=_MARKER_SIZE,
                color=colours[state],
                rasterized=True,
            )
        if subset is not None:
            ax.set_xlim(*_limits(rows[x_col]))
            ax.set_ylim(*_limits(rows[y_col]))
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}", fontsize="medium")
        apply_grid(ax)
    handles = [
        Line2D([], [], linestyle="none", marker="o", color=colours[s], label=f"{s} ({hours[s]:,.0f} h)")
        for s in order
        if s in hours.index
    ]
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 4), title=legend_title)
    index = pd.DatetimeIndex(rows.index)
    title = title or f"{turbine}: operating states, {index.min():%Y-%m-%d} to {index.max():%Y-%m-%d}"
    fig.suptitle(title if handles else f"{title} (no records)", fontsize="large")
    path = out_dir / f"{name}_{turbine}.png"
    save_fig(fig, path)
    return path


def _missing_reason(rows: pd.DataFrame, *, roles: tuple[str, str], columns: ColumnSchema) -> str | None:
    """Return why the pair of ``roles`` cannot be drawn from ``rows``, or ``None`` when it can."""
    reasons = []
    for role in roles:
        label = role.replace("_", " ")
        column = getattr(columns, role)
        if column is None:
            reasons.append(f"no {label} field in the schema")
        elif column not in rows.columns:
            reasons.append(f"{label} ({column}) not loaded")
        elif rows[column].isna().all():
            reasons.append(f"{label} ({column}) has no values")
    return "\n".join(reasons) or None


def _limits(values: pd.Series) -> tuple[float, float]:
    """Return axis limits covering the finite ``values`` with a small margin."""
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    pad = _LIMIT_MARGIN * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def plot_monthly_state_hours(
    labelled: pd.DataFrame,
    *,
    columns: ColumnSchema,
    timebase: pd.Timedelta,
    out_dir: Path,
    changeovers: Mapping[str, Sequence[pd.Timestamp]] | None = None,
) -> Path | None:
    """Write ``operating_state_hours.png``: each turbine's hours per state per calendar month, stacked.

    The states other than normal operation sit at the bottom, so curtailment and downtime periods
    are visible over time. Each turbine's ``changeovers`` are dashed lines. ``None`` when ``labelled``
    carries no states.
    """
    changeovers = changeovers or {}
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
        for changeover in changeovers.get(turbine, ()):
            stamp = pd.Timestamp(changeover)
            ax.axvline(
                stamp.tz_convert("UTC").tz_localize(None) if stamp.tz is not None else stamp,
                color="k",
                linestyle="--",
                linewidth=1.0,
            )
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
    dashed = "; dashed: changeover" if changeovers else ""
    fig.suptitle(f"hours per operating state per month, {naive.min():%Y-%m-%d} to {naive.max():%Y-%m-%d}{dashed}")
    path = out_dir / "operating_state_hours.png"
    save_fig(fig, path)
    return path


def write_operating_state_plots(
    labelled: pd.DataFrame,
    *,
    columns: ColumnSchema,
    timebase: pd.Timedelta,
    out_dir: Path,
    changeovers: Mapping[str, Sequence[pd.Timestamp]] | None = None,
    per_turbine_plots: bool = True,
) -> list[Path]:
    """Write every turbine's operating-state scatters, the hours CSV and the monthly hours plot.

    :param labelled: long-format SCADA with the operating-state columns, indexed by timestamp
    :param columns: the schema ``labelled`` is keyed by
    :param timebase: the records' period
    :param out_dir: the folder written to
    :param changeovers: each turbine's changeover dates, marked on the monthly hours plot
    :param per_turbine_plots: one scatter per turbine; when False, one scatter of every turbine's records
        together, ``operating_states_all_turbines.png``
    :return: the files written; none when ``labelled`` carries no states
    """
    if STATE_COL not in labelled.columns:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    turbine_names = labelled[columns.turbine].astype(str)
    written: list[Path] = []
    for turbine in sorted(turbine_names.unique()) if per_turbine_plots else [ALL_TURBINES]:
        rows = labelled[(turbine_names == turbine).to_numpy()] if per_turbine_plots else labelled
        index = pd.DatetimeIndex(rows.index)
        title = (
            None
            if per_turbine_plots
            else f"All turbines: operating states, {index.min():%Y-%m-%d} to {index.max():%Y-%m-%d}"
        )
        path = plot_operating_states(
            rows, turbine=turbine, columns=columns, timebase=timebase, out_dir=out_dir, title=title
        )
        if path is not None:
            written.append(path)
    csv = out_dir / STATE_HOURS_CSV
    state_hours(labelled, columns=columns, timebase=timebase).to_csv(csv, index=False)
    written.append(csv)
    path = plot_monthly_state_hours(
        labelled, columns=columns, timebase=timebase, out_dir=out_dir, changeovers=changeovers
    )
    if path is not None:
        written.append(path)
    logger.info("Wrote %d operating-state files to %s", len(written), out_dir)
    return written


def plot_run_operating_states(ctx: DiagnosticContext) -> list[Path]:
    """Write the scatters for a run's test turbine and power references over the span the run sees."""
    if STATE_COL not in ctx.scada_df.columns:
        return []
    references = ctx.plotted_references()
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


def _view_title(turbine: str, *, view: ValidityView, rows: pd.DataFrame, role: str | None = None) -> str:
    index = pd.DatetimeIndex(rows.index)
    period = f"{index.min():%Y-%m-%d} to {index.max():%Y-%m-%d}"
    if role is None:
        return f"{turbine}: records {view.label}, {period}"
    return f"{turbine} ({role}): records {view.label} over the span, {period}"


def write_validity_plots(
    labelled: pd.DataFrame,
    *,
    views: Sequence[ValidityView],
    columns: ColumnSchema,
    timebase: pd.Timedelta,
    out_dir: Path,
) -> list[Path]:
    """Write each turbine's operating-state scatters once per view, each drawing only that view's records.

    :param labelled: long-format SCADA with the operating-state columns, indexed by timestamp
    :param views: the subsets to draw, e.g. :data:`NORTHING_VIEWS`
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
        for view in views:
            path = plot_operating_states(
                rows,
                turbine=turbine,
                columns=columns,
                timebase=timebase,
                out_dir=out_dir,
                title=_view_title(turbine, view=view, rows=rows),
                name=view.name,
                subset=(rows[view.column] == view.value).to_numpy(),
            )
            if path is not None:
                written.append(path)
    logger.info("Wrote %d validity plots to %s", len(written), out_dir)
    return written


def uplift_records(ctx: DiagnosticContext, turbine: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Return ``turbine``'s rows over the span, which of them step 11 used, and why each was not used.

    The reasons are ``None`` when the method gives none.
    """
    rows = ctx.scada_df[(ctx.scada_df[ctx.turbine_col] == turbine).to_numpy()]
    stamps = pd.DatetimeIndex(rows.index)
    used = pd.Series(ctx.used_mask(turbine), index=ctx.index).reindex(stamps, fill_value=False).to_numpy(dtype=bool)
    if ctx.loss_reasons is None:
        return rows, used, None
    reasons = pd.Series(ctx.loss_reasons, index=ctx.index).reindex(stamps).to_numpy(dtype=object)
    return rows, used, reasons


def plot_run_uplift_validity(ctx: DiagnosticContext) -> list[Path]:
    """Write the test turbine's and each power reference's records used for uplift by step 11, and those not used.

    The records used are coloured by operating state; those not used by the reason step 11 dropped them,
    when the method gives one.
    """
    if STATE_COL not in ctx.scada_df.columns:
        return []
    references = ctx.plotted_references()
    out_dir = ctx.stage_dir(stages.VALID_RECORDS)
    written: list[Path] = []
    for turbine in [ctx.test_wtg, *sorted(r for r in references if r != ctx.test_wtg)]:
        rows, used, reasons = uplift_records(ctx, turbine)
        if rows.empty:
            continue
        role = "test turbine" if turbine == ctx.test_wtg else "power reference"
        index = pd.DatetimeIndex(rows.index)
        period = f"{index.min():%Y-%m-%d} to {index.max():%Y-%m-%d}"
        for name, subset, label in (
            ("uplift_used", used, "used for uplift by step 11"),
            ("uplift_not_used", ~used, "not used for uplift by step 11"),
        ):
            by_reason = name == "uplift_not_used" and reasons is not None
            path = plot_operating_states(
                rows,
                turbine=turbine,
                columns=ctx.columns,
                timebase=ctx.timebase,
                out_dir=out_dir,
                title=f"{turbine} ({role}): {int(subset.sum()):,} records {label}, {period}",
                name=name,
                subset=subset,
                colour_by=reasons if by_reason else None,
                colour_order=CATEGORIES if by_reason else (),
                legend_title="reason not used" if by_reason else "operating state",
            )
            if path is not None:
                written.append(path)
    return written
