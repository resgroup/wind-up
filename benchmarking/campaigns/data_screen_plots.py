"""Plots of the data screen, one set per test turbine."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from benchmarking.campaigns.data_screen import DROPPED, KEPT, SPARE
from benchmarking.diagnostics.style import apply_grid, save_fig
from benchmarking.harness.record_selection import PAIR_CATEGORIES, USABLE
from benchmarking.harness.waking import DIRECTION_COL
from wind_up.geodesy import local_east_north
from wind_up.layout import LATITUDE_COL, LONGITUDE_COL, NAME_COL

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    import pandas as pd
    from matplotlib.artist import Artist
    from matplotlib.legend import Legend
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.transforms import Transform

    from benchmarking.campaigns.data_screen import DataScreen
    from benchmarking.harness.record_selection import ChangedInterval
    from wind_up.layout import Layout

VERDICT_COLOURS = {KEPT: "C2", SPARE: "C0", DROPPED: "C3"}
REASON_COLOURS = dict(zip(PAIR_CATEGORIES, ("C7", "C1", "C4", "C3", "C6", "C8", "C9", "C5", "C2"), strict=True))
OTHER_COLOUR = "0.75"
_ROSE_SECTORS = 16
_MIN_COLOUR_SPAN = 0.1


def write_data_screen_plots(
    screen: DataScreen, *, layout: Layout, scada_df: pd.DataFrame, turbine_col: str, out_dir: Path
) -> None:
    """Write the five data screen plots of one test turbine under ``out_dir``."""
    test = screen.plan.turbine
    save_fig(plot_data_screen(screen), out_dir / f"data_screen_{test}.png")
    save_fig(plot_record_losses(screen), out_dir / f"record_losses_{test}.png")
    save_fig(plot_reference_hours(screen), out_dir / f"reference_hours_{test}.png")
    save_fig(
        plot_reference_map(screen, layout=layout, scada_df=scada_df, turbine_col=turbine_col),
        out_dir / f"reference_map_{test}.png",
    )
    save_fig(plot_correlation_map(screen, layout=layout), out_dir / f"correlation_map_{test}.png")


def plot_data_screen(screen: DataScreen) -> plt.Figure:
    """Hours before and after, and power correlation, per candidate in rank order."""
    table = screen.table
    x = np.arange(len(table))
    colours = [VERDICT_COLOURS[v] for v in table["verdict"]]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(max(6.0, 0.9 * len(table) + 2.0), 7.5), sharex=True)
    width = 0.38
    top.bar(x - width / 2, table["hours_before"], width=width, color=colours, edgecolor="black")
    top.bar(x + width / 2, table["hours_after"], width=width, color=colours, edgecolor="black", hatch="//", alpha=0.7)
    fraction = screen.settings.hours_fraction
    if len(table):
        top.axhline(fraction * table["median_hours_before"].iloc[0], color="black", linestyle="--", linewidth=1)
        top.axhline(fraction * table["median_hours_after"].iloc[0], color="black", linestyle=":", linewidth=1)
    top.set_ylabel("usable hours")
    top.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", label="before"),
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="//", label="after"),
            Line2D([], [], color="black", linestyle="--", label=f"{fraction:g} x median before"),
            Line2D([], [], color="black", linestyle=":", label=f"{fraction:g} x median after"),
            *_verdict_handles(),
        ],
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    bottom.scatter(x, table["correlation"], c=colours, edgecolors="black", s=60, zorder=3)
    if len(table) and np.isfinite(table["median_correlation"].iloc[0]):
        threshold = table["median_correlation"].iloc[0] - screen.settings.correlation_margin
        bottom.axhline(threshold, color="black", linestyle="--", linewidth=1, label="median - margin")
        bottom.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    bottom.set_ylabel("power correlation with the test turbine")
    bottom.set_xticks(x, [f"{c}\n{d:.1f} D" for c, d in zip(table["candidate"], table["distance_d"], strict=True)])
    for ax in (top, bottom):
        apply_grid(ax)
        _mark_cutoff(ax, table=table)
    top.set_title(f"Data screen of the candidate references for {screen.plan.turbine}")
    return fig


def plot_record_losses(screen: DataScreen) -> plt.Figure:
    """Each candidate's span records split by the reason they are lost, or usable."""
    losses = screen.losses
    candidates = list(screen.table["candidate"])
    counts = losses.pivot_table(index="candidate", columns="reason", values="records", aggfunc="sum", observed=False)
    counts = counts.reindex(index=candidates, columns=list(PAIR_CATEGORIES), fill_value=0).fillna(0)
    totals = counts.sum(axis=1).replace(0, 1)
    shares = counts.div(totals, axis=0) * 100.0
    fig, ax = plt.subplots(figsize=(9.0, max(3.0, 0.5 * len(candidates) + 1.5)))
    y = np.arange(len(candidates))
    left = np.zeros(len(candidates))
    for reason in PAIR_CATEGORIES:
        ax.barh(y, shares[reason], left=left, color=REASON_COLOURS[reason], label=reason, edgecolor="white")
        left += shares[reason].to_numpy()
    ax.set_yticks(y, candidates)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the span's records [%]")
    ax.set_title(f"Records of {screen.plan.turbine} paired with each candidate, by loss reason")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    apply_grid(ax)
    return fig


def plot_reference_hours(screen: DataScreen) -> plt.Figure:
    """Joint usable hours before and after against the number of references."""
    hours = screen.hours
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(hours["n_references"], hours["hours_before"], marker="o", label="before")
    ax.plot(hours["n_references"], hours["hours_after"], marker="s", label="after")
    chosen = len(screen.plan.power_references)
    if chosen:
        ax.axvline(chosen, color="black", linestyle="--", linewidth=1, label=f"chosen ({chosen})")
    ax.set_xlabel("number of references, nearest first")
    ax.set_ylabel("joint usable hours")
    ax.set_xticks(list(hours["n_references"]))
    ax.set_ylim(bottom=0)
    ax.set_title(f"Usable hours of {screen.plan.turbine} against the number of references")
    ax.legend(fontsize=8)
    apply_grid(ax)
    return fig


def plot_reference_map(screen: DataScreen, *, layout: Layout, scada_df: pd.DataFrame, turbine_col: str) -> plt.Figure:
    """Draw the layout with the candidates coloured by verdict, the changed turbines marked and a wind rose."""
    names, x, y = _positions(layout)
    table = screen.table.set_index("candidate")
    test = screen.plan.turbine
    usable = _usable_fraction(screen)
    fig = plt.figure(figsize=(12.0, 8.0))
    ax = fig.add_axes((0.06, 0.08, 0.62, 0.84))
    others = [i for i, n in enumerate(names) if n not in table.index and n != test]
    ax.scatter(x[others], y[others], s=20, color=OTHER_COLOUR, zorder=2)
    for i in others:
        ax.annotate(names[i], (x[i], y[i]), xytext=(3, 3), textcoords="offset points", fontsize=6, color="0.5")
    for i, name in enumerate(names):
        if name not in table.index:
            continue
        ax.scatter(x[i], y[i], s=90, color=VERDICT_COLOURS[table.loc[name, "verdict"]], edgecolors="black", zorder=3)
        ax.annotate(f"{name} {usable[name]:.0%}", (x[i], y[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    throughout = [i for i, n in enumerate(names) if n in screen.changed and _throughout(screen.changed[n])]
    ax.scatter(x[throughout], y[throughout], s=220, facecolors="none", edgecolors="C1", linewidths=2.0, zorder=4)
    partly = [i for i, n in enumerate(names) if n in screen.changed and not _throughout(screen.changed[n])]
    ax.scatter(
        x[partly], y[partly], s=220, facecolors="none", edgecolors="C1", linewidths=1.5, linestyles="--", zorder=4
    )
    for i in partly:
        ax.annotate(
            _changed_label(screen.changed[names[i]]),
            (x[i], y[i]),
            xytext=(5, -11),
            textcoords="offset points",
            fontsize=6,
            color="C1",
        )
    t = names.index(test)
    ax.scatter(x[t], y[t], s=300, marker="*", color="gold", edgecolors="black", zorder=5)
    ax.annotate(test, (x[t], y[t]), xytext=(6, -12), textcoords="offset points", fontsize=9, weight="bold")
    _finish_map(ax, x=x, y=y)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="*",
                linestyle="",
                markersize=14,
                color="gold",
                markeredgecolor="black",
                label="test turbine",
            ),
            *_verdict_handles(),
            _ring("changed throughout", edge="C1", dashed=False),
            _ring("changed from / until a date", edge="C1", dashed=True),
            Line2D([], [], marker="o", linestyle="", color=OTHER_COLOUR, label="other"),
        ],
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        handler_map={Circle: _RingHandler()},
    )
    plan = screen.plan
    ax.set_title(
        f"Candidate references of {test}, labelled with their usable share of the span\n"
        f"span {plan.start:%Y-%m-%d} to {plan.end:%Y-%m-%d}, {test} works {plan.works[0]:%Y-%m-%d} to "
        f"{plan.works[1]:%Y-%m-%d}",
        fontsize="medium",
    )
    _wind_rose(fig, screen=screen, scada_df=scada_df, turbine_col=turbine_col)
    return fig


def _ring(label: str, *, edge: str, dashed: bool) -> Circle:
    """Return a legend entry drawn as an open ring, dashed or solid."""
    return Circle(
        (0.0, 0.0),
        radius=1.0,
        facecolor="none",
        edgecolor=edge,
        linewidth=1.5,
        linestyle="--" if dashed else "-",
        label=label,
    )


class _RingHandler(HandlerPatch):
    """Draw a :class:`Circle` legend entry as a circle, keeping its line style."""

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> list[Artist]:
        """Return one ring centred in the legend entry's box."""
        del fontsize
        ring = Circle((width / 2 - xdescent, height / 2 - ydescent), radius=0.6 * height)
        self.update_prop(ring, orig_handle, legend)
        ring.set_linestyle(cast("Circle", orig_handle).get_linestyle())
        ring.set_transform(trans)
        return [ring]


def _throughout(times: tuple[ChangedInterval, ...]) -> bool:
    """Return whether a turbine counts as changed over the whole span."""
    return any(t.start is None and t.end is None for t in times)


def _changed_label(times: tuple[ChangedInterval, ...]) -> str:
    """Return when a turbine counts as changed, as ``from <date>`` and ``until <date>`` parts."""
    parts = [f"from {t.start:%Y-%m-%d}" if t.start is not None else f"until {t.end:%Y-%m-%d}" for t in times]
    return ", ".join(parts)


def plot_correlation_map(screen: DataScreen, *, layout: Layout) -> plt.Figure:
    """Draw the layout with each candidate as a bubble whose colour and size show its correlation."""
    names, x, y = _positions(layout)
    table = screen.table.set_index("candidate")
    test = screen.plan.turbine
    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    others = [i for i, n in enumerate(names) if n not in table.index and n != test]
    ax.scatter(x[others], y[others], s=20, color=OTHER_COLOUR, zorder=2)
    correlation = table["correlation"].astype(float)
    finite = correlation[np.isfinite(correlation)]
    low, high = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    if high - low < _MIN_COLOUR_SPAN:
        low, high = low - _MIN_COLOUR_SPAN / 2, high + _MIN_COLOUR_SPAN / 2
    mappable = None
    for dropped in (False, True):
        rows = [i for i, n in enumerate(names) if n in table.index and (table.loc[n, "verdict"] == DROPPED) == dropped]
        if not rows:
            continue
        values = np.array([correlation[names[i]] for i in rows])
        sizes = 80.0 + 520.0 * np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0.0, 1.0)
        mappable = ax.scatter(
            x[rows],
            y[rows],
            c=values,
            s=sizes,
            cmap="viridis",
            vmin=low,
            vmax=high,
            edgecolors="black",
            linewidths=1.5,
            linestyles="--" if dropped else "-",
            plotnonfinite=True,
            zorder=3,
        )
        for i in rows:
            ax.annotate(names[i], (x[i], y[i]), xytext=(8, 8), textcoords="offset points", fontsize=8)
    t = names.index(test)
    ax.scatter(x[t], y[t], s=300, marker="*", color="gold", edgecolors="black", zorder=5)
    ax.annotate(test, (x[t], y[t]), xytext=(6, -12), textcoords="offset points", fontsize=9, weight="bold")
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, shrink=0.7, label="power correlation with the test turbine")
    _finish_map(ax, x=x, y=y)
    ax.legend(
        handles=[
            _ring("kept or spare", edge="black", dashed=False),
            _ring("dropped (dashed outline)", edge="black", dashed=True),
            Line2D([], [], marker="o", linestyle="", color=OTHER_COLOUR, label="not a candidate"),
        ],
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        handler_map={Circle: _RingHandler()},
    )
    ax.set_title(f"Power correlation of each candidate reference for {test}")
    return fig


def _positions(layout: Layout) -> tuple[list[str], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    frame = layout.frame
    x, y = local_east_north(latitudes=frame[LATITUDE_COL], longitudes=frame[LONGITUDE_COL])
    return [str(n) for n in frame[NAME_COL]], x, y


def _finish_map(ax: plt.Axes, *, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
    ax.set_aspect("equal")
    pad = 0.08 * max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    apply_grid(ax)


def _verdict_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", linestyle="", color=colour, markeredgecolor="black", label=verdict)
        for verdict, colour in VERDICT_COLOURS.items()
    ]


def _mark_cutoff(ax: plt.Axes, *, table: pd.DataFrame) -> None:
    """Draw a line after the last kept candidate."""
    kept = np.flatnonzero((table["verdict"] == KEPT).to_numpy())
    if len(kept):
        ax.axvline(kept[-1] + 0.5, color="0.3", linestyle="-.", linewidth=1)
        ax.annotate(
            "k",
            (kept[-1] + 0.5, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -12),
            textcoords="offset points",
            fontsize=8,
        )


def _usable_fraction(screen: DataScreen) -> dict[str, float]:
    losses = screen.losses
    totals = losses.groupby("candidate")["records"].sum()
    usable = losses[losses["reason"] == USABLE].set_index("candidate")["records"]
    return {str(c): float(usable.get(c, 0)) / float(totals[c]) if totals[c] else 0.0 for c in totals.index}


def _wind_rose(fig: plt.Figure, *, screen: DataScreen, scada_df: pd.DataFrame, turbine_col: str) -> None:
    """Draw the span's waking direction at the test turbine as a rose beside the map, when the frame has it."""
    if DIRECTION_COL not in scada_df.columns:
        return
    plan = screen.plan
    rows = scada_df[scada_df[turbine_col].astype(str) == plan.turbine]
    rows = rows[(rows.index >= plan.start) & (rows.index < plan.end)]
    direction = rows[DIRECTION_COL].to_numpy(dtype=float)
    direction = direction[np.isfinite(direction)]
    if not len(direction):
        return
    edges = np.linspace(0.0, 360.0, _ROSE_SECTORS + 1)
    counts, _ = np.histogram(direction % 360.0, bins=edges)
    rose = cast("PolarAxes", fig.add_axes((0.74, 0.08, 0.22, 0.33), projection="polar"))
    rose.bar(
        np.deg2rad(edges[:-1] + 360.0 / _ROSE_SECTORS / 2),
        counts / counts.sum(),
        width=np.deg2rad(360.0 / _ROSE_SECTORS),
        color="C0",
        edgecolor="white",
    )
    rose.set_theta_zero_location("N")
    rose.set_theta_direction(-1)
    rose.set_yticklabels([])
    rose.tick_params(labelsize=7)
    rose.set_title("waking direction at the test turbine", fontsize=8)
