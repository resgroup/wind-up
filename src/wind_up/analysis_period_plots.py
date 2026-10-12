"""Draw one test turbine's analysis timeline: every turbine's data, works and exclusions, and the span over them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from wind_up.analysis_period import distances_in_diameters

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from matplotlib.figure import Figure

    from wind_up.analysis_period import AnalysisPlan, Exclusion, Window
    from wind_up.layout import Layout

_ROLE_COLOURS = {
    "candidate": "dimgray",
    "test": "black",
    "power_reference": "C0",
    "forced": "C4",
    "reserve": "C2",
    "waking_only": "dimgray",
}
_ROLE_WORDS = {
    "candidate": "",
    "test": "test turbine",
    "power_reference": "power reference",
    "forced": "forced reference",
    "reserve": "reserve",
    "waking_only": "waking only",
}
_DATA_COLOUR = "0.85"
_WORKS_COLOUR = "C3"
_EXCLUSION_COLOUR = "0.6"
_BEFORE_COLOUR = "C0"
_AFTER_COLOUR = "C1"
_ROW_HEIGHT = 0.6


def plot_analysis_timeline(
    turbine: str,
    *,
    layout: Layout,
    works: Mapping[str, Sequence[Window]],
    exclusions: Sequence[Exclusion],
    extents: Mapping[str, Window],
    plan: AnalysisPlan | None,
    reason: str = "",
    out_dir: Path | None = None,
) -> Figure:
    """Draw ``turbine``'s analysis timeline and return the figure.

    One row per turbine, the test turbine first and the rest nearest first: its data extent, its
    works windows and its own exclusions. Farm-wide exclusions span every row. With a ``plan`` the
    before and after periods are shaded and each row is labelled with its role; without one the
    title gives ``reason``.

    :param turbine: the test turbine
    :param layout: every turbine, rotor diameters included
    :param works: works windows per turbine
    :param exclusions: ``(turbine, start, end)``, turbine None for farm-wide
    :param extents: each turbine's data extent
    :param plan: the chosen plan, or None when none could be chosen
    :param reason: why no plan could be chosen
    :param out_dir: when given, the figure is saved as ``analysis_timeline_<turbine>.png``
    """
    distances = distances_in_diameters(layout, turbine=turbine)
    others = sorted(distances, key=lambda other: (distances[other], other))
    rows = [turbine, *others]
    roles = _roles(turbine, others=others, plan=plan)

    fig, ax = plt.subplots(figsize=(13, 1.6 + 0.3 * len(rows)))
    for y, name in enumerate(rows):
        extent = extents.get(name)
        if extent is not None:
            ax.broken_barh([_span(*extent)], (y - _ROW_HEIGHT / 2, _ROW_HEIGHT), facecolors=_DATA_COLOUR)
        ax.broken_barh(
            [_span(w0, w1) for w0, w1 in works.get(name, [])],
            (y - _ROW_HEIGHT / 2, _ROW_HEIGHT),
            facecolors=_WORKS_COLOUR,
        )
        own = [_span(s, e) for who, s, e in exclusions if who == name]
        ax.broken_barh(
            own, (y - _ROW_HEIGHT / 2, _ROW_HEIGHT), facecolors="none", edgecolors=_EXCLUSION_COLOUR, hatch="///"
        )
    for who, s, e in exclusions:
        if who is None:
            ax.axvspan(_num(s), _num(e), facecolor="none", edgecolor=_EXCLUSION_COLOUR, hatch="///", zorder=0)
    if plan is not None:
        ax.axvspan(*map(_num, plan.before), facecolor=_BEFORE_COLOUR, alpha=0.15, zorder=0)
        ax.axvspan(*map(_num, plan.after), facecolor=_AFTER_COLOUR, alpha=0.15, zorder=0)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_label(name, test=turbine, distances=distances, roles=roles) for name in rows])
    for tick, name in zip(ax.get_yticklabels(), rows, strict=True):
        tick.set_color(_ROLE_COLOURS[roles[name]])
        tick.set_fontweight("bold" if roles[name] in ("test", "power_reference", "forced") else "normal")
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(visible=True, axis="x", alpha=0.3)
    ax.set_title(_title(turbine, plan=plan, reason=reason))
    ax.legend(
        handles=_legend_handles(planned=plan is not None), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8
    )
    fig.tight_layout()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"analysis_timeline_{turbine}.png", dpi=130, bbox_inches="tight")
    return fig


def _roles(turbine: str, *, others: Sequence[str], plan: AnalysisPlan | None) -> dict[str, str]:
    roles = {turbine: "test"}
    for other in others:
        if plan is None:
            roles[other] = "candidate"
        elif other in plan.forced:
            roles[other] = "forced"
        elif other in plan.power_references:
            roles[other] = "power_reference"
        elif other in plan.reserves:
            roles[other] = "reserve"
        else:
            roles[other] = "waking_only"
    return roles


def _label(name: str, *, test: str, distances: Mapping[str, float], roles: Mapping[str, str]) -> str:
    if name == test:
        return f"{name}  {_ROLE_WORDS['test']}"
    return f"{name}  {distances[name]:.1f} D  {_ROLE_WORDS[roles[name]]}".rstrip()


def _title(turbine: str, *, plan: AnalysisPlan | None, reason: str) -> str:
    if plan is None:
        reason = reason.removeprefix(f"{turbine}: ")
        return f"{turbine}: not analysed. {reason}" if reason else f"{turbine}: not analysed"
    before_days = plan.pre / pd.Timedelta(days=1)
    after_days = plan.post / pd.Timedelta(days=1)
    pool = "met" if plan.pool_rule_met else f"not met ({plan.pool_rule_reason})"
    how = "declared" if plan.declared else "chosen"
    return (
        f"{turbine}: span {_day(plan.start)}..{_day(plan.end)} ({how}), "
        f"before {before_days:.0f} d, after {after_days:.0f} d, pool rule {pool}"
    )


def _legend_handles(*, planned: bool) -> list[Patch]:
    handles = [
        Patch(facecolor=_DATA_COLOUR, label="data"),
        Patch(facecolor=_WORKS_COLOUR, label="works"),
        Patch(facecolor="none", edgecolor=_EXCLUSION_COLOUR, hatch="///", label="exclusion"),
    ]
    if planned:
        handles += [
            Patch(facecolor=_BEFORE_COLOUR, alpha=0.15, label="before period"),
            Patch(facecolor=_AFTER_COLOUR, alpha=0.15, label="after period"),
        ]
    return handles


def _num(stamp: pd.Timestamp) -> float:
    return float(mdates.date2num(stamp.to_pydatetime()))


def _span(start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float]:
    x0, x1 = _num(start), _num(end)
    return x0, max(x1 - x0, 1e-6)


def _day(stamp: pd.Timestamp) -> str:
    return stamp.strftime("%Y-%m-%d")
