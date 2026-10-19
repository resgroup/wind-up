"""Maps of a campaign design: test turbines, their references, the wind they lose, and the turbines around."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge

from wind_up.geodesy import local_east_north
from wind_up.layout import LATITUDE_COL, LONGITUDE_COL, NAME_COL, WIND_FARM_COL

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    import pandas as pd
    from matplotlib.figure import Figure
    from matplotlib.projections.polar import PolarAxes

    from wind_up.campaign_design import CampaignDesign

TEST_COLOUR = "tab:red"
REFERENCE_COLOUR = "tab:blue"
UNUSED_COLOUR = "0.6"
FRONT_ROW_COLOUR = "#1b7837"
NOT_FRONT_ROW_COLOUR = "#762a83"
OTHER_FARM_COLOURS = ("tab:green", "tab:purple", "tab:brown", "tab:olive", "tab:cyan", "tab:pink")
# The view extends this fraction of the farm's larger span beyond the farm under design.
MARGIN_FRACTION = 0.08
# One colour per changed turbine whose wake change costs a test turbine wind, in test-turbine order.
WAKER_COLOURS = ("tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:green")
WEDGE_ALPHA = 0.08
COVERAGE_CMAP = "Blues"
ROSE_BIN_DEG = 10


def plot_design_map(design: CampaignDesign, *, latlon: bool = False) -> Figure:
    """Draw ``design`` on a map, in east/north metres or, with ``latlon``, in degrees.

    Metres are measured from the minimum easting and northing of every turbine in the layout. Test
    turbines are red and joined to their references (blue), labelled with their kept share; other
    available turbines are grey, excluded turbines crosses, and front-row turbines ringed.
    Turbines outside the farm under design are small, coloured by wind farm. The view is the farm
    under design plus a margin. The metre map draws each changed turbine's wake as a wedge from it
    out to each turbine it wakes, as wide there as the disturbed sector, one colour per changed
    turbine, with the wind rose beside it.
    """
    frame = design.layout.frame
    fig, ax, x, y, in_farm, side = _start_map(design, latlon=latlon, side=not latlon and design.wind_rose is not None)
    at = {str(n): (x[i], y[i]) for i, n in enumerate(frame[NAME_COL]) if n is not None}
    wakers = [] if latlon else _draw_loss_sectors(ax, design=design, at=at)
    for test, refs in design.references.items():
        for ref in refs:
            ax.plot(*zip(at[test], at[ref], strict=True), color=REFERENCE_COLOUR, lw=0.6, alpha=0.6, zorder=1)

    other_farms = _draw_other_farms(ax, frame=frame, x=x, y=y, in_farm=in_farm)

    tests = set(design.test_turbines)
    in_use = {r for refs in design.references.values() for r in refs}
    excluded = set(design.excluded)
    kept = dict(zip(design.compliance.table["test_turbine"], design.compliance.table["kept_share"], strict=True))
    for i in np.flatnonzero(in_farm):
        name = str(frame[NAME_COL].iloc[i])
        front = name in design.front_row
        if name in excluded:
            ax.scatter(x[i], y[i], marker="x", color="black", s=50, zorder=3)
        else:
            colour = TEST_COLOUR if name in tests else REFERENCE_COLOUR if name in in_use else UNUSED_COLOUR
            edge = "black" if front else colour
            ax.scatter(x[i], y[i], s=80, color=colour, edgecolors=edge, linewidths=2.0 if front else 0.0, zorder=3)
        label = f"{name} ({kept[name]:.0%} kept)" if name in kept else name
        ax.annotate(label, (x[i], y[i]), xytext=(4, 4), textcoords="offset points", fontsize=7)

    _finish_map(ax, design=design, x=x, y=y, in_farm=in_farm, latlon=latlon)
    ax.set_title(_title(design))
    handles = [*_legend(), *wakers, *other_farms]
    if side is None:
        ax.legend(handles=handles, loc="best", fontsize=8)
    else:
        key, rose = side
        key.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)
        _draw_rose(rose, design=design)
    fig.tight_layout()
    return fig


def plot_coverage_map(design: CampaignDesign) -> Figure:
    """Draw every available farm turbine coloured by its kept share as a lone test turbine, in east/north metres.

    The kept share is with the turbine's own best-ranked references; turbines with too few are grey
    crosses. Excluded turbines and those of other farms are small and grey. The wind rose is beside it.
    """
    frame = design.layout.frame
    fig, ax, x, y, in_farm, side = _start_map(design, latlon=False, side=design.wind_rose is not None)
    names = [None if n is None else str(n) for n in frame[NAME_COL]]
    shares = np.array([design.lone_kept_share.get(n, np.nan) if n is not None else np.nan for n in names])
    rated = np.isfinite(shares)
    rest = ~rated
    ax.scatter(x[rest], y[rest], s=20, marker="x", color=UNUSED_COLOUR, zorder=2)
    points = ax.scatter(
        x[rated],
        y[rated],
        s=110,
        c=shares[rated],
        cmap=COVERAGE_CMAP,
        vmin=0.0,
        vmax=1.0,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )
    for i in np.flatnonzero(in_farm):
        name = names[i]
        if name is not None:
            label = f"{name} {shares[i]:.0%}" if rated[i] else name
            ax.annotate(label, (x[i], y[i]), xytext=(5, 5), textcoords="offset points", fontsize=7)
    _finish_map(ax, design=design, x=x, y=y, in_farm=in_farm, latlon=False)
    label = design.wind_farm if design.wind_farm is not None else "Farm"
    ax.set_title(
        f"{label}: share of the wind a lone test turbine keeps clear of its own wake change\n"
        f"wake reach {design.wake_reach_d:g} D, rose: {design.rose_label}"
    )
    handles = [Line2D([], [], marker="x", ls="", color=UNUSED_COLOUR, label="too few references or not available")]
    if side is None:
        fig.colorbar(points, ax=ax, shrink=0.7, label="kept share as a lone test turbine")
        ax.legend(handles=handles, loc="best", fontsize=8)
    else:
        key, rose = side
        key.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)
        fig.colorbar(points, cax=key.inset_axes((0.05, 0.05, 0.1, 0.75)), label="kept share as a lone test turbine")
        _draw_rose(rose, design=design)
    fig.tight_layout()
    return fig


def plot_front_row_map(design: CampaignDesign) -> Figure:
    """Draw which turbines of the farm under design are front row, in east/north metres.

    Front-row turbines are green and the rest purple. Turbines outside the farm under design are
    small and grey; they only block wakes.
    """
    frame = design.layout.frame
    fig, ax, x, y, in_farm, _ = _start_map(design, latlon=False)
    others = np.flatnonzero(~in_farm)
    if len(others):
        ax.scatter(x[others], y[others], s=15, color=UNUSED_COLOUR, zorder=2)
        for i in others:
            name = frame[NAME_COL].iloc[i]
            if name is not None:
                ax.annotate(name, (x[i], y[i]), xytext=(3, 3), textcoords="offset points", fontsize=6)

    farm = np.flatnonzero(in_farm)
    names = [str(frame[NAME_COL].iloc[i]) for i in farm]
    colours = [FRONT_ROW_COLOUR if n in design.front_row else NOT_FRONT_ROW_COLOUR for n in names]
    ax.scatter(x[farm], y[farm], s=80, c=colours, zorder=3)
    for i, name in zip(farm, names, strict=True):
        ax.annotate(name, (x[i], y[i]), xytext=(4, 4), textcoords="offset points", fontsize=7)

    _finish_map(ax, design=design, x=x, y=y, in_farm=in_farm, latlon=False)
    label = design.wind_farm if design.wind_farm is not None else "Farm"
    ax.set_title(
        f"{label}: {len(design.front_row)} of {len(farm)} turbines front row\n"
        f"front row: at least {design.front_row_min_clear_deg:g}° of wind directions with no turbine upwind"
    )
    handles = [
        Line2D([], [], marker="o", ls="", color=FRONT_ROW_COLOUR, markersize=8, label="front row"),
        Line2D([], [], marker="o", ls="", color=NOT_FRONT_ROW_COLOUR, markersize=8, label="not front row"),
    ]
    if len(others):
        handles.append(Line2D([], [], marker="o", ls="", color=UNUSED_COLOUR, markersize=4, label="other farms"))
    ax.legend(handles=handles, loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def save_design_maps(design: CampaignDesign, *, out_dir: Path) -> None:
    """Save ``design_map.png``, ``design_map_latlon.png``, ``front_row_map.png`` and ``coverage_map.png``."""
    for latlon, filename in ((False, "design_map.png"), (True, "design_map_latlon.png")):
        fig = plot_design_map(design, latlon=latlon)
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)
    for plot, filename in ((plot_front_row_map, "front_row_map.png"), (plot_coverage_map, "coverage_map.png")):
        fig = plot(design)
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)


def _start_map(
    design: CampaignDesign, *, latlon: bool, side: bool = False
) -> tuple[
    Figure,
    plt.Axes,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    tuple[plt.Axes, PolarAxes] | None,
]:
    """Return a figure sized to the farm under design, its map axes, every turbine's x and y, and the farm mask.

    With ``side``, also a column beside the map: a key axes, blank, above a polar axes for the rose.
    """
    frame = design.layout.frame
    if latlon:
        x, y = frame[LONGITUDE_COL].to_numpy(dtype=float), frame[LATITUDE_COL].to_numpy(dtype=float)
    else:
        x, y = local_east_north(latitudes=frame[LATITUDE_COL], longitudes=frame[LONGITUDE_COL])
    in_farm = (frame[WIND_FARM_COL] == design.wind_farm).to_numpy() if design.wind_farm else np.ones(len(frame), bool)
    height_per_width = (np.ptp(y[in_farm]) + 1e-9) / (np.ptp(x[in_farm]) * _x_stretch(design, latlon=latlon) + 1e-9)
    height = float(np.clip(12 * height_per_width + 1.5, 5, 13))
    if not side:
        fig, ax = plt.subplots(figsize=(12, height))
        return fig, ax, x, y, in_farm, None
    fig = plt.figure(figsize=(15, max(height, 7.0)))
    grid = fig.add_gridspec(2, 2, width_ratios=(4, 1))
    ax = fig.add_subplot(grid[:, 0])
    key = fig.add_subplot(grid[0, 1])
    key.axis("off")
    rose = cast("PolarAxes", fig.add_subplot(grid[1, 1], projection="polar"))
    return fig, ax, x, y, in_farm, (key, rose)


def _finish_map(
    ax: plt.Axes,
    *,
    design: CampaignDesign,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    in_farm: npt.NDArray[np.bool_],
    latlon: bool,
) -> None:
    """Set the aspect, axis labels, grid, and a view of the farm under design plus a margin."""
    if latlon:
        ax.set_aspect(1 / _x_stretch(design, latlon=True))
        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("latitude [deg]")
    else:
        ax.set_aspect("equal")
        ax.set_xlabel("east [m]")
        ax.set_ylabel("north [m]")
    pad = MARGIN_FRACTION * max(np.ptp(x[in_farm]), np.ptp(y[in_farm]))
    ax.set_xlim(x[in_farm].min() - pad, x[in_farm].max() + pad)
    ax.set_ylim(y[in_farm].min() - pad, y[in_farm].max() + pad)
    ax.grid(visible=True, alpha=0.3)


def _x_stretch(design: CampaignDesign, *, latlon: bool) -> float:
    """Return how much shorter a degree of longitude is than one of latitude, or 1 in metres."""
    return float(np.cos(np.radians(design.layout.frame[LATITUDE_COL].mean()))) if latlon else 1.0


def _draw_other_farms(
    ax: plt.Axes, *, frame: pd.DataFrame, x: np.ndarray, y: np.ndarray, in_farm: np.ndarray
) -> list[Line2D]:
    """Draw the turbines outside the farm under design, coloured by wind farm; return their legend entries."""
    farms = frame[WIND_FARM_COL]
    handles: list[Line2D] = []
    others = sorted({f for f, inside in zip(farms, in_farm, strict=True) if not inside and f is not None})
    colours = {f: OTHER_FARM_COLOURS[k % len(OTHER_FARM_COLOURS)] for k, f in enumerate(others)}
    for farm in [*others, None]:
        mask = (~in_farm) & np.array([f == farm for f in farms])
        if mask.any():
            label = farm if farm is not None else "unknown farm"
            colour = colours.get(farm, UNUSED_COLOUR)
            ax.scatter(x[mask], y[mask], s=15, color=colour, zorder=2)
            handles.append(Line2D([], [], marker="o", ls="", color=colour, markersize=4, label=label))
            for i in np.flatnonzero(mask):
                name = frame[NAME_COL].iloc[i]
                if name is not None:
                    ax.annotate(name, (x[i], y[i]), xytext=(3, 3), textcoords="offset points", fontsize=6)
    return handles


def _draw_loss_sectors(ax: plt.Axes, *, design: CampaignDesign, at: dict[str, tuple[float, float]]) -> list[Patch]:
    """Draw each lost sector as a wake from its waker widening to the waked turbine; return the legend entries.

    The wedge has the sector's angle, its point at the waker and its axis towards the waked turbine.
    """
    losses = design.compliance.wake_losses
    if losses.empty:
        return []
    order = [t for t in design.test_turbines if t in set(losses["waker"])]
    colours = {w: WAKER_COLOURS[k % len(WAKER_COLOURS)] for k, w in enumerate(order)}
    for sector in losses.drop_duplicates(subset=["waked", "waker"]).itertuples():
        (x0, y0), (x1, y1) = at[str(sector.waked)], at[str(sector.waker)]
        downwind, half = float(sector.centre_deg) + 180.0, float(sector.half_width_deg)
        colour = colours[str(sector.waker)]
        ax.add_patch(
            Wedge(
                (x1, y1),
                float(np.hypot(x1 - x0, y1 - y0)),
                90.0 - (downwind + half),
                90.0 - (downwind - half),
                facecolor=colour,
                edgecolor="none",
                alpha=WEDGE_ALPHA,
                zorder=0,
            )
        )
    return [Patch(facecolor=colours[w], alpha=0.3, label=f"wake of {w}") for w in order]


def _draw_rose(ax: PolarAxes, *, design: CampaignDesign) -> None:
    """Draw the wind rose, north up and clockwise, as the share of the wind from each sector."""
    rose = np.asarray(design.wind_rose, dtype=float)
    shares = rose.reshape(-1, ROSE_BIN_DEG).sum(axis=1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    centres = np.radians(np.arange(0, 360, ROSE_BIN_DEG) + ROSE_BIN_DEG / 2)
    ax.bar(centres, shares, width=np.radians(ROSE_BIN_DEG), color="0.4", edgecolor="white", linewidth=0.5)
    ax.set_xticks(np.radians([0, 90, 180, 270]), labels=["N", "E", "S", "W"], fontsize=8)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title(f"wind rose, share per {ROSE_BIN_DEG}° sector\n{design.rose_label}", fontsize=8)


def _title(design: CampaignDesign) -> str:
    summary = design.compliance.summary
    farm = design.wind_farm if design.wind_farm is not None else "Farm"
    return (
        f"{farm}: {summary['test_turbines']} test turbines of {summary['available_turbines']} available "
        f"(most possible {design.max_test_turbines})\n"
        f"front row: {summary['front_row_test_turbines']} tested against a fair share of "
        f"{summary['fair_front_row_share']:.2f}; wake reach {design.wake_reach_d:g} D, total kept share "
        f"{summary['total_kept_share']:.2f}"
    )


def _legend() -> list[Line2D]:
    def dot(colour: str, label: str, **kwargs: object) -> Line2D:
        return Line2D([], [], marker="o", ls="", color=colour, markersize=8, label=label, **kwargs)  # type: ignore[arg-type]

    return [
        dot(TEST_COLOUR, "test"),
        dot(REFERENCE_COLOUR, "reference in use"),
        dot(UNUSED_COLOUR, "available, unused"),
        Line2D([], [], marker="x", ls="", color="black", label="excluded"),
        dot("white", "front row", markeredgecolor="black", markeredgewidth=2),
    ]
