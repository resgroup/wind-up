"""Maps of a campaign design: test turbines, their references, front row, and the turbines around."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from wind_up.geodesy import local_east_north
from wind_up.layout import LATITUDE_COL, LONGITUDE_COL, NAME_COL, WIND_FARM_COL

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    import pandas as pd
    from matplotlib.figure import Figure

    from wind_up.campaign_design import CampaignDesign

TEST_COLOUR = "tab:red"
REFERENCE_COLOUR = "tab:blue"
UNUSED_COLOUR = "0.6"
FRONT_ROW_COLOUR = "#1b7837"
NOT_FRONT_ROW_COLOUR = "#762a83"
OTHER_FARM_COLOURS = ("tab:green", "tab:purple", "tab:brown", "tab:olive", "tab:cyan", "tab:pink")
# The view extends this fraction of the farm's larger span beyond the farm under design.
MARGIN_FRACTION = 0.08


def plot_design_map(design: CampaignDesign, *, latlon: bool = False) -> Figure:
    """Draw ``design`` on a map, in east/north metres or, with ``latlon``, in degrees.

    Metres are measured from the minimum easting and northing of every turbine in the layout. Test
    turbines are red and joined to their references (blue); other available turbines are grey,
    excluded turbines crosses, and front-row turbines ringed.
    Turbines outside the farm under design are small, coloured by wind farm. The view is the farm
    under design plus a margin.
    """
    frame = design.layout.frame
    fig, ax, x, y, in_farm = _start_map(design, latlon=latlon)
    at = {str(n): (x[i], y[i]) for i, n in enumerate(frame[NAME_COL]) if n is not None}
    for test, refs in design.references.items():
        for ref in refs:
            ax.plot(*zip(at[test], at[ref], strict=True), color=REFERENCE_COLOUR, lw=0.6, alpha=0.6, zorder=1)

    other_farms = _draw_other_farms(ax, frame=frame, x=x, y=y, in_farm=in_farm)

    tests = set(design.test_turbines)
    in_use = {r for refs in design.references.values() for r in refs}
    excluded = set(design.excluded)
    for i in np.flatnonzero(in_farm):
        name = str(frame[NAME_COL].iloc[i])
        front = name in design.front_row
        if name in excluded:
            ax.scatter(x[i], y[i], marker="x", color="black", s=50, zorder=3)
        else:
            colour = TEST_COLOUR if name in tests else REFERENCE_COLOUR if name in in_use else UNUSED_COLOUR
            edge = "black" if front else colour
            ax.scatter(x[i], y[i], s=80, color=colour, edgecolors=edge, linewidths=2.0 if front else 0.0, zorder=3)
        ax.annotate(name, (x[i], y[i]), xytext=(4, 4), textcoords="offset points", fontsize=7)

    _finish_map(ax, design=design, x=x, y=y, in_farm=in_farm, latlon=latlon)
    ax.set_title(_title(design))
    ax.legend(handles=[*_legend(), *other_farms], loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_front_row_map(design: CampaignDesign) -> Figure:
    """Draw which turbines of the farm under design are front row, in east/north metres.

    Front-row turbines are green and the rest purple. Turbines outside the farm under design are
    small and grey; they only block wakes.
    """
    frame = design.layout.frame
    fig, ax, x, y, in_farm = _start_map(design, latlon=False)
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
    """Save ``design_map.png``, ``design_map_latlon.png`` and ``front_row_map.png`` in ``out_dir``."""
    for latlon, filename in ((False, "design_map.png"), (True, "design_map_latlon.png")):
        fig = plot_design_map(design, latlon=latlon)
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)
    fig = plot_front_row_map(design)
    fig.savefig(out_dir / "front_row_map.png", dpi=150)
    plt.close(fig)


def _start_map(
    design: CampaignDesign, *, latlon: bool
) -> tuple[Figure, plt.Axes, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Return a figure sized to the farm under design, its axes, every turbine's x and y, and the farm mask."""
    frame = design.layout.frame
    if latlon:
        x, y = frame[LONGITUDE_COL].to_numpy(dtype=float), frame[LATITUDE_COL].to_numpy(dtype=float)
    else:
        x, y = local_east_north(latitudes=frame[LATITUDE_COL], longitudes=frame[LONGITUDE_COL])
    in_farm = (frame[WIND_FARM_COL] == design.wind_farm).to_numpy() if design.wind_farm else np.ones(len(frame), bool)
    height_per_width = (np.ptp(y[in_farm]) + 1e-9) / (np.ptp(x[in_farm]) * _x_stretch(design, latlon=latlon) + 1e-9)
    fig, ax = plt.subplots(figsize=(12, float(np.clip(12 * height_per_width + 1.5, 5, 13))))
    return fig, ax, x, y, in_farm


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


def _title(design: CampaignDesign) -> str:
    summary = design.compliance.summary
    farm = design.wind_farm if design.wind_farm is not None else "Farm"
    return (
        f"{farm}: {summary['test_turbines']} test turbines of {summary['available_turbines']} available "
        f"(most possible {design.max_test_turbines})\n"
        f"front row: {summary['front_row_test_turbines']} tested against a fair share of "
        f"{summary['fair_front_row_share']:.2f}"
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
