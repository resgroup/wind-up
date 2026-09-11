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

    import pandas as pd
    from matplotlib.figure import Figure

    from wind_up.campaign_design import CampaignDesign

TEST_COLOUR = "tab:red"
REFERENCE_COLOUR = "tab:blue"
UNUSED_COLOUR = "0.6"
OTHER_FARM_COLOURS = ("tab:green", "tab:purple", "tab:brown", "tab:olive", "tab:cyan", "tab:pink")
# The view extends this fraction of the farm's larger span beyond the farm under design.
MARGIN_FRACTION = 0.08


def plot_design_map(design: CampaignDesign, *, latlon: bool = False) -> Figure:
    """Draw ``design`` on a map, in east/north metres or, with ``latlon``, in degrees.

    Metres are measured from the minimum easting and northing of every turbine in the layout. Test
    turbines are red and joined to their references (blue); other available turbines are grey,
    reference-only turbines hollow blue, excluded turbines crosses, and front-row turbines ringed.
    Turbines outside the farm under design are small, coloured by wind farm. The view is the farm
    under design plus a margin.
    """
    frame = design.layout.frame
    if latlon:
        x, y = frame[LONGITUDE_COL].to_numpy(), frame[LATITUDE_COL].to_numpy()
    else:
        x, y = local_east_north(latitudes=frame[LATITUDE_COL], longitudes=frame[LONGITUDE_COL])
    at = {str(n): (x[i], y[i]) for i, n in enumerate(frame[NAME_COL]) if n is not None}
    in_farm = (frame[WIND_FARM_COL] == design.wind_farm).to_numpy() if design.wind_farm else np.ones(len(frame), bool)
    x_stretch = np.cos(np.radians(frame[LATITUDE_COL].mean())) if latlon else 1.0
    height_per_width = (np.ptp(y[in_farm]) + 1e-9) / (np.ptp(x[in_farm]) * x_stretch + 1e-9)

    fig, ax = plt.subplots(figsize=(12, float(np.clip(12 * height_per_width + 1.5, 5, 13))))
    for test, refs in design.references.items():
        for ref in refs:
            ax.plot(*zip(at[test], at[ref], strict=True), color=REFERENCE_COLOUR, lw=0.6, alpha=0.6, zorder=1)

    other_farms = _draw_other_farms(ax, frame=frame, x=x, y=y, in_farm=in_farm)

    tests = set(design.test_turbines)
    in_use = {r for refs in design.references.values() for r in refs}
    reference_only, excluded = set(design.reference_only), set(design.excluded)
    for i in np.flatnonzero(in_farm):
        name = str(frame[NAME_COL].iloc[i])
        front = name in design.front_row
        if name in excluded:
            ax.scatter(x[i], y[i], marker="x", color="black", s=50, zorder=3)
        elif name in reference_only:
            edge = "black" if front else REFERENCE_COLOUR
            ax.scatter(
                x[i], y[i], s=80, facecolors="white", edgecolors=edge, linewidths=2.0 if front else 1.5, zorder=3
            )
        else:
            colour = TEST_COLOUR if name in tests else REFERENCE_COLOUR if name in in_use else UNUSED_COLOUR
            edge = "black" if front else colour
            ax.scatter(x[i], y[i], s=80, color=colour, edgecolors=edge, linewidths=2.0 if front else 0.0, zorder=3)
        ax.annotate(name, (x[i], y[i]), xytext=(4, 4), textcoords="offset points", fontsize=7)

    if latlon:
        ax.set_aspect(1 / x_stretch)
        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("latitude [deg]")
    else:
        ax.set_aspect("equal")
        ax.set_xlabel("east [m]")
        ax.set_ylabel("north [m]")
    pad = MARGIN_FRACTION * max(np.ptp(x[in_farm]), np.ptp(y[in_farm]))
    ax.set_xlim(x[in_farm].min() - pad, x[in_farm].max() + pad)
    ax.set_ylim(y[in_farm].min() - pad, y[in_farm].max() + pad)
    ax.set_title(_title(design))
    ax.legend(handles=[*_legend(), *other_farms], loc="best", fontsize=8)
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_design_maps(design: CampaignDesign, *, out_dir: Path) -> None:
    """Save ``design_map.png`` (east/north metres) and ``design_map_latlon.png`` (degrees) in ``out_dir``."""
    for latlon, filename in ((False, "design_map.png"), (True, "design_map_latlon.png")):
        fig = plot_design_map(design, latlon=latlon)
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)


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
        dot("white", "reference-only", markeredgecolor=REFERENCE_COLOUR),
        Line2D([], [], marker="x", ls="", color="black", label="excluded"),
        dot("white", "front row", markeredgecolor="black", markeredgewidth=2),
    ]
