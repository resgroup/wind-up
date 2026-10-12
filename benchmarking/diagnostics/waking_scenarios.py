"""Step 5 outputs: the waking direction, the upwind sectors and the hours in each waking scenario.

:func:`write_waking_outputs` writes, into one folder:

* ``waking_scenario_hours.csv``: each turbine's hours in each waking scenario;
* ``waking_scenarios.png``: each turbine's hours with every upwind turbine waking, with an upwind turbine
  not waking, with an upwind turbine part waking, and unknown;
* ``waking_direction.csv``: each turbine's hours by direction source, and how far its own direction is
  from its neighbours' where both are valid;
* ``waking_direction_difference.png``: that difference as a histogram per turbine;
* ``upwind_sectors.png``: which turbines are upwind of each turbine at every direction.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.diagnostics.style import apply_grid, save_fig
from benchmarking.harness.operating_state import PART_WAKING
from benchmarking.harness.waking import (
    DIRECTION_SOURCE_COL,
    NEIGHBOURS,
    NORMAL,
    OWN,
    UNKNOWN,
    WAKING_SCENARIO_COL,
    upwind_matrix,
)
from wind_up.circular_math import circ_diff
from wind_up.layout import NAME_COL, ROTOR_DIAMETER_COL, iec_disturbed_sector_deg

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.harness.waking import WakingDirections
    from benchmarking.synthetic import ColumnSchema
    from wind_up.layout import Layout

logger = logging.getLogger(__name__)

SCENARIO_HOURS_CSV = "waking_scenario_hours.csv"
DIRECTION_CSV = "waking_direction.csv"

_UPWIND_NOT_WAKING = "upwind not waking"
_UPWIND_PART_WAKING = "upwind part waking"
_CLASSES = (NORMAL, _UPWIND_NOT_WAKING, _UPWIND_PART_WAKING, UNKNOWN)
_CLASS_COLOURS = {NORMAL: "C0", _UPWIND_NOT_WAKING: "C3", _UPWIND_PART_WAKING: "C1", UNKNOWN: "C7"}
_PANEL_COLUMNS = 4
_DIFFERENCE_BINS = np.arange(-30.0, 30.5, 1.0)


def write_waking_outputs(
    labelled: pd.DataFrame,
    *,
    directions: WakingDirections,
    layout: Layout,
    columns: ColumnSchema,
    timebase: pd.Timedelta,
    out_dir: Path,
) -> list[Path]:
    """Write the step 5 tables and plots of a frame carrying the waking columns.

    :param labelled: long SCADA with the columns of :func:`~benchmarking.harness.waking.add_waking_scenarios`
    :param directions: the directions the waking columns were found from
    :param layout: the farm layout
    :param columns: the schema ``labelled`` is keyed by
    :param timebase: the records' period
    :param out_dir: the folder written to
    :return: the files written
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    hours_per_record = timebase / pd.Timedelta(hours=1)
    turbine = pd.Index(labelled[columns.turbine].astype(str), name="turbine")
    scenario = pd.Index(labelled[WAKING_SCENARIO_COL], name=WAKING_SCENARIO_COL)

    scenario_hours = (
        pd.Series(1, index=[turbine, scenario]).groupby(level=[0, 1]).size().mul(hours_per_record).rename("hours")
    )
    scenario_hours = scenario_hours.reset_index().sort_values(["turbine", "hours"], ascending=[True, False])
    written = [out_dir / SCENARIO_HOURS_CSV]
    scenario_hours.to_csv(written[-1], index=False)

    class_hours = pd.crosstab(turbine, scenario.map(_scenario_class)).mul(hours_per_record)
    written.append(_plot_classes(class_hours.reindex(columns=list(_CLASSES), fill_value=0.0), out_dir=out_dir))

    source_hours = pd.crosstab(turbine, pd.Index(labelled[DIRECTION_SOURCE_COL])).mul(hours_per_record)
    source_hours = source_hours.reindex(columns=[OWN, NEIGHBOURS, UNKNOWN], fill_value=0.0)
    difference = np.asarray(circ_diff(directions.own, directions.neighbours), dtype=float)
    abs_difference = pd.DataFrame(np.abs(difference), columns=list(directions.turbines))
    source_hours.columns = [f"hours_{c}" for c in source_hours.columns]
    source_hours["median_abs_own_minus_neighbours_deg"] = abs_difference.median()
    source_hours["p95_abs_own_minus_neighbours_deg"] = abs_difference.quantile(0.95)
    written.append(out_dir / DIRECTION_CSV)
    source_hours.rename_axis("turbine").to_csv(written[-1], float_format="%.2f")
    written.append(_plot_differences(difference, turbines=directions.turbines, out_dir=out_dir))

    written.append(_plot_upwind_sectors(layout, turbines=directions.turbines, out_dir=out_dir))
    logger.info("Wrote %d waking files to %s", len(written), out_dir)
    return written


def _scenario_class(scenario: str) -> str:
    """Return which of :data:`_CLASSES` a waking scenario falls in."""
    if scenario in (NORMAL, UNKNOWN):
        return scenario
    return _UPWIND_PART_WAKING if PART_WAKING in scenario else _UPWIND_NOT_WAKING


def _plot_classes(class_hours: pd.DataFrame, *, out_dir: Path) -> Path:
    """Bar each turbine's hours in each waking-scenario class other than normal, with the normal share above."""
    fig, ax = plt.subplots(figsize=(max(9.0, 0.5 * len(class_hours) + 3.0), 5.5))
    bottom = np.zeros(len(class_hours))
    x = np.arange(len(class_hours))
    normal_share = class_hours[NORMAL] / class_hours.sum(axis=1)
    for name in _CLASSES[1:]:
        ax.bar(x, class_hours[name], bottom=bottom, color=_CLASS_COLOURS[name], label=name)
        bottom += class_hours[name].to_numpy()
    for xi, top, share in zip(x, bottom, normal_share, strict=True):
        ax.annotate(f"{share:.0%}", (float(xi), float(top)), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, class_hours.index, rotation=90)
    ax.set_ylabel("hours")
    ax.set_title("Hours not in the normal waking scenario (step 5); share of hours normal above each bar")
    ax.legend(loc="upper right")
    apply_grid(ax)
    path = out_dir / "waking_scenarios.png"
    save_fig(fig, path)
    return path


def _plot_differences(difference: np.ndarray, *, turbines: tuple[str, ...], out_dir: Path) -> Path:
    """Histogram each turbine's own direction minus its neighbours' mean, where both are valid."""
    rows = math.ceil(len(turbines) / _PANEL_COLUMNS)
    fig, axes = plt.subplots(rows, _PANEL_COLUMNS, figsize=(4 * _PANEL_COLUMNS, 2.8 * rows), squeeze=False)
    fig.subplots_adjust(hspace=0.5)
    for ax, (j, turbine) in zip(axes.flat, enumerate(turbines), strict=False):
        values = difference[:, j]
        values = values[np.isfinite(values)]
        if len(values):
            ax.hist(np.clip(values, _DIFFERENCE_BINS[0], _DIFFERENCE_BINS[-1]), bins=_DIFFERENCE_BINS, color="C0")
            ax.set_title(f"{turbine}: median |diff| {np.median(np.abs(values)):.1f} deg", fontsize=9)
        else:
            ax.set_title(f"{turbine}: no records", fontsize=9)
        apply_grid(ax)
    for ax in axes.flat[len(turbines) :]:
        ax.set_visible(False)
    fig.suptitle("Own direction minus neighbours' mean direction (deg, clipped to +-30)")
    path = out_dir / "waking_direction_difference.png"
    save_fig(fig, path)
    return path


def _plot_upwind_sectors(layout: Layout, *, turbines: tuple[str, ...], out_dir: Path) -> Path:
    """Shade, for each turbine, which layout turbines are upwind of it at each whole degree."""
    names = [str(n) for n in layout.frame[NAME_COL]]
    order = np.argsort(names, kind="stable")
    diameters = layout.frame[ROTOR_DIAMETER_COL].to_numpy(dtype=float)
    directions = np.arange(360, dtype=float)
    rows = math.ceil(len(turbines) / _PANEL_COLUMNS)
    fig, axes = plt.subplots(rows, _PANEL_COLUMNS, figsize=(4 * _PANEL_COLUMNS, 3.6 * rows), squeeze=False)
    fig.subplots_adjust(hspace=0.55)
    for ax, turbine in zip(axes.flat, turbines, strict=False):
        target = layout.index_of(turbine)
        upwind = upwind_matrix(
            directions,
            bearing=layout.bearing_deg[target],
            sector=iec_disturbed_sector_deg(layout.distance_m[target] / diameters),
            target=target,
        )[:, order]
        ax.imshow(upwind.T, aspect="auto", interpolation="nearest", cmap="Greys", extent=(0, 360, len(names), 0))
        ax.set_yticks(np.arange(len(names)) + 0.5, [names[i] for i in order], fontsize=5)
        ax.set_xticks(range(0, 361, 90))
        ax.set_title(f"upwind of {turbine}", fontsize=9)
        ax.set_xlabel("direction (deg)", fontsize=8)
    for ax in axes.flat[len(turbines) :]:
        ax.set_visible(False)
    fig.suptitle("Turbines upwind (IEC disturbed sector) by direction")
    path = out_dir / "upwind_sectors.png"
    save_fig(fig, path)
    return path
