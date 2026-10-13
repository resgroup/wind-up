"""Draw the ``docs/northing.md`` figures on Hill of Towie: north error before and after each northing step.

The north error is the 14-day rolling circular median of the circular difference between each
turbine's nacelle position and ERA5 wind direction, over the rows northing uses
(:func:`~wind_up.northing.yaw_usable`). Each step's tables are rebuilt the way
:func:`~wind_up.northing.north_farm` builds them:

* ``north_error_before_after.png`` -- raw, then after every step;
* ``step_reanalysis_anchor.png`` -- raw, then after reanalysis-anchor;
* ``step_changepoints_v_consensus.png`` -- after reanalysis-anchor, then after changepoints-v-consensus;
* ``step_changepoints_v_reanalysis.png`` -- the same for a two-turbine subset, northed against reanalysis;
* ``wake_nadir_bubble.png`` and ``wake_nadir_pair_<up>_<down>.png`` -- the wake-nadir shift.

Needs the cached Hill of Towie open data (see :mod:`benchmarking.baselines.study_wake_nadir_golden`).

    python -m benchmarking.baselines.study_northing_doc_plot
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.baselines.study_wake_nadir_golden import hot_inputs
from benchmarking.diagnostics.style import apply_grid, save_fig, series_style
from wind_up.circular_math import circ_diff, circ_median
from wind_up.northing import DEFAULT_NORTHING, add_wake_nadir_shift, apply_north_table, estimate_north_table, north_farm
from wind_up.northing_plots import plot_wake_nadir_farm, plot_wake_nadir_pair
from wind_up.wake_nadir import wake_pair_curves

if TYPE_CHECKING:
    from wind_up.layout import Layout

logger = logging.getLogger("northing_doc_plot")

REPO = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO / "docs" / "images" / "northing"
WINDOW = ("2016-01-01", "2021-01-01")
# A two-turbine subset for changepoints-v-reanalysis: both have large steps.
REANALYSIS_SUBSET = ("T12", "T19")
# The wake drawn before and after the wake-nadir shift.
WAKE_PAIR = ("T21", "T19")


def rolling_error(
    direction: np.ndarray, *, reference: np.ndarray, usable: np.ndarray, index: pd.DatetimeIndex
) -> pd.Series:
    """Return the 14-day centred rolling circular median of ``circ_diff(direction, reference)`` over ``usable`` rows.

    Evaluated twice a day; a window with fewer than a third of its rows usable gives NaN.
    """
    rows_per_day = 144
    half_window = 7 * rows_per_day
    step = rows_per_day // 2
    error = np.where(usable, circ_diff(direction, reference), np.nan)
    finite = np.concatenate([[0], np.cumsum(np.isfinite(error))])
    centres = np.arange(0, len(error), step)
    values = np.full(len(centres), np.nan)
    for k, centre in enumerate(centres):
        lo, hi = max(int(centre) - half_window, 0), min(int(centre) + half_window, len(error))
        if finite[hi] - finite[lo] >= 2 * half_window // 3:
            values[k] = circ_median(error[lo:hi], range_360=False)
    return pd.Series(values, index=index[centres])


def before_after_figure(
    inputs: dict,
    *,
    before: dict[str, pd.DataFrame] | None,
    after: dict[str, pd.DataFrame],
    title: str,
    panel_titles: tuple[str, str],
    output: Path,
) -> None:
    """Save each turbine in ``after``, northed by ``before`` (raw when ``None``) above and by ``after`` below."""
    index = inputs["index"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, sharey=True)
    for position, turbine in enumerate(sorted(after)):
        raw = inputs["direction"][turbine]
        colour, dash = series_style(position)
        for ax, tables in zip(axes, (before, after), strict=True):
            direction = raw if tables is None else apply_north_table(index, raw, north_table=tables[turbine])
            error = rolling_error(
                direction, reference=inputs["reference"], usable=inputs["usable"][turbine], index=index
            )
            values = error.to_numpy()
            wrap_deg = 180.0
            values[1:][np.abs(np.diff(values)) > wrap_deg] = np.nan  # break the line where it wraps
            ax.plot(error.index, values, color=colour, linestyle=dash, linewidth=1.2, label=turbine)

    fig.suptitle(title)
    for ax, panel_title in zip(axes, panel_titles, strict=True):
        ax.set_title(panel_title)
        ax.set_ylabel("14-day rolling north error [deg]")
        ax.set_ylim(-180, 180)
        ax.set_yticks(range(-180, 181, 45))
        apply_grid(ax)
    axes[1].set_xlabel("date")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small", title="turbine")
    fig.tight_layout()
    save_fig(fig, output)
    logger.info("wrote %s", output)


def reanalysis_anchor(inputs: dict, *, turbines: list[str]) -> dict[str, pd.DataFrame]:
    """Return the reanalysis-anchor tables: one constant offset per turbine, as ``north_farm`` builds them."""
    anchoring = replace(DEFAULT_NORTHING, changepoints_per_year=0.0, min_changepoints=0)
    return {
        t: estimate_north_table(
            inputs["index"],
            inputs["direction"][t],
            reference_deg=inputs["reference"],
            usable=inputs["usable"][t],
            settings=anchoring,
        )
        for t in turbines
    }


def wake_nadir_figures(
    layout: Layout, inputs: dict, *, before: dict[str, pd.DataFrame], after: dict[str, pd.DataFrame], corrections: dict
) -> None:
    """Save the wake-nadir bubble map and one pair's wake before and after the shift."""
    figure = plot_wake_nadir_farm(layout, corrections=corrections, out_dir=OUTPUT_DIR)
    plt.close(figure)
    upstream, downstream = WAKE_PAIR
    curves = [
        wake_pair_curves(
            layout,
            upstream=upstream,
            downstream=downstream,
            northed_direction=apply_north_table(
                inputs["index"], inputs["direction"][upstream], north_table=tables[upstream]
            ),
            power=inputs["power"],
            wind_speed=inputs["wind_speed"],
            usable=inputs["usable"],
        )
        for tables in (before, after)
    ]
    if curves[0] is None or curves[1] is None:
        msg = f"the {upstream}->{downstream} wake is too thinly sampled to draw"
        raise ValueError(msg)
    figure = plot_wake_nadir_pair(curves[0], after=curves[1], out_dir=OUTPUT_DIR)
    plt.close(figure)
    logger.info("wrote the wake-nadir figures")


def main() -> None:
    """Load Hill of Towie, north it step by step, and save every figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    layout, inputs = hot_inputs(start=WINDOW[0], end=WINDOW[1])
    turbines = sorted(inputs["direction"])
    common = {"usable": inputs["usable"], "reanalysis_deg": inputs["reference"]}

    anchor = reanalysis_anchor(inputs, turbines=turbines)
    consensus = north_farm(inputs["index"], direction_deg=inputs["direction"], layout=layout, **common)
    final, corrections = add_wake_nadir_shift(
        consensus,
        layout=layout,
        index=inputs["index"],
        direction_deg=inputs["direction"],
        power=inputs["power"],
        wind_speed=inputs["wind_speed"],
        usable=inputs["usable"],
    )
    subset = {t: inputs["direction"][t] for t in REANALYSIS_SUBSET}
    v_reanalysis = north_farm(inputs["index"], direction_deg=subset, layout=None, **common)

    raw_title = "circular difference of raw nacelle position and ERA5 wind direction"
    before_after_figure(
        inputs,
        before=None,
        after=final,
        title="Hill of Towie: north error per turbine, before and after northing",
        panel_titles=(
            f"Before northing: {raw_title}",
            "After northing: circular difference of northed nacelle position and ERA5 wind direction",
        ),
        output=OUTPUT_DIR / "north_error_before_after.png",
    )
    before_after_figure(
        inputs,
        before=None,
        after=anchor,
        title="Hill of Towie: north error per turbine, before and after reanalysis-anchor",
        panel_titles=(f"Before: {raw_title}", "After reanalysis-anchor: one constant offset per turbine"),
        output=OUTPUT_DIR / "step_reanalysis_anchor.png",
    )
    before_after_figure(
        inputs,
        before=anchor,
        after=consensus,
        title="Hill of Towie: north error per turbine, before and after changepoints-v-consensus",
        panel_titles=("Before: after reanalysis-anchor", "After changepoints-v-consensus"),
        output=OUTPUT_DIR / "step_changepoints_v_consensus.png",
    )
    before_after_figure(
        inputs,
        before={t: anchor[t] for t in REANALYSIS_SUBSET},
        after=v_reanalysis,
        title=f"Hill of Towie, {' and '.join(REANALYSIS_SUBSET)} alone: before and after changepoints-v-reanalysis",
        panel_titles=("Before: after reanalysis-anchor", "After changepoints-v-reanalysis"),
        output=OUTPUT_DIR / "step_changepoints_v_reanalysis.png",
    )
    wake_nadir_figures(layout, inputs, before=consensus, after=final, corrections=corrections)


if __name__ == "__main__":
    main()
