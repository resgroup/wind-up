"""Draw the ``docs/northing.md`` figure: every Hill of Towie turbine's north error, before and after northing.

The error is the 14-day rolling circular median of the circular difference between each turbine's
nacelle position and ERA5 wind direction, over the rows northing uses
(:func:`~wind_up.northing.yaw_usable`). The top panel uses the raw nacelle position; the bottom one
applies the golden north table
(``tests/test_data/hot/northing/golden_northing_corrections_hill_of_towie.yaml``).

Needs the cached Hill of Towie open data (see :mod:`benchmarking.baselines.study_wake_nadir_golden`).

    python -m benchmarking.baselines.study_northing_doc_plot
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.baselines.study_northing_degradation import load_north_table_yaml
from benchmarking.baselines.study_wake_nadir_golden import NORTHING_DIR, hot_inputs
from benchmarking.diagnostics.style import apply_grid, save_fig, series_style
from wind_up.circular_math import circ_diff, circ_median
from wind_up.northing import apply_north_table

logger = logging.getLogger("northing_doc_plot")

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "docs" / "images" / "northing" / "north_error_before_after.png"
WINDOW = ("2016-01-01", "2021-01-01")


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
        lo, hi = max(centre - half_window, 0), min(centre + half_window, len(error))
        if finite[hi] - finite[lo] >= 2 * half_window // 3:
            values[k] = circ_median(error[lo:hi], range_360=False)
    return pd.Series(values, index=index[centres])


def main() -> None:
    """Load the data, compute each turbine's rolling error before and after northing, and save the figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _, inputs = hot_inputs(start=WINDOW[0], end=WINDOW[1])
    tables = load_north_table_yaml(NORTHING_DIR / "golden_northing_corrections_hill_of_towie.yaml")
    index = inputs["index"]
    reference = inputs["reference"]
    turbines = sorted(inputs["direction"])

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, sharey=True)
    for position, turbine in enumerate(turbines):
        raw = inputs["direction"][turbine]
        northed = apply_north_table(index, raw, north_table=tables[turbine])
        colour, dash = series_style(position)
        for ax, direction in zip(axes, (raw, northed), strict=True):
            error = rolling_error(direction, reference=reference, usable=inputs["usable"][turbine], index=index)
            values = error.to_numpy()
            wrap_deg = 180.0
            values[1:][np.abs(np.diff(values)) > wrap_deg] = np.nan  # break the line where it wraps
            ax.plot(error.index, values, color=colour, linestyle=dash, linewidth=1.2, label=turbine)
        logger.info("%s done", turbine)

    fig.suptitle("Hill of Towie: north error per turbine, before and after northing")
    axes[0].set_title("Before northing: circular difference of raw nacelle position and ERA5 wind direction")
    axes[1].set_title("After northing: circular difference of northed nacelle position and ERA5 wind direction")
    for ax in axes:
        ax.set_ylabel("14-day rolling north error [deg]")
        ax.set_ylim(-180, 180)
        ax.set_yticks(range(-180, 181, 45))
        apply_grid(ax)
    axes[1].set_xlabel("date")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small", title="turbine")
    fig.tight_layout()
    save_fig(fig, OUTPUT)
    logger.info("wrote %s", OUTPUT)


if __name__ == "__main__":
    main()
