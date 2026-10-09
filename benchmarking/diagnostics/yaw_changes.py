"""Possible yaw-alignment changes (method step 4): the candidates found in the north tables, with apparent Cp.

Writes ``possible_yaw_changes.csv`` and, for each turbine with a candidate, ``apparent_cp_<turbine>.png``:
the shift in the turbine's apparent Cp across each date of its record, with its candidates and declared
changeovers marked. Apparent Cp is read from the rows used for northing that are also valid for
uplift. The report is for inspection only; nothing downstream reads it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.diagnostics.style import apply_grid, save_fig
from benchmarking.harness.northing import northing_rows
from benchmarking.harness.operating_state import VALID_UPLIFT_COL
from wind_up.yaw_changes import CANDIDATE_COLUMNS, CP_WINDOW, possible_yaw_changes

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)

POSSIBLE_YAW_CHANGES_CSV = "possible_yaw_changes.csv"
REPORT_COLUMNS = (*CANDIDATE_COLUMNS, "days_from_declared_change", "in_analysis_period")


def write_possible_yaw_changes(
    labelled: pd.DataFrame,
    *,
    tables: Mapping[str, pd.DataFrame],
    columns: ColumnSchema,
    rated_power_kw: float,
    timebase: pd.Timedelta,
    changeovers: Mapping[str, Sequence[pd.Timestamp]],
    analysis_period: tuple[pd.Timestamp, pd.Timestamp] | None,
    out_dir: Path,
) -> pd.DataFrame:
    """Write the possible yaw-alignment changes of every turbine in ``tables`` and return them.

    :param labelled: long-format SCADA with the operating-state columns, every record, indexed by timestamp
    :param tables: each turbine's north table
    :param columns: the schema ``labelled`` is keyed by
    :param rated_power_kw: turbine rating, for the rows used for northing
    :param timebase: the records' period
    :param changeovers: each turbine's declared changeover dates
    :param analysis_period: the outer bounds of the analysis period, or ``None`` when undeclared
    :param out_dir: the folder written to
    :return: one row per candidate (:data:`REPORT_COLUMNS`)
    """
    index = pd.DatetimeIndex(labelled.index.unique()).sort_values()
    rows = northing_rows(labelled, columns=columns, rated_power_kw=rated_power_kw) & labelled[VALID_UPLIFT_COL].fillna(
        value=False
    ).to_numpy(dtype=bool)
    turbine_of = labelled[columns.turbine].astype(str)
    power: dict[str, np.ndarray] = {}
    wind_speed: dict[str, np.ndarray] = {}
    usable: dict[str, np.ndarray] = {}
    for turbine in tables:
        is_turbine = (turbine_of == turbine).to_numpy()
        frame = labelled[is_turbine].assign(_rows=rows[is_turbine])
        frame = frame[~frame.index.duplicated()].reindex(index)
        power[turbine] = frame[columns.active_power].to_numpy(dtype=float)
        wind_speed[turbine] = frame[columns.wind_speed].to_numpy(dtype=float)
        usable[turbine] = frame["_rows"].fillna(value=False).to_numpy(dtype=bool)
    found, scans = possible_yaw_changes(
        tables, index=index, power=power, wind_speed=wind_speed, usable=usable, timebase=timebase
    )
    found["days_from_declared_change"] = [
        _days_from_nearest(t, changeovers.get(turbine, ()))
        for turbine, t in zip(found["turbine"], found["timestamp"], strict=True)
    ]
    found["in_analysis_period"] = [
        analysis_period is not None and analysis_period[0] <= t < analysis_period[1] for t in found["timestamp"]
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    found = found[list(REPORT_COLUMNS)]
    found.to_csv(out_dir / POSSIBLE_YAW_CHANGES_CSV, index=False)
    for turbine, scan in scans.items():
        _plot_scan(
            scan,
            turbine=turbine,
            candidates=found[found["turbine"] == turbine],
            changeovers=changeovers.get(turbine, ()),
            out_dir=out_dir,
        )
    logger.info("Found %d possible yaw-alignment change(s) on %d turbine(s)", len(found), len(scans))
    return found


def _days_from_nearest(t: pd.Timestamp, dates: Sequence[pd.Timestamp]) -> float:
    """Signed days from the nearest of ``dates`` to ``t``; NaN when there are none."""
    if not dates:
        return float("nan")
    nearest = min(dates, key=lambda d: abs(t - d))
    return (t - nearest) / pd.Timedelta(days=1)


def _plot_scan(
    scan: pd.Series,
    *,
    turbine: str,
    candidates: pd.DataFrame,
    changeovers: Sequence[pd.Timestamp],
    out_dir: Path,
) -> None:
    """Write ``apparent_cp_<turbine>.png``: the apparent-Cp shift scan with candidates and changeovers marked."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(scan.index, 100 * scan.to_numpy(), color="C0", lw=1, label=f"shift across date (±{CP_WINDOW.days} d)")
    null_max = float(candidates["cp_null_max"].iloc[0])
    if np.isfinite(null_max):
        for sign in (-1, 1):
            ax.axhline(sign * 100 * null_max, color="grey", ls=":", lw=1)
    for i, (_, row) in enumerate(candidates.iterrows()):
        ax.axvline(row["timestamp"], color="C3", lw=1.2, label="candidate" if i == 0 else None)
        ax.plot(row["timestamp"], 100 * row["cp_shift"], "o", color="C3")
        ax.annotate(
            f"{row['north_step_deg']:+.1f}°, ratio {row['cp_ratio']:.1f}",
            (row["timestamp"], 100 * row["cp_shift"]),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=8,
            color="C3",
        )
    for i, changeover in enumerate(changeovers):
        ax.axvline(changeover, color="k", ls="--", lw=1, label="declared changeover" if i == 0 else None)
    ax.set_ylabel("apparent Cp shift [%]")
    ax.set_title(
        f"{turbine}: possible yaw-alignment changes (north step, |Cp shift| / largest elsewhere); "
        "dotted: largest shift elsewhere"
    )
    apply_grid(ax)
    ax.legend(loc="upper left", fontsize=8)
    save_fig(fig, out_dir / f"apparent_cp_{turbine}.png")
