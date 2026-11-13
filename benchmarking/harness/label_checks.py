"""Checks that the operating-state labels match what the turbines did, warned about in the run log.

The labels decide which records every later step uses, so a wrong label moves the result while the run
still reports a confident number: on a real campaign, a test turbine's eight-day parked stop counted as
normal operation moved the farm headline by 4 pp. These checks read power, rotor speed and pitch -- to
check the labels, never to make them -- and warn per turbine when a mislabel is large enough to matter:

* :data:`PARKED_AS_NORMAL` -- normal operation with the rotor at rest and power about zero in wind that
  would turn it. The availability counter can read available through a stop it does not account for
  (a licensing, grid or owner stop); a parked-pitch threshold or a site label fixes it.
* :data:`COUNTER_CONTRADICTS_POWER` -- the counter reads zero while the turbine generates. wind-up labels
  these records missing, since one of the two signals is wrong; many of them mean the counter, or the
  power signal, is unreliable on that turbine.

:func:`label_checks` returns one row per turbine and check with any records; :func:`warn_label_checks`
logs those above :data:`WARN_HOURS`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.harness.operating_state import NORMAL_OPERATION, STATE_COL, counter_contradicts_power

if TYPE_CHECKING:
    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)

PARKED_AS_NORMAL = "parked as normal operation"
COUNTER_CONTRADICTS_POWER = "counter contradicts power"

# At rest: rotor speed below this share of the turbine's own 99th-percentile rotor speed, and power at most
# this share of rated, in nacelle wind speed at or above WIND_TURNING_MS.
AT_REST_RPM_SHARE = 0.05
AT_REST_POWER_RATED = 0.01
WIND_TURNING_MS = 5.0
# A check warns for a turbine with at least this many hours.
WARN_HOURS = {PARKED_AS_NORMAL: 6.0, COUNTER_CONTRADICTS_POWER: 1.0}

LABEL_CHECKS_CSV = "label_checks.csv"
_COLUMNS = ("turbine", "check", "hours", "first", "last", "longest_run_start", "longest_run_hours", "median_pitch_deg")


def label_checks(
    labelled: pd.DataFrame, *, columns: ColumnSchema, timebase: pd.Timedelta, rated_power_kw: float
) -> pd.DataFrame:
    """Return the hours of each suspect label per turbine, with when it happened.

    :param labelled: long SCADA with the operating-state columns, indexed by period-start timestamps
    :param columns: the schema ``labelled`` is keyed by
    :param timebase: the records' period
    :param rated_power_kw: rated power
    :return: one row per turbine and check with any records, columns ``_COLUMNS``
    """
    power = labelled[columns.active_power].to_numpy(dtype=float)
    flags = {
        COUNTER_CONTRADICTS_POWER: counter_contradicts_power(
            power=power,
            counter=labelled[columns.availability].to_numpy(dtype=float),
            rated_power_kw=rated_power_kw,
        )
    }
    if columns.gen_rpm in labelled.columns and columns.wind_speed in labelled.columns:
        flags[PARKED_AS_NORMAL] = _parked_as_normal(labelled, columns=columns, rated_power_kw=rated_power_kw)
    rows = []
    turbines = labelled[columns.turbine].astype(str).to_numpy()
    stamps = pd.DatetimeIndex(labelled.index)
    pitch = labelled[columns.pitch].to_numpy(dtype=float) if columns.pitch in labelled.columns else None
    for check, flag in flags.items():
        for turbine in sorted(set(turbines[flag])):
            mine = flag & (turbines == turbine)
            when = stamps[mine].sort_values()
            start, length = _longest_run(when, timebase=timebase)
            rows.append(
                {
                    "turbine": turbine,
                    "check": check,
                    "hours": mine.sum() * timebase / pd.Timedelta(hours=1),
                    "first": when[0],
                    "last": when[-1],
                    "longest_run_start": start,
                    "longest_run_hours": length * timebase / pd.Timedelta(hours=1),
                    "median_pitch_deg": float(np.nanmedian(pitch[mine])) if pitch is not None else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=list(_COLUMNS))


def warn_label_checks(checks: pd.DataFrame) -> None:
    """Log a warning for each turbine and check at or above its :data:`WARN_HOURS`."""
    for row in checks.itertuples(index=False):
        if row.hours < WARN_HOURS[row.check]:
            continue
        where = (
            f"{row.first:%Y-%m-%d} to {row.last:%Y-%m-%d}, longest run {row.longest_run_hours:.0f} h "
            f"from {row.longest_run_start:%Y-%m-%d %H:%M}"
        )
        if row.check == PARKED_AS_NORMAL:
            logger.warning(
                "%s: %.0f h of normal operation with the rotor at rest in wind of %.0f m/s or more (%s; median "
                "pitch %.0f deg). The availability counter reads available through these stops: declare "
                "operating_state.parked_pitch_above_deg or label them, then check 02_operating_states.",
                row.turbine,
                row.hours,
                WIND_TURNING_MS,
                where,
                row.median_pitch_deg,
            )
        else:
            logger.warning(
                "%s: %.0f h where the availability counter reads zero while the turbine generates (%s); "
                "labelled missing. Check whether the counter or the power signal is wrong there.",
                row.turbine,
                row.hours,
                where,
            )


def _parked_as_normal(labelled: pd.DataFrame, *, columns: ColumnSchema, rated_power_kw: float) -> np.ndarray:
    """Return True on normal operation with the rotor at rest and power about zero in turning wind."""
    rpm = labelled[columns.gen_rpm].to_numpy(dtype=float)
    turbines = labelled[columns.turbine].astype(str)
    running_rpm = labelled[columns.gen_rpm].groupby(turbines.to_numpy()).transform(lambda s: s.quantile(0.99))
    with np.errstate(invalid="ignore"):
        return (
            (labelled[STATE_COL].to_numpy() == NORMAL_OPERATION)
            & (rpm < AT_REST_RPM_SHARE * running_rpm.to_numpy(dtype=float))
            & (labelled[columns.active_power].to_numpy(dtype=float) <= AT_REST_POWER_RATED * rated_power_kw)
            & (labelled[columns.wind_speed].to_numpy(dtype=float) >= WIND_TURNING_MS)
        )


def _longest_run(when: pd.DatetimeIndex, *, timebase: pd.Timedelta) -> tuple[pd.Timestamp, int]:
    """Return the start and record count of the longest run of consecutive timestamps in sorted ``when``."""
    breaks = np.flatnonzero(np.diff(when.as_unit("ns").asi8) != timebase.as_unit("ns").value) + 1
    starts = np.concatenate([[0], breaks])
    lengths = np.diff(np.concatenate([starts, [len(when)]]))
    best = int(np.argmax(lengths))
    return when[starts[best]], int(lengths[best])
