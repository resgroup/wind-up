from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from benchmarking.harness.label_checks import (
    COUNTER_CONTRADICTS_POWER,
    PARKED_AS_NORMAL,
    label_checks,
    warn_label_checks,
)
from benchmarking.harness.operating_state import OperatingStateConfig, label_operating_states
from benchmarking.synthetic import HOT_COLUMNS as COLS

TIMEBASE = pd.Timedelta(minutes=10)
RATED = 2000.0
FULL = TIMEBASE.total_seconds()


def running(n: int, *, turbine: str = "T01") -> pd.DataFrame:
    """``n`` records of a turbine generating, every signal varying."""
    index = pd.date_range("2020-01-01", periods=n, freq=TIMEBASE, tz="UTC")
    values = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            COLS.turbine: turbine,
            COLS.active_power: 1000.0 + values,
            COLS.wind_speed: 8.0 + values / 1000,
            COLS.gen_rpm: 1600.0 + values / 100,
            COLS.availability: FULL,
            COLS.pitch: -2.0 + values / 1000,
        },
        index=index,
    )


def checks_of(scada: pd.DataFrame, config: OperatingStateConfig | None = None) -> pd.DataFrame:
    labelled = label_operating_states(
        scada, columns=COLS, config=config or OperatingStateConfig(), timebase=TIMEBASE, rated_power_kw=RATED
    )
    return label_checks(labelled, columns=COLS, timebase=TIMEBASE, rated_power_kw=RATED)


def parked_stop(scada: pd.DataFrame, rows: slice) -> pd.DataFrame:
    """The same turbine parked over ``rows`` with its counter still reading available."""
    out = scada.copy()
    out.iloc[rows, out.columns.get_loc(COLS.active_power)] = -5.0
    out.iloc[rows, out.columns.get_loc(COLS.gen_rpm)] = 0.0
    out.iloc[rows, out.columns.get_loc(COLS.pitch)] = 86.0
    return out


def test_a_parked_stop_counted_as_normal_is_found_with_its_run() -> None:
    scada = parked_stop(running(200), slice(100, 160))
    checks = checks_of(scada)
    row = checks[checks["check"] == PARKED_AS_NORMAL].iloc[0]
    assert row["hours"] == pytest.approx(10.0)
    assert row["longest_run_start"] == scada.index[100]
    assert row["longest_run_hours"] == pytest.approx(10.0)
    assert row["median_pitch_deg"] == pytest.approx(86.0)


def test_a_parked_pitch_rule_silences_it() -> None:
    scada = parked_stop(running(200), slice(100, 160))
    assert checks_of(scada, OperatingStateConfig(parked_pitch_above_deg=40.0)).empty


def test_calm_rest_is_not_flagged() -> None:
    """A rotor at rest in wind too light to turn it is normal operation, not a missed stop."""
    scada = parked_stop(running(200), slice(100, 160))
    scada.iloc[100:160, scada.columns.get_loc(COLS.wind_speed)] = 2.0
    assert (checks_of(scada)["check"] != PARKED_AS_NORMAL).all()


def test_the_counter_contradicting_power_is_counted() -> None:
    scada = running(50)
    scada.iloc[10:22, scada.columns.get_loc(COLS.availability)] = 0.0
    checks = checks_of(scada)
    assert checks.set_index("check").loc[COUNTER_CONTRADICTS_POWER, "hours"] == pytest.approx(2.0)


def test_warnings_name_the_turbine_only_above_the_threshold(caplog: pytest.LogCaptureFixture) -> None:
    long_stop = parked_stop(running(200, turbine="T01"), slice(100, 160))
    short_stop = parked_stop(running(200, turbine="T02"), slice(100, 106))
    with caplog.at_level(logging.WARNING):
        warn_label_checks(checks_of(pd.concat([long_stop, short_stop])))
    assert "T01: 10 h of normal operation with the rotor at rest" in caplog.text
    assert "parked_pitch_above_deg" in caplog.text
    assert "T02" not in caplog.text


def test_a_clean_record_has_no_checks() -> None:
    assert checks_of(running(100)).empty
