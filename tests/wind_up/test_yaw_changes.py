"""Tests for the possible yaw-alignment change report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_up.northing import NORTH_OFFSET_COL, TIMESTAMP_COL
from wind_up.yaw_changes import apparent_cp_scan, possible_yaw_changes, yaw_change_candidates

TIMEBASE = pd.Timedelta(minutes=10)


def _table(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            TIMESTAMP_COL: pd.DatetimeIndex([t for t, _ in rows], tz="UTC"),
            NORTH_OFFSET_COL: [offset for _, offset in rows],
        }
    )


END = pd.Timestamp("2024-01-01", tz="UTC")


def test_in_frame_steps_are_candidates() -> None:
    table = _table([("2022-01-01", 10.0), ("2022-06-01", 14.0), ("2023-01-01", 11.5)])
    found = yaw_change_candidates(table, record_end=END)
    assert list(found[TIMESTAMP_COL]) == [pd.Timestamp("2022-06-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")]
    assert found["north_step_deg"].to_numpy() == pytest.approx([4.0, -2.5])


def test_an_excursion_and_its_imprecise_return_are_not_candidates() -> None:
    # out by 60 deg for two weeks, a small step while out, back 3 deg from where it left
    table = _table(
        [
            ("2022-01-01", 0.0),
            ("2022-05-01", 60.0),
            ("2022-05-08", 64.0),
            ("2022-05-15", 3.0),
            ("2022-09-01", 7.0),
        ]
    )
    found = yaw_change_candidates(table, record_end=END)
    assert list(found[TIMESTAMP_COL]) == [pd.Timestamp("2022-09-01", tz="UTC")]
    assert found["north_step_deg"].to_numpy() == pytest.approx([4.0])


def test_a_long_out_of_frame_run_is_a_recalibration_and_the_new_frame() -> None:
    table = _table([("2022-01-01", 0.0), ("2022-03-01", 90.0), ("2022-09-01", 95.0)])
    found = yaw_change_candidates(table, record_end=END)
    assert list(found[TIMESTAMP_COL]) == [pd.Timestamp("2022-09-01", tz="UTC")]
    assert found["north_step_deg"].to_numpy() == pytest.approx([5.0])


def test_a_constant_table_has_no_candidates() -> None:
    assert yaw_change_candidates(_table([("2022-01-01", 5.0)]), record_end=END).empty


def _turbine(*, change: pd.Timestamp, factor: float, seed: int = 0) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    index = pd.date_range("2022-01-01", "2023-12-31", freq=TIMEBASE, tz="UTC")
    rng = np.random.default_rng(seed)
    ws = rng.uniform(3.0, 13.0, len(index))
    power = 0.5 * ws**3 * (1.0 + 0.01 * rng.standard_normal(len(index)))
    power[index >= change] *= factor
    return index, power, ws


def test_cp_scan_reads_a_power_step_where_it_happens() -> None:
    change = pd.Timestamp("2023-01-01", tz="UTC")
    index, power, ws = _turbine(change=change, factor=1.1)
    usable = np.ones(len(index), dtype=bool)
    at = pd.DatetimeIndex([change, pd.Timestamp("2022-06-01", tz="UTC")])
    shifts = apparent_cp_scan(index, power=power, wind_speed=ws, usable=usable, timebase=TIMEBASE, at=at)
    assert shifts.iloc[0] == pytest.approx(0.1, abs=0.005)
    assert shifts.iloc[1] == pytest.approx(0.0, abs=0.005)


def test_cp_scan_skips_bins_without_enough_hours() -> None:
    index, power, ws = _turbine(change=pd.Timestamp("2023-01-01", tz="UTC"), factor=1.0)
    usable = np.zeros(len(index), dtype=bool)
    at = pd.DatetimeIndex([pd.Timestamp("2023-01-01", tz="UTC")])
    shifts = apparent_cp_scan(index, power=power, wind_speed=ws, usable=usable, timebase=TIMEBASE, at=at)
    assert np.isnan(shifts.iloc[0])


def test_possible_yaw_changes_compares_each_candidate_with_the_rest_of_the_record() -> None:
    change = pd.Timestamp("2023-01-01", tz="UTC")
    index, power, ws = _turbine(change=change, factor=1.1)
    tables = {
        "T01": _table([("2022-01-01", 0.0), ("2023-01-01", 5.0)]),
        "T02": _table([("2022-01-01", 0.0)]),
    }
    arrays = {"T01": power, "T02": power}
    found, scans = possible_yaw_changes(
        tables,
        index=index,
        power=arrays,
        wind_speed={"T01": ws, "T02": ws},
        usable={"T01": np.ones(len(index), dtype=bool), "T02": np.ones(len(index), dtype=bool)},
        timebase=TIMEBASE,
    )
    assert list(found["turbine"]) == ["T01"]
    assert list(scans) == ["T01"]
    row = found.iloc[0]
    assert row["north_step_deg"] == pytest.approx(5.0)
    assert row["cp_shift"] == pytest.approx(0.1, abs=0.005)
    assert row["cp_null_max"] < 0.02  # well below the step
    assert row["cp_ratio"] > 5  # the step stands out from the rest of the record
