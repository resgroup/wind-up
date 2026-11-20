"""Tests for the possible yaw-alignment change report written in data preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.diagnostics.yaw_changes import POSSIBLE_YAW_CHANGES_CSV, REPORT_COLUMNS, write_possible_yaw_changes
from benchmarking.harness.operating_state import VALID_NORTHING_COL, VALID_UPLIFT_COL
from benchmarking.synthetic import HOT_COLUMNS
from wind_up.northing import NORTH_OFFSET_COL, TIMESTAMP_COL

if TYPE_CHECKING:
    from pathlib import Path

_CHANGE = pd.Timestamp("2023-01-01", tz="UTC")


def _labelled(*, valid_uplift_after: bool = True) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", "2023-12-31", freq="10min", tz="UTC")
    rng = np.random.default_rng(0)
    frames = []
    for turbine in ("T01", "T02"):
        ws = rng.uniform(3.0, 13.0, len(index))
        power = 0.5 * ws**3
        if turbine == "T01":
            power[index >= _CHANGE] *= 1.1
        frames.append(
            pd.DataFrame(
                {
                    HOT_COLUMNS.turbine: turbine,
                    HOT_COLUMNS.active_power: power,
                    HOT_COLUMNS.wind_speed: ws,
                    VALID_NORTHING_COL: True,
                    VALID_UPLIFT_COL: valid_uplift_after | (index < _CHANGE),
                },
                index=index,
            )
        )
    return pd.concat(frames)


def _tables() -> dict[str, pd.DataFrame]:
    start = pd.Timestamp("2022-01-01", tz="UTC")
    return {
        "T01": pd.DataFrame({TIMESTAMP_COL: pd.DatetimeIndex([start, _CHANGE]), NORTH_OFFSET_COL: [0.0, 4.0]}),
        "T02": pd.DataFrame({TIMESTAMP_COL: pd.DatetimeIndex([start]), NORTH_OFFSET_COL: [0.0]}),
    }


def _write(labelled: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    return write_possible_yaw_changes(
        labelled,
        tables=_tables(),
        columns=HOT_COLUMNS,
        rated_power_kw=2300.0,
        timebase=pd.Timedelta(minutes=10),
        changeovers={"T01": [_CHANGE + pd.Timedelta(days=2)]},
        analysis_period=(pd.Timestamp("2022-06-01", tz="UTC"), pd.Timestamp("2023-06-01", tz="UTC")),
        out_dir=out_dir,
    )


def test_each_candidate_is_reported_with_its_apparent_cp_shift(tmp_path: Path) -> None:
    found = _write(_labelled(), tmp_path)
    assert list(found.columns) == list(REPORT_COLUMNS)
    (row,) = found.itertuples()
    assert row.turbine == "T01"
    assert row.north_step_deg == pytest.approx(4.0)
    assert row.cp_shift == pytest.approx(0.1, abs=0.005)
    assert row.days_from_declared_change == pytest.approx(-2.0)
    assert row.in_analysis_period
    assert (tmp_path / POSSIBLE_YAW_CHANGES_CSV).exists()
    assert (tmp_path / "apparent_cp_T01.png").exists()
    assert not (tmp_path / "apparent_cp_T02.png").exists()


def test_apparent_cp_reads_only_rows_valid_for_uplift(tmp_path: Path) -> None:
    found = _write(_labelled(valid_uplift_after=False), tmp_path)
    assert np.isnan(found["cp_shift"].iloc[0])
