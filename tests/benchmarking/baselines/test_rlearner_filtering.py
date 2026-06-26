"""Tests for the R-learner test-turbine normal-operation filter.

The outcome ``Y`` is the test turbine's power, so abnormal operation (downtime, curtailment,
frozen/stuck sensors) unrelated to the upgrade would otherwise be attributed to it. These
cover the stuck-data and downtime filters, and the central correctness rule: filter on
*cause* (operational flags) never *effect* (low power).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarking.baselines.rlearner.filtering import NormalOperationFilter

_POWER = "wtc_ActPower_mean"
_WS = "wtc_AcWindSp_mean"
_AVAIL = "wtc_ScReToOp_timeon"  # seconds ready to operate in the period
_TIMEBASE = pd.Timedelta(minutes=10)
_FULL = _TIMEBASE.total_seconds()  # 600s = fully available


def _test_rows(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC", name="timestamp")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            _POWER: rng.uniform(200, 900, n),
            _WS: rng.uniform(5, 12, n),
            _AVAIL: np.full(n, _FULL),
        },
        index=idx,
    )


class TestDowntime:
    def test_drops_partial_availability(self) -> None:
        rows = _test_rows()
        rows.loc[rows.index[2], _AVAIL] = 300.0  # only half the period available
        keep = NormalOperationFilter(active_power_col=_POWER, availability_col=_AVAIL).keep_mask(
            rows, timebase=_TIMEBASE
        )
        assert not keep.iloc[2]
        assert keep.drop(rows.index[2]).all()

    def test_drops_nan_availability(self) -> None:
        rows = _test_rows()
        rows.loc[rows.index[4], _AVAIL] = np.nan
        keep = NormalOperationFilter(active_power_col=_POWER, availability_col=_AVAIL).keep_mask(
            rows, timebase=_TIMEBASE
        )
        assert not keep.iloc[4]

    def test_no_availability_col_keeps_all(self) -> None:
        rows = _test_rows()
        keep = NormalOperationFilter(active_power_col=_POWER, apply_stuck_filter=False).keep_mask(
            rows, timebase=_TIMEBASE
        )
        assert keep.all()


class TestStuckData:
    def test_flags_frozen_rows_in_normal_wind(self) -> None:
        rows = _test_rows()
        # rows 5,6,7 frozen (every signal identical to row 4) in normal wind -> stuck
        for i in (5, 6, 7):
            rows.iloc[i] = rows.iloc[4]
        keep = NormalOperationFilter(active_power_col=_POWER, wind_speed_col=_WS).keep_mask(rows, timebase=_TIMEBASE)
        assert not keep.iloc[[5, 6, 7]].any()
        assert keep.iloc[4]  # the first of the run is not itself a repeat

    def test_low_wind_calm_is_not_stuck(self) -> None:
        rows = _test_rows()
        # frozen run but at very low wind: a genuine calm, must NOT be filtered as stuck
        for i in (5, 6, 7):
            rows.iloc[i] = rows.iloc[4]
            rows.iloc[i, rows.columns.get_loc(_WS)] = 0.5
        rows.iloc[4, rows.columns.get_loc(_WS)] = 0.5
        keep = NormalOperationFilter(active_power_col=_POWER, wind_speed_col=_WS).keep_mask(rows, timebase=_TIMEBASE)
        assert keep.iloc[[5, 6, 7]].all()


class TestCauseNotEffect:
    def test_low_power_but_normal_operation_is_kept(self) -> None:
        # the key rule: a genuinely low-power record that is fully available and not stuck must be
        # KEPT. Filtering it (because power is low) would remove real low-uplift records and bias.
        rows = _test_rows()
        rows.loc[rows.index[3], _POWER] = 5.0  # very low power, but available and not frozen
        keep = NormalOperationFilter(active_power_col=_POWER, wind_speed_col=_WS, availability_col=_AVAIL).keep_mask(
            rows, timebase=_TIMEBASE
        )
        assert keep.iloc[3]

    def test_nan_power_is_dropped(self) -> None:
        rows = _test_rows()
        rows.loc[rows.index[6], _POWER] = np.nan
        keep = NormalOperationFilter(active_power_col=_POWER).keep_mask(rows, timebase=_TIMEBASE)
        assert not keep.iloc[6]


class TestReturnType:
    def test_keep_mask_is_boolean_series_on_index(self) -> None:
        rows = _test_rows()
        keep = NormalOperationFilter(active_power_col=_POWER).keep_mask(rows, timebase=_TIMEBASE)
        assert isinstance(keep, pd.Series)
        assert keep.index.equals(rows.index)
        assert keep.dtype == bool
