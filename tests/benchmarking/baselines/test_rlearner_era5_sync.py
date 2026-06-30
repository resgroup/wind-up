"""Tests for the R-learner ERA5 sync helper.

ERA5 arrives hourly; SCADA is 10-min. These cover the upsample to the analysis timebase
and the wind-speed correlation lag sweep that aligns ERA5 to the SCADA, on small
hand-built frames (no network).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.rlearner.era5_sync import (
    ERA5_WD,
    ERA5_WS,
    Era5SyncResult,
    find_best_lag,
    sync_era5,
    upsample_era5_to_timebase,
)

_RAW_WS = "wind_speed_100m"
_RAW_WD = "wind_direction_100m"


def _hourly(n: int, *, start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=n, freq="1h", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {_RAW_WS: np.arange(n, dtype=float) + 1.0, _RAW_WD: np.linspace(0.0, 90.0, n)},
        index=idx,
    )


class TestUpsample:
    def test_expands_to_ten_minute_grid(self) -> None:
        out = upsample_era5_to_timebase(_hourly(3), timebase=pd.Timedelta(minutes=10))
        # 3 hours -> 00:00..02:50 on a 10-min grid = 18 rows (last hour's 5 trailing slots filled)
        assert len(out) == 18
        assert out.index.freq is None or len(out) == 18

    def test_passes_through_raw_columns_and_adds_aliases(self) -> None:
        out = upsample_era5_to_timebase(_hourly(2), timebase=pd.Timedelta(minutes=10))
        # raw Open-Meteo columns are preserved (no renaming) and neutral ws/wd aliases are added
        assert set(out.columns) == {_RAW_WS, _RAW_WD, ERA5_WS, ERA5_WD}
        assert out[ERA5_WS].equals(out[_RAW_WS])
        assert out[ERA5_WD].equals(out[_RAW_WD])

    def test_forward_fills_within_the_hour(self) -> None:
        out = upsample_era5_to_timebase(_hourly(2), timebase=pd.Timedelta(minutes=10))
        # first hour's six 10-min slots all carry the first hour's raw value (1.0)
        assert out[ERA5_WS].to_numpy()[:6] == pytest.approx(1.0)
        assert out[ERA5_WS].to_numpy()[6:12] == pytest.approx(2.0)


class TestFindBestLag:
    def test_recovers_known_positive_lag(self) -> None:
        idx = pd.date_range("2020-01-01", periods=300, freq="10min", tz="UTC")
        rng = np.random.default_rng(0)
        era5_ws = pd.Series(rng.normal(8.0, 2.0, size=len(idx)), index=idx)
        # reference lags ERA5 by 3 rows: reference[t] == era5[t-3]
        reference_ws = era5_ws.shift(3)
        best_lag, best_corr, sweep = find_best_lag(
            reference_ws=reference_ws, era5_ws=era5_ws, timebase=pd.Timedelta(minutes=10)
        )
        assert best_lag == 3
        assert best_corr == pytest.approx(1.0, abs=1e-6)
        assert {"shift_rows", "corr"} <= set(sweep.columns)

    def test_zero_lag_when_aligned(self) -> None:
        idx = pd.date_range("2020-01-01", periods=300, freq="10min", tz="UTC")
        rng = np.random.default_rng(1)
        era5_ws = pd.Series(rng.normal(8.0, 2.0, size=len(idx)), index=idx)
        best_lag, _, _ = find_best_lag(reference_ws=era5_ws.copy(), era5_ws=era5_ws, timebase=pd.Timedelta(minutes=10))
        assert best_lag == 0


class TestSyncEra5:
    def test_returns_aligned_frame_on_target_index(self) -> None:
        target = pd.date_range("2020-01-01 00:00", periods=120, freq="10min", tz="UTC")
        # 24 hours of ERA5 covering the target window
        era5 = _hourly(24)
        rng = np.random.default_rng(2)
        reference_ws = pd.Series(rng.normal(8.0, 2.0, size=len(target)), index=target)
        result = sync_era5(era5, target_index=target, reference_ws=reference_ws)
        assert isinstance(result, Era5SyncResult)
        assert {_RAW_WS, _RAW_WD, ERA5_WS, ERA5_WD} <= set(result.aligned.columns)
        assert result.aligned.index.equals(target)

    def test_applies_recovered_lag_to_columns(self) -> None:
        target = pd.date_range("2020-01-01 00:00", periods=144, freq="10min", tz="UTC")
        # random (non-monotonic) hourly ws so the lag is identifiable
        hourly_idx = pd.date_range("2020-01-01", periods=36, freq="1h", tz="UTC", name="timestamp")
        rng = np.random.default_rng(3)
        era5 = pd.DataFrame(
            {_RAW_WS: rng.normal(8.0, 2.0, size=36), _RAW_WD: rng.uniform(0.0, 360.0, size=36)},
            index=hourly_idx,
        )
        up = upsample_era5_to_timebase(era5, timebase=pd.Timedelta(minutes=10)).reindex(target)
        # reference lags the upsampled ERA5 ws by 2 rows -> sync should shift ERA5 forward by 2
        reference_ws = up[ERA5_WS].shift(2)
        result = sync_era5(era5, target_index=target, reference_ws=reference_ws)
        assert result.best_lag_rows == 2
        expected = up[ERA5_WS].shift(2)
        pd.testing.assert_series_equal(result.aligned[ERA5_WS], expected, check_names=False)
