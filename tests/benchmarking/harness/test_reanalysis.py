"""Tests for method step 3: aligning reanalysis to the SCADA timebase and checking the alignment."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from benchmarking.harness.operating_state import VALID_NORTHING_COL
from benchmarking.harness.reanalysis import (
    TimestampConvention,
    check_shift,
    interpolate_era5,
    normalise_timestamps,
    prepare_reanalysis,
    uncovered_spans,
)
from benchmarking.synthetic import HOT_COLUMNS

TIMEBASE = pd.Timedelta(minutes=10)


def hourly(n: int = 3, *, start: str = "2020-01-01", **fields: list[float]) -> pd.DataFrame:
    """An hourly ERA5 frame of ``n`` hours with ``fields``; 100 m wind speed and direction by default."""
    index = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    data = {"wind_speed_100m": [8.0] * n, "wind_direction_100m": [180.0] * n} | fields
    return pd.DataFrame(data, index=index)


def scada_index(n: int, *, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=TIMEBASE, tz="UTC")


class TestInterpolate:
    def test_instantaneous_field_is_linear_at_the_period_centre(self) -> None:
        era5 = hourly(2, temperature_2m=[0.0, 6.0])
        out = interpolate_era5(era5, index=scada_index(6), timebase=TIMEBASE)
        np.testing.assert_allclose(out["temperature_2m"], [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

    def test_direction_interpolates_across_north(self) -> None:
        era5 = hourly(2, wind_direction_100m=[350.0, 10.0])
        out = interpolate_era5(era5, index=scada_index(6), timebase=TIMEBASE)
        wd = out["wind_direction_100m"].to_numpy()
        assert ((wd > 345) | (wd < 15)).all()
        assert out["wind_direction_100m"].iloc[2] == pytest.approx(358.33, abs=0.05)

    def test_speed_is_interpolated_as_a_scalar(self) -> None:
        era5 = hourly(2, wind_speed_100m=[8.0, 8.0], wind_direction_100m=[0.0, 180.0])
        out = interpolate_era5(era5, index=scada_index(6), timebase=TIMEBASE)
        np.testing.assert_allclose(out["wind_speed_100m"], 8.0)

    def test_hour_ending_field_takes_the_value_of_the_hour_ending_after_the_period(self) -> None:
        era5 = hourly(3, precipitation=[1.0, 2.0, 3.0])
        out = interpolate_era5(era5, index=scada_index(12), timebase=TIMEBASE)
        np.testing.assert_allclose(out["precipitation"].iloc[:6], 2.0)
        np.testing.assert_allclose(out["precipitation"].iloc[6:], 3.0)

    def test_no_extrapolation_outside_the_record(self) -> None:
        era5 = hourly(2, start="2020-01-01 01:00", temperature_2m=[0.0, 6.0])
        out = interpolate_era5(era5, index=scada_index(18), timebase=TIMEBASE)
        assert out["temperature_2m"].iloc[:6].isna().all()
        assert out["temperature_2m"].iloc[6:12].notna().all()
        assert out["temperature_2m"].iloc[12:].isna().all()

    def test_adds_the_neutral_aliases(self) -> None:
        out = interpolate_era5(hourly(2), index=scada_index(6), timebase=TIMEBASE)
        np.testing.assert_allclose(out["era5_ws"], out["wind_speed_100m"])
        np.testing.assert_allclose(out["era5_wd"], out["wind_direction_100m"])

    def test_a_frame_already_on_the_timebase_is_only_reindexed(self) -> None:
        index = scada_index(6)
        aligned = interpolate_era5(hourly(2, temperature_2m=[0.0, 6.0]), index=index, timebase=TIMEBASE)
        again = interpolate_era5(aligned, index=index[1:], timebase=TIMEBASE)
        pd.testing.assert_frame_equal(again, aligned.iloc[1:])

    def test_trailing_all_nan_rows_are_trimmed_not_a_gap(self) -> None:
        era5 = hourly(3, temperature_2m=[0.0, 6.0, np.nan])
        era5.iloc[2] = np.nan
        out = interpolate_era5(era5, index=scada_index(12), timebase=TIMEBASE)
        assert out["temperature_2m"].iloc[6:].isna().all()

    def test_unclassified_column_raises(self) -> None:
        with pytest.raises(ValueError, match="mystery"):
            interpolate_era5(hourly(2, mystery=[1.0, 2.0]), index=scada_index(6), timebase=TIMEBASE)

    def test_missing_hour_raises(self) -> None:
        era5 = hourly(4).drop(index=pd.Timestamp("2020-01-01 02:00", tz="UTC"))
        with pytest.raises(ValueError, match="gap"):
            interpolate_era5(era5, index=scada_index(6), timebase=TIMEBASE)

    def test_interior_nan_raises(self) -> None:
        era5 = hourly(3, temperature_2m=[0.0, np.nan, 1.0])
        with pytest.raises(ValueError, match="temperature_2m"):
            interpolate_era5(era5, index=scada_index(6), timebase=TIMEBASE)

    def test_direction_without_speed_raises(self) -> None:
        era5 = hourly(2, wind_direction_10m=[0.0, 10.0])
        with pytest.raises(ValueError, match="wind_speed_10m"):
            interpolate_era5(era5, index=scada_index(6), timebase=TIMEBASE)

    def test_naive_reanalysis_index_raises(self) -> None:
        era5 = hourly(2)
        era5.index = era5.index.tz_localize(None)
        with pytest.raises(ValueError, match="UTC"):
            interpolate_era5(era5, index=scada_index(6), timebase=TIMEBASE)


class TestCoverage:
    def test_uncovered_spans_before_and_after(self) -> None:
        era5 = hourly(2, start="2020-01-01 01:00")
        index = scada_index(18)
        spans = uncovered_spans(interpolate_era5(era5, index=index, timebase=TIMEBASE), index=index)
        assert spans == [
            (index[0], index[5]),
            (index[12], index[17]),
        ]

    def test_full_coverage_has_no_spans(self) -> None:
        index = scada_index(6)
        assert uncovered_spans(interpolate_era5(hourly(2), index=index, timebase=TIMEBASE), index=index) == []


class TestShiftCheck:
    @staticmethod
    def signals(lag_minutes: int) -> tuple[pd.Series, pd.Series]:
        rng = np.random.default_rng(0)
        n = 6 * 24 * 20
        index = scada_index(n)
        base = pd.Series(np.cumsum(rng.normal(size=n + 200)), dtype=float)
        era5_ws = pd.Series(base.iloc[100 : 100 + n].to_numpy(), index=index)
        lag = lag_minutes // 10
        site_ws = pd.Series(base.iloc[100 - lag : 100 - lag + n].to_numpy(), index=index)
        return era5_ws, site_ws

    def test_aligned_signals_pass_quietly(self, caplog: pytest.LogCaptureFixture) -> None:
        era5_ws, site_ws = self.signals(0)
        with caplog.at_level(logging.WARNING):
            result = check_shift(era5_ws=era5_ws, site_ws=site_ws, timebase=TIMEBASE)
        assert result.best_shift == pd.Timedelta(0)
        assert not caplog.records

    def test_forty_minutes_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        era5_ws, site_ws = self.signals(40)
        with caplog.at_level(logging.WARNING):
            result = check_shift(era5_ws=era5_ws, site_ws=site_ws, timebase=TIMEBASE)
        assert abs(result.best_shift) == pd.Timedelta(minutes=40)
        assert any("shift" in r.message for r in caplog.records)

    def test_sixty_minutes_raises(self) -> None:
        era5_ws, site_ws = self.signals(60)
        with pytest.raises(ValueError, match="time zone"):
            check_shift(era5_ws=era5_ws, site_ws=site_ws, timebase=TIMEBASE)

    def test_fifty_minutes_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        era5_ws, site_ws = self.signals(50)
        with caplog.at_level(logging.WARNING):
            check_shift(era5_ws=era5_ws, site_ws=site_ws, timebase=TIMEBASE)
        assert any("shift" in r.message for r in caplog.records)

    def test_seventy_minutes_raises(self) -> None:
        era5_ws, site_ws = self.signals(70)
        with pytest.raises(ValueError, match="time zone"):
            check_shift(era5_ws=era5_ws, site_ws=site_ws, timebase=TIMEBASE)

    def test_positive_shift_means_the_site_lags_reanalysis(self) -> None:
        era5_ws, site_ws = self.signals(20)
        assert check_shift(era5_ws=era5_ws, site_ws=site_ws, timebase=TIMEBASE).best_shift == pd.Timedelta(minutes=20)


class TestNormaliseTimestamps:
    @staticmethod
    def scada(index: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame({"x": range(len(index))}, index=index)

    def test_naive_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            normalise_timestamps(
                self.scada(scada_index(3).tz_localize(None)), timestamps=TimestampConvention(), timebase=TIMEBASE
            )

    def test_declared_zone_mismatch_raises(self) -> None:
        index = scada_index(3).tz_convert("Europe/London")
        with pytest.raises(ValueError, match="Europe/London"):
            normalise_timestamps(self.scada(index), timestamps=TimestampConvention(), timebase=TIMEBASE)

    def test_declared_zone_is_converted_to_utc(self) -> None:
        index = pd.date_range("2020-07-01 01:00", periods=3, freq=TIMEBASE, tz="Europe/London")
        out = normalise_timestamps(
            self.scada(index), timestamps=TimestampConvention(time_zone="Europe/London"), timebase=TIMEBASE
        )
        assert out.index[0] == pd.Timestamp("2020-07-01 00:00", tz="UTC")
        assert str(out.index.tz) == "UTC"

    def test_period_end_moves_to_period_start(self) -> None:
        out = normalise_timestamps(
            self.scada(scada_index(3)), timestamps=TimestampConvention(convention="end"), timebase=TIMEBASE
        )
        assert out.index[0] == pd.Timestamp("2019-12-31 23:50", tz="UTC")

    def test_unknown_convention_raises(self) -> None:
        with pytest.raises(ValueError, match="middle"):
            TimestampConvention(convention="middle")  # type: ignore[arg-type]


class TestPrepare:
    def test_site_wind_speed_uses_unchanged_turbines_valid_for_northing(self) -> None:
        index = scada_index(6 * 24 * 4)
        hours = pd.date_range(index[0], index[-1] + pd.Timedelta(hours=1), freq="1h")

        def wind(times: pd.DatetimeIndex) -> np.ndarray:
            days = (times - index[0]) / pd.Timedelta(days=1)
            return 8 + 3 * np.sin(2 * np.pi * days / 1.7) + np.sin(2 * np.pi * days / 0.31)

        site = wind(index + TIMEBASE / 2)
        frames = [
            pd.DataFrame(
                {HOT_COLUMNS.turbine: wtg, HOT_COLUMNS.wind_speed: site * scale, VALID_NORTHING_COL: True},
                index=index,
            )
            for wtg, scale in (("T01", 1.0), ("T02", 1.0), ("T03", -5.0))
        ]
        scada = pd.concat(frames).sort_index()
        era5 = pd.DataFrame({"wind_speed_100m": wind(hours), "wind_direction_100m": 180.0}, index=hours)
        result = prepare_reanalysis(
            era5, scada_df=scada, columns=HOT_COLUMNS, unchanged=["T01", "T02"], timebase=TIMEBASE
        )
        assert result.best_shift == pd.Timedelta(0)
        assert result.best_corr > 0.9
        assert result.aligned.index.equals(index)
        assert result.uncovered == []
