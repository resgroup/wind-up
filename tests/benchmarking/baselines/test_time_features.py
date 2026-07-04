"""Tests for the explicit time features in ``benchmarking.baselines.time_features``.

Checks known-input values for each feature (a plain linear day-count, a solstice-anchored
sin/cos pair, and NOAA solar-position altitude/azimuth against hand-verified reference
points), plus the shared tz-naive-index error path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.time_features import (
    TIME_FEATURE_NAMES,
    days_since_campaign_start,
    season_sin_cos,
    solar_altitude_azimuth,
)

# Hill of Towie, Aberdeenshire -- a real wind_up test site, used as the solar-position fixture.
_HILL_OF_TOWIE_LATITUDE = 57.50
_HILL_OF_TOWIE_LONGITUDE = -3.25


def test_time_feature_names_vocabulary() -> None:
    assert TIME_FEATURE_NAMES == ("days_since_campaign_start", "season", "solar")


def test_days_since_campaign_start_exact_values() -> None:
    campaign_start = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
    index = pd.DatetimeIndex(["2019-12-31 00:00:00", "2020-01-01 00:00:00", "2020-01-02 12:00:00"], tz="UTC")

    result = days_since_campaign_start(index, campaign_start=campaign_start)

    assert result.name == "days_since_campaign_start"
    np.testing.assert_allclose(result.to_numpy(), [-1.0, 0.0, 1.5])
    assert result.index.equals(index)


def test_days_since_campaign_start_raises_on_tz_naive_campaign_start() -> None:
    index = pd.date_range("2018-06-21", periods=3, freq="D", tz="UTC")
    with pytest.raises(ValueError, match="campaign_start must be timezone-aware"):
        days_since_campaign_start(index, campaign_start=pd.Timestamp("2018-06-22"))


def test_days_since_campaign_start_raises_on_tz_naive_index() -> None:
    index = pd.DatetimeIndex(["2020-01-01 00:00:00"])
    with pytest.raises(ValueError, match="timezone-aware"):
        days_since_campaign_start(index, campaign_start=pd.Timestamp("2020-01-01", tz="UTC"))


def test_season_sin_cos_june_solstice_noon() -> None:
    index = pd.DatetimeIndex(["2018-06-21 12:00:00"], tz="UTC")

    result = season_sin_cos(index)

    assert list(result.columns) == ["season_sin", "season_cos"]
    np.testing.assert_allclose(result["season_cos"].to_numpy(), [1.0], atol=0.01)
    np.testing.assert_allclose(result["season_sin"].to_numpy(), [0.0], atol=0.02)


def test_season_sin_cos_december_solstice() -> None:
    index = pd.DatetimeIndex(["2018-12-21 00:00:00"], tz="UTC")

    result = season_sin_cos(index)

    np.testing.assert_allclose(result["season_cos"].to_numpy(), [-1.0], atol=0.01)


def test_season_sin_cos_is_on_unit_circle() -> None:
    index = pd.date_range("2018-01-01", periods=50, freq="7D", tz="UTC")

    result = season_sin_cos(index)

    magnitude_sq = result["season_sin"].to_numpy() ** 2 + result["season_cos"].to_numpy() ** 2
    np.testing.assert_allclose(magnitude_sq, np.ones_like(magnitude_sq))


def test_season_sin_cos_leap_year_anchor_stays_on_june21() -> None:
    # 2020 is a leap year: June 21 is day-of-year 173, and the anchor must move with it.
    index = pd.DatetimeIndex(["2020-06-21 12:00:00"], tz="UTC")

    result = season_sin_cos(index)

    np.testing.assert_allclose(result["season_cos"].to_numpy(), [1.0], atol=0.01)
    np.testing.assert_allclose(result["season_sin"].to_numpy(), [0.0], atol=0.02)


def test_season_sin_cos_non_utc_index_encodes_the_same_instants() -> None:
    utc = pd.DatetimeIndex(["2018-06-21 12:00:00"], tz="UTC")
    same_instant_elsewhere = utc.tz_convert("Australia/Sydney")

    np.testing.assert_allclose(
        season_sin_cos(same_instant_elsewhere).to_numpy(), season_sin_cos(utc).to_numpy(), atol=1e-12
    )


def test_season_sin_cos_raises_on_tz_naive_index() -> None:
    index = pd.DatetimeIndex(["2018-06-21 12:00:00"])
    with pytest.raises(ValueError, match="timezone-aware"):
        season_sin_cos(index)


def test_solar_altitude_azimuth_hill_of_towie_summer_noon() -> None:
    """Sun near its highest, roughly due south, at midsummer local noon."""
    index = pd.DatetimeIndex(["2018-06-21 12:00:00"], tz="UTC")

    result = solar_altitude_azimuth(index, latitude=_HILL_OF_TOWIE_LATITUDE, longitude=_HILL_OF_TOWIE_LONGITUDE)

    assert result["solar_altitude"].to_numpy()[0] == pytest.approx(55.7, abs=1.0)
    # Azimuth close to due south (180 degrees): sin small, cos close to -1.
    assert result["solar_azimuth_sin"].to_numpy()[0] == pytest.approx(0.1, abs=0.15)
    assert result["solar_azimuth_cos"].to_numpy()[0] == pytest.approx(-1.0, abs=0.05)


def test_solar_altitude_azimuth_hill_of_towie_midnight_is_below_horizon() -> None:
    index = pd.DatetimeIndex(["2018-06-21 00:00:00"], tz="UTC")

    result = solar_altitude_azimuth(index, latitude=_HILL_OF_TOWIE_LATITUDE, longitude=_HILL_OF_TOWIE_LONGITUDE)

    assert result["solar_altitude"].to_numpy()[0] < 0.0


def test_solar_altitude_azimuth_equator_equinox_noon_near_zenith() -> None:
    index = pd.DatetimeIndex(["2019-03-21 12:00:00"], tz="UTC")

    result = solar_altitude_azimuth(index, latitude=0.0, longitude=0.0)

    assert result["solar_altitude"].to_numpy()[0] > 85.0


def test_solar_azimuth_sin_cos_on_unit_circle() -> None:
    index = pd.date_range("2018-01-01", periods=100, freq="97min", tz="UTC")

    result = solar_altitude_azimuth(index, latitude=_HILL_OF_TOWIE_LATITUDE, longitude=_HILL_OF_TOWIE_LONGITUDE)

    magnitude_sq = result["solar_azimuth_sin"].to_numpy() ** 2 + result["solar_azimuth_cos"].to_numpy() ** 2
    np.testing.assert_allclose(magnitude_sq, np.ones_like(magnitude_sq))


def test_solar_altitude_azimuth_raises_on_tz_naive_index() -> None:
    index = pd.DatetimeIndex(["2018-06-21 12:00:00"])
    with pytest.raises(ValueError, match="timezone-aware"):
        solar_altitude_azimuth(index, latitude=_HILL_OF_TOWIE_LATITUDE, longitude=_HILL_OF_TOWIE_LONGITUDE)
