"""Explicitly-constructed time features for ML uplift methods.

The counterfactual power-model methods normally drop the timestamp entirely before
modelling: a bare timestamp is not a weather variable and folding it in naively risks
leaking the treatment (design-note SS3 -- anything that could reflect the upgrade must not
enter the model except through the treatment-invariant reference features). But time itself
carries real, physically meaningful signal that is *not* weather: reference-turbine
instrumentation drifts slowly over a campaign, the wind resource and air density have a
seasonal cycle, and the diurnal cycle (via solar heating -> boundary-layer shear/turbulence,
and directly via light for some effects) modulates conditions across the day. This module
gives a model an explicit, named handle on each of those axes instead of leaving it to guess
from an opaque clock value:

* :func:`days_since_campaign_start` -- a continuous linear clock (in days, negative before
  the campaign starts) for slow drift such as reference-anemometer calibration decay.
* :func:`season_sin_cos` -- a smooth, cyclical encoding of time-of-year (sin/cos pair, so the
  model sees December and January as adjacent rather than as opposite ends of a 0-364 ramp).
* :func:`solar_altitude_azimuth` -- the sun's position (altitude plus a cyclical azimuth
  encoding), a proxy for diurnal heating and boundary-layer state that is far more
  informative than clock hour alone because it also depends on latitude, longitude and
  season.

All functions are pure and vectorized over a :class:`pandas.DatetimeIndex` that must be
timezone-aware (the benchmarking harness works exclusively in UTC).

The solar-position calculation implements the NOAA solar-position algorithm (the public NOAA
Solar Calculator spreadsheet equations, itself based on Jean Meeus's *Astronomical
Algorithms*). It deliberately omits the atmospheric refraction correction that the NOAA
spreadsheet applies near the horizon -- refraction only matters for altitudes within a couple
of degrees of the horizon, and skipping it keeps the implementation a direct, checkable
transcription of the core geometry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Vocabulary of configurable time-feature groups; "season" and "solar" each expand to
# multiple columns (see season_sin_cos / solar_altitude_azimuth).
TIME_FEATURE_NAMES: tuple[str, ...] = ("days_since_campaign_start", "season", "solar")

# Day-of-year of June 21st in a non-leap year (one later in leap years); the season anchor.
_JUNE21_DAY_OF_YEAR_NON_LEAP = 172
# Mean length of a year in days (Gregorian calendar average, i.e. including leap years).
_DAYS_PER_YEAR = 365.25
_SECONDS_PER_DAY = 86_400.0
_MINUTES_PER_DAY = 1_440.0

# Julian date of the J2000.0 epoch (2000-01-01 12:00 UTC), the NOAA algorithm's time origin.
_J2000_JULIAN_DATE = 2_451_545.0
# Julian days per Julian century, used to convert Julian date into Julian centuries since J2000.
_JULIAN_DAYS_PER_CENTURY = 36_525.0

# Degrees of hour angle per minute of time (360 degrees / 1440 minutes per day).
_DEGREES_PER_MINUTE = 0.25
_MINUTES_PER_DEGREE_LONGITUDE = 4.0
_DEGREES_PER_CIRCLE = 360.0


def _require_tz_aware(index: pd.DatetimeIndex) -> None:
    """Raise ``ValueError`` unless ``index`` is timezone-aware.

    :param index: the index to check.
    """
    if index.tz is None:
        msg = "index must be timezone-aware (expected tz-aware UTC); got a tz-naive DatetimeIndex."
        raise ValueError(msg)


def days_since_campaign_start(index: pd.DatetimeIndex, *, campaign_start: pd.Timestamp) -> pd.Series:
    """Continuous days elapsed since ``campaign_start``, negative before it.

    A linear clock feature for slow, monotonic drift (e.g. reference-anemometer calibration
    decay) that a purely cyclical feature such as :func:`season_sin_cos` cannot represent.

    :param index: timezone-aware timestamps to featurize.
    :param campaign_start: the reference instant; must be comparable to ``index`` (i.e. also
        timezone-aware).
    :return: a :class:`pandas.Series` named ``"days_since_campaign_start"``, indexed by
        ``index``, holding ``(index - campaign_start) / 1 day`` as a float.
    """
    _require_tz_aware(index)
    if campaign_start.tzinfo is None:
        msg = "campaign_start must be timezone-aware (expected tz-aware UTC); got a tz-naive Timestamp."
        raise ValueError(msg)
    days = (index - campaign_start) / pd.Timedelta(days=1)
    return pd.Series(days, index=index, name="days_since_campaign_start")


def season_sin_cos(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Time-of-year as a sin/cos pair anchored on the June 21st solstice.

    The angle is ``2*pi * (fractional day-of-year offset from June 21) / 365.25``, so the
    encoding is smooth and cyclical (no discontinuity at year end) and ``season_cos`` peaks
    at +1 around the June solstice and -1 around the December solstice, giving a tree-based
    model a direct handle on the seasonal cycle without a sharp day-365-to-day-1 jump.

    :param index: timezone-aware timestamps to featurize.
    :return: a :class:`pandas.DataFrame` with columns ``"season_sin"`` and ``"season_cos"``,
        indexed by ``index``.
    """
    _require_tz_aware(index)
    index_utc = index.tz_convert("UTC")  # wall-clock fields below must read UTC, whatever tz came in
    fractional_day_of_year = (
        index_utc.dayofyear.to_numpy(dtype=float)
        + (
            index_utc.hour.to_numpy(dtype=float) * 3600.0
            + index_utc.minute.to_numpy(dtype=float) * 60.0
            + index_utc.second.to_numpy(dtype=float)
            + index_utc.microsecond.to_numpy(dtype=float) / 1.0e6
        )
        / _SECONDS_PER_DAY
    )
    # June 21 is day 172 in a non-leap year but 173 in a leap year (the extra Feb 29 shifts it).
    june21_day_of_year = _JUNE21_DAY_OF_YEAR_NON_LEAP + index_utc.is_leap_year.astype(float)
    offset_from_june21 = fractional_day_of_year - june21_day_of_year
    angle = 2.0 * np.pi * offset_from_june21 / _DAYS_PER_YEAR
    return pd.DataFrame({"season_sin": np.sin(angle), "season_cos": np.cos(angle)}, index=index)


def solar_altitude_azimuth(index: pd.DatetimeIndex, *, latitude: float, longitude: float) -> pd.DataFrame:
    """Solar altitude and azimuth via the NOAA solar-position algorithm.

    A vectorized (numpy-only, no per-row loops) transcription of the NOAA Solar Calculator
    spreadsheet equations: julian day/century from the UTC timestamps, the sun's geometric
    mean longitude and anomaly, the equation-of-center correction to true longitude, apparent
    longitude, mean and corrected obliquity of the ecliptic, solar declination, the equation
    of time, true solar time (from longitude and equation of time), hour angle, and finally
    solar zenith (-> altitude) and azimuth. Atmospheric refraction near the horizon is
    **not** applied, so altitude is the true geometric altitude rather than the
    apparent/refracted one; this keeps the implementation a direct, checkable transcription
    of the core geometry (refraction only matters within a couple of degrees of the horizon).

    :param index: timezone-aware timestamps to featurize (converted to UTC internally).
    :param latitude: observer latitude in degrees, positive north.
    :param longitude: observer longitude in degrees, positive east.
    :return: a :class:`pandas.DataFrame` with columns ``"solar_altitude"`` (degrees, negative
        below the horizon), ``"solar_azimuth_sin"`` and ``"solar_azimuth_cos"`` (sine and
        cosine of the azimuth in radians, clockwise from north -- encoded cyclically because
        azimuth wraps at 360 degrees and the downstream model is tree-based), indexed by
        ``index``.
    """
    _require_tz_aware(index)
    index_utc = index.tz_convert("UTC")

    julian_date = index_utc.to_julian_date().to_numpy(dtype=float)
    julian_century = (julian_date - _J2000_JULIAN_DATE) / _JULIAN_DAYS_PER_CENTURY

    geom_mean_long_sun = np.mod(280.46646 + julian_century * (36000.76983 + julian_century * 0.0003032), 360.0)
    geom_mean_anom_sun = 357.52911 + julian_century * (35999.05029 - 0.0001537 * julian_century)
    eccent_earth_orbit = 0.016708634 - julian_century * (0.000042037 + 0.0000001267 * julian_century)

    mean_anom_rad = np.radians(geom_mean_anom_sun)
    sun_eq_of_ctr = (
        np.sin(mean_anom_rad) * (1.914602 - julian_century * (0.004817 + 0.000014 * julian_century))
        + np.sin(2.0 * mean_anom_rad) * (0.019993 - 0.000101 * julian_century)
        + np.sin(3.0 * mean_anom_rad) * 0.000289
    )

    sun_true_long = geom_mean_long_sun + sun_eq_of_ctr
    sun_app_long = sun_true_long - 0.00569 - 0.00478 * np.sin(np.radians(125.04 - 1934.136 * julian_century))

    mean_obliq_ecliptic = (
        23.0
        + (26.0 + (21.448 - julian_century * (46.815 + julian_century * (0.00059 - julian_century * 0.001813))) / 60.0)
        / 60.0
    )
    obliq_corr = mean_obliq_ecliptic + 0.00256 * np.cos(np.radians(125.04 - 1934.136 * julian_century))

    sun_declin = np.degrees(np.arcsin(np.sin(np.radians(obliq_corr)) * np.sin(np.radians(sun_app_long))))

    var_y = np.tan(np.radians(obliq_corr / 2.0)) ** 2
    geom_mean_long_sun_rad = np.radians(geom_mean_long_sun)
    equation_of_time = 4.0 * np.degrees(
        var_y * np.sin(2.0 * geom_mean_long_sun_rad)
        - 2.0 * eccent_earth_orbit * np.sin(mean_anom_rad)
        + 4.0 * eccent_earth_orbit * var_y * np.sin(mean_anom_rad) * np.cos(2.0 * geom_mean_long_sun_rad)
        - 0.5 * var_y * var_y * np.sin(4.0 * geom_mean_long_sun_rad)
        - 1.25 * eccent_earth_orbit * eccent_earth_orbit * np.sin(2.0 * mean_anom_rad)
    )

    minutes_since_midnight = (
        index_utc.hour.to_numpy(dtype=float) * 60.0
        + index_utc.minute.to_numpy(dtype=float)
        + index_utc.second.to_numpy(dtype=float) / 60.0
        + index_utc.microsecond.to_numpy(dtype=float) / 60.0e6
    )
    true_solar_time = np.mod(
        minutes_since_midnight + equation_of_time + _MINUTES_PER_DEGREE_LONGITUDE * longitude, _MINUTES_PER_DAY
    )
    # true_solar_time is wrapped into [0, 1440) above, so the NOAA spreadsheet's negative branch
    # cannot occur and the hour angle reduces to the single expression over [-180, 180).
    hour_angle = true_solar_time * _DEGREES_PER_MINUTE - 180.0

    lat_rad = np.radians(latitude)
    declin_rad = np.radians(sun_declin)
    hour_angle_rad = np.radians(hour_angle)

    cos_zenith = np.clip(
        np.sin(lat_rad) * np.sin(declin_rad) + np.cos(lat_rad) * np.cos(declin_rad) * np.cos(hour_angle_rad),
        -1.0,
        1.0,
    )
    zenith = np.degrees(np.arccos(cos_zenith))
    altitude = 90.0 - zenith

    zenith_rad = np.radians(zenith)
    with np.errstate(divide="ignore", invalid="ignore"):
        azimuth_arg = np.clip(
            (np.sin(lat_rad) * np.cos(zenith_rad) - np.sin(declin_rad)) / (np.cos(lat_rad) * np.sin(zenith_rad)),
            -1.0,
            1.0,
        )
    azimuth_base = np.degrees(np.arccos(azimuth_arg))
    azimuth = np.where(
        hour_angle > 0.0,
        np.mod(azimuth_base + 180.0, _DEGREES_PER_CIRCLE),
        np.mod(540.0 - azimuth_base, _DEGREES_PER_CIRCLE),
    )
    # Directly overhead (or a pole latitude), azimuth is undefined; the division above yields
    # nan there, which np.mod propagates -- fall back to due-north (0 degrees) by convention.
    azimuth = np.where(np.isnan(azimuth), 0.0, azimuth)

    azimuth_rad = np.radians(azimuth)
    return pd.DataFrame(
        {
            "solar_altitude": altitude,
            "solar_azimuth_sin": np.sin(azimuth_rad),
            "solar_azimuth_cos": np.cos(azimuth_rad),
        },
        index=index,
    )
