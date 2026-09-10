"""Rough solar position for diurnal modulation of synthetic wake steering.

A cheap solar-elevation estimate from timestamp and site coordinates, with no external
dependency: day-of-year declination, longitude-based hour angle, no equation of time.
Timestamps are treated as UTC. Good enough to drive a day/night weighting, not for real
solar-engineering accuracy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import numpy.typing as npt


def sin_solar_elevation(index: pd.DatetimeIndex, *, lat: float, lon: float) -> npt.NDArray[np.float64]:
    """Sine of the solar elevation at each (UTC) timestamp for a site at ``(lat, lon)`` degrees.

    Timezone-aware inputs are converted to UTC first; naive inputs are assumed to be UTC.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    day = idx.dayofyear.to_numpy(dtype=float)
    utc_hour = idx.hour.to_numpy(dtype=float) + idx.minute.to_numpy(dtype=float) / 60.0
    declination = np.radians(23.45 * np.sin(np.radians(360.0 * (284.0 + day) / 365.0)))
    hour_angle = np.radians(15.0 * (utc_hour + lon / 15.0 - 12.0))
    lat_r = np.radians(lat)
    return np.sin(lat_r) * np.sin(declination) + np.cos(lat_r) * np.cos(declination) * np.cos(hour_angle)


def diurnal_factor(
    index: pd.DatetimeIndex,
    *,
    lat: float,
    lon: float,
    night_factor: float,
    day_factor: float,
    sun_ref: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Day/night multiplier: ``night_factor`` deep at night, ``day_factor`` at high sun.

    ``night_weight = clip(1 - max(sin_elev, 0) / sun_ref, 0, 1)`` is 1 when the sun is below the
    horizon and 0 once its elevation sine reaches ``sun_ref``; the factor interpolates linearly
    between ``day_factor`` and ``night_factor`` across that range.
    """
    sin_elev = sin_solar_elevation(index, lat=lat, lon=lon)
    night_weight = np.clip(1.0 - np.maximum(sin_elev, 0.0) / sun_ref, 0.0, 1.0)
    return day_factor + (night_factor - day_factor) * night_weight
