"""Circular math functions missing from numpy/scipy."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import circmean


def circ_diff(angle1: npt.ArrayLike, angle2: npt.ArrayLike) -> npt.NDArray[np.float64] | np.float64:
    """Signed circular difference ``angle1 - angle2`` in degrees, wrapped to [-180, 180).

    :param angle1: first angle in degrees
    :param angle2: second angle in degrees
    :return: the wrapped difference in degrees
    """
    return np.mod(np.subtract(angle1, angle2) + 180, 360) - 180


def circ_median(
    angles: npt.ArrayLike, axis: int | None = None, *, range_360: bool = True
) -> npt.NDArray[np.float64] | np.float64:
    """Circular median of angles in degrees, approximated by centring on the circular mean.

    NaNs are dropped; an input with no finite values gives NaN. Input may be in any range.

    :param angles: angles in degrees
    :param axis: axis to reduce over; ``None`` reduces the flattened input
    :param range_360: return in [0, 360) rather than [-180, 180)
    :return: the circular median in degrees
    """
    values = np.asarray(angles)
    if axis is not None:
        return np.apply_along_axis(lambda x: circ_median(x, axis=None, range_360=range_360), axis, values)

    values = values.flatten()
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.float64(np.nan)

    normalized = np.mod(values, 360)
    mean_angle = circmean(normalized, high=360, low=0)
    centered = np.mod(normalized - mean_angle + 180, 360)
    median_angle = np.mod(np.median(centered) - 180 + mean_angle, 360)

    if range_360:
        return median_angle
    return np.mod(median_angle + 180, 360) - 180


def rolling_circ_mean(
    series: pd.Series, *, window: int, min_periods: int, center: bool = False, range_360: bool = True
) -> pd.Series:
    """Efficient rolling circular mean for angles in degrees.

    :param series: Series of angles in degrees.
    :param window: Size of the rolling window.
    :param min_periods: Minimum number of observations required to have a value.
    :param center: If True, set the labels at the center of the window.
    :param range_360: If True, return result in [0, 360). If False, return result in [-180, 180).
    :return: Series with rolling circular mean.
    """
    rad_values = np.deg2rad(series)
    sin_series = pd.Series(np.sin(rad_values), index=series.index)
    cos_series = pd.Series(np.cos(rad_values), index=series.index)

    sin_rolling = sin_series.rolling(window=window, min_periods=min_periods, center=center).mean()
    cos_rolling = cos_series.rolling(window=window, min_periods=min_periods, center=center).mean()

    result = (np.rad2deg(np.arctan2(sin_rolling, cos_rolling)) + 360) % 360

    if not range_360:
        # Convert to [-180, 180)
        result = np.mod(result + 180, 360) - 180

    return result


def rolling_circ_median_approx(
    series: pd.Series, *, window: int, min_periods: int, center: bool = False, range_360: bool = True
) -> pd.Series:
    """Efficient rolling circular (approximate) median for angles in degrees.

    :param series: Series of angles in degrees.
    :param window: Size of the rolling window.
    :param min_periods: Minimum number of observations required to have a value.
    :param center: If True, set the labels at the center of the window.
    :param range_360: If True, return result in [0, 360). If False, return result in [-180, 180).
    :return: Series with rolling circular median.
    """
    rad_values = np.deg2rad(series)
    sin_series = pd.Series(np.sin(rad_values), index=series.index)
    cos_series = pd.Series(np.cos(rad_values), index=series.index)

    sin_rolling = sin_series.rolling(window=window, min_periods=min_periods, center=center).median()
    cos_rolling = cos_series.rolling(window=window, min_periods=min_periods, center=center).median()

    result = (np.rad2deg(np.arctan2(sin_rolling, cos_rolling)) + 360) % 360

    if not range_360:
        # Convert to [-180, 180)
        result = np.mod(result + 180, 360) - 180

    return result
