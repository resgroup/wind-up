"""Circular math functions missing from numpy/scipy."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import circmean


def circ_diff(angle1: float | npt.NDArray | list, angle2: float | npt.NDArray | list) -> float | npt.NDArray:
    """Calculate the circular difference between two angles.

    :param angle1: First angle in degrees.
    :param angle2: Second angle in degrees.
    :return: Circular difference between the two angles in degrees
    """
    # Convert list to numpy array
    if isinstance(angle1, list):
        angle1 = np.array(angle1)
    if isinstance(angle2, list):
        angle2 = np.array(angle2)

    return np.mod(angle1 - angle2 + 180, 360) - 180


def circ_median(angles: npt.NDArray, axis: int | None = None, *, range_360: bool = True) -> float | npt.NDArray:
    """Calculate the circular median of angles.

    Uses an efficient approximation: centers data around the circular mean,
    computes ordinary median, then rotates back.

    :param angles: Array of angles in degrees. Can be a numpy array, list, or pandas Series.
                   Input can be in any range; it will be normalized internally.
    :param axis: Axis along which to compute the median. If None, compute over flattened array.
    :param range_360: If True, return result in [0, 360). If False, return result in [-180, 180).
    :return: Circular median in degrees
    """
    # Convert to numpy array (handles lists, Series, etc.)
    angles = np.asarray(angles)

    # Handle axis parameter
    if axis is not None:
        return np.apply_along_axis(lambda x: circ_median(x, axis=None, range_360=range_360), axis, angles)

    # Flatten if needed
    angles = angles.flatten()

    # Remove NaN values
    angles = angles[~np.isnan(angles)]

    if len(angles) == 0:
        return np.nan

    # Normalize angles to [0, 360) for computation
    angles_normalized = np.mod(angles, 360)

    # Calculate circular mean (in radians for scipy, convert back to degrees)
    mean_angle = circmean(angles_normalized, high=360, low=0)

    # Center the data around 180 (subtract mean, add 180)
    centered_angles = np.mod(angles_normalized - mean_angle + 180, 360)

    # Compute ordinary median on centered data
    median_centered = np.median(centered_angles)

    # Rotate back (subtract 180, add mean back)
    median_angle = np.mod(median_centered - 180 + mean_angle, 360)

    # Convert to requested range
    if range_360:
        return median_angle
    # Convert to [-180, 180)
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
