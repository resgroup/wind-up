"""Circular math functions missing from numpy/scipy.

Re-exported from :mod:`wind_up.circular_math`, where the implementation now lives so the v1
northing core can use it without importing from the legacy package.
"""

from wind_up.circular_math import (
    circ_diff,
    circ_median,
    rolling_circ_mean,
    rolling_circ_median_approx,
)

__all__ = ["circ_diff", "circ_median", "rolling_circ_mean", "rolling_circ_median_approx"]
