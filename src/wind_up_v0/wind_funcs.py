"""Wind turbine related functions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def calc_cp(
    power_kw: float | pd.Series,
    ws_ms: float | pd.Series,
    air_density_kgpm3: float | pd.Series,
    rotor_diameter_m: float,
) -> float | pd.Series:
    """Calculate the power coefficient.

    :param power_kw: power in kilowatts
    :param ws_ms: wind speed in meters per second
    :param air_density_kgpm3: air density in kilograms per cubic meter
    :param rotor_diameter_m: turbine rotor diameter in meters
    :return: power coefficient
    """
    rotor_area_m2 = math.pi * (rotor_diameter_m / 2) ** 2
    return 1000.0 * power_kw / (0.5 * air_density_kgpm3 * rotor_area_m2 * ws_ms**3)
