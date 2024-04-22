import math

import pandas as pd


def calc_cp(
    power_kw: float | pd.Series,
    ws_ms: float | pd.Series,
    air_density_kgpm3: float | pd.Series,
    rotor_diameter_m: float,
) -> float | pd.Series:
    rotor_area_m2 = math.pi * (rotor_diameter_m / 2) ** 2
    return 1000.0 * power_kw / (0.5 * air_density_kgpm3 * rotor_area_m2 * ws_ms**3)
