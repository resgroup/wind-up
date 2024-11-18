from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from wind_up.wind_funcs import calc_cp

test_calc_cp_data = [
    (500, 8.2, 1.21, 62, 0.4964778),
    (pd.Series([500.1, -30.1]), pd.Series([8.4, 8.5]), 1.21, 62, pd.Series([0.4619451, -0.0268337])),
]


@pytest.mark.parametrize(("power_kw", "ws_ms", "air_density_kgpm3", "rotor_diamter_m", "expected"), test_calc_cp_data)
def test_calc_cp(
    power_kw: float | pd.Series,
    ws_ms: float | pd.Series,
    air_density_kgpm3: float | pd.Series,
    rotor_diamter_m: float,
    expected: float | pd.Series,
) -> None:
    if isinstance(expected, pd.Series):
        assert_series_equal(calc_cp(power_kw, ws_ms, air_density_kgpm3, rotor_diamter_m), (expected))
    else:
        assert calc_cp(power_kw, ws_ms, air_density_kgpm3, rotor_diamter_m) == pytest.approx(expected)

    cp = calc_cp(500, 8.2, 1.21, 62)
    assert cp == pytest.approx(0.4964778)
