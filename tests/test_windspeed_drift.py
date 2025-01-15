from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wind_up.constants import REANALYSIS_WS_COL
from wind_up.models import WindUpConfig
from wind_up.windspeed_drift import _calculate_rolling_windspeed_diff, check_windspeed_drift


def test_check_windspeed_drift(test_lsa_t13_config: WindUpConfig) -> None:
    cfg = test_lsa_t13_config
    test_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_test_df.parquet")
    test_name = "LSA_T13"
    test_ws_col = "test_ws_est_blend"
    test_max_ws_drift, test_max_ws_drift_pp_period = check_windspeed_drift(
        wtg_df=test_df,
        wtg_name=test_name,
        ws_col=test_ws_col,
        reanalysis_ws_col="test_" + REANALYSIS_WS_COL,
        cfg=cfg,
        plot_cfg=None,
    )
    assert test_max_ws_drift == pytest.approx(0.45289044075068974)
    assert test_max_ws_drift_pp_period == pytest.approx(0.42913942378401204)


def test_calc_rolling_windspeed_diff() -> None:
    n_values = 50
    timestep = pd.Timedelta("6h")
    ts_index = pd.date_range("2020-01-01", periods=n_values, freq=timestep)
    ws_col_vals = np.linspace(5, 15, n_values)
    test_df = pd.DataFrame({"ws_col": ws_col_vals, "reanalysis_ws_col": ws_col_vals[::-1]}, index=ts_index)

    original = test_df.copy()
    actual = _calculate_rolling_windspeed_diff(
        wtg_df=test_df,
        ws_col="ws_col",
        reanalysis_ws_col="reanalysis_ws_col",
        timebase_s=int(timestep / pd.Timedelta("1s")),
    )

    expected = pd.Series(np.nan, index=ts_index)
    expected[-17:] = np.linspace(-2.2448979591836746, 1.0204081632653068, 17)
    pd.testing.assert_series_equal(actual, expected)

    # checking original dataframe is not modified
    pd.testing.assert_frame_equal(test_df, original)
