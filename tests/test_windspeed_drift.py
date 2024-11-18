from pathlib import Path

import pandas as pd
import pytest

from wind_up.constants import REANALYSIS_WS_COL
from wind_up.models import WindUpConfig
from wind_up.windspeed_drift import check_windspeed_drift


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
