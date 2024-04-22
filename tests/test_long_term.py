from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from wind_up.long_term import calc_turbine_lt_dfs_raw_filt
from wind_up.models import WindUpConfig


def test_calc_turbine_lt_dfs_raw_filt(test_lsa_t13_config: WindUpConfig) -> None:
    cfg = test_lsa_t13_config
    test_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_test_df.parquet")
    test_df.columns = test_df.columns.str.replace("test_", "")
    test_name = "LSA_T13"
    lt_wtg_df_raw, lt_wtg_df_filt = calc_turbine_lt_dfs_raw_filt(
        wtg_name=test_name,
        cfg=cfg,
        wtg_df=test_df,
        ws_col="WindSpeedMean",
        pw_col="ActivePowerMean",
        plot_cfg=None,
    )

    expected_raw_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_lt_wtg_df_raw.parquet")
    expected_filt_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_lt_wtg_df_filt.parquet")
    assert_frame_equal(lt_wtg_df_raw, expected_raw_df)
    assert_frame_equal(lt_wtg_df_filt, expected_filt_df)
