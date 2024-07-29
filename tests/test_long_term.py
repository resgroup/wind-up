from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from wind_up.long_term import calc_lt_dfs_raw_filt
from wind_up.models import WindUpConfig


def test_calc_turbine_lt_dfs_raw_filt(test_lsa_t13_config: WindUpConfig) -> None:
    cfg = test_lsa_t13_config
    test_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_test_df.parquet")
    test_df.columns = test_df.columns.str.replace("test_", "")
    test_name = "LSA_T13"
    lt_wtg_df_raw, lt_wtg_df_filt = calc_lt_dfs_raw_filt(
        wtg_or_wf_name=test_name,
        cfg=cfg,
        wtg_or_wf_df=test_df,
        ws_col="WindSpeedMean",
        pw_col="ActivePowerMean",
        one_turbine=True,
        plot_cfg=None,
    )

    expected_raw_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_lt_wtg_df_raw.parquet")
    expected_filt_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_lt_wtg_df_filt.parquet")
    assert_frame_equal(lt_wtg_df_raw, expected_raw_df)
    assert_frame_equal(lt_wtg_df_filt, expected_filt_df)


def test_calc_windfarm_lt_dfs_raw_filt(test_lsa_t13_config: WindUpConfig) -> None:
    cfg = test_lsa_t13_config
    test_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_test_df.parquet")
    test_df.columns = test_df.columns.str.replace("test_", "")
    # make a fake wf_df
    test_df["TurbineName"] = "LSA_T13"
    wf_df = test_df.copy().set_index(["TurbineName"], append=True).swaplevel()
    for fake_wtg_name in ["LSA_T14", "LSA_T15"]:
        new_df = test_df.copy()
        new_df["TurbineName"] = fake_wtg_name
        new_df = new_df.set_index(["TurbineName"], append=True).swaplevel()
        wf_df = pd.concat([wf_df, new_df])
    cfg.asset.wtgs = [x for x in cfg.asset.wtgs if x.name in {"LSA_T13", "LSA_T14", "LSA_T15"}]

    lt_wtg_df_raw, lt_wtg_df_filt = calc_lt_dfs_raw_filt(
        wtg_or_wf_name=cfg.asset.name,
        cfg=cfg,
        wtg_or_wf_df=wf_df,
        ws_col="WindSpeedMean",
        pw_col="ActivePowerMean",
        one_turbine=False,
        plot_cfg=None,
    )

    expected_raw_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_lt_wtg_df_raw.parquet")
    expected_raw_df["observed_hours"] *= 3
    expected_raw_df["observed_mwh"] *= 3
    assert_frame_equal(lt_wtg_df_raw, expected_raw_df)

    expected_filt_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_lt_wtg_df_filt.parquet")
    expected_filt_df["observed_hours"] *= 3
    expected_filt_df["observed_mwh"] *= 3
    assert_frame_equal(lt_wtg_df_filt, expected_filt_df)
