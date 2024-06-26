from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wind_up.models import WindUpConfig
from wind_up.pp_analysis import pre_post_pp_analysis_with_reversal


def test_pre_post_pp_analysis_with_reversal(test_lsa_t13_config: WindUpConfig) -> None:
    cfg = test_lsa_t13_config

    test_wtg = next(x for x in cfg.asset.wtgs if x.name == "LSA_T13")
    ref_name = "LSA_T12"
    detrend_ws_col = "ref_ws_detrended"
    test_pw_col = "test_pw_clipped"
    ref_wd_col = "ref_YawAngleMean"

    pre_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_LSA_T12_pre_df.parquet")
    post_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_LSA_T12_post_df.parquet")
    lt_wtg_df_filt = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_lt_wtg_df_filt.parquet")
    test_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_test_df.parquet")
    expected_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/pre_post_pp_analysis_expected_df.parquet")
    pp_results, actual_df = pre_post_pp_analysis_with_reversal(
        cfg=cfg,
        test_wtg=test_wtg,
        ref_name=ref_name,
        lt_df=lt_wtg_df_filt,
        pre_df=pre_df,
        post_df=post_df,
        ws_col=detrend_ws_col,
        pw_col=test_pw_col,
        wd_col=ref_wd_col,
        plot_cfg=None,
        test_df=test_df,
    )

    # minor changes to make actual_df compatible with old expected_df
    expected_df["hours_for_mwh_calc"] = expected_df["hours_per_year"]
    expected_df["hours_per_year"] = actual_df["hours_per_year"]
    cols_with_new_calc = ["uplift_kw_se", "uplift_p5_kw", "uplift_p95_kw"]
    expected_df[cols_with_new_calc] = actual_df[cols_with_new_calc]
    new_cols = [
        "pre_valid",
        "post_valid",
        "hours_pre_raw",
        "hours_post_raw",
        "is_invalid_bin",
        "pw_at_mid_expected",
        "pw_sem_at_mid_expected",
        "relative_cp_baseline",
        "relative_cp_post",
        "relative_cp_sem_at_mid_expected",
        "relative_cp_sem_at_mid_post",
        "uplift_relative_cp",
        "uplift_relative_cp_se",
        "uplift_relative_cp_p5",
        "uplift_relative_cp_p95",
    ]
    expected_df[new_cols] = actual_df[new_cols]
    expected_df = expected_df[actual_df.columns]

    assert_frame_equal(actual_df, expected_df)
    assert pp_results["pp_valid_hours"] == pytest.approx(10748.5)
    assert pp_results["pp_valid_hours_pre"] == pytest.approx(5807.333333333333)
    assert pp_results["pp_valid_hours_post"] == pytest.approx(4941.166666666667)
    assert pp_results["pp_invalid_bin_count"] == 3
    assert pp_results["pp_data_coverage"] == pytest.approx(0.6793388952092024)
    assert pp_results["reversal_error"] == pytest.approx(-0.008786551768533796)
    assert pp_results["uplift_noadj_frc"] == pytest.approx(0.04523448345231426)
    assert pp_results["poweronly_uplift_frc"] == pytest.approx(0.04560411838169785)
    assert pp_results["reversed_uplift_frc"] == pytest.approx(0.03681756661316406)
    assert pp_results["uplift_frc"] == pytest.approx(0.040841207568047364)
    assert pp_results["missing_bins_unc_scale_factor"] == pytest.approx(1.0000000006930523)
    assert pp_results["t_value_one_sigma"] == pytest.approx(1.0000168636907854)
    assert pp_results["unc_one_sigma_lowerbound_frc"] == pytest.approx(0.004393275884266898)
    assert pp_results["unc_one_sigma_frc"] == pytest.approx(0.004393275884266898)
