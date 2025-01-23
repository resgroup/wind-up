from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wind_up.models import WindUpConfig
from wind_up.pp_analysis import _pre_post_pp_analysis_with_reversal


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
    pp_results, actual_df = _pre_post_pp_analysis_with_reversal(
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

    assert_frame_equal(actual_df, expected_df)
    assert pp_results["pp_valid_hours"] == pytest.approx(10745.83333333333)
    assert pp_results["pp_valid_hours_pre"] == pytest.approx(5804.666666666666)
    assert pp_results["pp_valid_hours_post"] == pytest.approx(4941.166666666667)
    assert pp_results["pp_invalid_bin_count"] == 4
    assert pp_results["pp_data_coverage"] == pytest.approx(0.679170353516201)
    assert pp_results["reversal_error"] == pytest.approx(-0.008804504211352544)
    assert pp_results["uplift_noadj_frc"] == pytest.approx(0.04523639558619659)
    assert pp_results["poweronly_uplift_frc"] == pytest.approx(0.045608171004022265)
    assert pp_results["reversed_uplift_frc"] == pytest.approx(0.03680366679266972)
    assert pp_results["uplift_frc"] == pytest.approx(0.04083414348052032)
    assert pp_results["missing_bins_unc_scale_factor"] == pytest.approx(1.0000004657300203)
    assert pp_results["t_value_one_sigma"] == pytest.approx(1.0000168659661646)
    assert pp_results["unc_one_sigma_lowerbound_frc"] == pytest.approx(0.004402252105676272)
    assert pp_results["unc_one_sigma_frc"] == pytest.approx(0.004402252105676272)
