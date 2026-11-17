import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.conftest import TEST_DATA_FLD
from wind_up.models import WindUpConfig
from wind_up.pp_analysis import (
    _pre_post_pp_analysis_with_reversal,
    pre_post_pp_analysis_with_reversal_and_bootstrapping,
)


def test_pre_post_pp_analysis_with_reversal(test_lsa_t13_config: WindUpConfig) -> None:
    cfg = test_lsa_t13_config

    test_wtg = next(x for x in cfg.asset.wtgs if x.name == "LSA_T13")
    ref_name = "LSA_T12"
    detrend_ws_col = "ref_ws_detrended"
    test_pw_col = "test_pw_clipped"
    ref_wd_col = "ref_YawAngleMean"

    pre_df = pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_pre_df.parquet")
    post_df = pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_post_df.parquet")
    lt_wtg_df_filt = pd.read_parquet(TEST_DATA_FLD / "LSA_T13_lt_wtg_df_filt.parquet")
    test_df = pd.read_parquet(TEST_DATA_FLD / "LSA_T13_test_df.parquet")
    expected_df = pd.read_parquet(TEST_DATA_FLD / "pre_post_pp_analysis_expected_df.parquet")
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


@pytest.mark.parametrize(
    ("reference", "reverse_gap_kw"),
    [("different_type", 1200.0), ("same_type", 600.0), ("non_turbine", 600.0)],
)
def test_site_mean_gap_fill_uses_type_of_turbine_producing_power(
    test_lsa_t13_config: WindUpConfig, reference: str, reverse_gap_kw: float
) -> None:
    cfg = test_lsa_t13_config
    cfg.ws_bin_width = 1.0
    cfg.bootstrap_runs_override = 3
    test_wtg = cfg.test_wtgs[0]
    ref_wtg = cfg.ref_wtgs[0]
    ref_name = ref_wtg.name if reference != "non_turbine" else "mast"
    if reference == "different_type":
        ref_wtg.turbine_type = ref_wtg.turbine_type.model_copy(
            update={"turbine_type": "reference_type", "rated_power_kw": 2500.0},
        )

    # Only the 5.5 m/s bin has observations. The empty 8.5 m/s bin has the same long-term weight.
    pre_df = pd.DataFrame(
        {
            "ref_ws_detrended": 5.5,
            "ref_ws_est_from_power_only": 5.5,
            "test_ws_est_from_power_only": 5.5,
            "ws_rom": 1.0,
            "test_pw_clipped": 500.0,
            "ref_pw_clipped": 600.0,
            "ref_wd": 270.0,
        },
        index=pd.date_range("2020-01-01", periods=48, freq="10min", tz="UTC"),
    )
    post_df = pre_df.assign(test_pw_clipped=525.0, ref_pw_clipped=570.0).shift(freq="1D")
    lt_df = pd.DataFrame({"bin_mid": [5.5, 8.5], "hours_per_year": [4380, 4380]})
    test_curve = pd.DataFrame({"bin_mid": [0.5, 5.5, 8.5], "pw_clipped": [0.0, 500.0, 600.0]})
    ref_curve = pd.DataFrame({"bin_mid": [0.5, 5.5, 8.5], "pw_clipped": [0.0, 570.0, 1200.0]})

    results, pp_df = pre_post_pp_analysis_with_reversal_and_bootstrapping(
        cfg=cfg,
        test_wtg=test_wtg,
        ref_name=ref_name,
        lt_df=lt_df,
        pre_df=pre_df,
        post_df=post_df,
        ws_col="ref_ws_detrended",
        pw_col="test_pw_clipped",
        wd_col="ref_wd",
        plot_cfg=None,
        random_seed=87,
        site_mean_pc_per_ttype={test_wtg.turbine_type.turbine_type: test_curve, "reference_type": ref_curve},
    )

    assert pp_df.loc[8.5, "pw_at_mid_post"] == pytest.approx(600.0)
    assert results["uplift_noadj_frc"] == pytest.approx(25 / (500 + 600))
    assert results["poweronly_uplift_frc"] == pytest.approx(25 / (500 + 600))
    assert results["reversed_uplift_frc"] == pytest.approx(30 / (570 + reverse_gap_kw))
