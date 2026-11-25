import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.conftest import TEST_DATA_FLD
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.pp_analysis import (
    _block_bootstrap_sample,
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


def _two_bin_gap_fill_inputs(cfg: WindUpConfig, ref_name: str) -> dict:
    """Inputs where only the 5.5 m/s bin has data and the empty 8.5 m/s bin has the same long-term weight."""
    cfg.ws_bin_width = 1.0
    cfg.bootstrap_runs_override = 3
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
    return {
        "cfg": cfg,
        "test_wtg": cfg.test_wtgs[0],
        "ref_name": ref_name,
        "lt_df": pd.DataFrame({"bin_mid": [5.5, 8.5], "hours_per_year": [4380, 4380]}),
        "pre_df": pre_df,
        "post_df": post_df,
        "ws_col": "ref_ws_detrended",
        "pw_col": "test_pw_clipped",
        "wd_col": "ref_wd",
        "plot_cfg": None,
        "random_seed": 87,
    }


TEST_CURVE = pd.DataFrame({"bin_mid": [0.5, 5.5, 8.5], "pw_clipped": [0.0, 500.0, 600.0]})
REF_CURVE = pd.DataFrame({"bin_mid": [0.5, 5.5, 8.5], "pw_clipped": [0.0, 570.0, 1200.0]})


def test_custom_confidence_level_is_used_for_uplift_diagnostics(
    test_lsa_t13_config: WindUpConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    inputs = _two_bin_gap_fill_inputs(test_lsa_t13_config, test_lsa_t13_config.ref_wtgs[0].name)
    ambient_temp = np.tile([9, 11], 24)
    inputs["pre_df"] = inputs["pre_df"].assign(
        ref_ws_detrended=5.75, test_pw_clipped=np.tile([490, 510], 24), ref_AmbientTemp=ambient_temp
    )
    inputs["post_df"] = inputs["post_df"].assign(ref_ws_detrended=5.75, ref_AmbientTemp=ambient_temp)
    test_name, ref_name = inputs["test_wtg"].name, inputs["ref_name"]
    plot_dir = tmp_path / test_name / ref_name
    plot_dir.mkdir(parents=True)
    inputs["plot_cfg"] = PlotConfig(show_plots=False, save_plots=True, plots_dir=tmp_path)

    with caplog.at_level(logging.INFO, logger="wind_up.pp_analysis"):
        results, pp_df = pre_post_pp_analysis_with_reversal_and_bootstrapping(
            **inputs, confidence_level=0.8, site_mean_pc_df=TEST_CURVE
        )

    wind_bin = pp_df.loc[5.5]
    assert wind_bin["uplift_p90_kw"] < wind_bin["uplift_kw"] < wind_bin["uplift_p10_kw"]
    assert (plot_dir / f"test={test_name} ref={ref_name} uplift [kW] and 80% CI.png").is_file()
    assert f"final P90 uplift estimate = {100 * results['uplift_p90_frc']:.1f} %" in caplog.text
    assert f"final P10 uplift estimate = {100 * results['uplift_p10_frc']:.1f} %" in caplog.text


def test_confidence_level_with_fractional_percentiles_is_labelled_exactly(
    test_lsa_t13_config: WindUpConfig, caplog: pytest.LogCaptureFixture
) -> None:
    inputs = _two_bin_gap_fill_inputs(test_lsa_t13_config, test_lsa_t13_config.ref_wtgs[0].name)
    inputs["plot_cfg"] = PlotConfig(show_plots=False, save_plots=False, plots_dir=Path())

    with caplog.at_level(logging.INFO, logger="wind_up.pp_analysis"):
        results, pp_df = pre_post_pp_analysis_with_reversal_and_bootstrapping(
            **inputs, confidence_level=0.95, site_mean_pc_df=TEST_CURVE
        )

    assert {"uplift_p2.5_kw", "uplift_p97.5_kw"} <= set(pp_df.columns)
    assert results["uplift_p97.5_frc"] <= results["uplift_frc"] <= results["uplift_p2.5_frc"]
    assert "final P97.5 uplift estimate" in caplog.text
    assert "final P2.5 uplift estimate" in caplog.text


@pytest.mark.parametrize("confidence_level", [0, 1])
def test_invalid_confidence_level_is_rejected(test_lsa_t13_config: WindUpConfig, confidence_level: float) -> None:
    inputs = _two_bin_gap_fill_inputs(test_lsa_t13_config, test_lsa_t13_config.ref_wtgs[0].name)

    with pytest.raises(ValueError, match="confidence_level must be greater than 0 and less than 1"):
        pre_post_pp_analysis_with_reversal_and_bootstrapping(**inputs, confidence_level=confidence_level)


@pytest.mark.parametrize(
    ("reference", "reverse_gap_kw"),
    [("different_type", 1200.0), ("same_type", 600.0), ("non_turbine", 600.0)],
)
def test_site_mean_gap_fill_uses_type_of_turbine_producing_power(
    test_lsa_t13_config: WindUpConfig, reference: str, reverse_gap_kw: float
) -> None:
    cfg = test_lsa_t13_config
    ref_name = cfg.ref_wtgs[0].name if reference != "non_turbine" else "mast"
    if reference == "different_type":
        # reverse mode looks the reference up in cfg.asset.wtgs, so change its type there
        ref_wtg = next(x for x in cfg.asset.wtgs if x.name == ref_name)
        ref_wtg.turbine_type = ref_wtg.turbine_type.model_copy(
            update={"turbine_type": "reference_type", "rated_power_kw": 2500.0},
        )
    inputs = _two_bin_gap_fill_inputs(cfg, ref_name)
    test_ttype = inputs["test_wtg"].turbine_type.turbine_type

    results, pp_df = pre_post_pp_analysis_with_reversal_and_bootstrapping(
        **inputs,
        site_mean_pc_per_ttype={test_ttype: TEST_CURVE, "reference_type": REF_CURVE},
    )

    assert pp_df.loc[8.5, "pw_at_mid_post"] == pytest.approx(600.0)
    assert results["uplift_noadj_frc"] == pytest.approx(25 / (500 + 600))
    assert results["poweronly_uplift_frc"] == pytest.approx(25 / (500 + 600))
    assert results["reversed_uplift_frc"] == pytest.approx(30 / (570 + reverse_gap_kw))


def test_site_mean_gap_fill_rejects_both_curve_arguments(test_lsa_t13_config: WindUpConfig) -> None:
    inputs = _two_bin_gap_fill_inputs(test_lsa_t13_config, test_lsa_t13_config.ref_wtgs[0].name)
    test_ttype = inputs["test_wtg"].turbine_type.turbine_type
    with pytest.raises(ValueError, match="only one of site_mean_pc_df and site_mean_pc_per_ttype"):
        pre_post_pp_analysis_with_reversal_and_bootstrapping(
            **inputs, site_mean_pc_df=TEST_CURVE, site_mean_pc_per_ttype={test_ttype: TEST_CURVE}
        )


def test_site_mean_gap_fill_names_missing_turbine_type(test_lsa_t13_config: WindUpConfig) -> None:
    inputs = _two_bin_gap_fill_inputs(test_lsa_t13_config, test_lsa_t13_config.ref_wtgs[0].name)
    test_ttype = inputs["test_wtg"].turbine_type.turbine_type
    with pytest.raises(ValueError, match=f"no power curve for turbine type '{test_ttype}'"):
        pre_post_pp_analysis_with_reversal_and_bootstrapping(
            **inputs, site_mean_pc_per_ttype={"other_type": TEST_CURVE}
        )


EXPECTED_POST_WS = [103, 104, 105, 106, 107, 108, 110, 110, 110, 111, 111, 111, 112, 114, 115, 116, 117, 119, 122, 124]


@pytest.mark.parametrize(
    ("pre_order", "expected_pre_ws"),
    [
        (list(range(12)), [1, 2, 3, 4, 4, 4, 6, 7, 8, 11]),
        ([5, 0, 11, 3, 8, 1, 10, 2, 7, 4, 9, 6], [2, 3, 5, 6, 6, 6, 7, 8, 9, 11]),
        ([0, 1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11], [1, 2, 2, 3, 3, 3, 5, 7, 7, 9]),
    ],
    ids=["sorted", "shuffled", "repeated timestamps"],
)
def test_block_bootstrap_sample(pre_order: list[int], expected_pre_ws: list[float]) -> None:
    pre_index = pd.date_range("2020-01-01", periods=12, freq="10min", tz="UTC")
    pre_df = pd.DataFrame({"ws": np.arange(12.0)}, index=pre_index).iloc[pre_order]
    post_df = pd.DataFrame(
        {"ws": 100 + np.arange(25.0)}, index=pd.date_range("2021-01-01", periods=25, freq="10min", tz="UTC")
    )
    rng = np.random.default_rng(1)

    pre_sample, post_sample = _block_bootstrap_sample(pre_df, post_df, rng=rng)

    assert pre_sample["ws"].tolist() == expected_pre_ws
    assert post_sample["ws"].tolist() == EXPECTED_POST_WS
    assert rng.integers(1_000_000) == 613003  # the same random numbers are drawn as with wind-up 0.7.1


@pytest.mark.parametrize(
    ("ref_name", "expected_bootstrap_unc", "expected_p95"),
    [
        ("LSA_T12", 0.0034729424641318754, 0.03359308313774399),
        ("LSA_T13", 0.0032152424480777573, 0.03994779238394744),
    ],
)
def test_bootstrapping_results_are_unchanged(
    test_lsa_t13_config: WindUpConfig, ref_name: str, expected_bootstrap_unc: float, expected_p95: float
) -> None:
    cfg = test_lsa_t13_config
    cfg.bootstrap_runs_override = 8

    pp_results, _ = pre_post_pp_analysis_with_reversal_and_bootstrapping(
        cfg=cfg,
        test_wtg=next(x for x in cfg.asset.wtgs if x.name == "LSA_T13"),
        ref_name=ref_name,
        lt_df=pd.read_parquet(TEST_DATA_FLD / "LSA_T13_lt_wtg_df_filt.parquet"),
        pre_df=pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_pre_df.parquet"),
        post_df=pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_post_df.parquet"),
        ws_col="ref_ws_detrended",
        pw_col="test_pw_clipped",
        wd_col="ref_YawAngleMean",
        plot_cfg=None,
        random_seed=0,
    )

    assert pp_results["unc_one_sigma_bootstrap_frc"] == expected_bootstrap_unc
    assert pp_results["uplift_p95_frc"] == expected_p95
