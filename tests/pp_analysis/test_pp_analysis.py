import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.conftest import TEST_DATA_FLD
from wind_up.models import WindUpConfig
from wind_up.pp_analysis import (
    _block_bootstrap_sample,
    _bootstrap_columns,
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


def _label_based_block_bootstrap_sample(
    pre_df: pd.DataFrame, post_df: pd.DataFrame, *, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw a block bootstrap sample by index label, as wind-up 0.7.1 did."""
    num_blocks = rng.choice([9, 10, 11])
    block_size_pre = math.floor(len(pre_df) / num_blocks)
    block_size_post = math.floor(len(post_df) / num_blocks)
    pre_rows_to_keep = rng.choice(pre_df.index, size=num_blocks * block_size_pre, replace=False)
    post_rows_to_keep = rng.choice(post_df.index, size=num_blocks * block_size_post, replace=False)
    pre_df_trim = pre_df.loc[pre_rows_to_keep].sort_index()
    post_df_trim = post_df.loc[post_rows_to_keep].sort_index()
    blocks = rng.choice(num_blocks, size=num_blocks, replace=True)
    pre_sample_ilocs = (block_size_pre * blocks[:, None] + np.arange(block_size_pre)[None, :]).flatten()
    post_sample_ilocs = (block_size_post * blocks[:, None] + np.arange(block_size_post)[None, :]).flatten()
    return (
        pre_df_trim.iloc[pre_sample_ilocs].sort_index().reset_index(),
        post_df_trim.iloc[post_sample_ilocs].sort_index().reset_index(),
    )


def _bootstrap_test_frame(*, rows: int, seed: int, index_kind: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=rows, freq="10min", tz="UTC", name="TimeStamp_StartFormat")
    df = pd.DataFrame({"ws": rng.uniform(0, 25, rows), "pw": rng.uniform(0, 2000, rows)}, index=index)
    if index_kind == "shuffled":
        return df.iloc[rng.permutation(rows)]
    if index_kind == "duplicated":
        return pd.concat([df, df.iloc[: rows // 7]]).sort_index()
    return df


@pytest.mark.parametrize("index_kind", ["sorted", "shuffled", "duplicated"])
@pytest.mark.parametrize(("pre_rows", "post_rows"), [(15000, 9000), (997, 1203), (10, 30), (5, 0)])
def test_block_bootstrap_sample_matches_label_based_sampling(index_kind: str, pre_rows: int, post_rows: int) -> None:
    pre_df = _bootstrap_test_frame(rows=pre_rows, seed=1, index_kind=index_kind)
    post_df = _bootstrap_test_frame(rows=post_rows, seed=2, index_kind=index_kind)
    rng = np.random.default_rng(seed=3)
    expected_rng = np.random.default_rng(seed=3)

    for _ in range(5):
        actual_pre, actual_post = _block_bootstrap_sample(pre_df, post_df, rng=rng)
        expected_pre, expected_post = _label_based_block_bootstrap_sample(pre_df, post_df, rng=expected_rng)

        assert_frame_equal(actual_pre[["ws", "pw"]], expected_pre[["ws", "pw"]], check_exact=True)
        assert_frame_equal(actual_post[["ws", "pw"]], expected_post[["ws", "pw"]], check_exact=True)
    assert rng.bit_generator.state == expected_rng.bit_generator.state


def test_bootstrap_columns() -> None:
    assert _bootstrap_columns(test_name="T1", ref_name="T2", ws_col="ref_ws_detrended", pw_col="test_pw_clipped") == [
        "ref_ws_detrended",
        "test_pw_clipped",
        "ref_ws_est_from_power_only",
        "test_ws_est_from_power_only",
        "ws_rom",
        "ref_pw_clipped",
    ]
    assert _bootstrap_columns(test_name="T1", ref_name="T1", ws_col="ref_ws_detrended", pw_col="test_pw_clipped") == [
        "ref_ws_detrended",
        "test_pw_clipped",
    ]


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

    assert pp_results["unc_one_sigma_bootstrap_frc"] == pytest.approx(expected_bootstrap_unc, rel=1e-9)
    assert pp_results["uplift_p95_frc"] == pytest.approx(expected_p95, rel=1e-9)
