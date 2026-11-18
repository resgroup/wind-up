import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.conftest import TEST_DATA_FLD
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.pp_analysis import (
    _pre_post_pp_analysis,
    _pre_post_pp_analysis_with_reversal,
    pre_post_pp_analysis_with_reversal_and_bootstrapping,
)

UPLIFT_COLS = [
    "uplift_kw",
    "uplift_kw_se",
    "uplift_p5_kw",
    "uplift_p95_kw",
    "relative_cp_baseline",
    "relative_cp_post",
    "relative_cp_sem_at_mid_expected",
    "relative_cp_sem_at_mid_post",
]


def _lsa_t13_inputs(cfg: WindUpConfig, *, ref_name: str = "LSA_T12") -> dict:
    return {
        "cfg": cfg,
        "test_wtg": next(x for x in cfg.asset.wtgs if x.name == "LSA_T13"),
        "ref_name": ref_name,
        "lt_df": pd.read_parquet(TEST_DATA_FLD / "LSA_T13_lt_wtg_df_filt.parquet"),
        "pre_df": pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_pre_df.parquet"),
        "post_df": pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_post_df.parquet"),
        "ws_col": "ref_ws_detrended",
        "pw_col": "test_pw_clipped",
        "wd_col": "ref_YawAngleMean",
        "plot_cfg": None,
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_pre_post_pp_analysis_without_uplift_cols(test_lsa_t13_config: WindUpConfig, *, reverse: bool) -> None:
    inputs = _lsa_t13_inputs(test_lsa_t13_config)
    if reverse:
        for period in ("pre_df", "post_df"):
            inputs[period] = inputs[period].assign(test_ws_detrended=inputs[period]["test_ws_est_from_power_only"])

    expected_results, expected_df = _pre_post_pp_analysis(**inputs, reverse=reverse)
    actual_results, actual_df = _pre_post_pp_analysis(**inputs, reverse=reverse, add_uplift_cols=False)

    assert set(UPLIFT_COLS) <= set(expected_df.columns)
    assert_frame_equal(actual_df, expected_df.drop(columns=UPLIFT_COLS), check_exact=True)
    assert {k: v for k, v in actual_results.items() if k != "time_calculated"} == {
        k: v for k, v in expected_results.items() if k != "time_calculated"
    }


@pytest.mark.parametrize("ref_name", ["LSA_T12", "LSA_T13"])
def test_pre_post_pp_analysis_with_reversal_without_uplift_cols(
    test_lsa_t13_config: WindUpConfig, ref_name: str
) -> None:
    inputs = _lsa_t13_inputs(test_lsa_t13_config, ref_name=ref_name)

    expected_results, expected_df = _pre_post_pp_analysis_with_reversal(**inputs)
    actual_results, actual_df = _pre_post_pp_analysis_with_reversal(**inputs, add_uplift_cols=False)

    assert_frame_equal(actual_df, expected_df.drop(columns=UPLIFT_COLS), check_exact=True)
    assert {k: v for k, v in actual_results.items() if k != "time_calculated"} == pytest.approx(
        {k: v for k, v in expected_results.items() if k != "time_calculated"}, rel=0, abs=0, nan_ok=True
    )


def test_pre_post_pp_analysis_without_uplift_cols_cannot_plot(test_lsa_t13_config: WindUpConfig, tmp_path) -> None:  # noqa: ANN001
    inputs = _lsa_t13_inputs(test_lsa_t13_config) | {"plot_cfg": PlotConfig(plots_dir=tmp_path)}

    with pytest.raises(ValueError, match="plot_cfg must be None when add_uplift_cols is False"):
        _pre_post_pp_analysis(**inputs, add_uplift_cols=False)


@pytest.mark.parametrize(
    ("ref_name", "expected_bootstrap_unc", "expected_p95"),
    [
        ("LSA_T12", 0.0034729424641318754, 0.03359308313774399),
        ("LSA_T13", 0.0032152424480777573, 0.03994779238394744),
    ],
)
def test_bootstrapping_results_are_unchanged_without_uplift_cols(
    test_lsa_t13_config: WindUpConfig, ref_name: str, expected_bootstrap_unc: float, expected_p95: float
) -> None:
    cfg = test_lsa_t13_config
    cfg.bootstrap_runs_override = 8

    pp_results, _ = pre_post_pp_analysis_with_reversal_and_bootstrapping(
        **_lsa_t13_inputs(cfg, ref_name=ref_name), random_seed=0
    )

    assert pp_results["unc_one_sigma_bootstrap_frc"] == pytest.approx(expected_bootstrap_unc, rel=1e-12)
    assert pp_results["uplift_p95_frc"] == pytest.approx(expected_p95, rel=1e-12)
