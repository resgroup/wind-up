from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.conftest import TEST_DATA_FLD
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.pp_analysis import _pre_post_pp_analysis


def _lsa_t13_inputs(cfg: WindUpConfig) -> dict:
    return {
        "cfg": cfg,
        "test_wtg": next(x for x in cfg.asset.wtgs if x.name == "LSA_T13"),
        "ref_name": "LSA_T12",
        "lt_df": pd.read_parquet(TEST_DATA_FLD / "LSA_T13_lt_wtg_df_filt.parquet"),
        "pre_df": pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_pre_df.parquet"),
        "post_df": pd.read_parquet(TEST_DATA_FLD / "LSA_T13_LSA_T12_post_df.parquet"),
        "ws_col": "ref_ws_detrended",
        "pw_col": "test_pw_clipped",
        "wd_col": "ref_YawAngleMean",
        "plot_cfg": None,
    }


def test_pre_post_pp_analysis_without_uplift_cols(test_lsa_t13_config: WindUpConfig) -> None:
    inputs = _lsa_t13_inputs(test_lsa_t13_config)

    expected_results, expected_df = _pre_post_pp_analysis(**inputs)
    actual_results, actual_df = _pre_post_pp_analysis(**inputs, add_uplift_cols=False)

    uplift_cols = [c for c in expected_df.columns if c.startswith(("uplift_", "relative_cp_"))]
    assert len(uplift_cols) == 8
    assert_frame_equal(actual_df, expected_df.drop(columns=uplift_cols), check_exact=True)
    del actual_results["time_calculated"], expected_results["time_calculated"]
    assert actual_results == expected_results


def test_pre_post_pp_analysis_without_uplift_cols_cannot_plot(
    test_lsa_t13_config: WindUpConfig, tmp_path: Path
) -> None:
    inputs = _lsa_t13_inputs(test_lsa_t13_config) | {"plot_cfg": PlotConfig(plots_dir=tmp_path)}

    with pytest.raises(ValueError, match="plot_cfg must be None when add_uplift_cols is False"):
        _pre_post_pp_analysis(**inputs, add_uplift_cols=False)
