from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wind_up.detrend import _calc_wsratio_v_wd, apply_wsratio_v_wd_scen, calc_wsratio_v_wd_scen, check_applied_detrend
from wind_up.models import WindUpConfig
from wind_up.smart_data import add_smart_lat_long_to_cfg


def test_apply_wsratio_v_wd_scen_pre() -> None:
    expected_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_LSA_T12_pre_df.parquet")
    pre_df = expected_df.drop(columns=["ws_rom", "ref_ws_detrended"])
    wsratio_v_dir_scen = pd.read_parquet(
        Path(__file__).parents[0] / "test_data/LSA_T13_LSA_T12_wsratio_v_dir_scen.parquet",
    )
    ref_ws_col = "ref_ws_est_blend"
    ref_wd_col = "ref_YawAngleMean"
    actual_df = apply_wsratio_v_wd_scen(pre_df, wsratio_v_dir_scen, ref_ws_col=ref_ws_col, ref_wd_col=ref_wd_col)
    assert_frame_equal(actual_df, expected_df)


def test_apply_wsratio_v_wd_scen_post() -> None:
    expected_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_LSA_T12_post_df.parquet")
    post_df = expected_df.drop(columns=["ws_rom", "ref_ws_detrended"])
    wsratio_v_dir_scen = pd.read_parquet(
        Path(__file__).parents[0] / "test_data/LSA_T13_LSA_T12_wsratio_v_dir_scen.parquet",
    )
    ref_ws_col = "ref_ws_est_blend"
    ref_wd_col = "ref_YawAngleMean"
    actual_df = apply_wsratio_v_wd_scen(post_df, wsratio_v_dir_scen, ref_ws_col=ref_ws_col, ref_wd_col=ref_wd_col)
    assert_frame_equal(actual_df, expected_df)


def test_check_applied_detrend(test_lsa_t13_config: WindUpConfig) -> None:
    cfg = test_lsa_t13_config
    pre_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_LSA_T12_pre_df.parquet")
    post_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_LSA_T12_post_df.parquet")
    test_name = "LSA_T13"
    ref_name = "LSA_T12"
    ref_wtg = next(x for x in cfg.asset.wtgs if x.name == ref_name)
    ref_lat = ref_wtg.latitude
    ref_long = ref_wtg.longitude
    test_ws_col = "test_ws_est_blend"
    ref_ws_col = "ref_ws_est_blend"
    ref_wd_col = "ref_YawAngleMean"
    detrend_ws_col = "ref_ws_detrended"

    detrend_pre_r2_improvement, detrend_post_r2_improvement = check_applied_detrend(
        test_name=test_name,
        ref_name=ref_name,
        ref_lat=ref_lat,
        ref_long=ref_long,
        pre_df=pre_df,
        post_df=post_df,
        test_ws_col=test_ws_col,
        ref_ws_col=ref_ws_col,
        detrend_ws_col=detrend_ws_col,
        ref_wd_col=ref_wd_col,
        cfg=cfg,
        plot_cfg=None,
    )

    assert detrend_pre_r2_improvement == pytest.approx(0.03464757681986863)
    assert detrend_post_r2_improvement == pytest.approx(0.03776561982402227)


def test_calc_wsratio_v_wd_scen(test_lsa_t13_config: WindUpConfig) -> None:
    # this test case borrows logic and results from check_applied_detrend where data which has already been detrended
    # is used to calculate the wsratio_v_wd_scen again to check it is flat
    cfg = test_lsa_t13_config
    md_df = pd.read_csv(Path(__file__).parents[0] / "test_data/smart_data/Lisa Wind Farm/Lisa Wind Farm_md.csv")
    cfg = add_smart_lat_long_to_cfg(md=md_df, cfg=cfg)
    pre_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_LSA_T12_pre_df.parquet")
    test_name = "LSA_T13"
    ref_name = "LSA_T12"
    ref_wtg = next(x for x in cfg.asset.wtgs if x.name == ref_name)
    ref_lat = ref_wtg.latitude
    ref_long = ref_wtg.longitude
    test_ws_col = "test_ws_est_blend"
    ref_wd_col = "ref_YawAngleMean"
    detrend_ws_col = "ref_ws_detrended"

    expected_pre_df = pd.read_parquet(
        Path(__file__).parents[0] / "test_data/LSA_T13_LSA_T12_check_pre_wsratio_v_dir_scen.parquet",
    )
    actual_pre_df = calc_wsratio_v_wd_scen(
        test_name=test_name,
        ref_name=ref_name,
        ref_lat=ref_lat,
        ref_long=ref_long,
        detrend_df=pre_df,
        test_ws_col=test_ws_col,
        ref_ws_col=detrend_ws_col,  # different intentionally
        ref_wd_col=ref_wd_col,
        cfg=cfg,
        plot_cfg=None,
    )

    assert_frame_equal(actual_pre_df, expected_pre_df)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, object])
def test_calc_wsratio_v_wd_by_dtype(dtype: type) -> None:
    """Float64 and float32 wind speeds are summed on the arrays; other dtypes take the pandas mean."""
    ws = np.tile([6.0, 15.9], 3000).astype(dtype)
    detrend_df = pd.DataFrame({"test_ws": ws, "ref_ws": ws, "ref_wd": np.full(len(ws), 180.0)})

    result = _calc_wsratio_v_wd(
        detrend_df=detrend_df,
        test_ws_col="test_ws",
        ref_ws_col="ref_ws",
        ref_wd_col="ref_wd",
        min_hours=24,
        timebase_s=600,
    )

    assert result.set_index("direction").loc[180, "ws_rom"] == 1.0
