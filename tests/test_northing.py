from pathlib import Path

import pandas as pd
import pytest

from wind_up.constants import REANALYSIS_WD_COL
from wind_up.models import WindUpConfig
from wind_up.northing import _calc_max_abs_north_errs, apply_northing_corrections
from wind_up.scada_funcs import _scada_multi_index


def test_apply_northing_corrections(test_lsa_t13_config: WindUpConfig) -> None:
    cfg = test_lsa_t13_config
    test_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_test_df.parquet")
    test_df.columns = test_df.columns.str.replace("test_", "")
    test_df["TurbineName"] = "LSA_T07"
    wf_df = test_df.copy()
    for wtg_name in ["LSA_T09", "LSA_T12", "LSA_T13", "LSA_T14"]:
        wtg_df = test_df.copy()
        wtg_df["TurbineName"] = wtg_name
        wf_df = pd.concat([wf_df, wtg_df])
    wf_df = _scada_multi_index(wf_df)
    wf_df_after_northing = apply_northing_corrections(wf_df, cfg=cfg, north_ref_wd_col=REANALYSIS_WD_COL, plot_cfg=None)

    median_yaw_before_northing = wf_df.groupby("TurbineName")["YawAngleMean"].median()
    median_yaw_after_northing = wf_df_after_northing.groupby("TurbineName")["YawAngleMean"].median()
    assert median_yaw_before_northing.min() == pytest.approx(232.67355346679688)
    assert median_yaw_before_northing.max() == pytest.approx(232.67355346679688)
    assert median_yaw_after_northing["LSA_T07"] == pytest.approx(219.69126892089844)
    assert median_yaw_after_northing["LSA_T09"] == pytest.approx(245.49956512451172)
    assert median_yaw_after_northing["LSA_T12"] == pytest.approx(177.40547943115234)
    assert median_yaw_after_northing["LSA_T13"] == pytest.approx(235.22855377197266)
    assert median_yaw_after_northing["LSA_T14"] == pytest.approx(224.92881774902344)

    abs_north_errs_before_northing = _calc_max_abs_north_errs(
        wf_df, north_ref_wd_col=REANALYSIS_WD_COL, timebase_s=cfg.timebase_s
    )
    abs_north_errs_after_northing = _calc_max_abs_north_errs(
        wf_df_after_northing, north_ref_wd_col=REANALYSIS_WD_COL, timebase_s=cfg.timebase_s
    )
    assert abs_north_errs_before_northing.min() == pytest.approx(7.88920288085938)
    assert abs_north_errs_before_northing.max() == pytest.approx(7.88920288085938)
    assert abs_north_errs_after_northing["LSA_T07"] == pytest.approx(18.400006103515604)
    assert abs_north_errs_after_northing["LSA_T09"] == pytest.approx(23.889203)
    assert abs_north_errs_after_northing["LSA_T12"] == pytest.approx(162.918431)
    assert abs_north_errs_after_northing["LSA_T13"] == pytest.approx(10.889203)
    assert abs_north_errs_after_northing["LSA_T14"] == pytest.approx(12.400006)
