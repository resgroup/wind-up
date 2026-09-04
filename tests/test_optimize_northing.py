from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import TEST_DATA_FLD
from wind_up.circular_math import circ_median
from wind_up_v0.constants import RAW_DOWNTIME_S_COL, RAW_POWER_COL, RAW_YAWDIR_COL
from wind_up_v0.models import WindUpConfig
from wind_up_v0.optimize_northing import auto_northing_corrections
from wind_up_v0.reanalysis_data import ReanalysisDataset, add_reanalysis_data

wind_direction_offsets = [
    0,
    343,  # chosen to create lots of 0-360 wraps in the original data
    290,  # chosen to create lots of 0-360 wraps in the northed data
]


def _homer_wf_df(cfg: WindUpConfig, *, wind_direction_offset: float) -> pd.DataFrame:
    """The July 2023 Homer month, with every direction column rotated by ``wind_direction_offset``."""
    wf_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/Homer Wind Farm_July2023_scada_improved.parquet")
    reanalysis_datasets = [
        ReanalysisDataset(id=fp.stem, data=pd.read_parquet(fp))
        for fp in (TEST_DATA_FLD / "reanalysis" / "Homer Wind Farm").glob("*.parquet")
    ]
    wf_df = add_reanalysis_data(wf_df, cfg=cfg, plot_cfg=None, reanalysis_datasets=reanalysis_datasets)

    # add required columns not included in original test file
    wf_df[RAW_POWER_COL] = wf_df["ActivePowerMean"]
    wf_df[RAW_YAWDIR_COL] = wf_df["YawAngleMean"]
    wf_df[RAW_DOWNTIME_S_COL] = wf_df["ShutdownDuration"]

    for col in {RAW_YAWDIR_COL, "YawAngleMean", "reanalysis_wd"}:
        wf_df[col] = (wf_df[col] + wind_direction_offset) % 360
    if wind_direction_offset != 0:
        # in this case YawAngleMin and YawAngleMax will be incorrect, so nan them out
        wf_df["YawAngleMin"] = np.nan
        wf_df["YawAngleMax"] = np.nan
    return wf_df


def _median_yaw(wf_df: pd.DataFrame) -> pd.Series:
    return wf_df.groupby("TurbineName", observed=True)["YawAngleMean"].apply(circ_median)


@pytest.mark.slow
@pytest.mark.parametrize(("wind_direction_offset"), wind_direction_offsets)
def test_auto_northing_corrections(test_homer_config: WindUpConfig, wind_direction_offset: float) -> None:
    cfg = test_homer_config
    cfg.lt_first_dt_utc_start = pd.Timestamp("2023-07-01 00:00:00", tz="UTC")
    cfg.analysis_last_dt_utc_start = pd.Timestamp("2023-07-31 23:50:00", tz="UTC")
    wf_df = _homer_wf_df(cfg, wind_direction_offset=wind_direction_offset)

    northed_wf_df = auto_northing_corrections(wf_df, cfg=cfg, plot_cfg=None)

    median_yaw_before_northing = _median_yaw(wf_df)
    median_yaw_after_northing = _median_yaw(northed_wf_df)

    expected_t1_yaw_after_northing = (290 + wind_direction_offset) % 360
    expected_t2_yaw_after_northing = (295 + wind_direction_offset) % 360
    assert median_yaw_before_northing["HMR_T01"] == pytest.approx((343 + wind_direction_offset) % 360)
    assert median_yaw_before_northing["HMR_T02"] == pytest.approx((173 + wind_direction_offset) % 360)
    assert median_yaw_after_northing["HMR_T01"] == pytest.approx(expected_t1_yaw_after_northing, abs=1.0)
    assert median_yaw_after_northing["HMR_T02"] == pytest.approx(expected_t2_yaw_after_northing, abs=1.0)


@pytest.mark.slow
@pytest.mark.parametrize(("wind_direction_offset"), wind_direction_offsets)
def test_auto_northing_corrections_with_changepoints(
    test_homer_config: WindUpConfig, wind_direction_offset: float
) -> None:
    """Two injected step changes per turbine, on top of a 180 degree rotation, are all recovered.

    A month is short enough that the per-year rate alone would allow only one changepoint; the
    settings' floor is what leaves room for both.
    """
    cfg = test_homer_config
    cfg.lt_first_dt_utc_start = pd.Timestamp("2023-07-01 00:00:00", tz="UTC")
    cfg.analysis_last_dt_utc_start = pd.Timestamp("2023-07-31 23:50:00", tz="UTC")
    wf_df = _homer_wf_df(cfg, wind_direction_offset=wind_direction_offset)

    wf_df[RAW_YAWDIR_COL] = (wf_df[RAW_YAWDIR_COL] + 180) % 360

    # add a change point in for each turbine
    sign = 1
    for wtg_name in wf_df.index.unique(level="TurbineName"):
        idx = pd.IndexSlice[wtg_name, pd.Timestamp("2023-07-10 00:00:00+00:00") :]
        wf_df.loc[idx, RAW_YAWDIR_COL] = (wf_df.loc[idx, RAW_YAWDIR_COL] + sign * 30) % 360
        sign *= -1
    # add another change point for each turbine
    for wtg_name in wf_df.index.unique(level="TurbineName"):
        idx = pd.IndexSlice[wtg_name, pd.Timestamp("2023-07-20 00:00:00+00:00") :]
        wf_df.loc[idx, RAW_YAWDIR_COL] = (wf_df.loc[idx, RAW_YAWDIR_COL] + sign * 30) % 360

    wf_df["YawAngleMean"] = wf_df[RAW_YAWDIR_COL]

    northed_wf_df = auto_northing_corrections(wf_df, cfg=cfg, plot_cfg=None)

    median_yaw_after_northing = _median_yaw(northed_wf_df)
    assert median_yaw_after_northing["HMR_T01"] == pytest.approx((290 + wind_direction_offset) % 360, abs=1.5)
    assert median_yaw_after_northing["HMR_T02"] == pytest.approx((295 + wind_direction_offset) % 360, abs=1.5)
