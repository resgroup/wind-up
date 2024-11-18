from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.conftest import TEST_DATA_FLD
from wind_up.constants import RAW_DOWNTIME_S_COL, RAW_POWER_COL, RAW_YAWDIR_COL, TIMESTAMP_COL
from wind_up.models import WindUpConfig
from wind_up.optimize_northing import auto_northing_corrections, clip_wtg_north_table
from wind_up.reanalysis_data import ReanalysisDataset, add_reanalysis_data


def test_clip_wtg_north_table_entries_before() -> None:
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=3, freq="10min")
    idx = pd.Index(tstamps)
    wtg_df = pd.DataFrame(
        data={
            "ActivePowerMean": [3.14] * 3,
            "some_col_with_nans": [np.nan] * 3,
        },
        index=idx,
    )
    tstamps_for_wtg_north_table = [
        tstamps[0] - pd.Timedelta(days=2),
        tstamps[0] - pd.Timedelta(days=1),
        tstamps[-1],
        tstamps[-1] + pd.Timedelta(days=1),
    ]
    initial_wtg_north_table = pd.DataFrame(
        data={
            TIMESTAMP_COL: tstamps_for_wtg_north_table,
            "north_offset": list(range(len(tstamps_for_wtg_north_table))),
        },
    )
    expected_wtg_north_table = pd.DataFrame(
        data={
            TIMESTAMP_COL: [tstamps[0], tstamps[-1], tstamps[-1] + pd.Timedelta(days=1)],
            "north_offset": [1, 2, 3],
        },
    )
    actual_wtg_north_table = clip_wtg_north_table(initial_wtg_north_table, wtg_df=wtg_df)
    assert_frame_equal(actual_wtg_north_table, expected_wtg_north_table)


def test_clip_wtg_north_table_entry_exactly_at_start() -> None:
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=3, freq="10min")
    idx = pd.Index(tstamps)
    wtg_df = pd.DataFrame(
        data={
            "ActivePowerMean": [3.14] * 3,
            "some_col_with_nans": [np.nan] * 3,
        },
        index=idx,
    )
    tstamps_for_wtg_north_table = [
        tstamps[0] - pd.Timedelta(days=1),
        tstamps[0],
        tstamps[-1],
        tstamps[-1] + pd.Timedelta(days=1),
    ]
    initial_wtg_north_table = pd.DataFrame(
        data={
            TIMESTAMP_COL: tstamps_for_wtg_north_table,
            "north_offset": list(range(len(tstamps_for_wtg_north_table))),
        },
    )
    expected_wtg_north_table = pd.DataFrame(
        data={
            TIMESTAMP_COL: [tstamps[0], tstamps[-1], tstamps[-1] + pd.Timedelta(days=1)],
            "north_offset": [1, 2, 3],
        },
    )
    actual_wtg_north_table = clip_wtg_north_table(initial_wtg_north_table, wtg_df=wtg_df)
    assert_frame_equal(actual_wtg_north_table, expected_wtg_north_table)


def test_clip_wtg_north_table_entry_after_start() -> None:
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=3, freq="10min")
    idx = pd.Index(tstamps)
    wtg_df = pd.DataFrame(
        data={
            "ActivePowerMean": [3.14] * 3,
            "some_col_with_nans": [np.nan] * 3,
        },
        index=idx,
    )
    tstamps_for_wtg_north_table = [
        tstamps[-1] + pd.Timedelta(days=1),
    ]
    initial_wtg_north_table = pd.DataFrame(
        data={
            TIMESTAMP_COL: tstamps_for_wtg_north_table,
            "north_offset": list(range(len(tstamps_for_wtg_north_table))),
        },
    )
    expected_wtg_north_table = pd.DataFrame(
        data={
            TIMESTAMP_COL: [tstamps[0]],
            "north_offset": [0],
        },
    )
    actual_wtg_north_table = clip_wtg_north_table(initial_wtg_north_table, wtg_df=wtg_df)
    assert_frame_equal(actual_wtg_north_table, expected_wtg_north_table)


def test_auto_northing_corrections(test_homer_config: WindUpConfig) -> None:
    cfg = test_homer_config
    cfg.lt_first_dt_utc_start = pd.Timestamp("2023-07-01 00:00:00", tz="UTC")
    cfg.analysis_last_dt_utc_start = pd.Timestamp("2023-07-31 23:50:00", tz="UTC")
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

    northed_wf_df = auto_northing_corrections(wf_df, cfg=cfg, plot_cfg=None)

    median_yaw_before_northing = wf_df.groupby("TurbineName", observed=True)["YawAngleMean"].median()
    median_yaw_after_northing = northed_wf_df.groupby("TurbineName", observed=True)["YawAngleMean"].median()

    assert median_yaw_before_northing["HMR_T01"] == pytest.approx(191.0)
    assert median_yaw_before_northing["HMR_T02"] == pytest.approx(173.0)
    assert median_yaw_after_northing["HMR_T01"] == pytest.approx(269.6)
    assert median_yaw_after_northing["HMR_T02"] == pytest.approx(267.6)

    # try to mess up the yaw angles further and run again
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

    median_yaw_before_northing = wf_df.groupby("TurbineName", observed=True)["YawAngleMean"].median()
    median_yaw_after_northing = northed_wf_df.groupby("TurbineName", observed=True)["YawAngleMean"].median()

    assert median_yaw_before_northing["HMR_T01"] == pytest.approx(178.0)
    assert median_yaw_before_northing["HMR_T02"] == pytest.approx(206.0)
    assert median_yaw_after_northing["HMR_T01"] == pytest.approx(269.2)
    assert median_yaw_after_northing["HMR_T02"] == pytest.approx(269.2)
