from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from wind_up.constants import TIMESTAMP_COL
from wind_up.models import WindUpConfig
from wind_up.scada_funcs import (
    add_pw_clipped,
    filter_bad_pw_ws,
    filter_downtime,
    filter_exclusions,
    filter_missing_rpm_or_pt,
    filter_rpm_and_pt,
    filter_rpm_and_pt_oor_one_ttype,
    filter_stuck_data,
    filter_wrong_yaw,
    filter_yaw_exclusions,
    scada_multi_index,
    wrap_yaw_and_pitch,
)


def test_filter_stuck_data() -> None:
    wtgs = ["MRG_T01", "MRG_T02", "MRG_T03"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=3, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    adf = pd.DataFrame(
        data={
            "ActivePowerMean": [5.1, 5.1, 5.1, 0.0, 0.0, 0.0, 5.1, 5.1, 5.1],
            "WindSpeedMean": [5.1, 5.1, np.nan, 0.0, 0.0, 0.0, 5.1, 5.1, 5.1],
            "YawAngleMean": [5.1, 5.1, 5.1, 0.0, 0.0, 0.0, 5.1, 5.1, 17],
            "some_col_with_nans": [np.nan] * 9,
        },
        index=idx,
    )
    adf = filter_stuck_data(adf)
    edf = pd.DataFrame(
        data={
            "ActivePowerMean": [5.1, np.nan, np.nan, 0.0, 0.0, 0.0, 5.1, np.nan, 5.1],
            "WindSpeedMean": [5.1, np.nan, np.nan, 0.0, 0.0, 0.0, 5.1, np.nan, 5.1],
            "YawAngleMean": [5.1, np.nan, np.nan, 0.0, 0.0, 0.0, 5.1, np.nan, 17],
            "some_col_with_nans": [np.nan] * 9,
        },
        index=idx,
    )
    assert_frame_equal(edf, adf)


def test_filter_bad_pw_ws() -> None:
    wtgs = ["MRG_T01", "MRG_T02", "MRG_T03"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=3, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    adf = pd.DataFrame(
        data={
            "ActivePowerMean": [np.nan, np.nan, 0.0, 5.1, 4000, 5.1, -1000, -1001, 4001],
            "WindSpeedMean": [np.nan, 0.1, np.nan, -0.1, 98, 99, 0.0, 0.0, 0.0],
            "some_col_with_nans": [np.nan] * 9,
            "som_col_with_values": [5.1] * 9,
        },
        index=idx,
    )
    adf = filter_bad_pw_ws(adf, max_rated_power=2000)
    edf = pd.DataFrame(
        data={
            "ActivePowerMean": [np.nan, np.nan, np.nan, np.nan, 4000, np.nan, -1000, np.nan, np.nan],
            "WindSpeedMean": [np.nan, np.nan, np.nan, np.nan, 98, np.nan, 0.0, np.nan, np.nan],
            "some_col_with_nans": [np.nan] * 9,
            "som_col_with_values": [np.nan, np.nan, np.nan, np.nan, 5.1, np.nan, 5.1, np.nan, np.nan],
        },
        index=idx,
    )
    assert_frame_equal(edf, adf)


def test_wrap_yaw_and_pitch() -> None:
    wtgs = ["MRG_T01", "MRG_T02", "MRG_T03"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=3, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    adf = pd.DataFrame(
        data={
            "YawAngleMean": [-181, -180, -179, -1, 0, 1, 359, 360, 361],
            "YawAngleMin": [-182, -181, -180, -2, -1, 0, 358, 359, 360],
            "PitchAngleMean": reversed([-181.1, -180.1, -179.1, -1.1, 0.1, 1.1, 359.1, 360.1, 361.1]),
        },
        index=idx,
    )
    adf = wrap_yaw_and_pitch(adf)
    edf = pd.DataFrame(
        data={
            "YawAngleMean": [179, 180, 181, 359, 0, 1, 359, 0, 1],
            "YawAngleMin": [0, 0, 0, 0, 0, 0, 358, 359, 0],
            "PitchAngleMean": reversed([178.9, 179.9, -179.1, -1.1, 0.1, 1.1, -0.9, 0.1, 1.1]),
        },
        index=idx,
    )
    assert_frame_equal(edf, adf)


def test_filter_wrong_yaw() -> None:
    wtgs = ["MRG_T01", "MRG_T02", "MRG_T03"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=3, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    adf = pd.DataFrame(
        data={
            "YawAngleMean": [180, 180, 180, 180, 15.1, 344.9, np.nan, 180, 180],
            "YawAngleMin": [170, 170, 190, 1, 1, 1, 180, np.nan, 180],
            "YawAngleMax": [190, 170, 190, 359, 359, 359, 180, 180, np.nan],
        },
        index=idx,
    )
    adf = filter_wrong_yaw(adf)
    edf = pd.DataFrame(
        data={
            "YawAngleMean": [180, 180, 180, np.nan, np.nan, np.nan, np.nan, 180, 180],
            "YawAngleMin": [170, np.nan, np.nan, np.nan, np.nan, np.nan, 180, np.nan, 180],
            "YawAngleMax": [190, np.nan, np.nan, np.nan, np.nan, np.nan, 180, 180, np.nan],
        },
        index=idx,
    )
    assert_frame_equal(edf, adf)


def test_filter_exclusions() -> None:
    wtgs = ["MRG_T01", "MRG_T02", "MRG_T03"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=6, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    adf = pd.DataFrame(
        data={
            "ActivePowerMean": [1.0] * len(idx),
            "YawAngleMean": [1.0] * len(idx),
        },
        index=idx,
    )
    exclusion_periods_utc = [
        ("MRG_T01", pd.Timestamp("2020-12-31 00:00:00", tz="UTC"), pd.Timestamp("2021-01-01 00:20:00", tz="UTC")),
        ("MRG_T02", pd.Timestamp("2021-01-01 00:29:00", tz="UTC"), pd.Timestamp("2021-01-01 00:31:00", tz="UTC")),
        ("ALL", pd.Timestamp("2021-01-01 00:30:00", tz="UTC"), pd.Timestamp("2022-02-02 00:00:00", tz="UTC")),
    ]
    adf = filter_exclusions(adf.copy(), exclusion_periods_utc)
    expected_values = [
        np.nan,
        np.nan,
        1.0,
        np.nan,
        np.nan,
        np.nan,
        1.0,
        1.0,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        1.0,
        1.0,
        1.0,
        np.nan,
        np.nan,
        np.nan,
    ]
    edf = pd.DataFrame(
        data={
            "ActivePowerMean": expected_values,
            "YawAngleMean": expected_values,
        },
        index=idx,
    )
    assert_frame_equal(edf, adf)


def test_filter_yaw_exclusions() -> None:
    wtgs = ["MRG_T01", "MRG_T02", "MRG_T03"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=6, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    adf = pd.DataFrame(
        data={
            "ActivePowerMean": [1.0] * len(idx),
            "YawAngleMean": [1.0] * len(idx),
        },
        index=idx,
    )
    yaw_data_exclusions_utc = [
        ("MRG_T01", pd.Timestamp("2020-12-31 00:00:00", tz="UTC"), pd.Timestamp("2021-01-01 00:20:00", tz="UTC")),
        ("MRG_T02", pd.Timestamp("2021-01-01 00:29:00", tz="UTC"), pd.Timestamp("2021-01-01 00:31:00", tz="UTC")),
        ("ALL", pd.Timestamp("2021-01-01 00:30:00", tz="UTC"), pd.Timestamp("2022-02-02 00:00:00", tz="UTC")),
    ]
    adf = filter_yaw_exclusions(adf.copy(), yaw_data_exclusions_utc)
    edf = pd.DataFrame(
        data={
            "ActivePowerMean": [1.0] * len(idx),
            "YawAngleMean": [
                np.nan,
                np.nan,
                1.0,
                np.nan,
                np.nan,
                np.nan,
                1.0,
                1.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1.0,
                1.0,
                1.0,
                np.nan,
                np.nan,
                np.nan,
            ],
        },
        index=idx,
    )
    assert_frame_equal(edf, adf)


def test_filter_downtime() -> None:
    wtgs = ["MRG_T01", "MRG_T02"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=2, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    adf = pd.DataFrame(
        data={
            "ShutdownDuration": [np.nan, -1, 0, 1],
            "ActivePowerMean": [5.1] * 4,
            "some_col_with_nans": [np.nan] * 4,
        },
        index=idx,
    )
    adf = filter_downtime(adf)
    edf = pd.DataFrame(
        data={
            "ShutdownDuration": [np.nan, np.nan, 0, np.nan],
            "ActivePowerMean": [np.nan, np.nan, 5.1, np.nan],
            "some_col_with_nans": [np.nan] * 4,
        },
        index=idx,
    )
    assert_frame_equal(edf, adf)


def test_filter_missing_rpm_or_pt() -> None:
    wtgs = ["MRG_T01", "MRG_T02"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=2, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    adf = pd.DataFrame(
        data={
            "GenRpmMean": [np.nan, -1, np.nan, -1],
            "PitchAngleMean": [np.nan, np.nan, -1, -1],
            "ActivePowerMean": [5.1] * 4,
            "some_col_with_nans": [np.nan] * 4,
        },
        index=idx,
    )
    adf = filter_missing_rpm_or_pt(adf)
    edf = pd.DataFrame(
        data={
            "GenRpmMean": [np.nan, np.nan, np.nan, -1],
            "PitchAngleMean": [np.nan, np.nan, np.nan, -1],
            "ActivePowerMean": [np.nan, np.nan, np.nan, 5.1],
            "some_col_with_nans": [np.nan] * 4,
        },
        index=idx,
    )
    assert_frame_equal(edf, adf)


def test_filter_rpm_and_pt_oor_one_ttype() -> None:
    wtgs = ["MRG_T01", "MRG_T02"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=4, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    adf = pd.DataFrame(
        data={
            "GenRpmMean": [799, 800, 1600, 1601] + [1000] * 4,
            "PitchAngleMean": [0] * 4 + [-11, -10, 40, 41],
            "ActivePowerMean": [np.nan, np.nan] + [5.1] * 6,
            "some_col_with_nans": [np.nan] * 8,
        },
        index=idx,
    )
    adf, na_rows = filter_rpm_and_pt_oor_one_ttype(adf, rpm_lower=800, rpm_upper=1600, pt_lower=-10, pt_upper=40)
    edf = pd.DataFrame(
        data={
            "GenRpmMean": [np.nan, 800, 1600, np.nan, np.nan, 1000, 1000, np.nan],
            "PitchAngleMean": [np.nan, 0, 0, np.nan, np.nan, -10, 40, np.nan],
            "ActivePowerMean": [np.nan, np.nan, 5.1, np.nan, np.nan, 5.1, 5.1, np.nan],
            "some_col_with_nans": [np.nan] * 8,
        },
        index=idx,
    )
    assert na_rows == 3
    assert_frame_equal(edf, adf)


def test_add_pw_clipped(test_marge_config: WindUpConfig) -> None:
    wtgs = ["MRG_T01", "MRG_T02", "MRG_T03"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=3, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    adf = pd.DataFrame(
        data={
            "ActivePowerMean": [-1, 0, 1901, -1, np.nan, 1900, -1, 1900, 1901],
        },
        index=idx,
    )
    cfg = test_marge_config
    cfg.asset.wtgs[2].turbine_type.rated_power_kw = 100
    cfg.asset.wtgs = cfg.asset.wtgs[:3]
    adf = add_pw_clipped(adf, wtgs=cfg.asset.wtgs)
    edf = pd.DataFrame(
        data={
            "ActivePowerMean": [-1, 0, 1901, -1, np.nan, 1900, -1, 1900, 1901],
            "pw_clipped": [0, 0, 1900, 0, np.nan, 1900, 0, 100, 100],
        },
        index=idx,
    )
    assert_frame_equal(edf, adf)


def test_filter_rpm_and_pt(test_marge_config: WindUpConfig) -> None:
    cfg = test_marge_config
    adf = pd.read_parquet(
        Path(__file__).parents[0] / "test_data/smart_data/Marge Wind Farm/Marge Wind Farm_20230101_20230103.parquet"
    )
    adf = scada_multi_index(adf)
    adf = add_pw_clipped(adf, wtgs=cfg.asset.wtgs)
    df_ = filter_rpm_and_pt(
        input_df=adf,
        cfg=cfg,
        plot_cfg=None,
    )

    tt = cfg.asset.wtgs[0].turbine_type
    assert df_["GenRpmMean"].min() >= cfg.get_normal_operation_genrpm_range(tt)[0]
    assert df_["GenRpmMean"].max() <= cfg.get_normal_operation_genrpm_range(tt)[1]
    assert df_["PitchAngleMean"].min() >= cfg.get_normal_operation_pitch_range(tt)[0]
    assert df_["PitchAngleMean"].max() <= cfg.get_normal_operation_pitch_range(tt)[1]

    # the below index locations should definitely be filtered out because of normal operation range
    # MRG_T03	2023-01-02 07:10
    # MRG_T03	2023-01-02 07:20
    # MRG_T03	2023-01-02 07:30
    # MRG_T03	2023-01-02 07:40
    # MRG_T06	2023-01-02 18:00
    # MRG_T06	2023-01-02 18:10
    # MRG_T06	2023-01-02 18:20
    assert df_.loc[("MRG_T03", "2023-01-02 07:10"), :].isna().all()
    assert df_.loc[("MRG_T03", "2023-01-02 07:20"), :].isna().all()
    assert df_.loc[("MRG_T03", "2023-01-02 07:30"), :].isna().all()
    assert df_.loc[("MRG_T03", "2023-01-02 07:40"), :].isna().all()
    assert df_.loc[("MRG_T06", "2023-01-02 18:00"), :].isna().all()
    assert df_.loc[("MRG_T06", "2023-01-02 18:10"), :].isna().all()
    assert df_.loc[("MRG_T06", "2023-01-02 18:20"), :].isna().all()

    # the below are filtered out based on curve filtering so these results could change over time
    assert df_.loc[("MRG_T03", "2023-01-01 12:20"), :].isna().all()
    assert df_.loc[("MRG_T04", "2023-01-02 17:40"), :].isna().all()
    assert df_.loc[("MRG_T06", "2023-01-02 18:30"), :].isna().all()

    # this may be too strict but should catch any major change
    edf = pd.read_parquet(Path(__file__).parents[0] / "test_data/test_filter_rpm_and_pt.parquet")
    assert_frame_equal(edf, df_)
