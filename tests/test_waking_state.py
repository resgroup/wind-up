import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wind_up.constants import RAW_DOWNTIME_S_COL, RAW_POWER_COL, RAW_WINDSPEED_COL, TIMESTAMP_COL
from wind_up.models import WindUpConfig
from wind_up.scada_funcs import scada_multi_index
from wind_up.waking_state import (
    add_waking_scen,
    add_waking_state,
    calc_bearing,
    calc_distance,
    calc_scen_name_from_wtgs_not_waking,
    get_distance_and_bearing,
    get_iec_upwind_turbines,
    list_wtgs_offline_in_scen,
)


def test_get_bearing(test_homer_config: WindUpConfig) -> None:
    assert calc_bearing(lat1=0, long1=0, lat2=0, long2=1) == 90
    assert calc_bearing(lat1=0, long1=0, lat2=1, long2=0) == 0
    assert calc_bearing(lat1=0, long1=0, lat2=0, long2=-1) == 270
    cfg = test_homer_config
    t1lat = cfg.asset.wtgs[0].latitude
    t1long = cfg.asset.wtgs[0].longitude
    t2lat = cfg.asset.wtgs[1].latitude
    t2long = cfg.asset.wtgs[1].longitude
    expected = 245.02500888680734 - 180
    assert calc_bearing(lat1=t1lat, long1=t1long, lat2=t2lat, long2=t2long) == pytest.approx(expected)
    assert calc_bearing(lat1=t2lat, long1=t2long, lat2=t1lat, long2=t1long) == pytest.approx(
        (expected + 180) % 360,
        rel=1e-4,
    )
    assert calc_bearing(lat1=-t1lat, long1=t1long, lat2=-t2lat, long2=t2long) == pytest.approx((-expected + 180) % 360)
    assert calc_bearing(lat1=t1lat, long1=-t1long, lat2=t2lat, long2=-t2long) == pytest.approx((-expected) % 360)


def test_get_distance(test_homer_config: WindUpConfig) -> None:
    # circumference of earth at equator is about 40075km
    assert calc_distance(lat1=0, long1=0, lat2=0, long2=1 / 100) == pytest.approx(40075 * 1000 / 360 / 100)
    assert calc_distance(lat1=0, long1=0, lat2=0, long2=-1 / 100) == pytest.approx(40075 * 1000 / 360 / 100)
    assert calc_distance(lat1=0, long1=90, lat2=0, long2=90 + 1 / 100) == pytest.approx(40075 * 1000 / 360 / 100)
    cfg = test_homer_config
    t1lat = cfg.asset.wtgs[0].latitude
    t1long = cfg.asset.wtgs[0].longitude
    t2lat = cfg.asset.wtgs[1].latitude
    t2long = cfg.asset.wtgs[1].longitude
    expected = 270.894287973147
    assert calc_distance(lat1=t1lat, long1=t1long, lat2=t2lat, long2=t2long) == pytest.approx(expected)
    assert calc_distance(lat1=t2lat, long1=t2long, lat2=t1lat, long2=t1long) == pytest.approx(expected)
    assert calc_distance(lat1=-t1lat, long1=t1long, lat2=-t2lat, long2=t2long) == pytest.approx(expected)
    assert calc_distance(lat1=t1lat, long1=-t1long, lat2=t2lat, long2=-t2long) == pytest.approx(expected)


def test_get_distance_and_bearing(test_homer_config: WindUpConfig) -> None:
    cfg = test_homer_config
    t1lat = cfg.asset.wtgs[0].latitude
    t1long = cfg.asset.wtgs[0].longitude
    t2lat = cfg.asset.wtgs[1].latitude
    t2long = cfg.asset.wtgs[1].longitude
    expected_bearing = 245.02500888680734 - 180
    expected_distance = 270.894287973147
    start_time = time.perf_counter()
    distance_m, bearing_deg = get_distance_and_bearing(lat1=t1lat, long1=t1long, lat2=t2lat, long2=t2long)
    end_time = time.perf_counter()
    run_time_pre_cache = end_time - start_time
    assert distance_m == pytest.approx(expected_distance)
    assert bearing_deg == pytest.approx(expected_bearing)
    start_time = time.perf_counter()
    reps = 100
    for _x in range(reps):
        distance_m, bearing_deg = get_distance_and_bearing(lat1=t1lat, long1=t1long, lat2=t2lat, long2=t2long)
        assert distance_m == pytest.approx(expected_distance)
        assert bearing_deg == pytest.approx(expected_bearing)
    end_time = time.perf_counter()
    run_time_with_cache = (end_time - start_time) / reps
    assert run_time_with_cache < run_time_pre_cache * 3 / 10


def test_calc_scen_name_from_wtgs_not_waking() -> None:
    assert calc_scen_name_from_wtgs_not_waking(["T1", "T2"]) == "T1 T2 offline"


def test_list_wtgs_offline_in_scen() -> None:
    assert list_wtgs_offline_in_scen("T1 T2 offline") == ["T1", "T2"]


test_get_iec_upwind_turbines_data = [
    (0, 250 - 180, ["HMR_T00", "HMR_T02"]),
    (1, 250 - 180, ["HMR_T00"]),
    (0, 250, []),
    (1, 250, ["HMR_T01"]),
    (0, 250 - 180 - 90, []),
    (1, 250 - 180 - 90, []),
]


@pytest.mark.parametrize(("wtg_idx", "wind_direction", "expected"), test_get_iec_upwind_turbines_data)
def test_get_iec_upwind_turbines(
    wtg_idx: int,
    wind_direction: float,
    expected: list[str],
    test_homer_with_t00_config: WindUpConfig,
) -> None:
    cfg = test_homer_with_t00_config

    upwind_wtgs = get_iec_upwind_turbines(
        latlongs=[(cfg.asset.wtgs[wtg_idx].latitude, cfg.asset.wtgs[wtg_idx].longitude)],
        wind_direction=wind_direction,
        cfg=cfg,
        object_name=cfg.asset.wtgs[wtg_idx].name,
    )
    assert upwind_wtgs == expected


def test_add_waking_scen(test_homer_with_t00_config: WindUpConfig) -> None:
    cfg = test_homer_with_t00_config
    test_name = "HMR_T01"
    ref_name = "HMR_T02"
    ref_wtg = next(x for x in cfg.asset.wtgs if x.name == ref_name)
    ref_lat = ref_wtg.latitude
    ref_long = ref_wtg.longitude

    wtgs = ["HMR_T00", "HMR_T01", "HMR_T02"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=3, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    wf_df = pd.DataFrame(
        data={
            "YawAngleMean": [249.9 - 180] * len(idx),
            "waking": [True, False, False] + [True] * (len(idx) - 3),
            "not_waking": [False, True, False] + [False] * (len(idx) - 3),
            "unknown_waking": [False, False, True] + [False] * (len(idx) - 3),
        },
        index=idx,
    )
    test_df = wf_df.loc[test_name]
    test_df.columns = ["test_" + x for x in test_df.columns]
    ref_df = wf_df.loc[ref_name]
    ref_df.columns = ["ref_" + x for x in ref_df.columns]
    test_ref_df = test_df.merge(ref_df, how="left", left_index=True, right_index=True)

    expected_df = test_ref_df.copy()
    expected_df["waking_scenario"] = ["none offline", "HMR_T00 offline", "unknown"]

    actual_df = add_waking_scen(
        test_name=test_name,
        ref_name=ref_name,
        test_ref_df=test_ref_df,
        cfg=cfg,
        wf_df=wf_df,
        ref_wd_col="ref_YawAngleMean",
        ref_lat=ref_lat,
        ref_long=ref_long,
    )

    assert_frame_equal(actual_df, expected_df)


def test_add_waking_state_simple_case(test_homer_with_t00_config: WindUpConfig) -> None:
    cfg = test_homer_with_t00_config

    wtgs = ["HMR_T00", "HMR_T01", "HMR_T02"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=3, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    wf_df = pd.DataFrame(
        data={
            RAW_POWER_COL: [0.009 * cfg.asset.wtgs[0].turbine_type.rated_power_kw, 100, 100] + [500] * (len(idx) - 3),
            RAW_WINDSPEED_COL: [8] * len(idx),
            RAW_DOWNTIME_S_COL: [0] * len(idx),
        },
        index=idx,
    )
    wf_df["WindSpeedMean"] = wf_df[RAW_WINDSPEED_COL]
    wf_df["ActivePowerMean"] = wf_df[RAW_POWER_COL]
    wf_df.iloc[:3, -2:] = pd.NA
    wf_df.iloc[1, 2] = 600 * 0.71  # just enough ShutdownDuration for not_waking
    expected_df = wf_df.copy()
    expected_df["waking"] = [False] * 3 + [True] * (len(idx) - 3)
    expected_df["not_waking"] = [True] * 2 + [False] * (len(idx) - 2)
    expected_df["unknown_waking"] = [False, False, True] + [False] * (len(idx) - 3)
    actual_df = add_waking_state(cfg=cfg, wf_df=wf_df, plot_cfg=None)

    assert_frame_equal(actual_df, expected_df)


def test_add_waking_state_nan_cases(test_homer_with_t00_config: WindUpConfig) -> None:
    cfg = test_homer_with_t00_config

    wtgs = ["HMR_T00", "HMR_T01", "HMR_T02"]
    tstamps = pd.date_range(start="2021-01-01", tz="UTC", periods=3, freq="10min")
    idx = pd.MultiIndex.from_product([wtgs, tstamps], names=["TurbineName", TIMESTAMP_COL])
    pw_value = 100.1
    ws_value = 8.1
    wf_df = pd.DataFrame(
        data={
            RAW_POWER_COL: [np.nan] * 4 + [pw_value] * 4 + [np.nan],
            RAW_WINDSPEED_COL: list([np.nan] * 2 + [ws_value] * 2) * 2 + [np.nan],
            RAW_DOWNTIME_S_COL: [np.nan, 0] * 4 + [np.nan],
        },
        index=idx,
    )
    wf_df["WindSpeedMean"] = wf_df[RAW_WINDSPEED_COL]
    wf_df["ActivePowerMean"] = wf_df[RAW_POWER_COL]
    expected_df = wf_df.copy()
    expected_df["waking"] = [False] * 4 + [True] * 4 + [False]
    expected_df["not_waking"] = [False] * len(idx)
    expected_df["unknown_waking"] = [True] * 4 + [False] * 4 + [True]
    actual_df = add_waking_state(cfg=cfg, wf_df=wf_df, plot_cfg=None)

    assert_frame_equal(actual_df, expected_df)


def test_add_waking_state(test_lsa_t13_config: WindUpConfig) -> None:
    cfg = test_lsa_t13_config
    test_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/LSA_T13_test_df.parquet")
    test_df.columns = test_df.columns.str.replace("test_", "")
    test_df["TurbineName"] = "LSA_T01"
    wf_df = scada_multi_index(test_df)
    expected_df = wf_df.copy()
    wf_df = wf_df.drop(columns=["waking", "not_waking", "unknown_waking"])
    # remove all turbines from cfg apart from LSA_T01
    cfg.asset.wtgs = [cfg.asset.wtgs[0]]
    actual_df = add_waking_state(cfg=cfg, wf_df=wf_df, plot_cfg=None)

    # results not exactly the same because Cp is calculated from the whole dataset and used for waking calc
    # in the original calculation data from the entire wind farm for many years was used
    assert actual_df["not_waking"].sum() - expected_df["not_waking"].sum() == 0
    assert actual_df["waking"].sum() - expected_df["waking"].sum() == 30
    assert actual_df["unknown_waking"].sum() - expected_df["unknown_waking"].sum() == -30
