import datetime as dt

import pandas as pd
import pytest

from tests.conftest import TEST_DATA_FLD
from wind_up.backporting import strict_zip
from wind_up.constants import TIMESTAMP_COL
from wind_up.models import WindUpConfig
from wind_up.smart_data import (
    add_smart_lat_long_to_cfg,
    calc_last_xmin_datetime_in_month,
    calc_month_list_and_time_info,
    load_smart_md_from_file,
    load_smart_scada_and_md_from_file,
)

TIMEBASE_PD_TIMEDELTA = pd.Timedelta("10min")


def test_calc_last_xmin_datetime_in_month() -> None:
    inputs = [
        dt.datetime(2020, 1, 13),
        dt.datetime(2020, 2, 1),
        dt.datetime(2020, 2, 1, 0, 0),
        dt.datetime(2020, 2, 1, 0, 10),
        dt.datetime(2020, 2, 29, 23, 50),
    ]
    expected = [
        dt.datetime(2020, 1, 31, 23, 50),
        dt.datetime(2020, 2, 29, 23, 50),
        dt.datetime(2020, 2, 29, 23, 50),
        dt.datetime(2020, 2, 29, 23, 50),
        dt.datetime(2020, 2, 29, 23, 50),
    ]
    for i, e in strict_zip(inputs, expected):
        assert calc_last_xmin_datetime_in_month(i, TIMEBASE_PD_TIMEDELTA) == pd.Timestamp(e)


def test_add_smart_lat_long_to_cfg(test_marge_config: WindUpConfig) -> None:
    cfg = test_marge_config
    md_df = load_smart_md_from_file(asset_name="Marge Wind Farm", test_mode=True)
    cfg = add_smart_lat_long_to_cfg(md=md_df, cfg=cfg)
    assert cfg.asset.wtgs[0].latitude == pytest.approx(-53.741243623)
    assert cfg.asset.wtgs[0].longitude == pytest.approx(10.0580049240437)
    assert cfg.asset.wtgs[-1].latitude == pytest.approx(-53.7496477908)
    assert cfg.asset.wtgs[-1].longitude == pytest.approx(10.0752109226079)


def test_calc_month_list_and_time_info() -> None:
    md_df = load_smart_md_from_file(asset_name="Marge Wind Farm", test_mode=True)
    month_start_list_no_tz, last_smart_dt_no_tz, smart_tz, smart_tf = calc_month_list_and_time_info(
        asset_name="Marge Wind Farm",
        first_datetime_utc_start=pd.Timestamp("2019-12-25 00:20:00", tz="UTC"),
        last_datetime_utc_start=pd.Timestamp("2020-02-28 23:50:00", tz="UTC"),
        md=md_df,
        timebase_s=600,
    )
    assert month_start_list_no_tz == [
        pd.Timestamp("2019-12-25 00:00:00"),
        pd.Timestamp("2020-01-01 00:00:00"),
        pd.Timestamp("2020-02-01 00:00:00"),
    ]
    assert last_smart_dt_no_tz == pd.Timestamp("2020-02-29 00:00:00")
    assert smart_tz == "UTC"
    assert smart_tf == "End"


def test_load_smart_scada_and_md_from_file() -> None:
    test_data_dir = TEST_DATA_FLD / "smart_data" / "Marge Wind Farm"
    first_datetime_utc_start = pd.Timestamp("2020-02-26 23:50:00", tz="UTC")
    last_datetime_utc_start = pd.Timestamp("2020-02-29 23:40:00", tz="UTC")
    scada_raw, md = load_smart_scada_and_md_from_file(
        asset_name="Marge Wind Farm",
        scada_df=pd.concat([pd.read_parquet(i) for i in test_data_dir.glob("*.parquet")]),
        metadata_df=pd.read_csv(test_data_dir / "Marge Wind Farm_md.csv"),
        first_datetime_utc_start=first_datetime_utc_start,
        last_datetime_utc_start=last_datetime_utc_start,
        timebase_s=600,
    )
    assert len(scada_raw) == 3 * 9 * 24 * 6  # 3 days, 9 turbines
    assert scada_raw.index.name == TIMESTAMP_COL
    assert scada_raw.index.min() == first_datetime_utc_start
    assert scada_raw.index.max() == last_datetime_utc_start
    assert scada_raw["smart_dtTimeStamp"].min() == pd.Timestamp("2020-02-27 00:00:00")
    assert scada_raw["smart_dtTimeStamp"].max() == pd.Timestamp("2020-02-29 23:50:00")
