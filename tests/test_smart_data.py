from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tests.conftest import TEST_DATA_FLD
from wind_up.backporting import strict_zip
from wind_up.constants import TIMESTAMP_COL
from wind_up.smart_data import (
    add_smart_lat_long_to_cfg,
    calc_last_xmin_datetime_in_month,
    calc_month_list_and_time_info,
    check_and_convert_scada_raw,
    load_smart_md_from_file,
    load_smart_scada_and_md_from_file,
)

if TYPE_CHECKING:
    from wind_up.models import WindUpConfig

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


@pytest.mark.parametrize("timezone", [None, "UTC", "Europe/Berlin", "Europe/Paris"])
def test_load_smart_scada_and_md_from_file(timezone: str | None) -> None:
    test_data_dir = TEST_DATA_FLD / "smart_data" / "Marge Wind Farm"
    first_datetime_utc_start = pd.Timestamp("2020-02-26 23:50:00").tz_localize(timezone)
    last_datetime_utc_start = pd.Timestamp("2020-02-29 23:40:00").tz_localize(timezone)
    first_datetime_utc_start = (
        first_datetime_utc_start
        if first_datetime_utc_start.tzinfo is not None
        else first_datetime_utc_start.tz_localize("UTC")
    )
    last_datetime_utc_start = (
        last_datetime_utc_start
        if last_datetime_utc_start.tzinfo is not None
        else last_datetime_utc_start.tz_localize("UTC")
    )
    scada_df = pd.concat([pd.read_parquet(i) for i in test_data_dir.glob("*.parquet")])
    if timezone:
        scada_df = scada_df.tz_localize(timezone)
    scada_raw, _md = load_smart_scada_and_md_from_file(
        asset_name="Marge Wind Farm",
        scada_df=scada_df,
        metadata_df=pd.read_csv(test_data_dir / "Marge Wind Farm_md.csv"),
        first_datetime_utc_start=first_datetime_utc_start,
        last_datetime_utc_start=last_datetime_utc_start,
        timebase_s=600,
    )
    assert len(scada_raw) == 3 * 9 * 24 * 6  # 3 days, 9 turbines
    assert scada_raw.index.name == TIMESTAMP_COL
    assert scada_raw.index.min() == first_datetime_utc_start
    assert scada_raw.index.max() == last_datetime_utc_start
    assert scada_raw["smart_dtTimeStamp"].min() == pd.Timestamp("2020-02-27 00:00:00", tz=timezone)
    assert scada_raw["smart_dtTimeStamp"].max() == pd.Timestamp("2020-02-29 23:50:00", tz=timezone)


@pytest.mark.parametrize("timezone", ["UTC", "Europe/Berlin", "Europe/Paris"])
def test_check_and_convert_scada_raw(timezone: str) -> None:
    """Test that handles ambiguous and non-existent times correctly."""
    start_date = pd.Timestamp("2023-10-28 22:00:00")
    end_date = pd.Timestamp("2023-10-29 05:00:00")

    any_column = "AnyColumnName"
    freq = pd.Timedelta(minutes=10)

    idx = pd.date_range(start_date, end_date, freq=freq)
    scada_df = pd.DataFrame(
        index=idx,
        data={"TurbineName": ["Turbine01"] * len(idx), any_column: np.random.default_rng(42).random(len(idx))},
    )

    contingency_hours = pd.Timedelta(hours=3)
    first_datetime_utc_start = pd.Timestamp(start_date - contingency_hours, tz="UTC")
    last_datetime_utc_start = pd.Timestamp(end_date + contingency_hours, tz="UTC")

    actual = check_and_convert_scada_raw(
        scada_raw=scada_df,
        scada_data_timezone=timezone,
        scada_data_time_format="End",
        first_datetime_utc_start=first_datetime_utc_start,
        last_datetime_utc_start=last_datetime_utc_start,
        timebase_s=freq.seconds,
    )

    assert actual.shape[0] == scada_df.shape[0]
    assert actual.shape[1] == (
        scada_df.shape[1] + 1
    )  # +1 as the original timestamp column is preserved as a new column
    assert all(
        actual.loc[:, any_column].to_numpy() == scada_df.loc[:, any_column].to_numpy()
    )  # check the values in the column are the same as in the input df
