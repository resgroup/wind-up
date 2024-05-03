import datetime as dt
import logging

import pandas as pd

from wind_up.constants import PROJECTROOT_DIR, TIMEBASE_PD_TIMEDELTA, TIMEBASE_S, TIMESTAMP_COL, TURBINE_DATA_DIR
from wind_up.models import WindUpConfig
from wind_up.result_manager import result_manager

logger = logging.getLogger(__name__)


def calc_last_xmin_datetime_in_month(d: dt.datetime, td: pd.Timedelta) -> pd.Timestamp:
    return pd.Timestamp(
        pd.offsets.MonthBegin(normalize=True).rollforward(d + td) - td,
    )


def load_smart_scada_month_from_file(
    asset_name: str,
    first_datetime_no_tz: dt.datetime,
    last_datetime_no_tz: dt.datetime,
    *,
    test_mode: bool = False,
) -> tuple[pd.DataFrame, bool]:
    data_dir = PROJECTROOT_DIR / "tests" / "test_data" / "smart_data" if test_mode else TURBINE_DATA_DIR
    scada_month = pd.DataFrame()
    data_loaded_from_file = False
    success = False
    asset_dir = data_dir / asset_name

    try:
        fname = (
            f'{asset_name}_{first_datetime_no_tz.strftime("%Y%m%d")}_{last_datetime_no_tz.strftime("%Y%m%d")}.parquet'
        )
        scada_month = pd.read_parquet(asset_dir / fname)
        data_loaded_from_file = True
    except FileNotFoundError:
        pass

    # check for 1st of month file
    if not data_loaded_from_file and first_datetime_no_tz.strftime("%Y%m%d")[-2:] != "01":
        try:
            fname = (
                f'{asset_name}_{first_datetime_no_tz.strftime("%Y%m")}01_'
                f'{last_datetime_no_tz.strftime("%Y%m%d")}.parquet'
            )
            scada_month = pd.read_parquet(asset_dir / fname)
            data_loaded_from_file = True
        except FileNotFoundError:
            pass

    # check for end of month file
    if not data_loaded_from_file:
        end_of_month = calc_last_xmin_datetime_in_month(first_datetime_no_tz, TIMEBASE_PD_TIMEDELTA)
        if last_datetime_no_tz.strftime("%Y%m%d") != end_of_month.strftime("%Y%m%d"):
            try:
                fname = (
                    f'{asset_name}_{first_datetime_no_tz.strftime("%Y%m%d")}_{end_of_month.strftime("%Y%m%d")}.parquet'
                )
                scada_month = pd.read_parquet(asset_dir / fname)
                data_loaded_from_file = True
            except FileNotFoundError:
                pass

    if data_loaded_from_file:
        logger.info(f"loaded SMART scada data from {TURBINE_DATA_DIR / asset_name / fname}")
        success = True

    return scada_month, success


def add_smart_lat_long_to_cfg(md: pd.DataFrame, cfg: WindUpConfig) -> WindUpConfig:
    # confirm md has the same turbines as cfg.asset.wtgs
    if set(md["Name"].values) != {x.name for x in cfg.asset.wtgs}:
        msg = "md has different turbines than cfg.asset.wtgs"
        raise ValueError(msg)
    # add md lat longs to cfg
    for wtg in cfg.asset.wtgs:
        wtg.latitude = float(md.loc[md["Name"] == wtg.name, "Latitude"].iloc[0])
        wtg.longitude = float(md.loc[md["Name"] == wtg.name, "Longitude"].iloc[0])
    return cfg


def load_smart_md_from_file(asset_name: str, *, test_mode: bool = False) -> pd.DataFrame:
    if test_mode:
        md_dir = PROJECTROOT_DIR / "tests" / "test_data" / "smart_data" / asset_name
    else:
        md_dir = TURBINE_DATA_DIR / asset_name
    return pd.read_csv(md_dir / f"{asset_name}_md.csv", index_col=0)


def calc_month_list_and_time_info(
    asset_name: str,
    first_datetime_utc_start: pd.Timestamp,
    last_datetime_utc_start: pd.Timestamp,
    md: pd.DataFrame,
) -> tuple[list[dt.datetime], dt.datetime, str, str]:
    # make sure TimeSpanMinutes is equal to 10.0 for all turbines
    if len(md["TimeSpanMinutes"].unique()) > 1 or md["TimeSpanMinutes"].unique()[0] != (TIMEBASE_S / 60):
        msg = f"TimeSpanMinutes not 10 for all SMART metadata in {asset_name}: {md['TimeSpanMinutes'].unique()}"
        raise ValueError(
            msg,
        )

    # make sure there is only one TimeZone and TimeFormat
    if len(md["TimeZone"].unique()) > 1:
        msg = f"more than one TimeZone found in SMART metadata for {asset_name}: {md['TimeZone'].unique()}"
        raise ValueError(msg)
    if len(md["TimeFormat"].unique()) > 1:
        msg = f"more than one TimeFormat found in SMART metadata for {asset_name}: {md['TimeFormat'].unique()}"
        raise ValueError(
            msg,
        )

    smart_tz = md.loc[:, "TimeZone"].iloc[0]
    smart_tf = md.loc[:, "TimeFormat"].iloc[0]
    if smart_tf not in ("Start", "End"):
        msg = f"TimeFormat not Start or End in SMART metadata for {asset_name}: {smart_tf}"
        raise ValueError(msg)

    first_smart_dt_no_tz = first_datetime_utc_start.tz_convert(smart_tz).tz_localize(None).to_pydatetime()
    last_smart_dt_no_tz = last_datetime_utc_start.tz_convert(smart_tz).tz_localize(None).to_pydatetime()
    if smart_tf == "End":
        first_smart_dt_no_tz = first_smart_dt_no_tz + TIMEBASE_PD_TIMEDELTA
        last_smart_dt_no_tz = last_smart_dt_no_tz + TIMEBASE_PD_TIMEDELTA

    month_start_list_no_tz = pd.date_range(
        first_smart_dt_no_tz.strftime("%Y-%m-%d"),
        last_smart_dt_no_tz.strftime("%Y-%m-%d"),
        freq="MS",
    ).to_list()
    if len(month_start_list_no_tz) == 0:
        month_start_list_no_tz.insert(0, pd.Timestamp(first_smart_dt_no_tz.strftime("%Y-%m-%d")))
    if first_smart_dt_no_tz < month_start_list_no_tz[0]:
        month_start_list_no_tz.insert(0, pd.Timestamp(first_smart_dt_no_tz.strftime("%Y-%m-%d")))
    return month_start_list_no_tz, last_smart_dt_no_tz, smart_tz, smart_tf


def check_and_convert_scada_raw(
    scada_raw: pd.DataFrame,
    *,
    scada_data_timezone: str,
    scada_data_time_format: str,
    first_datetime_utc_start: pd.Timestamp,
    last_datetime_utc_start: pd.Timestamp,
) -> pd.DataFrame:
    if scada_raw["TurbineName"].isna().any():
        msg = f"NaNs in TurbineName column of scada_raw: {scada_raw['TurbineName'].isna().sum()}"
        raise RuntimeError(msg)
    scada_raw["TurbineName"] = scada_raw["TurbineName"].astype("category")

    scada_raw = scada_raw.sort_index()

    # save a copy of scada_raw index as a column
    scada_raw["smart_dtTimeStamp"] = scada_raw.index

    # convert scada_raw index from SMART tz to the wind-up timezone (UTC) and Start format
    if not isinstance(scada_raw.index, pd.DatetimeIndex):
        msg = f"scada_raw.index is not a pd.DatetimeIndex: {type(scada_raw.index)}"
        raise TypeError(msg)

    if scada_raw.index.tzinfo is not None:
        scada_raw.index = scada_raw.index.tz_localize(None)
    scada_raw.index = scada_raw.index.tz_localize(scada_data_timezone).tz_convert("UTC")
    if scada_data_time_format == "End":
        scada_raw.index = scada_raw.index - pd.Timedelta(minutes=10)
    scada_raw.index = scada_raw.index.rename(TIMESTAMP_COL)

    # clip to originally requested datetime range
    scada_raw = scada_raw.loc[first_datetime_utc_start:last_datetime_utc_start]

    # verify there is only one row per turbine per timestamp
    scada_raw.set_index([scada_raw.index, "TurbineName"], verify_integrity=True)

    turbine_rows = scada_raw.groupby("TurbineName", observed=False)["TurbineName"].count().to_frame()
    rows_per_turbine = turbine_rows.max().iloc[0]
    if rows_per_turbine != turbine_rows.min().iloc[0]:
        msg = f"turbines have different number of rows: {rows_per_turbine} != {turbine_rows.min().iloc[0]}"
        result_manager.warning(msg)
        logger.info("attempting to repair")
        rows_before = len(scada_raw)
        scada_raw_reset = scada_raw.reset_index()
        unique_wtgs = scada_raw_reset["TurbineName"].unique()
        complete_index = pd.date_range(
            start=scada_raw.index.min(),
            end=scada_raw.index.max(),
            freq=TIMEBASE_PD_TIMEDELTA,
        )
        reindexed_df = pd.DataFrame()
        for wtg in unique_wtgs:
            temp_df = scada_raw_reset[scada_raw_reset["TurbineName"] == wtg]
            temp_df = temp_df.set_index(TIMESTAMP_COL)
            temp_df = temp_df.reindex(complete_index)
            temp_df["TurbineName"] = wtg
            reindexed_df = pd.concat([reindexed_df, temp_df])
        reindexed_df.index.name = TIMESTAMP_COL
        rows_after = len(reindexed_df)

        turbine_rows = reindexed_df.groupby("TurbineName", observed=False)["TurbineName"].count().to_frame()
        new_rows_per_turbine = turbine_rows.max().iloc[0]
        if new_rows_per_turbine == turbine_rows.min().iloc[0] and new_rows_per_turbine == rows_per_turbine:
            logger.info(f"repair successful. rows before: {rows_before}, rows after: {rows_after}")
            scada_raw = reindexed_df
        else:
            msg = f"turbines have different number of rows: {new_rows_per_turbine} != {turbine_rows.min().iloc[0]}"
            raise RuntimeError(msg)

    logger.info(f"loaded {len(turbine_rows)} turbines, {rows_per_turbine / 6 / 24 / 365.25:.1f} years per turbine")
    return scada_raw


def load_smart_scada_and_md_from_file(
    asset_name: str,
    scada_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    first_datetime_utc_start: pd.Timestamp,
    last_datetime_utc_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(
        f"running load_smart_scada_and_md_from_file for {first_datetime_utc_start} to {last_datetime_utc_start}"
    )
    md = metadata_df

    month_start_list_no_tz, last_smart_dt_no_tz, smart_tz, smart_tf = calc_month_list_and_time_info(
        asset_name=asset_name,
        first_datetime_utc_start=first_datetime_utc_start,
        last_datetime_utc_start=last_datetime_utc_start,
        md=md,
    )
    scada_raw = check_and_convert_scada_raw(
        scada_df,
        scada_data_timezone=smart_tz,
        scada_data_time_format=smart_tf,
        first_datetime_utc_start=first_datetime_utc_start,
        last_datetime_utc_start=last_datetime_utc_start,
    )

    logger.info(f"finished load_smart_scada_and_md for {first_datetime_utc_start} to {last_datetime_utc_start}")
    return scada_raw, md
