"""Module to handle reanalysis data."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from wind_up.constants import (
    REANALYSIS_WD_COL,
    REANALYSIS_WS_COL,
    TIMESTAMP_COL,
)
from wind_up.plots.reanalysis_plots import plot_find_best_shift_and_corr, plot_wf_and_reanalysis_sample_timeseries

if TYPE_CHECKING:
    from wind_up.models import PlotConfig, WindUpConfig
logger = logging.getLogger(__name__)


def _reanalysis_upsample(reanalysis_raw_df: pd.DataFrame, *, timebase_s: int) -> pd.DataFrame:
    reanalysis_df = reanalysis_raw_df.resample(pd.Timedelta(seconds=timebase_s), label="left").last()
    upsample_factor = round(len(reanalysis_df) / len(reanalysis_raw_df))
    # extend the end of the index e.g. by 50 minutes for 10 minute timebase
    reanalysis_df = pd.concat(
        [
            reanalysis_df,
            pd.DataFrame(
                index=pd.date_range(
                    start=reanalysis_df.index[-1] + pd.Timedelta(seconds=timebase_s),
                    periods=upsample_factor - 1,
                    freq=pd.Timedelta(seconds=timebase_s),
                ),
            ),
        ],
    )
    reanalysis_df = reanalysis_df.ffill(limit=upsample_factor - 1)
    if "100_m_hws_mean_mps" in reanalysis_df.columns and "100_m_hwd_mean_deg-n_true" in reanalysis_df.columns:
        reanalysis_df = reanalysis_df.rename(
            columns={"100_m_hws_mean_mps": REANALYSIS_WS_COL, "100_m_hwd_mean_deg-n_true": REANALYSIS_WD_COL},
        )
    elif "50_m_hws_mean_mps" in reanalysis_df.columns and "50_m_hwd_mean_deg-n_true" in reanalysis_df.columns:
        reanalysis_df = reanalysis_df.rename(
            columns={"50_m_hws_mean_mps": REANALYSIS_WS_COL, "50_m_hwd_mean_deg-n_true": REANALYSIS_WD_COL},
        )
    else:
        msg = "reanalysis wind speed and direction columns not found"
        raise RuntimeError(msg)
    reanalysis_df.index.name = TIMESTAMP_COL
    return reanalysis_df


def _find_best_shift_and_corr(
    *,
    wf_ws_df: pd.DataFrame,
    reanalysis_df: pd.DataFrame,
    wf_name: str,
    datastream_id: str,
    timebase_s: int,
    plot_cfg: PlotConfig | None,
    wf_ws_lower_limit: float = 0,
) -> tuple[float, int]:
    ws_filt_df = wf_ws_df["WindSpeedMean"][wf_ws_df["WindSpeedMean"] >= wf_ws_lower_limit].to_frame()
    best_s = -99
    best_corr = -99.0
    shifts = []
    corrs = []
    rows_per_hour = 3600 / timebase_s
    for s in range(round(-24 * rows_per_hour), round(24 * rows_per_hour), math.ceil(rows_per_hour / 6)):
        this_corr = float(ws_filt_df.corrwith(reanalysis_df[REANALYSIS_WS_COL].shift(s)).squeeze())
        shifts.append(s)
        corrs.append(this_corr)
        if this_corr > best_corr:
            best_corr = this_corr
            best_s = s
    if plot_cfg is not None:
        plot_find_best_shift_and_corr(
            wf_ws_df=ws_filt_df,
            reanalysis_df=reanalysis_df,
            shifts=shifts,
            corrs=corrs,
            wf_name=wf_name,
            datastream_id=datastream_id,
            best_corr=best_corr,
            best_s=best_s,
            timebase_s=timebase_s,
            plot_cfg=plot_cfg,
        )

    logger.info(f"{datastream_id} best correlation is {best_corr:.6f} with a shift of {best_s}")

    return best_corr, best_s


def _calc_wf_mean_wind_speed_df(
    wf_df: pd.DataFrame,
    *,
    num_turbines: int,
    allowed_data_coverage_width: float,
) -> pd.DataFrame:
    """Calculate wind farm mean wind speed.

    :param wf_df: wind farm data
    :param num_turbines: number of turbines
    :param allowed_data_coverage_width: allowed data coverage width
    :return: wind farm mean wind speed and data coverage
    """
    wf_ws_df = wf_df.groupby(TIMESTAMP_COL).agg(
        WindSpeedMean=pd.NamedAgg(column="WindSpeedMean", aggfunc=lambda x: x.mean()),
        data_coverage=pd.NamedAgg(column="WindSpeedMean", aggfunc=lambda x: x.count() / num_turbines),
    )
    median_data_coverage = wf_ws_df["data_coverage"].median()
    wf_ws_df = wf_ws_df[(wf_ws_df["data_coverage"] - median_data_coverage).abs() < allowed_data_coverage_width / 2]
    return wf_ws_df.dropna()


def get_dsid_and_dates_from_filename(filename: str) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    """Get reanalysis dataset id and dates from filename.

    :param filename:
        reanalysis data filename, e.g. 'ERA5T_47.50N_-3.25E_100m_1hr_19900101_20231031.parquet'
    :return: tuple of dataset id and start and end dates
    """
    fname = filename.replace(".parquet", "")
    date_to = fname.split("_")[-1]
    date_from = fname.split("_")[-2]
    dsid = "_".join(fname.split("_")[:-2])
    return (
        dsid,
        pd.to_datetime(date_from, format="%Y%m%d").tz_localize("UTC"),
        pd.to_datetime(date_to, format="%Y%m%d").tz_localize("UTC"),
    )


@dataclass
class ReanalysisDataset:
    """Class to store reanalysis data."""

    id: str
    data: pd.DataFrame


def add_reanalysis_data(
    wf_df: pd.DataFrame,
    *,
    reanalysis_datasets: list[ReanalysisDataset],
    cfg: WindUpConfig,
    plot_cfg: PlotConfig | None,
    require_full_coverage: bool = False,
) -> pd.DataFrame:
    """Add reanalysis data to the wind farm data.

    :param wf_df: wind farm data
    :param reanalysis_datasets: list of ReanalysisDataset objects
    :param cfg: WindUpConfig object
    :param plot_cfg: PlotConfig object
    :param require_full_coverage: whether to require full coverage
    :return: wind farm and reanalysis data
    """
    data_coverage_width = 0.1
    wf_ws_df = _calc_wf_mean_wind_speed_df(
        wf_df,
        num_turbines=len(cfg.asset.wtgs),
        allowed_data_coverage_width=data_coverage_width,
    )
    max_data_coverage_width = 0.3
    rows_per_hour = 3600 / cfg.timebase_s
    while len(wf_ws_df) < (60 * 24 * rows_per_hour) and data_coverage_width < max_data_coverage_width:
        data_coverage_width += 0.05
        wf_ws_df = _calc_wf_mean_wind_speed_df(
            wf_df,
            num_turbines=len(cfg.asset.wtgs),
            allowed_data_coverage_width=data_coverage_width,
        )

    best_id = None
    best_df = None
    best_corr = -99.0
    best_s = None
    for reanalysis_dataset in reanalysis_datasets:
        _starts_later = reanalysis_dataset.data.index.min() > cfg.lt_first_dt_utc_start
        _ends_earlier = reanalysis_dataset.data.index.max() < cfg.analysis_last_dt_utc_start
        if (_starts_later or _ends_earlier) and require_full_coverage:
            continue  # skip to next dataset

        dsid = reanalysis_dataset.id
        this_reanalysis_df = _reanalysis_upsample(reanalysis_dataset.data, timebase_s=cfg.timebase_s)

        this_corr, this_s = _find_best_shift_and_corr(
            wf_ws_df=wf_ws_df,
            reanalysis_df=this_reanalysis_df,
            wf_name=cfg.asset.name,
            datastream_id=dsid,
            timebase_s=cfg.timebase_s,
            plot_cfg=plot_cfg,
        )
        if this_corr > best_corr:
            best_id = dsid
            best_df = this_reanalysis_df
            best_corr = this_corr
            best_s = this_s

    if best_df is None:
        msg = "no best_id found"
        raise RuntimeError(msg)

    logger.info(f"{best_id} has best correlation: {best_corr:.3f} with a shift of {best_s}")

    wf_and_reanalysis_df = wf_df.merge(
        best_df.shift(best_s)[[REANALYSIS_WS_COL, REANALYSIS_WD_COL]],
        how="left",
        left_index=True,
        right_index=True,
    )
    if plot_cfg is not None:
        plot_wf_and_reanalysis_sample_timeseries(wf_df=wf_and_reanalysis_df, plot_cfg=plot_cfg)
    return wf_and_reanalysis_df
