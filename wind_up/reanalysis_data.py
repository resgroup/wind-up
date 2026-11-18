"""Module to handle reanalysis data."""

from __future__ import annotations

import importlib.util
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
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
    elif "wind_speed_100m" in reanalysis_df.columns and "wind_direction_100m" in reanalysis_df.columns:
        reanalysis_df = reanalysis_df.rename(
            columns={"wind_speed_100m": REANALYSIS_WS_COL, "wind_direction_100m": REANALYSIS_WD_COL},
        )
    else:
        msg = "reanalysis wind speed and direction columns not found"
        raise RuntimeError(msg)
    reanalysis_df.index.name = TIMESTAMP_COL
    return reanalysis_df


def _corr_with_shifted_reanalysis(ws_df: pd.DataFrame, reanalysis_ws: pd.Series, *, shifts: list[int]) -> list[float]:
    """Correlate the single column of `ws_df` with `reanalysis_ws` shifted by each number of rows in `shifts`."""
    ws = ws_df.iloc[:, 0]
    if ws_df.shape[1] != 1 or not all(
        x.is_unique and x.is_monotonic_increasing for x in (ws_df.index, reanalysis_ws.index)
    ):
        return [float(ws_df.corrwith(reanalysis_ws.shift(s)).squeeze()) for s in shifts]

    # every shift pairs the same timestamps, so they are paired once and the shift is applied by row position
    _, ws_paired = reanalysis_ws.align(ws, join="inner")
    reanalysis_rows = reanalysis_ws.index.get_indexer(ws_paired.index)
    reanalysis_values = reanalysis_ws.to_numpy(dtype=float, na_value=np.nan)
    ws_paired = pd.Series(ws_paired.to_numpy(dtype=float, na_value=np.nan))
    corrs = []
    for s in shifts:
        shifted_rows = reanalysis_rows - s
        in_range = (shifted_rows >= 0) & (shifted_rows < len(reanalysis_values))
        shifted_values = np.full(len(shifted_rows), np.nan)
        shifted_values[in_range] = reanalysis_values[shifted_rows[in_range]]
        corrs.append(float(pd.Series(shifted_values).corr(ws_paired)))
    return corrs


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
    rows_per_hour = 3600 / timebase_s
    shifts = list(range(round(-24 * rows_per_hour), round(24 * rows_per_hour), math.ceil(rows_per_hour / 6)))
    corrs = _corr_with_shifted_reanalysis(ws_filt_df, reanalysis_df[REANALYSIS_WS_COL], shifts=shifts)
    for s, this_corr in zip(shifts, corrs, strict=True):
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


def _wind_speed_by_timestamp_and_turbine(wf_df: pd.DataFrame) -> tuple[pd.Index, np.ndarray] | None:
    """Arrange the wind speed of `wf_df` with a row per timestamp and a column per turbine.

    :return: the timestamps and the wind speed array, or None unless `wf_df` is sorted by turbine and has the same
        sorted timestamps for every turbine
    """
    index = wf_df.index
    if (
        not isinstance(index, pd.MultiIndex)
        or index.nlevels != 2  # noqa: PLR2004
        or index.names[1] != TIMESTAMP_COL
        or len(wf_df) == 0
        or wf_df["WindSpeedMean"].dtype != np.float64
    ):
        return None
    turbine_codes, timestamp_codes = (np.asarray(x) for x in index.codes)
    num_timestamps = int((turbine_codes == turbine_codes[0]).sum())
    if len(wf_df) % num_timestamps != 0:
        return None
    turbine_codes = turbine_codes.reshape(-1, num_timestamps)
    timestamp_codes = timestamp_codes.reshape(-1, num_timestamps)
    timestamps = index.levels[1].take(timestamp_codes[0])
    if (
        (turbine_codes != turbine_codes[:, [0]]).any()
        or len(np.unique(turbine_codes[:, 0])) != len(turbine_codes)
        or (timestamp_codes != timestamp_codes[0]).any()
        or not (timestamps.is_unique and timestamps.is_monotonic_increasing)
    ):
        return None
    wind_speed = wf_df["WindSpeedMean"].to_numpy().reshape(-1, num_timestamps)
    if isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps, freq=None)
    return timestamps.rename(TIMESTAMP_COL), np.array(wind_speed.T, order="C")


def _calc_wf_mean_wind_speed_and_data_coverage(wf_df: pd.DataFrame, *, num_turbines: int) -> pd.DataFrame:
    """Calculate the mean wind speed of the turbines and the fraction of turbines with a wind speed, by timestamp."""
    pandas_sums_with_numpy = importlib.util.find_spec("bottleneck") is None or not pd.get_option(
        "compute.use_bottleneck"
    )
    by_timestamp_and_turbine = _wind_speed_by_timestamp_and_turbine(wf_df) if pandas_sums_with_numpy else None
    if by_timestamp_and_turbine is None:
        return wf_df.groupby(TIMESTAMP_COL).agg(
            WindSpeedMean=pd.NamedAgg(column="WindSpeedMean", aggfunc=lambda x: x.mean()),
            data_coverage=pd.NamedAgg(column="WindSpeedMean", aggfunc=lambda x: x.count() / num_turbines),
        )
    timestamps, wind_speed = by_timestamp_and_turbine
    is_missing = np.isnan(wind_speed)
    count = (~is_missing).sum(axis=1)
    # missing values count as zero in the sum of each row
    wind_speed[is_missing] = 0
    wind_speed_sum = wind_speed.sum(axis=1)
    wind_speed_mean = np.full(len(timestamps), np.nan)
    np.divide(wind_speed_sum, count, out=wind_speed_mean, where=count > 0)
    return pd.DataFrame({"WindSpeedMean": wind_speed_mean, "data_coverage": count / num_turbines}, index=timestamps)


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
    wf_ws_df = _calc_wf_mean_wind_speed_and_data_coverage(wf_df, num_turbines=num_turbines)
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


@dataclass
class MastOrLiDARDataset:
    """Class to store mast or LiDAR data."""

    id: str
    data: pd.DataFrame
