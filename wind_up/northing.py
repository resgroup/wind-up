import logging

import numpy as np
import pandas as pd
from scipy.stats import circmean

from wind_up.constants import (
    RAW_YAWDIR_COL,
    REANALYSIS_WD_COL,
    TIMESTAMP_COL,
    WINDFARM_YAWDIR_COL,
)
from wind_up.math_funcs import circ_diff
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.northing_utils import add_ok_yaw_col
from wind_up.plots.northing_plots import plot_and_print_northing_error, plot_northing_changepoint, plot_northing_error

logger = logging.getLogger(__name__)


def add_rolling_northing_error(wf_df: pd.DataFrame, *, north_ref_wd_col: str, timebase_s: int) -> pd.DataFrame:
    wf_df = wf_df.copy()
    wf_df["apparent_northing_error"] = circ_diff(wf_df["YawAngleMean"], wf_df[north_ref_wd_col])
    ws_ll = 6
    wf_df.loc[~(wf_df["WindSpeedMean"] >= ws_ll), "apparent_northing_error"] = np.nan
    rolling_days = 20
    rows_per_day = 24 * 3600 / timebase_s
    wf_df["rolling_northing_error"] = (
        wf_df["apparent_northing_error"]
        .rolling(window=round(rolling_days * rows_per_day), min_periods=round(rolling_days * rows_per_day // 3))
        .median()
    )
    return wf_df


def calc_max_abs_north_errs(wf_df: pd.DataFrame, *, north_ref_wd_col: str, timebase_s: int) -> pd.DataFrame:
    wf_df = add_rolling_northing_error(wf_df.copy(), north_ref_wd_col=north_ref_wd_col, timebase_s=timebase_s)
    return wf_df.groupby("TurbineName", observed=True)["rolling_northing_error"].agg(lambda x: x.abs().max())


def calc_northed_col_name(north_ref_wd_col: str) -> str:
    return f"yawdir_northed_to_{north_ref_wd_col}"


def apply_northing_corrections(
    wf_df: pd.DataFrame,
    *,
    cfg: WindUpConfig,
    north_ref_wd_col: str,
    plot_cfg: PlotConfig | None,
    wf_north_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    wf_df = wf_df.copy()
    northed_col = calc_northed_col_name(north_ref_wd_col)

    if plot_cfg is not None:
        plot_and_print_northing_error(
            add_rolling_northing_error(wf_df, north_ref_wd_col=north_ref_wd_col, timebase_s=cfg.timebase_s),
            cfg=cfg,
            abs_north_errs=calc_max_abs_north_errs(wf_df, north_ref_wd_col=north_ref_wd_col, timebase_s=cfg.timebase_s),
            title_end=f"vs {north_ref_wd_col} before northing",
            plot_cfg=plot_cfg,
        )

    if wf_north_table is None:
        wf_north_table = pd.DataFrame(
            data=cfg.northing_corrections_utc,
            columns=["TurbineName", TIMESTAMP_COL, "north_offset"],
        )

    len_corrs = len(wf_north_table)
    wf_df[northed_col] = wf_df[RAW_YAWDIR_COL]
    if len_corrs == 0:
        if plot_cfg is not None:
            logger.info("no northing corrections to apply")
    else:
        for nc in wf_north_table.sort_values(by=["TurbineName", TIMESTAMP_COL]).itertuples():
            northing_turbine = nc.TurbineName
            northing_datetime_utc = getattr(nc, TIMESTAMP_COL)
            northing_offset = nc.north_offset
            df_idx = pd.IndexSlice[northing_turbine, northing_datetime_utc:]
            wf_df.loc[df_idx, northed_col] = (wf_df.loc[df_idx, RAW_YAWDIR_COL] + northing_offset) % 360
            if plot_cfg is not None:
                plot_northing_changepoint(
                    wf_df,
                    northing_turbine=northing_turbine,
                    northed_col=northed_col,
                    north_ref_wd_col=north_ref_wd_col,
                    northing_datetime_utc=northing_datetime_utc,
                    cfg=cfg,
                    plot_cfg=plot_cfg,
                )
            if "YawAngleMin" in wf_df.columns:
                wf_df.loc[df_idx, "YawAngleMin"] = pd.NA
            if "YawAngleMax" in wf_df.columns:
                wf_df.loc[df_idx, "YawAngleMax"] = pd.NA
    logger.info(f"applied {len_corrs} northing corrections")
    wf_df["YawAngleMean"] = wf_df[northed_col]
    if plot_cfg is not None:
        plot_and_print_northing_error(
            add_rolling_northing_error(wf_df, north_ref_wd_col=north_ref_wd_col, timebase_s=cfg.timebase_s),
            cfg=cfg,
            abs_north_errs=calc_max_abs_north_errs(wf_df, north_ref_wd_col=north_ref_wd_col, timebase_s=cfg.timebase_s),
            title_end=f"vs {north_ref_wd_col} after northing",
            plot_cfg=plot_cfg,
        )
    return wf_df


def format_wtg_df_like_wf_df(wtg_df: pd.DataFrame, *, wtg_name: str) -> pd.DataFrame:
    wtg_df = wtg_df.copy()
    wtg_df["TurbineName"] = wtg_name
    return wtg_df.set_index(["TurbineName", wtg_df.index])


def check_wtg_northing(
    wtg_df: pd.DataFrame,
    *,
    wtg_name: str,
    north_ref_wd_col: str,
    timebase_s: int,
    plot_cfg: PlotConfig | None,
    sub_dir: str | None = None,
) -> float:
    wtg_wf_df = format_wtg_df_like_wf_df(wtg_df, wtg_name=wtg_name)
    max_northing_error = calc_max_abs_north_errs(
        wtg_wf_df, north_ref_wd_col=north_ref_wd_col, timebase_s=timebase_s
    ).squeeze()
    if plot_cfg is not None:
        plot_northing_error(
            wf_df=add_rolling_northing_error(wtg_wf_df, north_ref_wd_col=north_ref_wd_col, timebase_s=timebase_s),
            title_end=f"{wtg_name} vs {north_ref_wd_col}",
            plot_cfg=plot_cfg,
            sub_dir=wtg_name if sub_dir is None else sub_dir,
        )
    return max_northing_error


def calc_wf_yawdir_df(
    wf_df: pd.DataFrame,
    *,
    best_yaw_dir_col: str,
    min_num_turbines: int,
    cfg: WindUpConfig,
) -> pd.DataFrame:
    wf_df = wf_df.copy()
    wf_df = add_ok_yaw_col(
        wf_df,
        new_col_name="ok_for_yawdir",
        wd_col=best_yaw_dir_col,
        rated_power=cfg.get_max_rated_power(),
        timebase_s=cfg.timebase_s,
    )
    wf_df = wf_df.loc[wf_df["ok_for_yawdir"]].dropna(subset=[best_yaw_dir_col])
    wf_yawdir_df = wf_df.groupby(TIMESTAMP_COL).agg(
        wf_yawdir=pd.NamedAgg(
            column=best_yaw_dir_col,
            aggfunc=lambda x: (
                ((x - circmean(x, low=0, high=360, nan_policy="omit") + 180) % 360).median(skipna=True)
                + circmean(x, low=0, high=360, nan_policy="omit")
                - 180
            )
            % 360,
        ),
        num_turbines=pd.NamedAgg(column=best_yaw_dir_col, aggfunc=lambda x: x.count()),
    )
    wf_yawdir_df = wf_yawdir_df[wf_yawdir_df["num_turbines"] >= min_num_turbines]
    return wf_yawdir_df.dropna()


def add_wf_yawdir(wf_df: pd.DataFrame, *, cfg: WindUpConfig) -> pd.DataFrame:
    wf_yawdir_df = calc_wf_yawdir_df(
        wf_df,
        best_yaw_dir_col=calc_northed_col_name(REANALYSIS_WD_COL),
        min_num_turbines=3,
        cfg=cfg,
    )

    wf_df_with_wf_yawdir = wf_df.drop(columns=WINDFARM_YAWDIR_COL, errors="ignore").merge(
        wf_yawdir_df[[WINDFARM_YAWDIR_COL]],
        how="left",
        left_index=True,
        right_index=True,
    )
    wf_df_with_wf_yawdir[WINDFARM_YAWDIR_COL] = wf_df_with_wf_yawdir[WINDFARM_YAWDIR_COL].fillna(
        wf_df_with_wf_yawdir[REANALYSIS_WD_COL],
    )
    return wf_df_with_wf_yawdir
