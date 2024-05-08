import logging

import numpy as np
import pandas as pd

from wind_up.constants import HOURS_PER_YEAR, RAW_POWER_COL, RAW_WINDSPEED_COL
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.plots.long_term_plots import plot_lt_ws, plot_lt_ws_raw_filt
from wind_up.result_manager import result_manager

logger = logging.getLogger(__name__)


def calc_lt_df(
    df_for_lt: pd.DataFrame,
    *,
    num_turbines: int,
    years_for_lt_distribution: int,
    ws_col: str,
    ws_bin_width: float,
    pw_col: str,
    timebase_s: int,
) -> pd.DataFrame:
    years_of_data = (df_for_lt.index.max() - df_for_lt.index.min()).total_seconds() / 3600 / 24 / 365.25
    if years_of_data < 11.5 / 12:
        result_manager.warning(f"years_of_data for long term is too small: {years_of_data:.1f}")
    max_years_error = 0.1
    if years_for_lt_distribution - years_of_data > max_years_error:
        result_manager.warning(
            "years_of_data for long term is small, "
            f"expected {years_for_lt_distribution:.1f} actual {years_of_data:.1f}",
        )

    if not isinstance(df_for_lt.index, pd.DatetimeIndex):
        msg = f"df_for_lt must have a DatetimeIndex, got {type(df_for_lt.index)}"
        raise TypeError(msg)
    counts_by_month = df_for_lt.index.month.value_counts().to_frame()
    if len(counts_by_month) < 12:  # noqa PLR2004
        result_manager.warning(f"only {len(counts_by_month)} months represented in data for long term")
    counts_by_hour = df_for_lt.index.hour.value_counts().to_frame()
    if len(counts_by_hour) < 24:  # noqa PLR2004
        result_manager.warning(f"only {len(counts_by_hour)} hours represented in data for long term")

    ws_bin_edges = np.arange(0, df_for_lt[ws_col].max() + ws_bin_width, ws_bin_width)

    rows_per_hour = 3600 / timebase_s
    lt_df = (
        df_for_lt.dropna(subset=[ws_col, pw_col])
        .groupby(by=pd.cut(df_for_lt[ws_col], bins=ws_bin_edges, retbins=False), observed=False)
        .agg(
            ws_mean=pd.NamedAgg(column=ws_col, aggfunc=lambda x: x.mean()),
            observed_hours=pd.NamedAgg(column=ws_col, aggfunc=lambda x: len(x) / rows_per_hour),
            observed_mwh=pd.NamedAgg(column=pw_col, aggfunc=lambda x: sum(x) / rows_per_hour / 1000),
        )
    )
    lt_df = lt_df.rename(columns={"ws_mean": f"{ws_col}_mean"})
    lt_df["bin_left"] = [x.left for x in lt_df.index]
    lt_df["bin_mid"] = [x.mid for x in lt_df.index]
    lt_df["bin_right"] = [x.right for x in lt_df.index]
    lt_df["bin_closed_right"] = [x.closed_right for x in lt_df.index]

    lt_df = lt_df.set_index("bin_mid", drop=False, verify_integrity=True)
    lt_df.index.name = f"{ws_col}_bin_mid"

    lt_df[f"{ws_col}_mean"] = lt_df[f"{ws_col}_mean"].fillna(lt_df["bin_mid"])

    lt_df["fraction_of_time"] = lt_df["observed_hours"] / lt_df["observed_hours"].sum()
    if years_for_lt_distribution > 0:
        data_coverage = lt_df["observed_hours"].sum() / (years_for_lt_distribution * HOURS_PER_YEAR) / num_turbines
    else:
        data_coverage = lt_df["observed_hours"].sum() / (years_of_data * HOURS_PER_YEAR) / num_turbines
        result_manager.warning("years_for_lt_distribution is 0, using years_of_data instead to calculate data coverage")
    if data_coverage > 1:
        msg = f"Data coverage for long term ws distribution is >1, {data_coverage:.2f}"
        raise RuntimeError(msg)
    lt_df["hours_per_year"] = lt_df["fraction_of_time"] * HOURS_PER_YEAR * data_coverage
    lt_df["mwh_per_year_per_turbine"] = (
        lt_df["hours_per_year"] * lt_df["observed_mwh"] / lt_df["observed_hours"]
    ).fillna(0)

    if lt_df.isna().any().any():
        msg = "lt_df has missing values"
        raise RuntimeError(msg)

    logger.info(f"long term distribution uses data from {df_for_lt.index.min()} to {df_for_lt.index.max()}")
    logger.info(f"long term distribution data coverage: {data_coverage*100:.1f}%")

    return lt_df


def calc_turbine_lt_df(
    wtg_name: str,
    cfg: WindUpConfig,
    wtg_df: pd.DataFrame,
    *,
    ws_col: str,
    pw_col: str,
    title_end: str = "",
    plot_cfg: PlotConfig | None = None,
) -> pd.DataFrame:
    workings_df = wtg_df.copy()
    if not isinstance(workings_df.index, pd.DatetimeIndex):
        msg = f"wtg_df must have a DatetimeIndex, got {type(workings_df.index)}"
        raise TypeError(msg)

    ok_for_lt = (workings_df.index >= cfg.lt_first_dt_utc_start) & (workings_df.index <= cfg.lt_last_dt_utc_start)

    lt_df = calc_lt_df(
        df_for_lt=workings_df[ok_for_lt],
        num_turbines=1,
        years_for_lt_distribution=cfg.years_for_lt_distribution,
        ws_col=ws_col,
        ws_bin_width=cfg.ws_bin_width,
        pw_col=pw_col,
        timebase_s=cfg.timebase_s,
    )
    if plot_cfg is not None:
        plot_lt_ws(lt_df=lt_df, turbine_or_wf_name=wtg_name, title_end=title_end, plot_cfg=plot_cfg, one_turbine=True)

    return lt_df


def calc_turbine_lt_dfs_raw_filt(
    wtg_name: str,
    cfg: WindUpConfig,
    wtg_df: pd.DataFrame,
    *,
    ws_col: str,
    pw_col: str,
    plot_cfg: PlotConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lt_wf_df_raw = calc_turbine_lt_df(
        wtg_name=wtg_name,
        cfg=cfg,
        wtg_df=wtg_df,
        ws_col=RAW_WINDSPEED_COL,
        pw_col=RAW_POWER_COL,
        title_end="before filter",
        plot_cfg=plot_cfg,
    )
    lt_wf_df_filt = calc_turbine_lt_df(
        wtg_name=wtg_name,
        cfg=cfg,
        wtg_df=wtg_df,
        ws_col=ws_col,
        pw_col=pw_col,
        title_end="after filter",
        plot_cfg=plot_cfg,
    )
    if plot_cfg is not None:
        plot_lt_ws_raw_filt(
            lt_df_raw=lt_wf_df_raw,
            lt_df_filt=lt_wf_df_filt,
            turbine_or_wf_name=wtg_name,
            plot_cfg=plot_cfg,
            one_turbine=True,
        )

    return lt_wf_df_raw, lt_wf_df_filt


def calc_windfarm_lt_df(
    cfg: WindUpConfig,
    wf_df: pd.DataFrame,
    *,
    ws_col: str,
    pw_col: str,
    title_end: str = "",
    plot_cfg: PlotConfig | None = None,
) -> pd.DataFrame:
    workings_df = wf_df.copy()
    if len(workings_df.index.levels) == 2:  # noqa PLR2004
        workings_df.index = workings_df.index.droplevel("TurbineName")

    ok_for_lt = (workings_df.index >= cfg.lt_first_dt_utc_start) & (workings_df.index <= cfg.lt_last_dt_utc_start)

    lt_df = calc_lt_df(
        df_for_lt=workings_df[ok_for_lt],
        num_turbines=len(cfg.asset.wtgs),
        years_for_lt_distribution=cfg.years_for_lt_distribution,
        ws_col=ws_col,
        ws_bin_width=cfg.ws_bin_width,
        pw_col=pw_col,
        timebase_s=cfg.timebase_s,
    )

    if plot_cfg is not None:
        plot_lt_ws(lt_df=lt_df, turbine_or_wf_name=cfg.asset.name, title_end=title_end, plot_cfg=plot_cfg)

    return lt_df


def calc_windfarm_lt_dfs_raw_filt(
    cfg: WindUpConfig,
    wf_df_raw: pd.DataFrame,
    wf_df_filt: pd.DataFrame,
    *,
    ws_col: str,
    pw_col: str,
    plot_cfg: PlotConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lt_wf_df_raw = calc_windfarm_lt_df(
        cfg=cfg,
        wf_df=wf_df_raw,
        ws_col=ws_col,
        pw_col=pw_col,
        title_end="before filter",
        plot_cfg=plot_cfg,
    )
    lt_wf_df_filt = calc_windfarm_lt_df(
        cfg=cfg,
        wf_df=wf_df_filt,
        ws_col=ws_col,
        pw_col=pw_col,
        title_end="after filter",
        plot_cfg=plot_cfg,
    )
    if plot_cfg is not None:
        plot_lt_ws_raw_filt(
            lt_df_raw=lt_wf_df_raw,
            lt_df_filt=lt_wf_df_filt,
            turbine_or_wf_name=cfg.asset.name,
            plot_cfg=plot_cfg,
        )

    return lt_wf_df_raw, lt_wf_df_filt
