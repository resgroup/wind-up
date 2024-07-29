import numpy as np
import pandas as pd

from wind_up.models import PlotConfig, WindUpConfig
from wind_up.plots.windspeed_drift_plots import plot_rolling_windspeed_diff_one_wtg
from wind_up.result_manager import result_manager


def add_rolling_windspeed_diff(
    wtg_df: pd.DataFrame, *, ws_col: str, reanalysis_ws_col: str, timebase_s: int
) -> pd.DataFrame:
    wtg_df = wtg_df.copy()

    # check for ws drift issue
    wtg_df["ws_diff_to_renalysis"] = wtg_df[ws_col] - wtg_df[reanalysis_ws_col]
    ws_ll = 6
    ws_ul = 15
    wtg_df.loc[wtg_df[ws_col] < ws_ll, "ws_diff_to_renalysis"] = np.nan
    wtg_df.loc[wtg_df[ws_col] > ws_ul, "ws_diff_to_renalysis"] = np.nan
    rolling_days = 90
    rows_per_day = 24 * 3600 / timebase_s
    wtg_df["rolling_windspeed_diff"] = (
        wtg_df["ws_diff_to_renalysis"]
        .rolling(window=round(rolling_days * rows_per_day), min_periods=round(rolling_days * rows_per_day // 3))
        .median()
    )
    min_roll_days = 14
    while rolling_days >= (min_roll_days * 2) and len(wtg_df["rolling_windspeed_diff"].dropna()) == 0:
        rolling_days = rolling_days // 2
        wtg_df["rolling_windspeed_diff"] = (
            wtg_df["ws_diff_to_renalysis"]
            .rolling(window=round(rolling_days * rows_per_day), min_periods=round(rolling_days * rows_per_day // 3))
            .median()
        )
    if len(wtg_df["rolling_windspeed_diff"].dropna()) == 0:
        result_manager.warning("could not calculate rolling windspeed diff")
    return wtg_df


def calc_max_abs_relative_rolling_windspeed_diff(wtg_df: pd.DataFrame) -> float:
    return (wtg_df["rolling_windspeed_diff"] - wtg_df["rolling_windspeed_diff"].median()).abs().max()


def check_windspeed_drift(
    *,
    wtg_df: pd.DataFrame,
    wtg_name: str,
    ws_col: str,
    reanalysis_ws_col: str,
    cfg: WindUpConfig,
    plot_cfg: PlotConfig | None,
    sub_dir: str | None = None,
) -> tuple[float, float]:
    wtg_df = wtg_df.copy()
    wtg_df = add_rolling_windspeed_diff(
        wtg_df, ws_col=ws_col, reanalysis_ws_col=reanalysis_ws_col, timebase_s=cfg.timebase_s
    )
    if plot_cfg is not None:
        plot_rolling_windspeed_diff_one_wtg(
            wtg_df=wtg_df, wtg_name=wtg_name, ws_col=ws_col, plot_cfg=plot_cfg, sub_dir=sub_dir
        )

    max_abs_rel_diff = calc_max_abs_relative_rolling_windspeed_diff(wtg_df)
    max_abs_rel_diff_pp_period = calc_max_abs_relative_rolling_windspeed_diff(
        wtg_df.loc[cfg.analysis_first_dt_utc_start : cfg.analysis_last_dt_utc_start],  # type: ignore[misc]
    )

    ws_diff_ul = 1
    if max_abs_rel_diff > ws_diff_ul:
        result_manager.warning(f"possible wind speed drift of {max_abs_rel_diff:.1f} m/s for {wtg_name}")
    if max_abs_rel_diff_pp_period > ws_diff_ul:
        result_manager.warning(
            f"possible wind speed drift of {max_abs_rel_diff_pp_period:.1f} m/s for {wtg_name} "
            f"DURING POWER PERFORMANCE PERIOD",
        )

    return max_abs_rel_diff, max_abs_rel_diff_pp_period
