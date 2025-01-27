"""Module for calculating wind speed drift."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wind_up.plots.windspeed_drift_plots import plot_rolling_windspeed_diff_one_wtg
from wind_up.result_manager import result_manager

if TYPE_CHECKING:
    import pandas as pd

    from wind_up.models import PlotConfig, WindUpConfig


def _calculate_rolling_windspeed_diff(
    wtg_df: pd.DataFrame,
    *,
    ws_col: str,
    reanalysis_ws_col: str,
    timebase_s: int,
    ws_ll: float = 6,
    ws_ul: float = 15,
    rolling_period: float = 90,
    min_roll_days: float = 14,
    min_rolling_coverage: float = 1 / 3,
) -> pd.Series:
    ws_diff_to_renalysis = wtg_df[ws_col] - wtg_df[reanalysis_ws_col]
    ws_diff_to_renalysis.loc[(wtg_df[ws_col] < ws_ll) | (wtg_df[ws_col] > ws_ul)] = np.nan

    rows_per_day = 24 * 3600 / timebase_s

    def _rolling_specs(rolling_period: float) -> dict[str, int]:
        return {
            "window": round(rolling_period * rows_per_day),
            "min_periods": round(rolling_period * rows_per_day * min_rolling_coverage),
        }

    rolling_windspeed_diff = ws_diff_to_renalysis.rolling(**_rolling_specs(rolling_period)).median()

    while rolling_period >= (min_roll_days * 2) and len(rolling_windspeed_diff.dropna()) == 0:
        rolling_period = rolling_period // 2
        rolling_windspeed_diff = ws_diff_to_renalysis.rolling(**_rolling_specs(rolling_period)).median()

    if len(rolling_windspeed_diff.dropna()) == 0:
        result_manager.warning("could not calculate rolling windspeed diff")

    return rolling_windspeed_diff


def calc_max_abs_relative_rolling_windspeed_diff(ser: pd.Series) -> float:
    """Calculate the maximum absolute relative wind speed difference."""
    return (ser - ser.median()).abs().max()


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
    """Calculate wind speed drift.

    :param wtg_df: time series of turbine measurement data
    :param wtg_name: name of the turbine
    :param ws_col: wind speed column name in `wtg_df`
    :param reanalysis_ws_col: reanalysis wind speed column name in `wtg_df`
    :param cfg: wind up configuration
    :param plot_cfg: plot configuration
    :param sub_dir: subdirectory in which to save plots
    :return:
        tuple containing

            - maximum absolute relative wind speed difference
            - maximum absolute relative wind speed drift during power performance period
    """
    rolling_windspeed_diff = _calculate_rolling_windspeed_diff(
        wtg_df, ws_col=ws_col, reanalysis_ws_col=reanalysis_ws_col, timebase_s=cfg.timebase_s
    )

    if plot_cfg is not None:
        plot_rolling_windspeed_diff_one_wtg(
            ser=rolling_windspeed_diff, wtg_name=wtg_name, ws_col=ws_col, plot_cfg=plot_cfg, sub_dir=sub_dir
        )

    max_abs_rel_diff = calc_max_abs_relative_rolling_windspeed_diff(rolling_windspeed_diff)
    max_abs_rel_diff_pp_period = calc_max_abs_relative_rolling_windspeed_diff(
        rolling_windspeed_diff.loc[cfg.analysis_first_dt_utc_start : cfg.analysis_last_dt_utc_start],  # type: ignore[misc]
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
