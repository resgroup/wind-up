"""Helpers for running wind-up analysis."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from wind_up.caching import with_pickle_cache
from wind_up.constants import REANALYSIS_WD_COL
from wind_up.northing import add_wf_yawdir, apply_northing_corrections
from wind_up.optimize_northing import auto_northing_corrections
from wind_up.reanalysis_data import ReanalysisDataset, add_reanalysis_data
from wind_up.scada_funcs import filter_scada_df, get_raw_scada_and_cfg_from_file
from wind_up.scada_power_curve import calc_pc_and_rated_ws
from wind_up.smart_data import add_smart_lat_long_to_cfg
from wind_up.waking_state import add_waking_state
from wind_up.ws_est import add_ws_est

if TYPE_CHECKING:
    from pathlib import Path

    from wind_up.models import PlotConfig, WindUpConfig
logger = logging.getLogger(__name__)


class PrePostSplitter:
    """Class to split wind farm data into pre- and post-analysis periods."""

    def __init__(self, cfg: WindUpConfig, toggle_df: pd.DataFrame | None = None):
        """Initialise PrePostSplitter."""
        self.cfg = cfg
        self.toggle_df = toggle_df

    def split(self, df: pd.DataFrame, test_wtg_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split wind farm data into pre- and post-analysis periods.

        :param df: wind farm SCADA data
        :param test_wtg_name: wind turbine name
        :return:
            tuple of dataframes containing:

                - test-turbine SCADA data
                - pre-analysis SCADA data
                - post-analysis SCADA data
        """
        if (self.cfg.prepost is not None) and self.cfg.toggle is None:
            test_df = df.copy()
            pre_df = df[df.index <= self.cfg.prepost.pre_last_dt_utc_start].copy()
            post_df = df[df.index >= self.cfg.prepost.post_first_dt_utc_start].copy()
        elif (self.cfg.prepost is None) and self.cfg.toggle is not None:
            if not isinstance(self.toggle_df, pd.DataFrame):
                raise ValueError("toggle_df must be a pd.DataFrame")  # noqa TRY003
            test_df = add_toggle_signals(df, toggle_df=self.toggle_df, wtg_name=test_wtg_name, cfg=self.cfg)
            test_df = test_df.rename(columns={"toggle_off": "test_toggle_off", "toggle_on": "test_toggle_on"})
            pre_df = test_df[test_df["test_toggle_off"].fillna(value=False)].copy()
            post_df = test_df[test_df["test_toggle_on"].fillna(value=False)].copy()
        else:
            msg = "cfg must have exactly one of prepost or toggle"
            raise ValueError(msg)
        pre_df = pre_df[pre_df.index >= self.cfg.analysis_first_dt_utc_start]
        post_df = post_df[post_df.index >= self.cfg.analysis_first_dt_utc_start]
        return test_df, pre_df, post_df


def add_toggle_signals(
    input_df: pd.DataFrame, toggle_df: pd.DataFrame, wtg_name: str, cfg: WindUpConfig
) -> pd.DataFrame:
    """Add toggle signals to `input_df` based on `toggle_df`.

    :param input_df: wind farm SCADA data
    :param toggle_df: toggle data
    :param wtg_name: wind turbine name
    :param cfg: wind-up configuration
    :return: `input_df` with toggle signals added
    """
    toggle_df = toggle_df.copy()
    if cfg.toggle is None:
        msg = "add_toggle_signals cannot be run if cfg.toggle is None"
        raise ValueError(msg)
    if (toggle_df["toggle_on"] & toggle_df["toggle_off"]).any():
        msg = "toggle_on and toggle_off cannot be True at the same time"
        raise RuntimeError(msg)
    if toggle_df.index.nlevels == 2:  # noqa PLR2004
        toggle_df = toggle_df.loc[wtg_name]
    if cfg.toggle.toggle_change_settling_filter_seconds > 0:
        # apply toggle_change_settling_filter_seconds
        # first find toggle change times
        toggle_on_rows_before = toggle_df["toggle_on"].sum()
        toggle_off_rows_before = toggle_df["toggle_off"].sum()

        toggle_on_change_times = toggle_df.index[(toggle_df["toggle_on"] & toggle_df.shift(1)["toggle_off"])]
        toggle_off_change_times = toggle_df.index[(toggle_df["toggle_off"] & toggle_df.shift(1)["toggle_on"])]
        toggle_change_times = pd.DatetimeIndex(
            toggle_on_change_times.to_list() + toggle_off_change_times.to_list(),
        ).sort_values()
        rows_to_filter = math.ceil(cfg.toggle.toggle_change_settling_filter_seconds / cfg.timebase_s)
        while rows_to_filter > 0:
            toggle_df.loc[toggle_change_times, "toggle_on"] = False
            toggle_df.loc[toggle_change_times, "toggle_off"] = False
            toggle_change_times = toggle_change_times + pd.Timedelta(seconds=cfg.timebase_s)
            rows_to_filter -= 1

        toggle_on_rows_after = toggle_df["toggle_on"].sum()
        toggle_off_rows_after = toggle_df["toggle_off"].sum()

        logger.info(
            f"changed {toggle_on_rows_before - toggle_on_rows_after} "
            f"[{100 * (toggle_on_rows_before - toggle_on_rows_after) / toggle_on_rows_before:.1f}%] "
            f"rows from toggle_on True to False because toggle_change_settling_filter_seconds "
            f"= {cfg.toggle.toggle_change_settling_filter_seconds}",
        )
        logger.info(
            f"changed {toggle_off_rows_before - toggle_off_rows_after} "
            f"[{100 * (toggle_off_rows_before - toggle_off_rows_after) / toggle_off_rows_before:.1f}%] "
            f"rows from toggle_off True to False because toggle_change_settling_filter_seconds "
            f"= {cfg.toggle.toggle_change_settling_filter_seconds}",
        )

    if isinstance(toggle_df.index, pd.DatetimeIndex):
        return input_df.merge(
            toggle_df[["toggle_off", "toggle_on"]],
            how="left",
            left_index=True,
            right_index=True,
        )
    msg = f"toggle_df.index must be (or can be coerced to) a pd.DatetimeIndex, got {type(toggle_df.index)}"
    raise TypeError(msg)


@dataclass
class AssessmentInputs:
    """Container for inputs to the wind-up assessment."""

    wf_df: pd.DataFrame
    pc_per_ttype: dict[str, pd.DataFrame]
    cfg: WindUpConfig
    plot_cfg: PlotConfig
    pre_post_splitter: PrePostSplitter

    @classmethod
    def from_cfg(
        cls,
        cfg: WindUpConfig,
        plot_cfg: PlotConfig,
        *,
        scada_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        toggle_df: pd.DataFrame | None = None,
        reanalysis_datasets: list[ReanalysisDataset],
        cache_dir: Path | None = None,
    ) -> AssessmentInputs:
        """Construct instance of AssessmentInputs from configuration objects and data.

        :param cfg: wind-up configuration
        :param plot_cfg: plot configuration
        :param scada_df: wind farm SCADA data
        :param metadata_df: wind farm metadata
        :param toggle_df: wind farm toggle data
        :param reanalysis_datasets: reanalysis datasets
        :param cache_dir: directory for caching
        :return: instance of AssessmentInputs
        """
        func = preprocess if cache_dir is None else with_pickle_cache(cache_dir / "preprocess.pickle")(preprocess)
        wf_df, pc_per_ttype = func(
            cfg=cfg,
            plot_cfg=plot_cfg,
            scada_df=scada_df,
            metadata_df=metadata_df,
            reanalysis_datasets=reanalysis_datasets,
        )
        cfg = add_smart_lat_long_to_cfg(md=metadata_df, cfg=cfg)
        pre_post_splitter = PrePostSplitter(cfg=cfg, toggle_df=toggle_df)
        return cls(
            wf_df=wf_df,
            pc_per_ttype=pc_per_ttype,
            cfg=cfg,
            plot_cfg=plot_cfg,
            pre_post_splitter=pre_post_splitter,
        )


def _get_filtered_wf_df_and_cfg_with_latlongs(
    cfg: WindUpConfig,
    plot_cfg: PlotConfig,
    *,
    scada_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    reanalysis_datasets: list[ReanalysisDataset],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], WindUpConfig]:
    wf_df, cfg = get_raw_scada_and_cfg_from_file(
        cfg=cfg,
        scada_df=scada_df,
        metadata_df=metadata_df,
        first_datetime_utc_start=cfg.lt_first_dt_utc_start,
        last_datetime_utc_start=cfg.analysis_last_dt_utc_start,
        plot_cfg=plot_cfg,
    )

    wf_df = filter_scada_df(wf_df, cfg=cfg, plot_cfg=plot_cfg)
    wf_df = add_reanalysis_data(wf_df, cfg=cfg, plot_cfg=plot_cfg, reanalysis_datasets=reanalysis_datasets)
    if cfg.optimize_northing_corrections:
        wf_df = auto_northing_corrections(wf_df, cfg=cfg, plot_cfg=plot_cfg)
    else:
        wf_df = apply_northing_corrections(wf_df, cfg=cfg, north_ref_wd_col=REANALYSIS_WD_COL, plot_cfg=plot_cfg)
    wf_df = add_wf_yawdir(wf_df, cfg=cfg)

    pc_per_ttype, rated_ws_per_ttype = calc_pc_and_rated_ws(
        cfg=cfg,
        wf_df=wf_df,
        x_col="WindSpeedMean",
        y_col="pw_clipped",
        x_bin_width=cfg.ws_bin_width / 2,
        plot_cfg=plot_cfg,
    )

    wf_df = add_ws_est(cfg=cfg, wf_df=wf_df, pc_per_ttype=pc_per_ttype, plot_cfg=plot_cfg)
    wf_df = add_waking_state(cfg=cfg, wf_df=wf_df, plot_cfg=plot_cfg)

    return wf_df, pc_per_ttype, cfg


def preprocess(
    cfg: WindUpConfig,
    plot_cfg: PlotConfig,
    *,
    scada_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    reanalysis_datasets: list[ReanalysisDataset],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Get filtered wind farm data and power curves for a given wind-up configuration.

    Specifically these are mean SCADA power curves rather than power curves suitable for uplift calculation.

    :param cfg: wind-up configuration
    :param plot_cfg: plot configuration
    :param scada_df: wind farm SCADA data
    :param metadata_df: wind farm metadata
    :param reanalysis_datasets: reanalysis datasets
    :return: wind farm SCADA data (post filtering) and per turbine type power curves
    """
    logger.info(f"running wind_up analysis for {cfg.assessment_name}")

    wf_df, pc_per_ttype, cfg = _get_filtered_wf_df_and_cfg_with_latlongs(
        cfg=cfg,
        plot_cfg=plot_cfg,
        scada_df=scada_df,
        metadata_df=metadata_df,
        reanalysis_datasets=reanalysis_datasets,
    )
    if cfg.ignore_turbine_anemometer_data:
        wf_df = wf_df.drop(columns=["ws_est_blend"])

    return wf_df, pc_per_ttype
