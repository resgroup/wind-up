"""Discover northing corrections from the data.

A thin v0 adapter over :mod:`wind_up.northing`: this module supplies v0's vocabulary (a
MultiIndex wind-farm frame, a :class:`~wind_up_v0.models.WindUpConfig`, the ``raw_`` column
names) and its reporting -- logging, plots and the corrections YAML -- while the estimation
itself is the shared v1 core.

The two-pass structure is unchanged: north every turbine to reanalysis wind direction, derive
the wind-farm yaw direction from the result, then north every turbine to that.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from wind_up.northing import (
    DEFAULT_NORTHING,
    NORTH_OFFSET_COL,
    NorthingSettings,
    against_reanalysis,
    apply_north_table,
    estimate_north_table,
    yaw_usable,
)
from wind_up_v0.circular_math import circ_diff, rolling_circ_median_approx
from wind_up_v0.constants import (
    RAW_DOWNTIME_S_COL,
    RAW_POWER_COL,
    RAW_YAWDIR_COL,
    REANALYSIS_WD_COL,
    TIMESTAMP_COL,
    WINDFARM_YAWDIR_COL,
)
from wind_up_v0.northing import (
    add_wf_yawdir,
    apply_northing_corrections,
    calc_northed_col_name,
    check_wtg_northing,
)
from wind_up_v0.northing_utils import add_ok_yaw_col
from wind_up_v0.plots.optimize_northing_plots import (
    plot_diff_to_north_ref_wd,
    plot_wf_yawdir_and_reanalysis_timeseries,
    plot_yaw_diff_vs_power,
)

if TYPE_CHECKING:
    from pathlib import Path

    from wind_up_v0.models import PlotConfig, WindUpConfig

logger = logging.getLogger(__name__)

# ``add_wf_yawdir`` falls back to reanalysis wherever it cannot form a farm direction (it needs
# three turbines). Below this share of rows differing from reanalysis, the "farm" reference is
# really reanalysis and must be treated as such.
_MIN_INDEPENDENT_FARM_SHARE = 0.5
# Below this the two directions are the same value, not merely close.
_SAME_DIRECTION_DEG = 1e-6


def _farm_reference_is_independent(wf_df: pd.DataFrame) -> bool:
    """Whether the wind-farm yaw direction is genuinely farm-derived rather than reanalysis.

    A farm consensus shares the site's common-mode direction error, which is what makes small
    steps attributable to a single turbine. Where it has fallen back to reanalysis it carries no
    such information, and the second pass must stay as conservative as the first.
    """
    if WINDFARM_YAWDIR_COL not in wf_df.columns:
        return False
    farm = wf_df[WINDFARM_YAWDIR_COL].to_numpy(dtype=float)
    reanalysis = wf_df[REANALYSIS_WD_COL].to_numpy(dtype=float)
    comparable = np.isfinite(farm) & np.isfinite(reanalysis)
    if not comparable.any():
        return False
    differs = np.abs(circ_diff(farm[comparable], reanalysis[comparable])) > _SAME_DIRECTION_DEG
    return bool(differs.mean() >= _MIN_INDEPENDENT_FARM_SHARE)


def _add_northing_ok_and_diff_cols(wtg_df: pd.DataFrame, *, north_ref_wd_col: str, northed_col: str) -> pd.DataFrame:
    """Add the raw and filtered yaw-minus-reference difference columns the plots read."""
    wtg_df = wtg_df.copy()
    wtg_df[f"yaw_diff_to_{north_ref_wd_col}"] = circ_diff(wtg_df[northed_col], wtg_df[north_ref_wd_col])
    wtg_df[f"filt_diff_to_{north_ref_wd_col}"] = wtg_df[f"yaw_diff_to_{north_ref_wd_col}"]
    wtg_df.loc[~wtg_df[f"ok_for_{north_ref_wd_col}_northing"], f"filt_diff_to_{north_ref_wd_col}"] = pd.NA
    return wtg_df


def _add_northed_ok_diff_and_rolling_cols(
    wtg_df: pd.DataFrame,
    *,
    north_ref_wd_col: str,
    timebase_s: int,
    north_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add the northed yaw column plus the difference and rolling-median columns the plots read."""
    wtg_df = wtg_df.copy()
    northed_col = calc_northed_col_name(north_ref_wd_col)
    index = pd.DatetimeIndex(wtg_df.index)
    raw = wtg_df[RAW_YAWDIR_COL].to_numpy(dtype=float)
    wtg_df[northed_col] = raw if north_table is None else apply_north_table(index, raw, north_table=north_table)

    wtg_df = _add_northing_ok_and_diff_cols(wtg_df, north_ref_wd_col=north_ref_wd_col, northed_col=northed_col)
    rows_per_hour = 3600 / timebase_s
    for label, rolling_hours in (("short", 6), ("long", 15 * 24)):
        window = round(rolling_hours * rows_per_hour)
        wtg_df[f"{label}_rolling_diff_to_{north_ref_wd_col}"] = rolling_circ_median_approx(
            wtg_df[f"filt_diff_to_{north_ref_wd_col}"],
            center=True,
            window=window,
            min_periods=round(window // 3),
            range_360=False,
        )
    return wtg_df


def _wtg_north_table(
    wtg_df: pd.DataFrame,
    *,
    north_ref_wd_col: str,
    rated_power: float,
    timebase_s: int,
    settings: NorthingSettings,
) -> pd.DataFrame:
    """Estimate one turbine's north table against ``north_ref_wd_col``, from its raw yaw."""
    index = pd.DatetimeIndex(wtg_df.index)
    reference = wtg_df[north_ref_wd_col].to_numpy(dtype=float)
    usable = yaw_usable(
        power=wtg_df[RAW_POWER_COL].to_numpy(dtype=float),
        downtime_s=wtg_df[RAW_DOWNTIME_S_COL].to_numpy(dtype=float),
        reference_deg=reference,
        rated_power=rated_power,
        timebase_s=timebase_s,
    )
    return estimate_north_table(
        index,
        wtg_df[RAW_YAWDIR_COL].to_numpy(dtype=float),
        reference_deg=reference,
        usable=usable,
        settings=settings,
    )


def _north_wf_table(
    wf_df: pd.DataFrame,
    *,
    north_ref_wd_col: str,
    cfg: WindUpConfig,
    plot_cfg: PlotConfig | None,
    settings: NorthingSettings = DEFAULT_NORTHING,
) -> pd.DataFrame:
    """Estimate every turbine's north table against ``north_ref_wd_col``, with v0's reporting."""
    wf_north_table = pd.DataFrame()
    for wtg_name in sorted(wf_df.index.unique(level="TurbineName").to_list()):
        wtg_obj = next(x for x in cfg.asset.wtgs if x.name == wtg_name)
        rated_power = wtg_obj.turbine_type.rated_power_kw
        wtg_df = wf_df.loc[wtg_name].copy()

        max_northing_error_before = check_wtg_northing(
            wtg_df, wtg_name=wtg_name, north_ref_wd_col=north_ref_wd_col, timebase_s=cfg.timebase_s, plot_cfg=None
        )

        wtg_north_table = _wtg_north_table(
            wtg_df,
            north_ref_wd_col=north_ref_wd_col,
            rated_power=rated_power,
            timebase_s=cfg.timebase_s,
            settings=settings,
        )

        if plot_cfg is not None:
            wtg_df = add_ok_yaw_col(
                wtg_df,
                new_col_name=f"ok_for_{north_ref_wd_col}_northing",
                wd_col=north_ref_wd_col,
                rated_power=rated_power,
                timebase_s=cfg.timebase_s,
            )
            before_df = _add_northed_ok_diff_and_rolling_cols(
                wtg_df, north_ref_wd_col=north_ref_wd_col, timebase_s=cfg.timebase_s
            )
            plot_yaw_diff_vs_power(before_df, wtg_name=wtg_name, north_ref_wd_col=north_ref_wd_col, plot_cfg=plot_cfg)
            plot_diff_to_north_ref_wd(
                before_df, wtg_name=wtg_name, north_ref_wd_col=north_ref_wd_col, loop_count=0, plot_cfg=plot_cfg
            )
            after_df = _add_northed_ok_diff_and_rolling_cols(
                wtg_df, north_ref_wd_col=north_ref_wd_col, timebase_s=cfg.timebase_s, north_table=wtg_north_table
            )
            plot_diff_to_north_ref_wd(
                after_df, wtg_name=wtg_name, north_ref_wd_col=north_ref_wd_col, loop_count=1, plot_cfg=plot_cfg
            )
            after_df["YawAngleMean"] = after_df[calc_northed_col_name(north_ref_wd_col)]
            max_northing_error_after = check_wtg_northing(
                after_df,
                wtg_name=wtg_name,
                north_ref_wd_col=north_ref_wd_col,
                timebase_s=cfg.timebase_s,
                plot_cfg=plot_cfg,
            )
            logger.info(
                f"{wtg_name} max_northing_error changed from {max_northing_error_before:.1f} to "
                f"{max_northing_error_after:.1f} [{max_northing_error_after - max_northing_error_before:.1f}]",
            )

        logger.info(f"{wtg_name} vs {north_ref_wd_col}: {len(wtg_north_table)} northing period(s)")
        logger.info(f"\n{wtg_north_table=}\n\n")

        wtg_north_table = wtg_north_table.rename(columns={"timestamp": TIMESTAMP_COL})
        wtg_north_table["TurbineName"] = wtg_name
        wf_north_table = (
            pd.concat([wf_north_table, wtg_north_table])
            .sort_values(by=["TurbineName", TIMESTAMP_COL])
            .reset_index(drop=True)
        )
    return wf_north_table


def _write_northing_yaml(wf_north_table: pd.DataFrame, *, fpath: Path) -> None:
    """Write a wind-farm north table as the YAML list ``northing_corrections_utc`` expects."""
    north_table_for_yaml = wf_north_table.copy()
    north_table_for_yaml[TIMESTAMP_COL] = north_table_for_yaml[TIMESTAMP_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    yaml_strings = [
        f"    - ['{row['TurbineName']}', {row[TIMESTAMP_COL]}, {row[NORTH_OFFSET_COL]}]"
        for _, row in north_table_for_yaml.iterrows()
    ]
    with fpath.open(mode="w") as yaml_file:
        yaml_file.write("\n".join(yaml_strings))


def auto_northing_corrections(
    wf_df: pd.DataFrame,
    *,
    cfg: WindUpConfig,
    plot_cfg: PlotConfig | None,
    settings: NorthingSettings = DEFAULT_NORTHING,
) -> pd.DataFrame:
    """Correct the northing of the wind farm to reanalysis data.

    :param wf_df: wind farm SCADA data
    :param cfg: wind farm configuration
    :param plot_cfg: plot configuration
    :param settings: how the changepoint search is bounded; the default suits a farm record
    :return: wind farm SCADA data with corrected northing
    """
    wf_df = wf_df.copy()

    reanalysis_wf_north_table = _north_wf_table(
        wf_df, north_ref_wd_col=REANALYSIS_WD_COL, cfg=cfg, plot_cfg=plot_cfg, settings=against_reanalysis(settings)
    )
    if plot_cfg is not None:
        reanalysis_wf_north_table.to_csv(cfg.out_dir / "reanalysis_wf_north_table.csv")
        _write_northing_yaml(reanalysis_wf_north_table, fpath=cfg.out_dir / "reanalysis_wf_north_table.yaml")

    wf_df = apply_northing_corrections(
        wf_df,
        wf_north_table=reanalysis_wf_north_table,
        north_ref_wd_col=REANALYSIS_WD_COL,
        cfg=cfg,
        plot_cfg=plot_cfg,
    )

    wf_df = add_wf_yawdir(wf_df, cfg=cfg)

    if plot_cfg is not None:
        plot_wf_yawdir_and_reanalysis_timeseries(wf_df, cfg=cfg, plot_cfg=plot_cfg)

    farm_settings = settings if _farm_reference_is_independent(wf_df) else against_reanalysis(settings)
    if farm_settings is not settings:
        logger.info("wind farm yaw direction fell back to reanalysis; northing conservatively")
    optimized_northing_corrections = _north_wf_table(
        wf_df, north_ref_wd_col=WINDFARM_YAWDIR_COL, cfg=cfg, plot_cfg=plot_cfg, settings=farm_settings
    )
    if plot_cfg is not None:
        optimized_northing_corrections.to_csv(cfg.out_dir / "optimized_northing_corrections.csv")
        _write_northing_yaml(optimized_northing_corrections, fpath=cfg.out_dir / "optimized_northing_corrections.yaml")

    return apply_northing_corrections(
        wf_df,
        wf_north_table=optimized_northing_corrections,
        north_ref_wd_col=WINDFARM_YAWDIR_COL,
        cfg=cfg,
        plot_cfg=plot_cfg,
    )
