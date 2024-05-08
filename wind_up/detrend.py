import logging

import numpy as np
import pandas as pd

from wind_up.math_funcs import circ_diff
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.plots.detrend_plots import (
    plot_check_applied_detrend,
    plot_detrend_ws_scatter,
    plot_detrend_wsratio_v_dir_scen,
)
from wind_up.result_manager import result_manager
from wind_up.waking_state import get_iec_upwind_turbines, lat_long_is_valid, list_wtgs_offline_in_scen

logger = logging.getLogger(__name__)


def calc_wsratio_v_wd(
    *,
    detrend_df: pd.DataFrame,
    test_ws_col: str,
    ref_ws_col: str,
    ref_wd_col: str,
    min_hours: int,
    timebase_s: int,
    dir_bin_width: float = 10.0,
) -> pd.DataFrame:
    detrend_df = detrend_df.dropna(subset=[test_ws_col, ref_ws_col, ref_wd_col]).copy()

    # IEC says only use 4-16 m/s
    test_ws_ll = 4
    test_ws_ul = 16
    ref_ws_ll = test_ws_ll * detrend_df[ref_ws_col].mean() / detrend_df[test_ws_col].mean()
    ref_ws_ul = test_ws_ul * detrend_df[ref_ws_col].mean() / detrend_df[test_ws_col].mean()
    detrend_df = detrend_df[(detrend_df[test_ws_col] >= test_ws_ll) & (detrend_df[test_ws_col] < test_ws_ul)]
    detrend_df = detrend_df[(detrend_df[ref_ws_col] >= ref_ws_ll) & (detrend_df[ref_ws_col] < ref_ws_ul)]

    directions = []
    hours = []
    test_rf_ws_roms = []
    for d in list(range(0, 360, 1)):
        detrend_df["circ_diff_to_d"] = circ_diff(detrend_df[ref_wd_col], d)
        detrend_df["within_dir_bin"] = detrend_df["circ_diff_to_d"].abs() < dir_bin_width / 2
        subsector_df = detrend_df[detrend_df["within_dir_bin"]].copy()
        if len(subsector_df) > 0:
            directions.append(d)
            rows_per_hour = 3600 / timebase_s
            hours.append(len(subsector_df) / rows_per_hour)
            # 61400-12-1 requires >=24h data, >=6h above 8m/s, >= below 8m/s
            min_count = min_hours * rows_per_hour
            accept_sector = len(subsector_df) >= min_count
            iec_ws_threshold = 8
            accept_sector = accept_sector and ((subsector_df[test_ws_col] < iec_ws_threshold).sum() >= (min_count / 4))
            accept_sector = accept_sector and ((subsector_df[test_ws_col] >= iec_ws_threshold).sum() >= (min_count / 4))
            if accept_sector:
                rom = subsector_df[test_ws_col].mean() / subsector_df[ref_ws_col].mean()
                test_rf_ws_roms.append(rom)
            else:
                test_rf_ws_roms.append(np.nan)

    return pd.DataFrame(
        {
            "direction": directions,
            "hours": hours,
            "ws_rom": test_rf_ws_roms,
        },
    )


def apply_wsratio_v_wd(
    p_df: pd.DataFrame,
    wsratio_v_dir: pd.DataFrame,
    *,
    ref_ws_col: str,
    ref_wd_col: str,
) -> pd.DataFrame:
    p_df = p_df.copy()
    wsratio_v_dir = wsratio_v_dir.copy()
    wsratio_v_dir["direction"] = wsratio_v_dir.index
    # add a 360 row to wsratio_v_dir
    rows_w_0_dir = wsratio_v_dir[wsratio_v_dir["direction"] == 0].copy()
    rows_w_0_dir["direction"] = 360
    wsratio_v_dir = pd.concat([wsratio_v_dir, rows_w_0_dir]).reset_index(drop=True)

    p_df["ws_rom"] = np.interp(p_df[ref_wd_col].to_numpy(), wsratio_v_dir["direction"], wsratio_v_dir["ws_rom"])

    p_df["ref_ws_detrended"] = p_df[ref_ws_col] * p_df["ws_rom"]

    return p_df


def remove_bad_detrend_results(
    wsratio_v_dir_scen: pd.DataFrame,
    test_name: str,
    ref_name: str,
    ref_lat: float,
    ref_long: float,
    cfg: WindUpConfig,
) -> pd.DataFrame:
    try:
        none_offline_df = wsratio_v_dir_scen.dropna(subset="ws_rom").loc["none offline"]
    except KeyError:
        result_manager.warning("cannot remove_bad_detrend_results, no 'none offline' rows with ws_rom defined")
        return wsratio_v_dir_scen

    try:
        test_wtg = next(x for x in cfg.asset.wtgs if x.name == test_name)
    except StopIteration as exc:
        msg = f"{test_name} not found in cfg.asset.wtgs"
        raise ValueError(msg) from exc

    remove_count = 0
    for wd in none_offline_df.index.unique():
        test_upwind_wtgs = get_iec_upwind_turbines(
            latlongs=test_wtg.get_latlongs(),
            wind_direction=wd,
            cfg=cfg,
            object_name=test_name,
        )
        if lat_long_is_valid(ref_lat, ref_long):
            ref_upwind_wtgs = get_iec_upwind_turbines(
                latlongs=[(ref_lat, ref_long)],
                wind_direction=wd,
                cfg=cfg,
                object_name=ref_name,
            )
        else:
            ref_upwind_wtgs = []

        scens_to_check = [
            x
            for x in wsratio_v_dir_scen.dropna(subset="ws_rom")
            .loc[pd.IndexSlice[:, wd], :]
            .index.unique(level="waking_scenario")
            if x != "none offline"
        ]
        for scen in scens_to_check:
            upwind_wtgs_offline = list_wtgs_offline_in_scen(scen)
            test_upwind_wtgs_offline = sorted(set(test_upwind_wtgs) & set(upwind_wtgs_offline))
            ref_upwind_wtgs_offline = sorted(set(ref_upwind_wtgs) & set(upwind_wtgs_offline))
            if (len(test_upwind_wtgs_offline) + len(ref_upwind_wtgs_offline) > 0) and (
                len(test_upwind_wtgs_offline) == 0 or len(ref_upwind_wtgs_offline) == 0
            ):
                sign = 1 if len(ref_upwind_wtgs_offline) == 0 else -1
                scen_ws_rom = wsratio_v_dir_scen.loc[pd.IndexSlice[scen, wd], "ws_rom"]
                none_offline_ws_rom = none_offline_df.loc[wd, "ws_rom"]
                abs_tol = 1e-6
                if sign * scen_ws_rom < (sign * none_offline_ws_rom - abs_tol):
                    wsratio_v_dir_scen.loc[pd.IndexSlice[scen, wd], "ws_rom"] = np.nan
                    remove_count += 1

    logger.info(f"removed {remove_count} bad detrend results")
    return wsratio_v_dir_scen


def calc_wsratio_v_wd_scen(
    *,
    test_name: str,
    ref_name: str,
    ref_lat: float,
    ref_long: float,
    detrend_df: pd.DataFrame,
    test_ws_col: str,
    ref_ws_col: str,
    ref_wd_col: str,
    cfg: WindUpConfig,
    plot_cfg: PlotConfig | None,
) -> pd.DataFrame:
    count_by_scen = detrend_df.dropna(subset=[test_ws_col, ref_ws_col, ref_wd_col])["waking_scenario"].value_counts()
    rows_per_hour = 3600 / cfg.timebase_s
    scens_to_detrend = count_by_scen[count_by_scen > (cfg.detrend_min_hours * rows_per_hour)]
    scens_to_detrend_list = [x for x in scens_to_detrend.index if x != "unknown"]

    wsratio_v_dir_scen = pd.DataFrame()
    if len(scens_to_detrend_list) == 0:
        result_manager.warning("no scenarios with enough data to detrend")
        return pd.DataFrame(columns=["hours", "ws_rom"])

    for scen in scens_to_detrend_list:
        scen_df = detrend_df[detrend_df["waking_scenario"] == scen]
        scen_df = scen_df.dropna(subset=[test_ws_col, ref_ws_col, ref_wd_col])
        wsratio_v_dir = calc_wsratio_v_wd(
            detrend_df=scen_df,
            test_ws_col=test_ws_col,
            ref_ws_col=ref_ws_col,
            ref_wd_col=ref_wd_col,
            min_hours=cfg.detrend_min_hours,
            timebase_s=cfg.timebase_s,
        )
        wsratio_v_dir["waking_scenario"] = scen
        wsratio_v_dir_scen = pd.concat([wsratio_v_dir_scen, wsratio_v_dir])

    wsratio_v_dir_scen = wsratio_v_dir_scen.set_index(["waking_scenario", "direction"])

    wsratio_v_dir_scen = remove_bad_detrend_results(
        wsratio_v_dir_scen,
        test_name=test_name,
        ref_name=ref_name,
        ref_lat=ref_lat,
        ref_long=ref_long,
        cfg=cfg,
    )

    if plot_cfg is not None:
        plot_detrend_ws_scatter(
            detrend_df=detrend_df,
            test_name=test_name,
            ref_name=ref_name,
            test_ws_col=test_ws_col,
            ref_ws_col=ref_ws_col,
            plot_cfg=plot_cfg,
        )

        plot_detrend_wsratio_v_dir_scen(
            wsratio_v_dir_scen=wsratio_v_dir_scen,
            test_name=test_name,
            ref_name=ref_name,
            ref_wd_col=ref_wd_col,
            plot_cfg=plot_cfg,
        )
    return wsratio_v_dir_scen


def apply_wsratio_v_wd_scen(
    p_df: pd.DataFrame,
    wsratio_v_dir_scen: pd.DataFrame,
    *,
    ref_ws_col: str,
    ref_wd_col: str,
) -> pd.DataFrame:
    scen_list = list(
        set(p_df.dropna(subset=[ref_ws_col, ref_wd_col])["waking_scenario"].unique())
        & set(wsratio_v_dir_scen.dropna(subset="ws_rom").index.unique(level="waking_scenario")),
    )
    all_scens_df = pd.DataFrame()
    for scen in scen_list:
        scen_df = p_df[p_df["waking_scenario"] == scen]
        scen_df = scen_df.dropna(subset=[ref_ws_col, ref_wd_col])
        wsratio_v_dir = wsratio_v_dir_scen.loc[scen].dropna(subset="ws_rom")
        scen_df = apply_wsratio_v_wd(scen_df, wsratio_v_dir, ref_ws_col=ref_ws_col, ref_wd_col=ref_wd_col)
        all_scens_df = pd.concat([all_scens_df, scen_df])

    columns_to_add = ["ws_rom", "ref_ws_detrended"]

    try:
        result_df = p_df.merge(all_scens_df[columns_to_add], how="left", left_index=True, right_index=True)
    except KeyError:
        result_manager.warning("no rows in p_df to merge with all_scens_df")
        result_df = p_df.copy()
        result_df[columns_to_add] = np.nan
    # print count of distinct scenario - directions
    count_detrend_applied_df = result_df.dropna(subset=[ref_ws_col, ref_wd_col]).copy()
    count_detrend_applied_df["rounded_wd"] = count_detrend_applied_df[ref_wd_col].round(0).mod(360)
    count_detrend_applied_df["scen_wd"] = (
        count_detrend_applied_df["waking_scenario"] + "_" + count_detrend_applied_df["rounded_wd"].astype(str)
    )
    logger.info(f"detrend applied to {len(count_detrend_applied_df['scen_wd'].unique())} scenario - directions")

    return result_df


def check_applied_detrend(
    *,
    test_name: str,
    ref_name: str,
    ref_lat: float,
    ref_long: float,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    test_ws_col: str,
    ref_ws_col: str,
    detrend_ws_col: str,
    ref_wd_col: str,
    cfg: WindUpConfig,
    plot_cfg: PlotConfig | None,
) -> tuple[float, float]:
    pre_df = pre_df.dropna(subset=[test_ws_col, ref_ws_col, detrend_ws_col, ref_wd_col]).copy()
    post_df = post_df.dropna(subset=[test_ws_col, ref_ws_col, detrend_ws_col, ref_wd_col]).copy()

    # confirm pre_df correlation has improved thanks to detrend
    pre_r2_before_detrend = pre_df[test_ws_col].corr(pre_df[ref_ws_col]) ** 2
    pre_r2_after_detrend = pre_df[test_ws_col].corr(pre_df[detrend_ws_col]) ** 2
    pre_r2_improvement = pre_r2_after_detrend - pre_r2_before_detrend
    if pre_r2_improvement >= 0:
        logger.info(
            f"detrend improved pre_df ws r2 by {pre_r2_improvement:.2f} "
            f"({pre_r2_before_detrend:.2f} to {pre_r2_after_detrend:.2f})",
        )
    else:
        msg = f"pre_r2_after_detrend < pre_r2_before_detrend, {pre_r2_after_detrend} < {pre_r2_before_detrend}"
        result_manager.warning(msg)

    # print post_df corr change
    post_r2_before_detrend = post_df[test_ws_col].corr(post_df[ref_ws_col]) ** 2
    post_r2_after_detrend = post_df[test_ws_col].corr(post_df[detrend_ws_col]) ** 2
    post_r2_improvement = post_r2_after_detrend - post_r2_before_detrend
    if post_r2_improvement >= 0:
        logger.info(
            f"detrend improved post_df ws r2 by {post_r2_improvement:.2f} "
            f"({post_r2_before_detrend:.2f} to {post_r2_after_detrend:.2f})",
        )
    else:
        result_manager.warning(
            f"post_r2_after_detrend < post_r2_before_detrend," f" {post_r2_after_detrend} < {post_r2_before_detrend}"
        )

    if plot_cfg is not None:
        pre_wsratio_v_dir_scen = calc_wsratio_v_wd_scen(
            test_name=test_name,
            ref_name=ref_name,
            ref_lat=ref_lat,
            ref_long=ref_long,
            detrend_df=pre_df,
            test_ws_col=test_ws_col,
            ref_ws_col=detrend_ws_col,
            ref_wd_col=ref_wd_col,
            cfg=cfg,
            plot_cfg=None,
        )
        post_wsratio_v_dir_scen = calc_wsratio_v_wd_scen(
            test_name=test_name,
            ref_name=ref_name,
            ref_lat=ref_lat,
            ref_long=ref_long,
            detrend_df=post_df,
            test_ws_col=test_ws_col,
            ref_ws_col=detrend_ws_col,
            ref_wd_col=ref_wd_col,
            cfg=cfg,
            plot_cfg=None,
        )

        plot_check_applied_detrend(
            pre_df=pre_df,
            post_df=post_df,
            test_name=test_name,
            ref_name=ref_name,
            test_ws_col=test_ws_col,
            ref_ws_col=ref_ws_col,
            detrend_ws_col=detrend_ws_col,
            ref_wd_col=ref_wd_col,
            pre_wsratio_v_dir_scen=pre_wsratio_v_dir_scen,
            post_wsratio_v_dir_scen=post_wsratio_v_dir_scen,
            plot_cfg=plot_cfg,
        )

    return pre_r2_improvement, post_r2_improvement
