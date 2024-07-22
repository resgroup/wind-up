import logging
import math

import numpy as np
import pandas as pd

import wind_up
from wind_up.constants import (
    CONFIG_DIR,
    PROJECTROOT_DIR,
    RANDOM_SEED,
    REANALYSIS_WD_COL,
    REANALYSIS_WS_COL,
    WINDFARM_YAWDIR_COL,
    DataColumns,
)
from wind_up.detrend import apply_wsratio_v_wd_scen, calc_wsratio_v_wd_scen, check_applied_detrend
from wind_up.interface import AssessmentInputs, add_toggle_signals
from wind_up.long_term import calc_lt_dfs_raw_filt
from wind_up.math_funcs import circ_diff
from wind_up.models import PlotConfig, Turbine, WindUpConfig
from wind_up.northing import (
    check_wtg_northing,
)
from wind_up.plots.data_coverage_plots import plot_detrend_data_cov, plot_pre_post_data_cov
from wind_up.plots.detrend_plots import plot_apply_wsratio_v_wd_scen
from wind_up.plots.scada_funcs_plots import compare_ops_curves_pre_post, print_filter_stats
from wind_up.plots.yaw_direction_plots import plot_yaw_direction_pre_post
from wind_up.pp_analysis import pre_post_pp_analysis_with_reversal_and_bootstrapping
from wind_up.result_manager import result_manager
from wind_up.waking_state import (
    add_waking_scen,
    get_distance_and_bearing,
    get_iec_upwind_turbines,
    lat_long_is_valid,
)
from wind_up.windspeed_drift import check_windspeed_drift

logger = logging.getLogger(__name__)


def get_config_objects(
    config_yaml_file_name: str,
    *,
    show_plots: bool,
    save_plots: bool,
) -> tuple[WindUpConfig, PlotConfig]:
    cfg = WindUpConfig.from_yaml(CONFIG_DIR / config_yaml_file_name)
    plot_cfg = PlotConfig(show_plots=show_plots, save_plots=save_plots, plots_dir=cfg.out_dir / "plots")
    return cfg, plot_cfg


def add_fake_power_data(
    ref_df: pd.DataFrame,
    *,
    ref_pw_col: str,
    ref_ws_col: str,
    scada_pc: pd.DataFrame,
) -> pd.DataFrame:
    ref_df[ref_pw_col] = np.interp(ref_df[ref_ws_col], scada_pc["WindSpeedMean"], scada_pc["pw_clipped"])
    return ref_df


def get_ref_lat_long(ref_name: str, cfg: WindUpConfig) -> tuple[float, float]:
    try:
        ref_wtg = next(x for x in cfg.asset.wtgs if x.name == ref_name)
        ref_lat = ref_wtg.latitude
        ref_long = ref_wtg.longitude
    except StopIteration as exc:
        if ref_name == "reanalysis":
            ref_lat = np.nan
            ref_long = np.nan
        elif ref_name in [x.name for x in cfg.asset.masts_and_lidars]:
            ref_obj = next(x for x in cfg.asset.masts_and_lidars if x.name == ref_name)
            ref_lat = ref_obj.latitude
            ref_long = ref_obj.longitude
        else:
            msg = "ref_name must be a wtg name or 'reanalysis' or exist in cfg.asset.masts_and_lidars"
            raise ValueError(msg) from exc
    return ref_lat, ref_long


def filter_ref_df_for_wake_free(
    ref_df: pd.DataFrame,
    *,
    ref_name: str,
    ref_wd_col: str,
    cfg: WindUpConfig,
    test_wtg: Turbine,
) -> pd.DataFrame:
    ref_df = ref_df.copy()
    ref_df["rounded_wd"] = ref_df[ref_wd_col].round(0) % 360
    if cfg.require_test_wake_free:
        rows_before = len(ref_df)
        test_latlongs = test_wtg.get_latlongs()
        if len(test_latlongs) == 1 and lat_long_is_valid(test_latlongs[0][0], test_latlongs[0][1]):
            test_wds_to_keep = []
            for wd in ref_df["rounded_wd"].unique():
                test_upwind_wtgs = get_iec_upwind_turbines(
                    latlongs=test_latlongs,
                    wind_direction=wd,
                    cfg=cfg,
                    object_name=test_wtg.name,
                )
                if len(test_upwind_wtgs) == 0:
                    test_wds_to_keep.append(wd)
            ref_df = ref_df[ref_df["rounded_wd"].isin(test_wds_to_keep)]
            if len(test_wds_to_keep) > 0:
                logger.info(
                    f"{test_wtg.name} wake free directions min={min(test_wds_to_keep)} max={max(test_wds_to_keep)}"
                )
            else:
                logger.info(f"{test_wtg.name} has no wake free directions with data")
        rows_after = len(ref_df)
        logger.info(
            f"removed {rows_before - rows_after} [{100 * (rows_before - rows_after) / rows_before:.1f}%] "
            f"rows from ref_df using require_test_wake_free",
        )
    if cfg.require_ref_wake_free:
        rows_before = len(ref_df)
        ref_lat, ref_long = get_ref_lat_long(ref_name, cfg)
        if lat_long_is_valid(ref_lat, ref_long):
            ref_wds_to_keep = []
            for wd in ref_df["rounded_wd"].unique():
                ref_upwind_wtgs = get_iec_upwind_turbines(
                    latlongs=[(ref_lat, ref_long)],
                    wind_direction=wd,
                    cfg=cfg,
                    object_name=ref_name,
                )
                if len(ref_upwind_wtgs) == 0:
                    ref_wds_to_keep.append(wd)
            ref_df = ref_df[ref_df["rounded_wd"].isin(ref_wds_to_keep)]
            if len(ref_wds_to_keep) > 0:
                logger.info(
                    f"{ref_name} wake free directions with data min={min(ref_wds_to_keep)} max={max(ref_wds_to_keep)}",
                )
            else:
                logger.info(f"{ref_name} has no wake free directions with data")
        rows_after = len(ref_df)
        logger.info(
            f"removed {rows_before - rows_after} [{100 * (rows_before - rows_after) / rows_before:.1f}%] "
            f"rows from ref_df using require_ref_wake_free",
        )
    return ref_df.drop(columns=["rounded_wd"])


def filter_ref_df_for_wd_and_hod(ref_df: pd.DataFrame, ref_wd_col: str, cfg: WindUpConfig) -> pd.DataFrame:
    ref_df = ref_df.copy()
    if cfg.ref_wd_filter is not None:
        rows_before = len(ref_df)
        if cfg.ref_wd_filter[0] < cfg.ref_wd_filter[1]:
            ref_df = ref_df[(ref_df[ref_wd_col] >= cfg.ref_wd_filter[0]) & (ref_df[ref_wd_col] <= cfg.ref_wd_filter[1])]
        else:
            ref_df = ref_df[(ref_df[ref_wd_col] >= cfg.ref_wd_filter[0]) | (ref_df[ref_wd_col] <= cfg.ref_wd_filter[1])]
        rows_after = len(ref_df)
        logger.info(
            f"removed {rows_before - rows_after} [{100 * (rows_before - rows_after) / rows_before:.1f}%] "
            f"rows from ref_df using ref_wd_filter",
        )

    if cfg.ref_hod_filter is not None:
        rows_before = len(ref_df)
        if cfg.ref_hod_filter[0] < cfg.ref_hod_filter[1]:
            ref_df = ref_df[(ref_df.index.hour >= cfg.ref_hod_filter[0]) & (ref_df.index.hour <= cfg.ref_hod_filter[1])]
        else:
            ref_df = ref_df[(ref_df.index.hour >= cfg.ref_hod_filter[0]) | (ref_df.index.hour <= cfg.ref_hod_filter[1])]
        rows_after = len(ref_df)
        logger.info(
            f"removed {rows_before - rows_after} [{100 * (rows_before - rows_after) / rows_before:.1f}%] "
            f"rows from ref_df using ref_hod_filter",
        )
    return ref_df


def get_ref_df(
    *,
    ref_name: str,
    wf_df: pd.DataFrame,
    ref_pw_col: str,
    ref_ws_col: str,
    ref_wd_col: str,
    scada_pc: pd.DataFrame,
    cfg: WindUpConfig,
    test_wtg: Turbine,
    toggle_df: pd.DataFrame | None = None,
    keep_only_toggle_off: bool = True,
) -> pd.DataFrame:
    if ref_name in [x.name for x in cfg.asset.wtgs]:
        ref_df = wf_df.loc[ref_name].copy()
        if cfg.toggle is not None:
            ref_df = add_toggle_signals(ref_df, wtg_name=ref_name, cfg=cfg, toggle_df=toggle_df)
            ref_df = ref_df.rename(columns={"toggle_off": "ref_toggle_off", "toggle_on": "ref_toggle_on"})
            if keep_only_toggle_off:
                ref_df = ref_df[ref_df["ref_toggle_off"].fillna(value=False)]
                if ref_df["ref_toggle_on"].any():
                    msg = "keep_only_toggle_off = True but ref_df['ref_toggle_on'] has True values"
                    raise RuntimeError(msg)
    else:
        if ref_name == "reanalysis":
            ref_df = wf_df.loc[cfg.test_wtgs[0].name, [REANALYSIS_WS_COL, REANALYSIS_WD_COL, WINDFARM_YAWDIR_COL]]
            original_ws_col = REANALYSIS_WS_COL
            original_wd_col = REANALYSIS_WD_COL
        elif ref_name in [x.name for x in cfg.asset.masts_and_lidars]:
            ref_obj = next(x for x in cfg.asset.masts_and_lidars if x.name == ref_name)
            ref_df = pd.read_parquet(
                PROJECTROOT_DIR / "input_data" / "masts_and_lidars" / cfg.asset.name / f"{ref_obj.data_file_name}",
            )
            northing_df = wf_df.loc[cfg.test_wtgs[0].name, [REANALYSIS_WS_COL, REANALYSIS_WD_COL, WINDFARM_YAWDIR_COL]]
            ref_df = ref_df.merge(northing_df, how="left", left_index=True, right_index=True)
            original_ws_col = ref_obj.wind_speed_column
            original_wd_col = ref_obj.wind_direction_column
        else:
            msg = "ref_name must be a wtg name or 'reanalysis' or exist in cfg.asset.masts_and_lidars"
            raise ValueError(msg)
        ref_df[ref_ws_col] = ref_df[original_ws_col]
        ref_df["WindSpeedMean"] = ref_df[original_ws_col]  # needed for northing analysis
        ref_df["ws_est_from_power_only"] = ref_df[original_ws_col]  # needed for reversal analysis
        ref_df[ref_wd_col] = ref_df[original_wd_col]
        ref_df = add_fake_power_data(ref_df, ref_pw_col=ref_pw_col, ref_ws_col=ref_ws_col, scada_pc=scada_pc)

    ref_df = filter_ref_df_for_wd_and_hod(ref_df, ref_wd_col=ref_wd_col, cfg=cfg)

    if cfg.require_test_wake_free or cfg.require_ref_wake_free:
        ref_df = filter_ref_df_for_wake_free(
            ref_df,
            ref_name=ref_name,
            ref_wd_col=ref_wd_col,
            cfg=cfg,
            test_wtg=test_wtg,
        )
    return ref_df


def make_extended_time_index(
    original_index: pd.DatetimeIndex,
    *,
    timebase: pd.Timedelta,
    max_timedelta_seconds: int,
) -> pd.DatetimeIndex:
    if not isinstance(original_index, pd.DatetimeIndex):
        msg = f"original_index must be a pd.DatetimeIndex, not {type(original_index)}"
        raise TypeError(msg)
    extended_index = original_index
    timedelta_multiple = -math.floor(max_timedelta_seconds / timebase.total_seconds())
    max_timedelta_multiple = math.floor(max_timedelta_seconds / timebase.total_seconds())
    while timedelta_multiple <= max_timedelta_multiple:
        shifted_index = original_index + (timebase * timedelta_multiple)
        extended_index = pd.DatetimeIndex(pd.concat([pd.Series(extended_index), pd.Series(shifted_index)]))
        timedelta_multiple += 1
    return extended_index.sort_values().drop_duplicates()


def toggle_pairing_filter(
    *,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    pairing_filter_method: str,
    pairing_filter_timedelta_seconds: int,
    detrend_ws_col: str,
    test_pw_col: str,
    ref_wd_col: str,
    timebase_s: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = [detrend_ws_col, test_pw_col, ref_wd_col]
    len_pre_before = len(pre_df.dropna(subset=required_cols))
    len_post_before = len(post_df.dropna(subset=required_cols))
    if pairing_filter_method == "none":
        filt_pre_df = pre_df
        filt_post_df = post_df
    elif pairing_filter_method == "one_to_one":
        msg = "pairing_filter_method one_to_one not implemented"
        raise NotImplementedError(msg)
    elif pairing_filter_method == "any_within_timedelta":
        valid_pre_df = pre_df.dropna(subset=required_cols)
        valid_post_df = post_df.dropna(subset=required_cols)
        filt_pre_df = valid_pre_df[
            valid_pre_df.index.isin(
                make_extended_time_index(
                    valid_post_df.index,
                    timebase=pd.Timedelta(seconds=timebase_s),
                    max_timedelta_seconds=pairing_filter_timedelta_seconds,
                ),
            )
        ]
        filt_post_df = valid_post_df[
            valid_post_df.index.isin(
                make_extended_time_index(
                    valid_pre_df.index,
                    timebase=pd.Timedelta(seconds=timebase_s),
                    max_timedelta_seconds=pairing_filter_timedelta_seconds,
                ),
            )
        ]
    else:
        msg = f"pairing_filter_method {pairing_filter_method} not recognised"
        raise ValueError(msg)
    len_pre_after = len(filt_pre_df)
    len_post_after = len(filt_post_df)
    logger.info(
        f"removed {len_pre_before - len_pre_after} [{100 * (len_pre_before - len_pre_after) / len_pre_before:.1f}%] "
        f"rows from pre_df using {pairing_filter_method} pairing filter",
    )
    logger.info(
        f"removed {len_post_before - len_post_after} "
        f"[{100 * (len_post_before - len_post_after) / len_post_before:.1f}%] "
        f"rows from post_df using {pairing_filter_method} pairing filter",
    )
    return filt_pre_df, filt_post_df


def yaw_error_results(pre_df: pd.DataFrame, post_df: pd.DataFrame, required_pp_cols: list[str]) -> dict:
    results = {}
    if "test_yaw_error_mean" in pre_df.columns:
        results["test_yaw_error_pre"] = pre_df.dropna(subset=required_pp_cols)["test_yaw_error_mean"].mean()
        results["test_yaw_error_post"] = post_df.dropna(subset=required_pp_cols)["test_yaw_error_mean"].mean()
    if "ref_yaw_error_mean" in pre_df.columns:
        results["ref_yaw_error_pre"] = pre_df.dropna(subset=required_pp_cols)["ref_yaw_error_mean"].mean()
        results["ref_yaw_error_post"] = post_df.dropna(subset=required_pp_cols)["ref_yaw_error_mean"].mean()
    return results


def yaw_offset_results(
    pre_df: pd.DataFrame, post_df: pd.DataFrame, required_pp_cols: list[str], ref_wd_col: str, test_wd_col: str
) -> dict:
    results = {}

    pre_yaw_offset = pd.Series(
        circ_diff(
            pre_df.dropna(subset=required_pp_cols)[ref_wd_col], pre_df.dropna(subset=required_pp_cols)[test_wd_col]
        )
    )
    post_yaw_offset = pd.Series(
        circ_diff(
            post_df.dropna(subset=required_pp_cols)[ref_wd_col], post_df.dropna(subset=required_pp_cols)[test_wd_col]
        )
    )

    if len(pre_yaw_offset) > 0 and len(post_yaw_offset) > 0:
        results["mean_test_yaw_offset_pre"] = pre_yaw_offset.mean()
        results["mean_test_yaw_offset_post"] = post_yaw_offset.mean()

    yaw_offset_ul = 1e-3
    if "test_yaw_offset_command" in pre_df.columns:
        result_name = "mean_test_yaw_offset_command_pre"
        results[result_name] = pre_df.dropna(subset=required_pp_cols)["test_yaw_offset_command"].mean()
        if results[result_name] > yaw_offset_ul:
            result_manager.warning(f"{result_name} > 0: " f"({results[result_name]})")

        results["mean_test_yaw_offset_command_post"] = post_df.dropna(subset=required_pp_cols)[
            "test_yaw_offset_command"
        ].mean()
    if "ref_yaw_offset_command" in pre_df.columns:
        result_name = "mean_ref_yaw_offset_command_pre"
        results[result_name] = pre_df.dropna(subset=required_pp_cols)["ref_yaw_offset_command"].mean()
        if results[result_name] > yaw_offset_ul:
            result_manager.warning(f"{result_name} > 0 for: " f"({results[result_name]})")

        result_name = "mean_ref_yaw_offset_command_pre"
        results[result_name] = post_df.dropna(subset=required_pp_cols)["ref_yaw_offset_command"].mean()
        if results[result_name] > yaw_offset_ul:
            result_manager.warning(f"{result_name} > 0 for: " f"({results[result_name]})")
    return results


def check_for_ops_curve_shift(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    *,
    wtg_name: str,
    scada_ws_col: str,
    pw_col: str,
    rpm_col: str,
    pt_col: str,
    cfg: WindUpConfig,
    plot_cfg: PlotConfig,
    sub_dir: str | None = None,
) -> dict[str, float]:
    results_dict = {
        "powercurve_shift": np.nan,
        "rpm_shift": np.nan,
        "pitch_shift": np.nan,
    }
    # check if all required columns are present
    required_cols = [scada_ws_col, pw_col, pt_col, rpm_col]
    for req_col in required_cols:
        if req_col not in pre_df.columns:
            msg = f"check_for_ops_curve_shift {wtg_name} pre_df missing required column {req_col}"
            result_manager.warning(msg)
            return results_dict
        if req_col not in post_df.columns:
            msg = f"check_for_ops_curve_shift {wtg_name} post_df missing required column {req_col}"
            result_manager.warning(msg)
            return results_dict
    pre_dropna_df = pre_df.dropna(subset=[scada_ws_col, pw_col, pt_col, rpm_col]).copy()
    post_dropna_df = post_df.dropna(subset=[scada_ws_col, pw_col, pt_col, rpm_col]).copy()

    warning_msg: str | None = None
    for descr, x_var, y_var, x_bin_width, warn_thresh in [
        ("powercurve_shift", scada_ws_col, pw_col, 1, 0.01),
        ("rpm_shift", pw_col, rpm_col, 0, 0.005),
        ("pitch_shift", scada_ws_col, pt_col, 1, 0.1),
    ]:
        bins = np.arange(0, pre_dropna_df[x_var].max() + x_bin_width, x_bin_width) if x_bin_width > 0 else 10
        mean_curve = pre_dropna_df.groupby(pd.cut(pre_dropna_df[x_var], bins=bins, retbins=False), observed=True).agg(
            x_mean=pd.NamedAgg(column=x_var, aggfunc="mean"),
            y_mean=pd.NamedAgg(column=y_var, aggfunc="mean"),
        )
        post_dropna_df["expected_y"] = np.interp(post_dropna_df[x_var], mean_curve["x_mean"], mean_curve["y_mean"])
        mean_df = post_dropna_df.mean()
        if y_var == pt_col:
            results_dict[descr] = mean_df[y_var] - mean_df["expected_y"]
        else:
            results_dict[descr] = (mean_df[y_var] / mean_df["expected_y"] - 1).clip(-1, 1)
        if abs(results_dict[descr]) > warn_thresh:
            if warning_msg is None:
                warning_msg = f"{wtg_name} check_for_ops_curve_shift warnings:"
            warning_msg += f" abs({descr}) > {warn_thresh}: {abs(results_dict[descr]):.3f}"
    if warning_msg is not None:
        result_manager.warning(warning_msg)

    compare_ops_curves_pre_post(
        pre_df=pre_df,
        post_df=post_df,
        wtg_name=wtg_name,
        ws_col=scada_ws_col,
        pw_col=pw_col,
        pt_col=pt_col,
        rpm_col=rpm_col,
        plot_cfg=plot_cfg,
        is_toggle_test=(cfg.toggle is not None),
        sub_dir=sub_dir,
    )

    return results_dict


def calc_test_ref_results(
    *,
    test_df: pd.DataFrame,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    long_term_df: pd.DataFrame | None,
    wf_df: pd.DataFrame,
    test_wtg: Turbine,
    ref_name: str,
    test_pw_col: str,
    test_ws_col: str,
    scada_pc: pd.DataFrame,
    cfg: WindUpConfig,
    plot_cfg: PlotConfig,
    random_seed: int,
    toggle_df: pd.DataFrame | None = None,
) -> dict:
    test_name = test_wtg.name
    (plot_cfg.plots_dir / test_name / ref_name).mkdir(exist_ok=True)
    ref_pw_col = "pw_clipped"
    if test_name == ref_name:
        ref_ws_col = "WindSpeedMean"
        test_ws_col = "test_WindSpeedMean"
    else:
        ref_ws_col = "ws_est_from_power_only" if cfg.ignore_turbine_anemometer_data else "ws_est_blend"
    ref_info = {
        "ref": ref_name,
    }
    ref_wd_col = "YawAngleMean"
    keep_only_toggle_off = False
    if cfg.toggle is not None:
        # if ref is test then keep all data, not just toggle off
        # if there is not a toggle file per turbine then keep all data
        keep_only_toggle_off = cfg.toggle.toggle_file_per_turbine and (ref_name != test_name)
    ref_df = get_ref_df(
        ref_name=ref_name,
        wf_df=wf_df,
        ref_pw_col=ref_pw_col,
        ref_ws_col=ref_ws_col,
        ref_wd_col=ref_wd_col,
        scada_pc=scada_pc,
        cfg=cfg,
        test_wtg=test_wtg,
        toggle_df=toggle_df,
        keep_only_toggle_off=keep_only_toggle_off,
    )
    if len(ref_df) == 0:
        result_manager.warning(f"ref_df is empty for {ref_name}")
        return ref_info
    ref_max_northing_error_v_reanalysis = check_wtg_northing(
        ref_df,
        wtg_name=ref_name,
        north_ref_wd_col=REANALYSIS_WD_COL,
        timebase_s=cfg.timebase_s,
        plot_cfg=plot_cfg,
        sub_dir=f"{test_name}/{ref_name}",
    )
    ref_max_northing_error_v_wf = check_wtg_northing(
        ref_df,
        wtg_name=ref_name,
        north_ref_wd_col=WINDFARM_YAWDIR_COL,
        timebase_s=cfg.timebase_s,
        plot_cfg=plot_cfg,
        sub_dir=f"{test_name}/{ref_name}",
    )

    ref_pw_col = "ref_" + ref_pw_col
    ref_ws_col = "ref_" + ref_ws_col
    ref_wd_col = "ref_" + ref_wd_col
    ref_df.columns = ["ref_" + x for x in ref_df.columns]

    test_wtg = next(x for x in cfg.asset.wtgs if x.name == test_name)
    test_lat = test_wtg.latitude
    test_long = test_wtg.longitude
    ref_lat, ref_long = get_ref_lat_long(ref_name, cfg)

    distance_m, bearing_deg = get_distance_and_bearing(
        lat1=test_lat,
        long1=test_long,
        lat2=ref_lat,
        long2=ref_long,
    )

    ref_max_ws_drift, ref_max_ws_drift_pp_period = check_windspeed_drift(
        wtg_df=ref_df,
        wtg_name=ref_name,
        ws_col=ref_ws_col,
        reanalysis_ws_col="ref_" + REANALYSIS_WS_COL,
        cfg=cfg,
        plot_cfg=plot_cfg,
        sub_dir=f"{test_name}/{ref_name}",
    )

    detrend_df = test_df.merge(ref_df, how="left", left_index=True, right_index=True)
    detrend_df = detrend_df[cfg.detrend_first_dt_utc_start : cfg.detrend_last_dt_utc_start]  # type: ignore[misc]
    if "test_toggle_on" in detrend_df.columns:
        # find the first time where "test_toggle_on" is true
        first_toggle_on = detrend_df[detrend_df["test_toggle_on"].fillna(value=False)].index.min()
        rows_before = len(detrend_df)
        detrend_df = detrend_df[
            (detrend_df.index < first_toggle_on) | (detrend_df["test_toggle_off"].fillna(value=False))
        ]
        rows_after = len(detrend_df)
        logger.info(
            f"removed {rows_before - rows_after} [{100 * (rows_before - rows_after) / rows_before:.1f}%] "
            f"rows from detrend_df where test_toggle_off was not True after the first toggle on time",
        )
    detrend_df = add_waking_scen(
        test_name=test_name,
        ref_name=ref_name,
        test_ref_df=detrend_df,
        cfg=cfg,
        wf_df=wf_df,
        ref_wd_col=ref_wd_col,
        ref_lat=ref_lat,
        ref_long=ref_long,
    )

    plot_detrend_data_cov(
        cfg=cfg,
        test_name=test_name,
        ref_name=ref_name,
        test_df=test_df,
        test_ws_col=test_ws_col,
        ref_df=ref_df,
        ref_ws_col=ref_ws_col,
        ref_wd_col=ref_wd_col,
        detrend_df=detrend_df,
        plot_cfg=plot_cfg,
    )

    wsratio_v_dir_scen = calc_wsratio_v_wd_scen(
        test_name=test_name,
        ref_name=ref_name,
        ref_lat=ref_lat,
        ref_long=ref_long,
        detrend_df=detrend_df,
        test_ws_col=test_ws_col,
        ref_ws_col=ref_ws_col,
        ref_wd_col=ref_wd_col,
        cfg=cfg,
        plot_cfg=plot_cfg,
    )

    pre_df = pre_df.merge(ref_df, how="left", left_index=True, right_index=True)
    post_df = post_df.merge(ref_df, how="left", left_index=True, right_index=True)

    ref_ops_curve_shift_dict = check_for_ops_curve_shift(
        pre_df,
        post_df,
        wtg_name=ref_name,
        scada_ws_col=f"ref_{DataColumns.wind_speed_mean}",
        pw_col=f"ref_{DataColumns.active_power_mean}",
        rpm_col=f"ref_{DataColumns.gen_rpm_mean}",
        pt_col=f"ref_{DataColumns.pitch_angle_mean}",
        cfg=cfg,
        plot_cfg=plot_cfg,
        sub_dir=f"{test_name}/{ref_name}",
    )

    pre_df = add_waking_scen(
        test_ref_df=pre_df,
        test_name=test_name,
        ref_name=ref_name,
        cfg=cfg,
        wf_df=wf_df,
        ref_wd_col=ref_wd_col,
        ref_lat=ref_lat,
        ref_long=ref_long,
    )
    post_df = add_waking_scen(
        test_ref_df=post_df,
        test_name=test_name,
        ref_name=ref_name,
        cfg=cfg,
        wf_df=wf_df,
        ref_wd_col=ref_wd_col,
        ref_lat=ref_lat,
        ref_long=ref_long,
    )

    detrend_ws_col = "ref_ws_detrended"
    pre_df = apply_wsratio_v_wd_scen(pre_df, wsratio_v_dir_scen, ref_ws_col=ref_ws_col, ref_wd_col=ref_wd_col)
    plot_apply_wsratio_v_wd_scen(
        pre_df.dropna(subset=[ref_ws_col, test_ws_col, detrend_ws_col, test_pw_col]),
        ref_ws_col=ref_ws_col,
        test_ws_col=test_ws_col,
        detrend_ws_col=detrend_ws_col,
        test_pw_col=test_pw_col,
        test_name=test_name,
        ref_name=ref_name,
        title_end="pre",
        plot_cfg=plot_cfg,
    )
    post_df = apply_wsratio_v_wd_scen(post_df, wsratio_v_dir_scen, ref_ws_col=ref_ws_col, ref_wd_col=ref_wd_col)
    plot_apply_wsratio_v_wd_scen(
        post_df.dropna(subset=[ref_ws_col, test_ws_col, detrend_ws_col, test_pw_col]),
        ref_ws_col=ref_ws_col,
        test_ws_col=test_ws_col,
        detrend_ws_col=detrend_ws_col,
        test_pw_col=test_pw_col,
        test_name=test_name,
        ref_name=ref_name,
        title_end="post",
        plot_cfg=plot_cfg,
    )

    detrend_pre_r2_improvement, detrend_post_r2_improvement = check_applied_detrend(
        test_name=test_name,
        ref_name=ref_name,
        ref_lat=ref_lat,
        ref_long=ref_long,
        pre_df=pre_df,
        post_df=post_df,
        test_ws_col=test_ws_col,
        ref_ws_col=ref_ws_col,
        detrend_ws_col=detrend_ws_col,
        ref_wd_col=ref_wd_col,
        cfg=cfg,
        plot_cfg=plot_cfg,
    )

    plot_pre_post_data_cov(
        cfg=cfg,
        test_name=test_name,
        ref_name=ref_name,
        test_df=test_df,
        test_pw_col=test_pw_col,
        test_ws_col=test_ws_col,
        ref_df=ref_df,
        ref_pw_col=ref_pw_col,
        ref_ws_col=ref_ws_col,
        detrend_ws_col=detrend_ws_col,
        ref_wd_col=ref_wd_col,
        pre_df=pre_df,
        post_df=post_df,
        plot_cfg=plot_cfg,
    )

    if cfg.toggle is not None:
        pre_df, post_df = toggle_pairing_filter(
            pre_df=pre_df,
            post_df=post_df,
            pairing_filter_method=cfg.toggle.pairing_filter_method,
            pairing_filter_timedelta_seconds=cfg.toggle.pairing_filter_timedelta_seconds,
            detrend_ws_col=detrend_ws_col,
            test_pw_col=test_pw_col,
            ref_wd_col=ref_wd_col,
            timebase_s=cfg.timebase_s,
        )

    if plot_cfg is not None:
        plot_yaw_direction_pre_post(
            pre_df=pre_df,
            post_df=post_df,
            test_name=test_name,
            ref_name=ref_name,
            ref_ws_col=ref_ws_col,
            ref_wd_col=ref_wd_col,
            plot_cfg=plot_cfg,
            toggle_name=cfg.toggle.name if cfg.toggle else None,
        )

    pp_results, pp_df = pre_post_pp_analysis_with_reversal_and_bootstrapping(
        cfg=cfg,
        test_wtg=test_wtg,
        ref_name=ref_name,
        lt_df=long_term_df,
        pre_df=pre_df,
        post_df=post_df,
        ws_col=detrend_ws_col,
        pw_col=test_pw_col,
        wd_col=ref_wd_col,
        plot_cfg=plot_cfg,
        test_df=test_df,
        random_seed=random_seed,
    )

    other_results = ref_info | {
        "ref_ws_col": ref_ws_col,
        "distance_m": distance_m,
        "bearing_deg": bearing_deg,
        "ref_max_northing_error_v_reanalysis": ref_max_northing_error_v_reanalysis,
        "ref_max_northing_error_v_wf": ref_max_northing_error_v_wf,
        "ref_max_ws_drift": ref_max_ws_drift,
        "ref_max_ws_drift_pp_period": ref_max_ws_drift_pp_period,
        "ref_powercurve_shift": ref_ops_curve_shift_dict["powercurve_shift"],
        "ref_rpm_shift": ref_ops_curve_shift_dict["rpm_shift"],
        "ref_pitch_shift": ref_ops_curve_shift_dict["pitch_shift"],
        "detrend_pre_r2_improvement": detrend_pre_r2_improvement,
        "detrend_post_r2_improvement": detrend_post_r2_improvement,
        "mean_power_pre": pre_df.dropna(subset=[detrend_ws_col, test_pw_col, ref_wd_col])[test_pw_col].mean(),
        "mean_power_post": post_df.dropna(subset=[detrend_ws_col, test_pw_col, ref_wd_col])[test_pw_col].mean(),
    }

    other_results = other_results | yaw_error_results(
        pre_df=pre_df, post_df=post_df, required_pp_cols=[detrend_ws_col, test_pw_col, ref_wd_col]
    )
    other_results = other_results | yaw_offset_results(
        pre_df=pre_df,
        post_df=post_df,
        required_pp_cols=[detrend_ws_col, test_pw_col, ref_wd_col],
        ref_wd_col=ref_wd_col,
        test_wd_col="test_YawAngleMean",
    )

    other_results["test_ref_warning_counts"] = len(result_manager.stored_warnings)
    result_manager.stored_warnings = []
    return other_results | pp_results


def results_per_test_ref_to_df(results_per_test_ref: list[pd.DataFrame]) -> pd.DataFrame:
    results_per_test_ref_df = pd.concat(results_per_test_ref)
    first_columns = [
        "wind_up_version",
        "time_calculated",
        "preprocess_warning_counts",
        "test_warning_counts",
        "test_ref_warning_counts",
        "test_wtg",
        "test_pw_col",
        "ref",
        "ref_ws_col",
        "uplift_frc",
        "unc_one_sigma_frc",
        "uplift_p95_frc",
        "uplift_p5_frc",
        "pp_data_coverage",
        "distance_m",
        "bearing_deg",
        "unc_one_sigma_noadj_frc",
        "unc_one_sigma_lowerbound_frc",
        "unc_one_sigma_bootstrap_frc",
    ]
    other_columns = [x for x in results_per_test_ref_df.columns if x not in first_columns]
    return results_per_test_ref_df[
        [col for col in first_columns + other_columns if col in results_per_test_ref_df.columns]
    ]


def run_wind_up_analysis(
    inputs: AssessmentInputs,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    preprocess_warning_counts = len(result_manager.stored_warnings)
    result_manager.stored_warnings = []

    wf_df = inputs.wf_df
    pc_per_ttype = inputs.pc_per_ttype
    cfg = inputs.cfg
    plot_cfg = inputs.plot_cfg
    pre_post_splitter = inputs.pre_post_splitter

    ref_name_list = sorted(set(cfg.non_wtg_ref_names + [x.name for x in cfg.ref_wtgs]))
    wtgs_to_test = cfg.test_wtgs.copy()
    if len(cfg.ref_wtgs) > 1:
        wtgs_to_test.extend([x for x in cfg.ref_wtgs if x not in cfg.test_wtgs])

    results_per_test_ref = []
    logger.info(f"test turbines: {[x.name for x in cfg.test_wtgs]}")
    logger.info(f"ref list: {ref_name_list}")
    logger.info(f"turbines to test: {[x.name for x in wtgs_to_test]}")
    for test_wtg_counter, test_wtg in enumerate(wtgs_to_test):
        test_name = test_wtg.name
        test_pw_col = "pw_clipped"
        test_ws_col = "ws_est_from_power_only" if cfg.ignore_turbine_anemometer_data else "ws_est_blend"
        test_df = wf_df.loc[test_wtg.name].copy()

        if cfg.filter_all_test_wtgs_together:
            for other_test_wtg in cfg.test_wtgs:
                if other_test_wtg.name == test_name:
                    continue
                pw_na_before = test_df["ActivePowerMean"].isna().sum()
                other_test_df = wf_df.loc[other_test_wtg.name]
                timestamps_to_filter = other_test_df[
                    other_test_df[test_pw_col].isna() | other_test_df[test_ws_col].isna()
                ].index
                cols_to_filter = list({test_pw_col, test_ws_col, "ActivePowerMean", "WindSpeedMean"})
                test_df.loc[timestamps_to_filter, cols_to_filter] = pd.NA
                pw_na_after = test_df["ActivePowerMean"].isna().sum()
                print_filter_stats(
                    filter_name=f"filter_all_test_wtgs_together {other_test_wtg.name}",
                    na_rows=pw_na_after - pw_na_before,
                    total_rows=len(test_df),
                )

        if cfg.use_lt_distribution:
            lt_df_raw, lt_df_filt = calc_lt_dfs_raw_filt(
                wtg_or_wf_name=test_name if cfg.use_test_wtg_lt_distribution else cfg.asset.name,
                cfg=cfg,
                wtg_or_wf_df=test_df if cfg.use_test_wtg_lt_distribution else wf_df,
                ws_col=test_ws_col,
                pw_col=test_pw_col,
                one_turbine=cfg.use_test_wtg_lt_distribution,
                plot_cfg=plot_cfg,
            )
        else:
            lt_df_raw = None
            lt_df_filt = None

        test_df.columns = ["test_" + x for x in test_df.columns]
        test_pw_col = "test_" + test_pw_col
        test_ws_col = "test_" + test_ws_col

        test_max_ws_drift, test_max_ws_drift_pp_period = check_windspeed_drift(
            wtg_df=test_df,
            wtg_name=test_name,
            ws_col=test_ws_col,
            reanalysis_ws_col="test_" + REANALYSIS_WS_COL,
            cfg=cfg,
            plot_cfg=plot_cfg,
        )

        test_df, pre_df, post_df = pre_post_splitter.split(test_df, test_wtg_name=test_name)

        test_ops_curve_shift_dict = check_for_ops_curve_shift(
            pre_df,
            post_df,
            wtg_name=test_name,
            scada_ws_col=f"test_{DataColumns.wind_speed_mean}",
            pw_col=f"test_{DataColumns.active_power_mean}",
            rpm_col=f"test_{DataColumns.gen_rpm_mean}",
            pt_col=f"test_{DataColumns.pitch_angle_mean}",
            cfg=cfg,
            plot_cfg=plot_cfg,
        )

        test_results = {
            "wind_up_version": wind_up.__version__,
            "test_wtg": test_name,
            "test_pw_col": test_pw_col,
            "lt_wtg_hours_raw": lt_df_raw["observed_hours"].sum() if lt_df_raw is not None else 0,
            "lt_wtg_hours_filt": lt_df_filt["observed_hours"].sum() if lt_df_filt is not None else 0,
            "test_max_ws_drift": test_max_ws_drift,
            "test_max_ws_drift_pp_period": test_max_ws_drift_pp_period,
            "test_powercurve_shift": test_ops_curve_shift_dict["powercurve_shift"],
            "test_rpm_shift": test_ops_curve_shift_dict["rpm_shift"],
            "test_pitch_shift": test_ops_curve_shift_dict["pitch_shift"],
            "preprocess_warning_counts": preprocess_warning_counts,
            "test_warning_counts": len(result_manager.stored_warnings),
        }
        result_manager.stored_warnings = []

        scada_pc = pc_per_ttype[test_wtg.turbine_type.turbine_type]
        for ref_name_counter, ref_name in enumerate(ref_name_list):
            loop_counter = test_wtg_counter * len(ref_name_list) + ref_name_counter
            logger.info(f"analysing {test_name} {ref_name}, loop_counter={loop_counter}")
            test_ref_results = calc_test_ref_results(
                test_df=test_df,
                pre_df=pre_df,
                post_df=post_df,
                long_term_df=lt_df_filt,
                wf_df=wf_df,
                toggle_df=inputs.pre_post_splitter.toggle_df,
                test_wtg=test_wtg,
                ref_name=ref_name,
                test_pw_col=test_pw_col,
                test_ws_col=test_ws_col,
                scada_pc=scada_pc,
                cfg=cfg,
                plot_cfg=plot_cfg,
                random_seed=random_seed,
            )
            test_ref_results = test_ref_results | test_results
            logger.info(test_ref_results)
            results_df = pd.DataFrame(test_ref_results, index=[loop_counter])
            results_per_test_ref.append(results_df)

            results_per_test_ref_df = results_per_test_ref_to_df(results_per_test_ref)

            results_per_test_ref_df.to_csv(cfg.out_dir / "results_interim.csv")

            try:
                msg = (
                    f"warning summary: preprocess_warning_counts={results_df['preprocess_warning_counts'].iloc[0]}, "
                    f"test_warning_counts={results_df['test_warning_counts'].iloc[0]}, "
                    f"test_ref_warning_counts={results_df['test_ref_warning_counts'].iloc[0]}"
                )
                logger.info(msg)
            except KeyError:
                pass
            logger.info(f"finished analysing {test_name} {ref_name}\n")

    results_per_test_ref_df.to_csv(
        cfg.out_dir / f"{cfg.assessment_name}_results_per_test_ref_"
        f"{pd.Timestamp.now('UTC').strftime('%Y%m%d_%H%M%S')}.csv",
    )
    return results_per_test_ref_df
