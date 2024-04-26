import math
import pprint

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
)
from wind_up.detrend import apply_wsratio_v_wd_scen, calc_wsratio_v_wd_scen, check_applied_detrend
from wind_up.interface import AssessmentInputs, add_toggle_signals
from wind_up.long_term import calc_turbine_lt_dfs_raw_filt
from wind_up.models import PlotConfig, Turbine, WindUpConfig
from wind_up.northing import (
    check_wtg_northing,
)
from wind_up.plots.data_coverage_plots import plot_detrend_data_cov, plot_pre_post_data_cov
from wind_up.plots.detrend_plots import plot_apply_wsratio_v_wd_scen
from wind_up.plots.scada_funcs_plots import compare_ops_curves_pre_post, print_filter_stats
from wind_up.pp_analysis import pre_post_pp_analysis_with_reversal_and_bootstrapping
from wind_up.waking_state import (
    add_waking_scen,
    get_distance_and_bearing,
    get_iec_upwind_turbines,
    lat_long_is_valid,
)
from wind_up.windspeed_drift import check_windspeed_drift


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
                print(f"{test_wtg.name} wake free directions min={min(test_wds_to_keep)} max={max(test_wds_to_keep)}")
            else:
                print(f"{test_wtg.name} has no wake free directions with data")
        rows_after = len(ref_df)
        print(
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
                print(
                    f"{ref_name} wake free directions with data min={min(ref_wds_to_keep)} max={max(ref_wds_to_keep)}",
                )
            else:
                print(f"{ref_name} has no wake free directions with data")
        rows_after = len(ref_df)
        print(
            f"removed {rows_before - rows_after} [{100 * (rows_before - rows_after) / rows_before:.1f}%] "
            f"rows from ref_df using require_ref_wake_free",
        )
    return ref_df.drop(columns=["rounded_wd"])


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

    if cfg.ref_wd_filter is not None:
        rows_before = len(ref_df)
        if cfg.ref_wd_filter[0] < cfg.ref_wd_filter[1]:
            ref_df = ref_df[(ref_df[ref_wd_col] >= cfg.ref_wd_filter[0]) & (ref_df[ref_wd_col] <= cfg.ref_wd_filter[1])]
        else:
            ref_df = ref_df[(ref_df[ref_wd_col] >= cfg.ref_wd_filter[0]) | (ref_df[ref_wd_col] <= cfg.ref_wd_filter[1])]
        rows_after = len(ref_df)
        print(
            f"removed {rows_before - rows_after} [{100 * (rows_before - rows_after) / rows_before:.1f}%] "
            f"rows from ref_df using ref_wd_filter",
        )

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
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    pairing_filter_method: str,
    pairing_filter_timedelta_seconds: int,
    detrend_ws_col: str,
    test_pw_col: str,
    ref_wd_col: str,
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
                    timebase=pd.Timedelta("10min"),
                    max_timedelta_seconds=pairing_filter_timedelta_seconds,
                ),
            )
        ]
        filt_post_df = valid_post_df[
            valid_post_df.index.isin(
                make_extended_time_index(
                    valid_pre_df.index,
                    timebase=pd.Timedelta("10min"),
                    max_timedelta_seconds=pairing_filter_timedelta_seconds,
                ),
            )
        ]
    else:
        msg = f"pairing_filter_method {pairing_filter_method} not recognised"
        raise ValueError(msg)
    len_pre_after = len(filt_pre_df)
    len_post_after = len(filt_post_df)
    print(
        f"removed {len_pre_before - len_pre_after} [{100 * (len_pre_before - len_pre_after) / len_pre_before:.1f}%] "
        f"rows from pre_df using {pairing_filter_method} pairing filter",
    )
    print(
        f"removed {len_post_before - len_post_after} "
        f"[{100 * (len_post_before - len_post_after) / len_post_before:.1f}%] "
        f"rows from post_df using {pairing_filter_method} pairing filter",
    )
    return filt_pre_df, filt_post_df


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
    ref_pw_col = "pw_clipped"
    if test_name == ref_name:
        ref_ws_col = "WindSpeedMean"
        test_ws_col = "test_WindSpeedMean"
    else:
        ref_ws_col = "ws_est_from_power_only" if cfg.ignore_turbine_anemometer_data else "ws_est_blend"
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
    ref_max_northing_error_v_reanalysis = check_wtg_northing(
        ref_df,
        wtg_name=ref_name,
        north_ref_wd_col=REANALYSIS_WD_COL,
        plot_cfg=plot_cfg,
    )
    ref_max_northing_error_v_wf = check_wtg_northing(
        ref_df,
        wtg_name=ref_name,
        north_ref_wd_col=WINDFARM_YAWDIR_COL,
        plot_cfg=plot_cfg,
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
        print(
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
            pre_df,
            post_df,
            pairing_filter_method=cfg.toggle.pairing_filter_method,
            pairing_filter_timedelta_seconds=cfg.toggle.pairing_filter_timedelta_seconds,
            detrend_ws_col=detrend_ws_col,
            test_pw_col=test_pw_col,
            ref_wd_col=ref_wd_col,
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

    other_results = {
        "ref": ref_name,
        "ref_ws_col": ref_ws_col,
        "distance_m": distance_m,
        "bearing_deg": bearing_deg,
        "ref_max_northing_error_v_reanalysis": ref_max_northing_error_v_reanalysis,
        "ref_max_northing_error_v_wf": ref_max_northing_error_v_wf,
        "ref_max_ws_drift": ref_max_ws_drift,
        "ref_max_ws_drift_pp_period": ref_max_ws_drift_pp_period,
        "detrend_pre_r2_improvement": detrend_pre_r2_improvement,
        "detrend_post_r2_improvement": detrend_post_r2_improvement,
        "mean_power_pre": pre_df.dropna(subset=[detrend_ws_col, test_pw_col, ref_wd_col])[test_pw_col].mean(),
        "mean_power_post": post_df.dropna(subset=[detrend_ws_col, test_pw_col, ref_wd_col])[test_pw_col].mean(),
    }

    return other_results | pp_results


def run_wind_up_analysis(
    inputs: AssessmentInputs,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
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
    print(f"test turbines: {[x.name for x in cfg.test_wtgs]}")
    print(f"ref list: {ref_name_list}")
    print(f"turbines to test: {[x.name for x in wtgs_to_test]}")
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
            lt_wtg_df_raw, lt_wtg_df_filt = calc_turbine_lt_dfs_raw_filt(
                wtg_name=test_name,
                cfg=cfg,
                wtg_df=test_df,
                ws_col=test_ws_col,
                pw_col=test_pw_col,
                plot_cfg=plot_cfg,
            )
        else:
            lt_wtg_df_raw = None
            lt_wtg_df_filt = None

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

        # compare ops curves of pre_df and post_df
        compare_ops_curves_pre_post(
            pre_df=pre_df,
            post_df=post_df,
            test_name=test_name,
            plot_cfg=plot_cfg,
            is_toggle_test=(cfg.toggle is not None),
        )

        test_results = {
            "wind_up_version": wind_up.__version__,
            "test_wtg": test_name,
            "test_pw_col": test_pw_col,
            "lt_wtg_hours_raw": lt_wtg_df_raw["observed_hours"].sum() if lt_wtg_df_raw is not None else 0,
            "lt_wtg_hours_filt": lt_wtg_df_filt["observed_hours"].sum() if lt_wtg_df_filt is not None else 0,
            "test_max_ws_drift": test_max_ws_drift,
            "test_max_ws_drift_pp_period": test_max_ws_drift_pp_period,
        }

        scada_pc = pc_per_ttype[test_wtg.turbine_type.turbine_type]
        for ref_name_counter, ref_name in enumerate(ref_name_list):
            loop_counter = test_wtg_counter * len(ref_name_list) + ref_name_counter
            print(f"\nanalysing {test_name} {ref_name}, loop_counter={loop_counter}\n")
            test_ref_results = calc_test_ref_results(
                test_df=test_df,
                pre_df=pre_df,
                post_df=post_df,
                long_term_df=lt_wtg_df_filt,
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
            pprint.pprint(test_ref_results)
            results_df = pd.DataFrame(test_ref_results, index=[loop_counter])
            first_columns = [
                "wind_up_version",
                "time_calculated",
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
            other_columns = [x for x in results_df.columns if x not in first_columns]
            results_df = results_df[first_columns + other_columns]

            results_per_test_ref.append(results_df)
            pd.concat(results_per_test_ref).to_csv(cfg.out_dir / "results_interim.csv")
            print(f"\nfinished analysing {test_name} {ref_name}\n")

    combined_results = pd.concat(results_per_test_ref)
    combined_results.to_csv(
        cfg.out_dir / f"{cfg.assessment_name}_results_per_test_ref_"
        f"{pd.Timestamp.now('UTC').strftime('%Y%m%d_%H%M%S')}.csv",
    )
    return combined_results
