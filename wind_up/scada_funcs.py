import datetime as dt
import logging
import warnings
from typing import TypeAlias

import numpy as np
import pandas as pd

from wind_up.constants import RAW_DOWNTIME_S_COL, RAW_POWER_COL, RAW_WINDSPEED_COL, RAW_YAWDIR_COL, DataColumns
from wind_up.math_funcs import circ_diff
from wind_up.models import PlotConfig, Turbine, WindUpConfig
from wind_up.plots.scada_funcs_plots import (
    plot_data_coverage_heatmap,
    plot_filter_rpm_and_pt_curve_one_ttype_or_wtg,
    plot_ops_curves_per_ttype,
    print_and_plot_capacity_factor,
    print_filter_stats,
)
from wind_up.smart_data import add_smart_lat_long_to_cfg, load_smart_scada_and_md_from_file

logger = logging.getLogger(__name__)
ExclusionPeriodsType: TypeAlias = list[tuple[str, dt.datetime, dt.datetime]]


def filter_stuck_data(df: pd.DataFrame) -> pd.DataFrame:
    diffdf = df.groupby("TurbineName", observed=False).ffill().fillna(0).diff()
    stuck_data = (diffdf == 0).all(axis=1)
    very_low_wind_threshold = 1.5
    very_low_wind = df["WindSpeedMean"] < very_low_wind_threshold
    stuck_filter = stuck_data & (~very_low_wind)
    df.loc[stuck_filter, :] = pd.NA
    print_filter_stats(filter_name="filter_stuck_data", na_rows=stuck_filter.sum(), total_rows=len(df))
    return df


def filter_bad_pw_ws(df: pd.DataFrame, max_rated_power: float) -> pd.DataFrame:
    na_rows_before = df["ActivePowerMean"].isna().sum()
    df.loc[df[["ActivePowerMean", "WindSpeedMean"]].isna().any(axis=1), :] = pd.NA
    df.loc[(df["ActivePowerMean"] < -0.5 * max_rated_power), :] = pd.NA
    df.loc[(df["ActivePowerMean"] > 2 * max_rated_power), :] = pd.NA
    df.loc[(df["WindSpeedMean"] < 0), :] = pd.NA
    ws_in_range_ul = 98
    df.loc[(df["WindSpeedMean"] > ws_in_range_ul), :] = pd.NA
    na_rows_after = df["ActivePowerMean"].isna().sum()
    na_rows = na_rows_after - na_rows_before
    print_filter_stats(filter_name="filter_bad_pw_ws", na_rows=na_rows, total_rows=len(df))
    return df


def wrap_yaw_and_pitch(df: pd.DataFrame) -> pd.DataFrame:
    yaw_lt0 = (df["YawAngleMean"] < 0).sum()
    yaw_ge360 = (df["YawAngleMean"] >= 360).sum()  # noqa PLR2004
    if yaw_lt0 > 0 or yaw_ge360 > 0:
        logger.info(f"rows with YawAngleMean lt 0: {yaw_lt0}")
        logger.info(f"rows with YawAngleMean ge 360: {yaw_ge360}")
        df["YawAngleMean"] = df["YawAngleMean"] % 360
        logger.info("wrapped YawAngleMean to 0-360")
        if "YawAngleMin" in df.columns:
            df["YawAngleMin"] = df["YawAngleMin"].clip(lower=0, upper=360) % 360
            logger.info("clipped YawAngleMin to 0-360")
        if "YawAngleMax" in df.columns:
            df["YawAngleMax"] = df["YawAngleMax"].clip(lower=0, upper=360) % 360
            logger.info("clipped YawAngleMax to 0-360")

    # ensure pitch data is -180 to 180
    pt_lt_180 = (df["PitchAngleMean"] < -180).sum()  # noqa PLR2004
    pt_ge_180 = (df["PitchAngleMean"] >= 180).sum()  # noqa PLR2004
    if pt_lt_180 > 0 or pt_ge_180 > 0:
        logger.info(f"rows with PitchAngleMean lt -180: {pt_lt_180}")
        logger.info(f"rows with PitchAngleMean ge 180: {pt_ge_180}")
        df["PitchAngleMean"] = ((df["PitchAngleMean"] + 180) % 360) - 180
        logger.info("wrapped PitchAngleMean to -180-180")
    return df


def filter_wrong_yaw(df: pd.DataFrame) -> pd.DataFrame:
    if "YawAngleMin" in df.columns and "YawAngleMax" in df.columns:
        min_wrong = (df["YawAngleMin"] > df["YawAngleMean"]) | (df["YawAngleMin"] > df["YawAngleMax"])
        max_wrong = (df["YawAngleMax"] < df["YawAngleMean"]) | (df["YawAngleMax"] < df["YawAngleMin"])
        min_max_filter = min_wrong | max_wrong
        df.loc[min_max_filter, ["YawAngleMin", "YawAngleMax"]] = pd.NA
        print_filter_stats(
            filter_name="filter_wrong_yaw",
            na_rows=min_max_filter.sum(),
            total_rows=len(df),
            just_yaw=True,
            just_min_max=True,
            reason="bad YawAngleMin/Max values",
        )

        # sometimes SCADA systems use arithmetic mean instead of circ mean
        wrap_rows = (df["YawAngleMin"] <= 1) & (df["YawAngleMax"] >= 359)  # noqa PLR2004
        mean_far_from_north = np.abs(circ_diff(df["YawAngleMean"].to_numpy(), 0)) >= 15  # noqa PLR2004
        bad_yaw_mean_filter = wrap_rows & mean_far_from_north
        yaw_cols = [col for col in df.columns if "Yaw" in col]
        df.loc[bad_yaw_mean_filter, yaw_cols] = pd.NA
        print_filter_stats(
            filter_name="filter_wrong_yaw",
            na_rows=bad_yaw_mean_filter.sum(),
            total_rows=len(df),
            just_yaw=True,
            reason="YawAngleMean appears to be wrong",
        )

    return df


def add_pw_clipped(df: pd.DataFrame, wtgs: list[Turbine]) -> pd.DataFrame:
    df["pw_clipped"] = df["ActivePowerMean"].clip(lower=0)
    for i, rp in zip(
        [x.name for x in wtgs],
        [x.turbine_type.rated_power_kw for x in wtgs],
        strict=True,
    ):
        df.loc[i, "pw_clipped"] = df.loc[i, "pw_clipped"].clip(upper=rp).to_numpy()
    return df


def filter_exclusions(
    input_df: pd.DataFrame,
    exclusion_periods_utc: ExclusionPeriodsType,
) -> pd.DataFrame:
    pw_na_before = input_df["ActivePowerMean"].isna().sum()
    for exclusion in exclusion_periods_utc:
        excl_turbine = exclusion[0]
        exclusion_period_start_utc = exclusion[1]
        exclusion_period_end_utc = exclusion[2]
        if excl_turbine == "ALL":
            input_df.loc[
                pd.IndexSlice[
                    :,
                    (exclusion_period_start_utc - pd.Timedelta(minutes=9, seconds=59)) : (
                        exclusion_period_end_utc - pd.Timedelta(seconds=1)
                    ),
                ],
                :,
            ] = pd.NA
        else:
            input_df.loc[
                pd.IndexSlice[
                    excl_turbine,
                    (exclusion_period_start_utc - pd.Timedelta(minutes=9, seconds=59)) : (
                        exclusion_period_end_utc - pd.Timedelta(seconds=1)
                    ),
                ],
                :,
            ] = pd.NA
    pw_na_after = input_df["ActivePowerMean"].isna().sum()
    print_filter_stats(filter_name="filter_exclusions", na_rows=pw_na_after - pw_na_before, total_rows=len(input_df))
    # also filter any row where "exclude_row" is True
    if "exclude_row" in input_df.columns:
        pw_na_before = input_df["ActivePowerMean"].isna().sum()
        input_df.loc[input_df["exclude_row"].fillna(value=0) == 1, :] = pd.NA
        pw_na_after = input_df["ActivePowerMean"].isna().sum()
        print_filter_stats(
            filter_name="filter_exclusions exclude_row", na_rows=pw_na_after - pw_na_before, total_rows=len(input_df)
        )
    return input_df


def filter_yaw_exclusions(
    input_df: pd.DataFrame,
    yaw_data_exclusions_utc: ExclusionPeriodsType,
) -> pd.DataFrame:
    yaw_cols = [col for col in input_df.columns if "Yaw" in col]
    yaw_na_before = input_df["YawAngleMean"].isna().sum()
    for exclusion in yaw_data_exclusions_utc:
        excl_turbine = exclusion[0]
        exclusion_period_start_utc = exclusion[1]
        exclusion_period_end_utc = exclusion[2]
        if excl_turbine == "ALL":
            input_df.loc[
                pd.IndexSlice[
                    :,
                    (exclusion_period_start_utc - pd.Timedelta(minutes=9, seconds=59)) : (
                        exclusion_period_end_utc - pd.Timedelta(seconds=1)
                    ),
                ],
                yaw_cols,
            ] = pd.NA
        else:
            input_df.loc[
                pd.IndexSlice[
                    excl_turbine,
                    (exclusion_period_start_utc - pd.Timedelta(minutes=9, seconds=59)) : (
                        exclusion_period_end_utc - pd.Timedelta(seconds=1)
                    ),
                ],
                yaw_cols,
            ] = pd.NA
    yaw_na_after = input_df["YawAngleMean"].isna().sum()
    print_filter_stats(
        filter_name="filter_yaw_exclusions",
        na_rows=yaw_na_after - yaw_na_before,
        total_rows=len(input_df),
        just_yaw=True,
    )
    return input_df


def filter_cfg_exclusions(
    input_df: pd.DataFrame, exclusion_periods_utc: ExclusionPeriodsType, yaw_data_exclusions_utc: ExclusionPeriodsType
) -> pd.DataFrame:
    input_df = filter_exclusions(input_df=input_df, exclusion_periods_utc=exclusion_periods_utc)
    return filter_yaw_exclusions(input_df=input_df, yaw_data_exclusions_utc=yaw_data_exclusions_utc)


def filter_downtime(df: pd.DataFrame) -> pd.DataFrame:
    pw_na_before = df["ActivePowerMean"].isna().sum()
    fully_available = df["ShutdownDuration"] == 0
    df.loc[~fully_available, :] = pd.NA
    pw_na_after = df["ActivePowerMean"].isna().sum()
    print_filter_stats(filter_name="filter_downtime", na_rows=pw_na_after - pw_na_before, total_rows=len(df))
    return df


def filter_rpm_and_pt_oor_one_ttype(
    df: pd.DataFrame,
    rpm_lower: float,
    rpm_upper: float,
    pt_lower: float,
    pt_upper: float,
) -> tuple[pd.DataFrame, int]:
    na_rows_before = df["ActivePowerMean"].isna().sum()
    df.loc[(df["GenRpmMean"] < rpm_lower), :] = pd.NA
    df.loc[(df["GenRpmMean"] > rpm_upper), :] = pd.NA
    df.loc[(df["PitchAngleMean"] < pt_lower), :] = pd.NA
    df.loc[(df["PitchAngleMean"] > pt_upper), :] = pd.NA
    na_rows_after = df["ActivePowerMean"].isna().sum()
    na_rows = na_rows_after - na_rows_before
    return df, na_rows


def filter_rpm_or_pt_curve(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    y_centile: float,
    x_bin_edges: list[float],
    reject_high: bool,
    y_margin: float,
    filter_last_bin: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # suppress RuntimeWarning due to empty slices, for example the high wind speed bins are often empty.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        filter_curve = df.groupby(by=pd.cut(df[x_col], bins=x_bin_edges, retbins=False), observed=False).agg(
            y_limit=pd.NamedAgg(column=y_col, aggfunc=lambda x: np.nanpercentile(x, y_centile)),
            min_y=pd.NamedAgg(column=y_col, aggfunc=lambda x: x.min()),
            max_y=pd.NamedAgg(column=y_col, aggfunc=lambda x: x.max()),
        )

    margin_sign = 1 if reject_high else -1
    filter_curve["y_limit"] = filter_curve["y_limit"] + (margin_sign * y_margin)

    if not filter_last_bin:
        if reject_high:
            filter_curve.loc[filter_curve.index[-1], "y_limit"] = filter_curve.iloc[-1]["max_y"]
        else:
            filter_curve.loc[filter_curve.index[-1], "y_limit"] = filter_curve.iloc[-1]["min_y"]

    filt_threshold = np.interp(df[x_col].values, [x.mid for x in filter_curve.index], filter_curve["y_limit"])
    if reject_high:
        df.loc[df[y_col] > filt_threshold, :] = pd.NA
    else:
        df.loc[df[y_col] < filt_threshold, :] = pd.NA

    return df, filter_curve


def filter_rpm_and_pt_curve_one_ttype(
    df: pd.DataFrame,
    *,
    ttype: str,
    pitch_to_stall: bool,
    rated_power_kw: float,
    rpm_v_pw_margin_factor: float,
    plot_cfg: PlotConfig | None,
) -> pd.DataFrame:
    na_rows_before = df["ActivePowerMean"].isna().sum()
    df_pre_filter = df.copy()

    if pitch_to_stall:
        pt_centile_pw = 10
        pt_centile_ws = 1
    else:
        pt_centile_pw = 90
        pt_centile_ws = 99
    rpm_centile_pw = 10
    rpm_centile_ws = 1
    pt_margin_pw = 2.5
    pt_margin_ws = 2.5
    rpm_margin_pw = df["GenRpmMean"].max() * rpm_v_pw_margin_factor
    rpm_margin_ws = df["GenRpmMean"].max() * 0.05

    pw_bin_width = rated_power_kw / 50
    pw_bin_edges = list(np.arange(0, rated_power_kw + pw_bin_width, pw_bin_width))

    ws_bin_width = 2
    ws_bin_edges = list(np.arange(0, df["WindSpeedMean"].max() + ws_bin_width, ws_bin_width))

    for _x in range(2):
        # the below filter steps are performed twice in case there is interaction between the filters
        df, pt_v_pw_curve = filter_rpm_or_pt_curve(
            df=df,
            x_col="pw_clipped",
            y_col="PitchAngleMean",
            y_centile=pt_centile_pw,
            x_bin_edges=pw_bin_edges,
            reject_high=not pitch_to_stall,
            y_margin=pt_margin_pw,
            filter_last_bin=False,
        )

        df, pt_v_ws_curve = filter_rpm_or_pt_curve(
            df=df,
            x_col="WindSpeedMean",
            y_col="PitchAngleMean",
            y_centile=pt_centile_ws,
            x_bin_edges=ws_bin_edges,
            reject_high=not pitch_to_stall,
            y_margin=pt_margin_ws,
        )

        df, rpm_v_pw_curve = filter_rpm_or_pt_curve(
            df=df,
            x_col="pw_clipped",
            y_col="GenRpmMean",
            y_centile=rpm_centile_pw,
            x_bin_edges=pw_bin_edges,
            reject_high=False,
            y_margin=rpm_margin_pw,
        )

        df, rpm_v_ws_curve = filter_rpm_or_pt_curve(
            df=df,
            x_col="WindSpeedMean",
            y_col="GenRpmMean",
            y_centile=rpm_centile_ws,
            x_bin_edges=ws_bin_edges,
            reject_high=False,
            y_margin=rpm_margin_ws,
        )

    if plot_cfg is not None:
        plot_filter_rpm_and_pt_curve_one_ttype_or_wtg(
            df=df_pre_filter,
            ttype_or_wtg=ttype,
            pt_v_pw_curve=pt_v_pw_curve,
            pt_v_ws_curve=pt_v_ws_curve,
            rpm_v_pw_curve=rpm_v_pw_curve,
            rpm_v_ws_curve=rpm_v_ws_curve,
            plot_cfg=plot_cfg,
        )
        if not plot_cfg.skip_per_turbine_plots:
            for wtg in df_pre_filter.index.unique(level="TurbineName"):
                plot_filter_rpm_and_pt_curve_one_ttype_or_wtg(
                    df=df_pre_filter.loc[wtg],
                    ttype_or_wtg=wtg,
                    pt_v_pw_curve=pt_v_pw_curve,
                    pt_v_ws_curve=pt_v_ws_curve,
                    rpm_v_pw_curve=rpm_v_pw_curve,
                    rpm_v_ws_curve=rpm_v_ws_curve,
                    plot_cfg=plot_cfg,
                )

    na_rows_after = df["ActivePowerMean"].isna().sum()
    na_rows = na_rows_after - na_rows_before
    return df, na_rows


def filter_missing_rpm_or_pt(df: pd.DataFrame) -> pd.DataFrame:
    pw_na_before = df["ActivePowerMean"].isna().sum()
    df.loc[df[["PitchAngleMean", "GenRpmMean"]].isna().any(axis=1), :] = pd.NA
    pw_na_after = df["ActivePowerMean"].isna().sum()
    print_filter_stats(
        filter_name="filter_missing_rpm_and_pt",
        na_rows=pw_na_after - pw_na_before,
        total_rows=len(df),
        reason="rpm or pitch are NA",
    )
    return df


def filter_rpm_and_pt(input_df: pd.DataFrame, cfg: WindUpConfig, plot_cfg: PlotConfig | None) -> pd.DataFrame:
    df_idx_before = input_df.index.copy()

    input_df = filter_missing_rpm_or_pt(df=input_df)

    # filter out of range pitch and rpm by turbine type
    df_prefilt = input_df.copy()
    filt_df = pd.DataFrame()
    na_rows = 0
    for ttype in cfg.list_unique_turbine_types():
        wtgs = cfg.list_turbine_ids_of_type(ttype)
        df_ttype = df_prefilt.loc[wtgs]
        rpm_lower, rpm_upper = cfg.get_normal_operation_genrpm_range(ttype=ttype)
        pt_lower, pt_upper = cfg.get_normal_operation_pitch_range(ttype=ttype)
        df_, na_rows_ = filter_rpm_and_pt_oor_one_ttype(
            df=df_ttype,
            rpm_lower=rpm_lower,
            rpm_upper=rpm_upper,
            pt_lower=pt_lower,
            pt_upper=pt_upper,
        )
        filt_df = pd.concat([filt_df, df_])
        na_rows += na_rows_
    filt_df = filt_df.sort_index()
    print_filter_stats(
        filter_name="filter_rpm_and_pt",
        na_rows=na_rows,
        total_rows=len(filt_df),
        reason="rpm or pitch are out of range",
    )

    # filter based on ops curves by turbine type
    df_prefilt = filt_df.copy()
    filt_df = pd.DataFrame()
    na_rows = 0
    for ttype in cfg.list_unique_turbine_types():
        wtgs = cfg.list_turbine_ids_of_type(ttype)
        df_ttype = df_prefilt.loc[wtgs]
        df_, na_rows_ = filter_rpm_and_pt_curve_one_ttype(
            df=df_ttype,
            ttype=ttype.turbine_type,
            pitch_to_stall=ttype.pitch_to_stall,
            rated_power_kw=ttype.rated_power_kw,
            rpm_v_pw_margin_factor=ttype.rpm_v_pw_margin_factor,
            plot_cfg=plot_cfg,
        )
        filt_df = pd.concat([filt_df, df_])
        na_rows += na_rows_
    filt_df = filt_df.sort_index()
    print_filter_stats(
        filter_name="filter_rpm_and_pt",
        na_rows=na_rows,
        total_rows=len(filt_df),
        reason="rpm or pitch curve filtering",
    )

    # check df.index is the same as df_idx_before
    if not filt_df.index.equals(df_idx_before):
        msg = "df.index has changed during filtering"
        raise ValueError(msg)
    return filt_df


def scada_multi_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("TurbineName", append=True).swaplevel(axis=0).sort_index()


def filter_scada_df(raw_df: pd.DataFrame, cfg: WindUpConfig, plot_cfg: PlotConfig) -> pd.DataFrame:
    ref_col = DataColumns.active_power_mean
    logger.info(
        f"{raw_df[ref_col].isna().sum()} rows "
        f"[{100 * raw_df[ref_col].isna().mean():.1f}%] "
        "of power data is missing before filtering",
    )
    filt_df = raw_df.copy()
    filt_df = filter_stuck_data(df=filt_df)
    filt_df = filter_bad_pw_ws(df=filt_df, max_rated_power=cfg.get_max_rated_power())
    filt_df = wrap_yaw_and_pitch(df=filt_df)
    filt_df = filter_wrong_yaw(df=filt_df)
    filt_df = filter_cfg_exclusions(
        input_df=filt_df,
        exclusion_periods_utc=cfg.exclusion_periods_utc,
        yaw_data_exclusions_utc=cfg.yaw_data_exclusions_utc,
    )
    filt_df = filter_downtime(df=filt_df)
    plot_ops_curves_per_ttype(cfg=cfg, df=filt_df, title_end="downtime filter", plot_cfg=plot_cfg)
    filt_df = add_pw_clipped(df=filt_df, wtgs=cfg.asset.wtgs)
    filt_df = filter_rpm_and_pt(input_df=filt_df, cfg=cfg, plot_cfg=plot_cfg)
    plot_data_coverage_heatmap(
        df=filt_df,
        plot_title=f"{cfg.asset.name} data coverage after filtering",
        plot_cfg=plot_cfg,
    )
    logger.info(
        f"{filt_df[ref_col].isna().sum()} rows [{100 * filt_df[ref_col].isna().mean():.1f}%] "
        "of power data is missing after filtering",
    )

    filt_df[RAW_POWER_COL] = raw_df[DataColumns.active_power_mean]
    filt_df[RAW_WINDSPEED_COL] = raw_df[DataColumns.wind_speed_mean]
    filt_df[RAW_DOWNTIME_S_COL] = raw_df[DataColumns.shutdown_duration]
    filt_df[RAW_YAWDIR_COL] = raw_df[DataColumns.yaw_angle_mean]

    plot_ops_curves_per_ttype(cfg=cfg, df=filt_df, title_end="after filtering", plot_cfg=plot_cfg)

    return filt_df


def get_raw_scada_and_cfg_from_file(
    cfg: WindUpConfig,
    scada_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    first_datetime_utc_start: pd.Timestamp,
    last_datetime_utc_start: pd.Timestamp,
    plot_cfg: PlotConfig | None,
) -> tuple[pd.DataFrame, WindUpConfig]:
    scada_raw, md = load_smart_scada_and_md_from_file(
        asset_name=cfg.asset.name,
        scada_df=scada_df,
        metadata_df=metadata_df,
        first_datetime_utc_start=first_datetime_utc_start,
        last_datetime_utc_start=last_datetime_utc_start,
        timebase_s=cfg.timebase_s,
    )
    cfg = add_smart_lat_long_to_cfg(md=md, cfg=cfg)
    if plot_cfg is not None:
        print_and_plot_capacity_factor(scada_df=scada_raw, cfg=cfg, plots_cfg=plot_cfg)
    scada_mi_df = scada_multi_index(scada_raw)
    del scada_raw
    if plot_cfg is not None:
        plot_ops_curves_per_ttype(cfg=cfg, df=scada_mi_df, title_end="before filtering", plot_cfg=plot_cfg)
        plot_data_coverage_heatmap(
            df=scada_mi_df,
            plot_title=f"{cfg.asset.name} data coverage before filtering",
            plot_cfg=plot_cfg,
        )
    return scada_mi_df, cfg
