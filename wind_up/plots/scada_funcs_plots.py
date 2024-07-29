import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from wind_up.constants import SCATTER_ALPHA, SCATTER_MARKERSCALE, SCATTER_S, DataColumns
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.plots.misc_plots import bubble_plot

logger = logging.getLogger(__name__)


def plot_data_coverage_heatmap(df: pd.DataFrame, plot_title: str, plot_cfg: PlotConfig) -> None:
    # calculate data coverage per turbine
    covdf = df.groupby("TurbineName", observed=False).agg(
        power=pd.NamedAgg(column="ActivePowerMean", aggfunc=lambda x: x.count() / x.size),
        windspeed=pd.NamedAgg(column="WindSpeedMean", aggfunc=lambda x: x.count() / x.size),
        yaw=pd.NamedAgg(column="YawAngleMean", aggfunc=lambda x: x.count() / x.size),
        rpm=pd.NamedAgg(column="GenRpmMean", aggfunc=lambda x: x.count() / x.size),
        pitch=pd.NamedAgg(column="PitchAngleMean", aggfunc=lambda x: x.count() / x.size),
    )

    plt.figure()
    sns.heatmap(covdf, annot=True, fmt=".2f", vmax=1, vmin=min(0.5, covdf.min().min()))
    plt.title(plot_title)
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / f"{plot_title}.png")
    if plot_cfg.show_plots:
        plt.show()
    plt.close()


def calc_cf_by_turbine(scada_df: pd.DataFrame, cfg: WindUpConfig) -> pd.DataFrame:
    rows_per_hour = 3600 / cfg.timebase_s
    cf_df = scada_df.groupby("TurbineName", observed=False).agg(
        hours=pd.NamedAgg(column="TurbineName", aggfunc=lambda x: x.count() / rows_per_hour),
        MWh=pd.NamedAgg(column="ActivePowerMean", aggfunc=lambda x: x.sum() / rows_per_hour / 1000),
    )
    for i, rp in zip(
        [x.name for x in cfg.asset.wtgs],
        [x.turbine_type.rated_power_kw for x in cfg.asset.wtgs],
        strict=True,
    ):
        cf_df.loc[i, "rated_power_kW"] = rp
    cf_df["CF"] = cf_df["MWh"] / (cf_df["hours"] * cf_df["rated_power_kW"] / 1000)
    return cf_df


def print_and_plot_capacity_factor(scada_df: pd.DataFrame, cfg: WindUpConfig, plots_cfg: PlotConfig) -> None:
    cf_df = calc_cf_by_turbine(scada_df=scada_df, cfg=cfg)
    title = f"{cfg.asset.name} capacity factor"
    plots_cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    bubble_plot(
        cfg=cfg,
        series=cf_df["CF"] * 100,
        title=f"{cfg.asset.name} capacity factor",
        cbarunits="%",
        save_path=plots_cfg.plots_dir / f"{title}.png",
        show_plot=plots_cfg.show_plots,
    )

    logger.info(f'average capacity factor: {cf_df["CF"].mean() * 100:.1f}%')
    _table = tabulate(
        (cf_df.sort_values(by="CF", ascending=False)["CF"][0:3] * 100).to_frame(),
        tablefmt="outline",
        floatfmt=".1f",
    )
    logger.info(f"top 3 capacity factor [%]:\n{_table}")
    _table = tabulate((cf_df.sort_values(by="CF")["CF"][0:3] * 100).to_frame(), tablefmt="outline", floatfmt=".1f")
    logger.info(f"bottom 3 capacity factor [%]:\n{_table}")


def plot_ops_curves_per_ttype(cfg: WindUpConfig, df: pd.DataFrame, title_end: str, plot_cfg: PlotConfig) -> None:
    for ttype in cfg.list_unique_turbine_types():
        wtgs = cfg.list_turbine_ids_of_type(ttype)
        df_ttype = df.loc[wtgs]
        plot_ops_curves_one_ttype_or_wtg(
            df=df_ttype,
            ttype_or_wtg=ttype.turbine_type,
            title_end=title_end,
            plot_cfg=plot_cfg,
        )
        if not plot_cfg.skip_per_turbine_plots:
            for wtg in wtgs:
                plot_ops_curves_one_ttype_or_wtg(
                    df=df_ttype.loc[[wtg]],
                    ttype_or_wtg=wtg,
                    title_end=title_end,
                    plot_cfg=plot_cfg,
                )
                plot_ops_curve_timelines_one_wtg(
                    wtg_df=df_ttype.loc[wtg],
                    wtg_name=wtg,
                    title_end=title_end,
                    plot_cfg=plot_cfg,
                )


def plot_ops_curves_one_ttype_or_wtg(df: pd.DataFrame, ttype_or_wtg: str, title_end: str, plot_cfg: PlotConfig) -> None:
    plt.figure()
    plt.scatter(df["WindSpeedMean"], df["ActivePowerMean"], s=SCATTER_S, alpha=SCATTER_ALPHA)
    plot_title = f"{ttype_or_wtg} power curve {title_end}"
    plt.title(plot_title)
    plt.xlabel("WindSpeedMean [m/s]")
    plt.ylabel("ActivePowerMean [kW]")
    plt.grid()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        t_dir = plot_cfg.plots_dir / ttype_or_wtg
        t_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(t_dir / f"{plot_title}.png")
    plt.close()

    # plot rpm and pitch vs power and wind speed in a 2 by 2 grid
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(df["ActivePowerMean"], df["GenRpmMean"], s=SCATTER_S, alpha=SCATTER_ALPHA)
    plt.xlabel("ActivePowerMean [kW]")
    plt.ylabel("GenRpmMean [RPM]")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.scatter(df["WindSpeedMean"], df["GenRpmMean"], s=SCATTER_S, alpha=SCATTER_ALPHA)
    plt.xlabel("WindSpeedMean [m/s]")
    plt.ylabel("GenRpmMean [RPM]")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.scatter(df["ActivePowerMean"], df["PitchAngleMean"], s=SCATTER_S, alpha=SCATTER_ALPHA)
    plt.xlabel("ActivePowerMean [kW]")
    plt.ylabel("PitchAngleMean [deg]")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.scatter(df["WindSpeedMean"], df["PitchAngleMean"], s=SCATTER_S, alpha=SCATTER_ALPHA)
    plt.xlabel("WindSpeedMean [m/s]")
    plt.ylabel("PitchAngleMean [deg]")
    plt.grid()

    plot_title = f"{ttype_or_wtg} ops curves, {title_end}"
    plt.suptitle(plot_title)
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(t_dir / f"{plot_title}.png")
    plt.close()


def plot_ops_curve_timelines_one_wtg(wtg_df: pd.DataFrame, wtg_name: str, title_end: str, plot_cfg: PlotConfig) -> None:
    dropna_df = wtg_df.dropna(
        subset=[
            DataColumns.wind_speed_mean,
            DataColumns.active_power_mean,
            DataColumns.gen_rpm_mean,
            DataColumns.pitch_angle_mean,
        ]
    )
    gen_df = dropna_df[dropna_df[DataColumns.active_power_mean] > 0].copy()

    for descr, x_var, y_var, x_bin_width in [
        ("power curve", DataColumns.wind_speed_mean, DataColumns.active_power_mean, 1),
        ("rpm v power", DataColumns.active_power_mean, DataColumns.gen_rpm_mean, 0),
        ("pitch v ws", DataColumns.wind_speed_mean, DataColumns.pitch_angle_mean, 1),
    ]:
        bins = np.arange(0, gen_df[x_var].max() + x_bin_width, x_bin_width) if x_bin_width > 0 else 10
        mean_curve = gen_df.groupby(pd.cut(gen_df[x_var], bins=bins, retbins=False), observed=True).agg(
            x_mean=pd.NamedAgg(column=x_var, aggfunc="mean"),
            y_mean=pd.NamedAgg(column=y_var, aggfunc="mean"),
        )
        gen_df["expected_y"] = np.interp(gen_df[x_var], mean_curve["x_mean"], mean_curve["y_mean"])

        daily_df = gen_df.resample("D").mean()
        monthly_df = gen_df.resample("ME").mean()
        if y_var == DataColumns.pitch_angle_mean:
            daily_df["relative_y"] = daily_df[y_var] - daily_df["expected_y"]
            monthly_df["relative_y"] = monthly_df[y_var] - monthly_df["expected_y"]
        else:
            daily_df["relative_y"] = (daily_df[y_var] / daily_df["expected_y"]).clip(0.5, 1.5)
            monthly_df["relative_y"] = (monthly_df[y_var] / monthly_df["expected_y"]).clip(0.5, 1.5)

        plt.figure()
        plt.plot(daily_df.index, daily_df["relative_y"], label="daily")
        plt.plot(monthly_df.index, monthly_df["relative_y"], label="monthly")
        plot_title = f"{wtg_name} relative {descr} timeline {title_end}"
        plt.title(plot_title)
        plt.xlabel("date")
        plt.ylabel(f"relative {descr}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if plot_cfg.show_plots:
            plt.show()
        if plot_cfg.save_plots:
            t_dir = plot_cfg.plots_dir / wtg_name
            plt.savefig(t_dir / f"{plot_title}.png")
        plt.close()


def plot_toggle_ops_curves_one_ttype_or_wtg(
    input_df: pd.DataFrame,
    *,
    ttype_or_wtg: str,
    title_end: str,
    toggle_name: str,
    ws_col: str,
    pw_col: str,
    pt_col: str,
    rpm_col: str,
    plot_cfg: PlotConfig,
    sub_dir: str | None = None,
) -> None:
    pd.set_option("future.no_silent_downcasting", True)  # noqa FBT003
    if "toggle_on" not in input_df.columns or "toggle_off" not in input_df.columns:
        df_off = input_df[input_df["test_toggle_off"].fillna(value=False).infer_objects(copy=False)]
        df_on = input_df[input_df["test_toggle_on"].fillna(value=False).infer_objects(copy=False)]
    else:
        df_off = input_df[input_df["toggle_off"].fillna(value=False).infer_objects(copy=False)]
        df_on = input_df[input_df["toggle_on"].fillna(value=False).infer_objects(copy=False)]

    plt.figure()
    plt.scatter(
        df_off[ws_col],
        df_off[pw_col],
        s=SCATTER_S,
        alpha=SCATTER_ALPHA,
        label=f"{toggle_name} OFF",
    )
    plt.scatter(
        df_on[ws_col],
        df_on[pw_col],
        s=SCATTER_S,
        alpha=SCATTER_ALPHA,
        label=f"{toggle_name} ON",
    )
    plot_title = f"{ttype_or_wtg} power curve by {toggle_name}, {title_end}"
    plt.title(plot_title)
    plt.xlabel(f"{ws_col} [m/s]")
    plt.ylabel(f"{pw_col} [kW]")
    plt.grid()
    plt.legend(loc="best", markerscale=SCATTER_MARKERSCALE)
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        t_dir = plot_cfg.plots_dir / ttype_or_wtg if sub_dir is None else plot_cfg.plots_dir / sub_dir
        t_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(t_dir / f"{plot_title}.png")
    plt.close()

    # plot rpm and pitch vs power and wind speed in a 2 by 2 grid
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(
        df_off[pw_col],
        df_off[rpm_col],
        s=SCATTER_S,
        alpha=SCATTER_ALPHA,
        label=f"{toggle_name} OFF",
    )
    plt.scatter(
        df_on[pw_col],
        df_on[rpm_col],
        s=SCATTER_S,
        alpha=SCATTER_ALPHA,
        label=f"{toggle_name} ON",
    )
    plt.xlabel(f"{pw_col} [kW]")
    plt.ylabel(f"{rpm_col} [RPM]")
    plt.grid()
    plt.legend(loc="best", markerscale=SCATTER_MARKERSCALE)

    plt.subplot(2, 2, 2)
    plt.scatter(df_off[ws_col], df_off[rpm_col], s=SCATTER_S, alpha=SCATTER_ALPHA)
    plt.scatter(df_on[ws_col], df_on[rpm_col], s=SCATTER_S, alpha=SCATTER_ALPHA)
    plt.xlabel(f"{ws_col} [m/s]")
    plt.ylabel(f"{rpm_col} [RPM]")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.scatter(df_off[pw_col], df_off[pt_col], s=SCATTER_S, alpha=SCATTER_ALPHA)
    plt.scatter(df_on[pw_col], df_on[pt_col], s=SCATTER_S, alpha=SCATTER_ALPHA)
    plt.xlabel(f"{pw_col} [kW]")
    plt.ylabel(f"{pt_col} [deg]")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.scatter(df_off[ws_col], df_off[pt_col], s=SCATTER_S, alpha=SCATTER_ALPHA)
    plt.scatter(df_on[ws_col], df_on[pt_col], s=SCATTER_S, alpha=SCATTER_ALPHA)
    plt.xlabel(f"{ws_col} [m/s]")
    plt.ylabel(f"{pt_col} [deg]")
    plt.grid()

    plot_title = f"{ttype_or_wtg} ops curves by {toggle_name}, {title_end}"
    plt.suptitle(plot_title)
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(t_dir / f"{plot_title}.png")
    plt.close()


def compare_ops_curves_pre_post(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    *,
    wtg_name: str,
    ws_col: str,
    pw_col: str,
    pt_col: str,
    rpm_col: str,
    plot_cfg: PlotConfig,
    is_toggle_test: bool,
    sub_dir: str | None = None,
) -> None:
    if is_toggle_test:
        plot_toggle_ops_curves_one_ttype_or_wtg(
            input_df=pd.concat([pre_df, post_df]),
            ttype_or_wtg=wtg_name,
            title_end="power performance data",
            toggle_name="toggle",
            ws_col=ws_col,
            pw_col=pw_col,
            pt_col=pt_col,
            rpm_col=rpm_col,
            plot_cfg=plot_cfg,
            sub_dir=sub_dir,
        )
    else:
        pre_df_fake_toggle = pre_df.copy()
        post_df_fake_toggle = post_df.copy()
        pre_df_fake_toggle["test_toggle_off"] = True
        post_df_fake_toggle["test_toggle_on"] = True
        plot_toggle_ops_curves_one_ttype_or_wtg(
            input_df=pd.concat([pre_df_fake_toggle, post_df_fake_toggle]),
            ttype_or_wtg=wtg_name,
            title_end="power performance data",
            toggle_name="upgrade",
            ws_col=ws_col,
            pw_col=pw_col,
            pt_col=pt_col,
            rpm_col=rpm_col,
            plot_cfg=plot_cfg,
            sub_dir=sub_dir,
        )


def plot_filter_rpm_and_pt_curve_one_ttype_or_wtg(
    df: pd.DataFrame,
    ttype_or_wtg: str,
    pt_v_pw_curve: pd.DataFrame,
    pt_v_ws_curve: pd.DataFrame,
    rpm_v_pw_curve: pd.DataFrame,
    rpm_v_ws_curve: pd.DataFrame,
    plot_cfg: PlotConfig,
) -> None:
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(df["pw_clipped"], df["GenRpmMean"], s=SCATTER_S, alpha=SCATTER_ALPHA)
    x = [rpm_v_pw_curve.index[0].left] + [x.mid for x in rpm_v_pw_curve.index] + [rpm_v_pw_curve.index[-1].right]
    y = [rpm_v_pw_curve["y_limit"].iloc[0], *list(rpm_v_pw_curve["y_limit"]), rpm_v_pw_curve["y_limit"].iloc[-1]]
    plt.plot(x, y, color="red")
    plt.xlabel("pw_clipped [kW]")
    plt.ylabel("GenRpmMean [deg]")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.scatter(df["WindSpeedMean"], df["GenRpmMean"], s=SCATTER_S, alpha=SCATTER_ALPHA)
    x = [rpm_v_ws_curve.index[0].left] + [x.mid for x in rpm_v_ws_curve.index] + [rpm_v_ws_curve.index[-1].right]
    y = [rpm_v_ws_curve["y_limit"].iloc[0], *list(rpm_v_ws_curve["y_limit"]), rpm_v_ws_curve["y_limit"].iloc[-1]]
    plt.plot(x, y, color="red")
    plt.xlabel("WindSpeedMean [m/s]")
    plt.ylabel("GenRpmMean [deg]")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.scatter(df["pw_clipped"], df["PitchAngleMean"], s=SCATTER_S, alpha=SCATTER_ALPHA)
    x = [pt_v_pw_curve.index[0].left] + [x.mid for x in pt_v_pw_curve.index] + [pt_v_pw_curve.index[-1].right]
    y = [pt_v_pw_curve["y_limit"].iloc[0], *list(pt_v_pw_curve["y_limit"]), pt_v_pw_curve["y_limit"].iloc[-1]]
    plt.plot(x, y, color="red")
    plt.xlabel("pw_clipped [kW]")
    plt.ylabel("PitchAngleMean [deg]")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.scatter(df["WindSpeedMean"], df["PitchAngleMean"], s=SCATTER_S, alpha=SCATTER_ALPHA)
    x = [pt_v_ws_curve.index[0].left] + [x.mid for x in pt_v_ws_curve.index] + [pt_v_ws_curve.index[-1].right]
    y = [pt_v_ws_curve["y_limit"].iloc[0], *list(pt_v_ws_curve["y_limit"]), pt_v_ws_curve["y_limit"].iloc[-1]]
    plt.plot(x, y, color="red")
    plt.xlabel("WindSpeedMean [m/s]")
    plt.ylabel("PitchAngleMean [deg]")
    plt.grid()

    plot_title = f"{ttype_or_wtg} rpm and pitch curve filters"
    plt.suptitle(plot_title)
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        (plot_cfg.plots_dir / ttype_or_wtg).mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_cfg.plots_dir / ttype_or_wtg / f"{plot_title}.png")
    plt.close()


def print_filter_stats(
    filter_name: str,
    na_rows: int,
    total_rows: int,
    *,
    just_yaw: bool = False,
    just_min_max: bool = False,
    reason: str = "",
) -> None:
    min_max_str = " Min & Max" if just_min_max else ""
    if len(reason) > 0:
        reason = f" because of {reason}"
    if just_yaw:
        logger.info(
            f"{filter_name} set {na_rows} rows [{100 * na_rows / total_rows:.1f}%] to NA yaw{min_max_str}{reason}"
        )
    else:
        logger.info(f"{filter_name} set {na_rows} rows [{100 * na_rows / total_rows:.1f}%] to NA{reason}")
