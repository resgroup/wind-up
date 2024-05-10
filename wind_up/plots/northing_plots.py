import logging

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from wind_up.constants import RAW_YAWDIR_COL
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.northing_utils import add_ok_yaw_col

logger = logging.getLogger(__name__)


def plot_northing_error(
    wf_df: pd.DataFrame,
    title_end: str,
    plot_cfg: PlotConfig,
    sub_dir: str | None = None,
) -> None:
    title_end = f" {title_end}" if len(title_end) > 0 else ""
    plt.figure(figsize=(10, 6))
    wtg_names = wf_df.index.unique(level="TurbineName")
    for wtg_name in wtg_names:
        plt.plot(wf_df.loc[wtg_name, "rolling_northing_error"].resample("6h").mean(), label=wtg_name)
    plt.legend(loc="best", ncol=max(1, round(len(wtg_names) // 5)), fontsize="small")
    plot_title = f"northing error{title_end}"
    plt.title(plot_title)
    plt.xlabel("datetime")
    plt.ylabel("rolling_northing_error [deg]")
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        if sub_dir is None:
            plot_cfg.plots_dir.mkdir(exist_ok=True)
            plt.savefig(plot_cfg.plots_dir / f"{plot_title}.png")
        else:
            (plot_cfg.plots_dir / sub_dir).mkdir(exist_ok=True)
            plt.savefig(plot_cfg.plots_dir / sub_dir / f"{plot_title}.png")
    plt.close()


def print_northing_error_summary(abs_north_errs: pd.DataFrame, *, wtgs_description: str, title_end: str) -> None:
    title_end = f" {title_end}" if len(title_end) > 0 else ""
    _table = tabulate((abs_north_errs.sort_values(ascending=False)[0:3]).to_frame(), tablefmt="outline", floatfmt=".1f")
    logger.info(f"top 3 {wtgs_description} needing northing correction{title_end}:\n{_table}")


def plot_and_print_northing_error(
    wf_df: pd.DataFrame,
    *,
    cfg: WindUpConfig,
    abs_north_errs: pd.DataFrame,
    title_end: str,
    plot_cfg: PlotConfig,
) -> None:
    print_northing_error_summary(abs_north_errs, wtgs_description="turbines", title_end=title_end)
    ref_abs_north_errs = abs_north_errs[[x.name for x in cfg.ref_wtgs]]
    print_northing_error_summary(ref_abs_north_errs, wtgs_description="REFERENCE turbines", title_end=title_end)

    plot_northing_error(
        wf_df=wf_df,
        title_end=title_end,
        plot_cfg=plot_cfg,
    )

    plot_northing_error(
        wf_df=wf_df.loc[[x.name for x in cfg.ref_wtgs]],
        title_end="REF wtgs " + title_end,
        plot_cfg=plot_cfg,
    )


def plot_northing_changepoint(
    wf_df: pd.DataFrame,
    *,
    northing_turbine: str,
    northed_col: str,
    north_ref_wd_col: str,
    northing_datetime_utc: pd.Timestamp,
    cfg: WindUpConfig,
    plot_cfg: PlotConfig,
) -> None:
    cols_to_plot = [RAW_YAWDIR_COL, northed_col, north_ref_wd_col]
    plot_df = wf_df.loc[northing_turbine].copy()
    days_to_plot = 2
    xmin = northing_datetime_utc - pd.Timedelta(days=days_to_plot)
    xmax = northing_datetime_utc + pd.Timedelta(days=days_to_plot)
    plot_df = plot_df.loc[xmin:xmax, :].dropna(subset=cols_to_plot, how="all")

    plt.figure()
    plt.plot(plot_df.index, plot_df[north_ref_wd_col], label=north_ref_wd_col)
    plt.plot(plot_df.index, plot_df[RAW_YAWDIR_COL], label=RAW_YAWDIR_COL)
    plt.plot(
        plot_df.index,
        plot_df[northed_col],
        label=northed_col,
    )
    plot_df = add_ok_yaw_col(
        plot_df,
        new_col_name="ok_for_yawdir",
        wd_col=northed_col,
        rated_power=next(x for x in cfg.asset.wtgs if x.name == northing_turbine).turbine_type.rated_power_kw,
        timebase_s=cfg.timebase_s,
    )
    plot_df = plot_df.loc[plot_df["ok_for_yawdir"]]
    plt.plot(
        plot_df.index,
        plot_df[northed_col],
        marker=".",
        linestyle="None",
        label=f"FILTERED yawdir_northed_to_{north_ref_wd_col}",
        color="darkgreen",
    )
    plt.axvline(northing_datetime_utc, color="grey", linestyle="--", label="northing correction")
    plt.xlabel("timestamp")
    plt.ylabel("direction [deg]")
    plt.xlim([xmin, xmax])
    plt.xticks(rotation=45)
    title = f"{northing_turbine} north_ref_wd_col={north_ref_wd_col} {northing_datetime_utc.strftime('%Y-%m-%d')}"
    plt.title(title)
    plt.legend(fontsize="small", ncol=2)
    plt.grid()
    plt.tight_layout()
    (plot_cfg.plots_dir / northing_turbine).mkdir(exist_ok=True)
    plt.savefig(plot_cfg.plots_dir / northing_turbine / f"{title}.png")
    plt.close()
