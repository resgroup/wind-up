import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_up.constants import RAW_WINDSPEED_COL, ROWS_PER_HOUR, SCATTER_ALPHA, SCATTER_S
from wind_up.models import PlotConfig

mpl.use("Agg")


def plot_pre_post_power_curves(
    *,
    test_name: str,
    ref_name: str,
    pp_df: pd.DataFrame,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    ws_col: str,
    pw_col: str,
    plot_cfg: PlotConfig,
) -> None:
    plt.figure()
    plt.scatter(pre_df[ws_col], pre_df[pw_col], s=SCATTER_S, alpha=SCATTER_ALPHA, label="pre upgrade")
    plt.scatter(post_df[ws_col], post_df[pw_col], s=SCATTER_S, alpha=SCATTER_ALPHA, label="post upgrade")
    plt.legend()
    plt.grid()
    plot_title = f"test={test_name} ref={ref_name} power curve data"
    plt.title(plot_title)
    plt.ylabel(f"{pw_col} [kW]")
    plt.xlabel(f"{ws_col} [m/s]")
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / f"{plot_title}.png")
    plt.close()

    plt.figure()
    plt.plot(pp_df["bin_mid"], pp_df["hours_pre"] / pp_df["hours_pre"].sum(), marker="s", label="pre")
    plt.plot(pp_df["bin_mid"], pp_df["hours_post"] / pp_df["hours_post"].sum(), marker="s", label="post")
    plt.plot(pp_df["bin_mid"], pp_df["f"], marker="s", label="long term")
    plt.legend()
    plt.grid()
    plot_title = f"test={test_name} ref={ref_name} relative frequency"
    plt.title(plot_title)
    plt.ylabel("fraction of time [-]")
    plt.xlabel("bin mid [m/s]")
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / f"{plot_title}.png")
    plt.close()

    plt.figure()
    plt.errorbar(pp_df["ws_mean_pre"], pp_df["pw_mean_pre_raw"], yerr=pp_df["pw_sem_pre"], label="pre raw", marker=".")
    plt.errorbar(
        pp_df["ws_mean_post"],
        pp_df["pw_mean_post_raw"],
        yerr=pp_df["pw_sem_post"],
        label="post raw",
        marker=".",
    )
    plt.legend()
    plt.grid()
    plot_title = f"test={test_name} ref={ref_name} raw power curve"
    plt.title(plot_title)
    plt.ylabel("mean power [kW]")
    plt.xlabel("mean wind speed [m/s]")
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / f"{plot_title}.png")
    plt.close()

    plt.figure()
    plt.errorbar(pp_df["bin_mid"], pp_df["pw_at_mid_pre"], yerr=pp_df["pw_sem_at_mid_pre"], label="pre", marker=".")
    plt.errorbar(
        pp_df["bin_mid"],
        pp_df["pw_at_mid_post"],
        yerr=pp_df["pw_sem_at_mid_post"],
        label="post",
        marker=".",
    )
    plt.legend()
    plt.grid()
    plot_title = f"test={test_name} ref={ref_name} cooked power curve"
    plt.title(plot_title)
    plt.ylabel("power at bin mid [kW]")
    plt.xlabel("wind speed bin mid [m/s]")
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / f"{plot_title}.png")
    plt.close()


def plot_pre_post_conditions(
    *,
    test_name: str,
    ref_name: str,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    wd_col: str,
    plot_cfg: PlotConfig,
) -> None:
    wd_width = 30
    plt.figure()
    plt.hist(
        pre_df[wd_col],
        weights=[1 / ROWS_PER_HOUR] * len(pre_df[wd_col]),
        bins=list(np.arange(0, 360 + wd_width / 2, wd_width)),
        label="pre",
    )
    plt.hist(
        post_df[wd_col],
        weights=[1 / ROWS_PER_HOUR] * len(post_df[wd_col]),
        bins=list(np.arange(0, 360 + wd_width / 2, wd_width)),
        alpha=0.5,
        label="post",
    )
    plot_title = f"{test_name} {ref_name} {wd_col} coverage"
    plt.title(plot_title)
    plt.xticks(np.arange(wd_width / 2, 360 + wd_width / 2, wd_width))
    plt.ylabel("hours")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / f"{plot_title}.png")
    plt.close()

    hod_width = 1
    plt.figure()
    plt.hist(
        pre_df.index.hour,
        weights=[1 / ROWS_PER_HOUR] * len(pre_df.index.hour),
        bins=list(np.arange(-hod_width / 2, 24 + hod_width / 2, hod_width)),
        label="pre",
    )
    plt.hist(
        post_df.index.hour,
        weights=[1 / ROWS_PER_HOUR] * len(post_df.index.hour),
        bins=list(np.arange(-hod_width / 2, 24 + hod_width / 2, hod_width)),
        alpha=0.5,
        label="post",
    )
    plot_title = f"{test_name} {ref_name} hour of day coverage"
    plt.title(plot_title)
    plt.xticks(np.arange(0, 25, 4))
    plt.ylabel("hours")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / f"{plot_title}.png")
    plt.close()

    moy_width = 1
    plt.figure()
    plt.hist(
        pre_df.index.month,
        weights=[1 / ROWS_PER_HOUR] * len(pre_df.index.month),
        bins=list(np.arange(0.5, 12 + moy_width, moy_width)),
        label="pre",
    )
    plt.hist(
        post_df.index.month,
        weights=[1 / ROWS_PER_HOUR] * len(post_df.index.month),
        bins=list(np.arange(0.5, 12 + moy_width, moy_width)),
        alpha=0.5,
        label="post",
    )
    plot_title = f"{test_name} {ref_name} month of year coverage"
    plt.title(plot_title)
    plt.xticks(np.arange(1, 13, 1))
    plt.ylabel("hours")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / f"{plot_title}.png")
    plt.close()


def plot_pre_post_pp_analysis(
    *,
    test_name: str,
    ref_name: str,
    pp_df: pd.DataFrame,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    ws_col: str,
    pw_col: str,
    wd_col: str,
    plot_cfg: PlotConfig,
    confidence_level: float,
) -> None:
    plot_pre_post_conditions(
        test_name=test_name,
        ref_name=ref_name,
        pre_df=pre_df.dropna(subset=[ws_col, pw_col]),
        post_df=post_df.dropna(subset=[ws_col, pw_col]),
        wd_col=wd_col,
        plot_cfg=plot_cfg,
    )

    pp_df = pp_df.copy()
    p_low = (1 - confidence_level) / 2
    p_high = 1 - p_low

    plot_pre_post_power_curves(
        test_name=test_name,
        ref_name=ref_name,
        pp_df=pp_df,
        pre_df=pre_df.dropna(subset=[ws_col, pw_col]),
        post_df=post_df.dropna(subset=[ws_col, pw_col]),
        ws_col=ws_col,
        pw_col=pw_col,
        plot_cfg=plot_cfg,
    )

    plt.figure()
    plt.plot(pp_df["bin_mid"], pp_df["uplift_kw"], color="b", marker="s")
    plt.plot(pp_df["bin_mid"], pp_df[f"uplift_p{p_low*100:.0f}_kw"], color="r", ls="--")
    plt.plot(pp_df["bin_mid"], pp_df[f"uplift_p{p_high*100:.0f}_kw"], color="r", ls="--")
    plt.grid()
    plot_title = f"test={test_name} ref={ref_name} uplift [kW] and {confidence_level*100:.0f}% CI"
    plt.title(plot_title)
    plt.ylabel("uplift [kW]")
    plt.xlabel("bin centre [m/s]")
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / f"{plot_title}.png")
    plt.close()

    plt.figure()
    plt.plot(
        pp_df["bin_mid"],
        pp_df["uplift_kw"] * pp_df["hours_per_year"].sum() * pp_df["f"] / 1000,
        color="b",
        marker="s",
    )
    plt.plot(
        pp_df["bin_mid"],
        pp_df[f"uplift_p{p_low*100:.0f}_kw"] * pp_df["hours_per_year"].sum() * pp_df["f"] / 1000,
        color="r",
        ls="--",
    )
    plt.plot(
        pp_df["bin_mid"],
        pp_df[f"uplift_p{p_high*100:.0f}_kw"] * pp_df["hours_per_year"].sum() * pp_df["f"] / 1000,
        color="r",
        ls="--",
    )
    plt.grid()
    plot_title = f"test={test_name} ref={ref_name} uplift [MWh] and {confidence_level*100:.0f}% CI"
    plt.title(plot_title)
    plt.ylabel("uplift [MWh]")
    plt.xlabel("bin centre [m/s]")
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / f"{plot_title}.png")
    plt.close()


def plot_pp_data_coverage(
    *,
    test_name: str,
    ref_name: str,
    pp_df: pd.DataFrame,
    test_df_pp_period: pd.DataFrame,
    ws_bin_width: float,
    plot_cfg: PlotConfig,
) -> None:
    ws_bin_edges = np.arange(0, test_df_pp_period["test_" + RAW_WINDSPEED_COL].max() + ws_bin_width, ws_bin_width)
    raw_df = test_df_pp_period.groupby(
        by=pd.cut(test_df_pp_period["test_" + RAW_WINDSPEED_COL], bins=ws_bin_edges, retbins=False),
        observed=False,
    ).agg(
        hours_raw=pd.NamedAgg(column="test_" + RAW_WINDSPEED_COL, aggfunc=lambda x: x.count() / ROWS_PER_HOUR),
    )
    raw_df["bin_mid"] = [x.mid for x in raw_df.index]

    plot_df = pp_df.merge(raw_df, on="bin_mid", how="left")
    plot_df["data_coverage"] = (
        ((plot_df["hours_pre"].fillna(0) + plot_df["hours_post"].fillna(0)) / plot_df["hours_raw"])
        .clip(upper=1)
        .fillna(1)
    )

    plt.figure()
    plt.plot(plot_df["bin_mid"], 100 * plot_df["data_coverage"], marker=".")
    plot_title = f"{test_name} {ref_name} data coverage"
    plt.title(plot_title)
    plt.ylabel("data coverage [%]")
    plt.xlabel("bin centre [m/s]")
    plt.ylim([0, 100])
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / f"{plot_title}.png")
    plt.close()
