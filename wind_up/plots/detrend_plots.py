import logging

import pandas as pd
from matplotlib import pyplot as plt

from wind_up.constants import SCATTER_ALPHA, SCATTER_MARKERSCALE, SCATTER_S
from wind_up.models import PlotConfig
from wind_up.result_manager import result_manager

logger = logging.getLogger(__name__)


def plot_detrend_ws_scatter(
    detrend_df: pd.DataFrame,
    test_name: str,
    ref_name: str,
    test_ws_col: str,
    ref_ws_col: str,
    plot_cfg: PlotConfig,
) -> None:
    plt.figure()
    plt.scatter(detrend_df[ref_ws_col], detrend_df[test_ws_col], s=SCATTER_S, alpha=SCATTER_ALPHA)
    title = f"{test_name} wind speed vs {ref_name} wind speed filtered for detrending"
    plt.title(title)
    plt.xlabel(f"{test_ws_col} [m/s]")
    plt.ylabel(f"{ref_ws_col} [m/s]")
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / ref_name / f"{title}.png")
    plt.close()


def plot_detrend_wsratio_v_dir_scen(
    wsratio_v_dir_scen: pd.DataFrame,
    *,
    test_name: str,
    ref_name: str,
    ref_wd_col: str,
    plot_cfg: PlotConfig,
) -> None:
    # figure out top scens to plot
    num_scens_to_plot = 5
    scens_to_plot = (
        wsratio_v_dir_scen.dropna(subset="ws_rom")
        .index.get_level_values("waking_scenario")
        .value_counts()[:num_scens_to_plot]
    )

    plt.figure()
    for scen in scens_to_plot.index:
        scen_df = wsratio_v_dir_scen.loc[scen]
        scen_df = scen_df.reindex(range(360), fill_value=pd.NA)
        plt.plot(scen_df.index, scen_df["hours"], marker=".", label=scen)
    title = f"{test_name} {ref_name} detrending hours"
    plt.title(title)
    plt.legend()
    plt.xlabel(f"{ref_wd_col} [deg]")
    plt.ylabel("hours")
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / ref_name / f"{title}.png")
    plt.close()

    plt.figure()
    for scen in scens_to_plot.index:
        scen_df = wsratio_v_dir_scen.loc[scen]
        scen_df = scen_df.reindex(range(360), fill_value=pd.NA)
        plt.plot(scen_df.index, scen_df["ws_rom"], marker=".", label=scen)
    title = f"{test_name} {ref_name} detrending ratio of mean ws"
    plt.title(title)
    plt.legend()
    plt.xlabel(f"{ref_wd_col} [deg]")
    plt.ylabel("ws_rom")
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / ref_name / f"{title}.png")
    plt.close()


def plot_apply_wsratio_v_wd_scen(
    p_df: pd.DataFrame,
    ref_ws_col: str,
    test_ws_col: str,
    detrend_ws_col: str,
    test_pw_col: str,
    test_name: str,
    ref_name: str,
    title_end: str,
    plot_cfg: PlotConfig,
) -> None:
    plt.figure()
    r2_before = p_df[ref_ws_col].corr(p_df[test_ws_col]) ** 2
    r2_after = p_df[detrend_ws_col].corr(p_df[test_ws_col]) ** 2
    plt.scatter(
        p_df[ref_ws_col],
        p_df[test_ws_col],
        s=SCATTER_S,
        alpha=SCATTER_ALPHA,
        label=f"before detrend, $r^2$ = {r2_before:.2f}",
    )
    plt.scatter(
        p_df[detrend_ws_col],
        p_df[test_ws_col],
        s=SCATTER_S,
        alpha=SCATTER_ALPHA,
        label=f"after detrend, $r^2$ = {r2_after:.2f}",
    )
    plot_title = f"{test_name} {ref_name} detrended wind speed {title_end}"
    plt.title(plot_title)
    plt.legend(loc="best", markerscale=SCATTER_MARKERSCALE)
    plt.xlabel("ref wind speed [m/s]")
    plt.ylabel(f"{test_ws_col} [m/s]")
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / ref_name / f"{plot_title}.png")
    plt.close()

    plt.figure()
    plt.scatter(p_df[ref_ws_col], p_df[test_pw_col], s=SCATTER_S, alpha=SCATTER_ALPHA, label="before detrend")
    plt.scatter(p_df[detrend_ws_col], p_df[test_pw_col], s=SCATTER_S, alpha=SCATTER_ALPHA, label="after detrend")
    plot_title = f"{test_name} {ref_name} detrended power curve {title_end}"
    plt.title(plot_title)
    plt.legend(loc="best", markerscale=SCATTER_MARKERSCALE)
    plt.xlabel("ref wind speed [m/s]")
    plt.ylabel(f"{test_pw_col} [kW]")
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / ref_name / f"{plot_title}.png")
    plt.close()


def plot_check_detrend_scatters(
    *,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    test_name: str,
    ref_name: str,
    test_ws_col: str,
    ref_ws_col: str,
    detrend_ws_col: str,
    plot_cfg: PlotConfig,
) -> None:
    plt.figure()
    plt.subplot(2, 1, 1)
    r2_before = pre_df[ref_ws_col].corr(pre_df[test_ws_col]) ** 2
    r2_after = pre_df[detrend_ws_col].corr(pre_df[test_ws_col]) ** 2
    plt.scatter(
        pre_df[ref_ws_col],
        pre_df[test_ws_col],
        s=SCATTER_S,
        alpha=SCATTER_ALPHA,
        label=f"pre-upgrade before detrend, $r^2$ = {r2_before:.2f}",
    )
    plt.scatter(
        pre_df[detrend_ws_col],
        pre_df[test_ws_col],
        s=SCATTER_S,
        alpha=SCATTER_ALPHA,
        label=f"pre-upgrade after detrend, $r^2$ = {r2_after:.2f}",
    )
    plt.xlabel("ref wind speed before/after detrend [m/s]")
    plt.ylabel(f"{test_ws_col} [m/s]")
    plt.grid()
    plt.legend(fontsize="small", loc="best", markerscale=SCATTER_MARKERSCALE)
    plt.subplot(2, 1, 2)
    r2_before = post_df[ref_ws_col].corr(post_df[test_ws_col]) ** 2
    r2_after = post_df[detrend_ws_col].corr(post_df[test_ws_col]) ** 2
    plt.scatter(
        post_df[ref_ws_col],
        post_df[test_ws_col],
        s=SCATTER_S,
        alpha=SCATTER_ALPHA,
        label=f"post-upgrade before detrend, $r^2$ = {r2_before:.2f}",
    )
    plt.scatter(
        post_df[detrend_ws_col],
        post_df[test_ws_col],
        s=SCATTER_S,
        alpha=SCATTER_ALPHA,
        label=f"post-upgrade after detrend, $r^2$ = {r2_after:.2f}",
    )
    plt.xlabel("ref wind speed before/after detrend [m/s]")
    plt.ylabel(f"{test_ws_col} [m/s]")
    plt.grid()
    plt.legend(fontsize="small", loc="best", markerscale=SCATTER_MARKERSCALE)
    plot_title = f"{test_name} vs {ref_name} wind speed detrend check"
    plt.suptitle(plot_title)
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / ref_name / f"{plot_title}.png")
    plt.close()


def plot_check_wsratio_v_dir(
    pre_wsratio_v_dir: pd.DataFrame,
    post_wsratio_v_dir: pd.DataFrame,
    test_name: str,
    ref_name: str,
    ref_wd_col: str,
    plot_cfg: PlotConfig,
) -> None:
    if pre_wsratio_v_dir.empty:
        logger.warning("pre_wsratio_v_dir is empty")
        return
    if post_wsratio_v_dir.empty:
        logger.warning("post_wsratio_v_dir is empty")
        return
    scen_to_plot = (
        pd.concat([pre_wsratio_v_dir, post_wsratio_v_dir])
        .dropna(subset="ws_rom")
        .groupby("waking_scenario")["hours"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )
    try:
        pre_plot_df = pre_wsratio_v_dir.loc[scen_to_plot].reindex(range(360), fill_value=pd.NA)
    except KeyError:
        pre_plot_df = pd.DataFrame(index=range(360), columns=["hours", "ws_rom"])
    try:
        post_plot_df = post_wsratio_v_dir.loc[scen_to_plot].reindex(range(360), fill_value=pd.NA)
    except KeyError:
        post_plot_df = pd.DataFrame(index=range(360), columns=["hours", "ws_rom"])

    plt.figure()
    plt.plot(pre_plot_df.index, pre_plot_df["hours"], marker=".", label=f"{scen_to_plot}, pre-upgrade")
    plt.plot(post_plot_df.index, post_plot_df["hours"], marker=".", label=f"{scen_to_plot}, post-upgrade")
    title = f"{test_name} {ref_name} check detrending hours"
    plt.title(title)
    plt.xlabel(f"{ref_wd_col} [deg]")
    plt.ylabel("hours")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / ref_name / f"{title}.png")
    plt.close()

    plt.figure()
    plt.plot(pre_plot_df.index, pre_plot_df["ws_rom"], marker=".", label=f"{scen_to_plot}, pre-upgrade")
    plt.plot(post_plot_df.index, post_plot_df["ws_rom"], marker=".", label=f"{scen_to_plot}, post-upgrade")
    title = f"{test_name} {ref_name} check detrending ratio of mean ws"
    plt.title(title)
    plt.xlabel(f"{ref_wd_col} [deg]")
    plt.ylabel("ws_rom")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / ref_name / f"{title}.png")
    plt.close()


def plot_check_applied_detrend(
    *,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    test_name: str,
    ref_name: str,
    test_ws_col: str,
    ref_ws_col: str,
    detrend_ws_col: str,
    ref_wd_col: str,
    pre_wsratio_v_dir_scen: pd.DataFrame,
    post_wsratio_v_dir_scen: pd.DataFrame,
    plot_cfg: PlotConfig,
) -> None:
    plot_check_detrend_scatters(
        pre_df=pre_df,
        post_df=post_df,
        test_name=test_name,
        ref_name=ref_name,
        test_ws_col=test_ws_col,
        ref_ws_col=ref_ws_col,
        detrend_ws_col=detrend_ws_col,
        plot_cfg=plot_cfg,
    )

    if (
        len(pre_wsratio_v_dir_scen.dropna(subset="ws_rom")) > 0
        or len(post_wsratio_v_dir_scen.dropna(subset="ws_rom")) > 0
    ):
        plot_check_wsratio_v_dir(
            pre_wsratio_v_dir=pre_wsratio_v_dir_scen,
            post_wsratio_v_dir=post_wsratio_v_dir_scen,
            test_name=test_name,
            ref_name=ref_name,
            ref_wd_col=ref_wd_col,
            plot_cfg=plot_cfg,
        )
    else:
        result_manager.warning("check detrending ratio of mean ws plots cannot be made, not enough data")
