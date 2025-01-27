from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from wind_up.circular_math import circ_diff

if TYPE_CHECKING:
    from wind_up.models import PlotConfig


def plot_yaw_direction_pre_post_per_signal(
    signal_name: str,
    test_wd_col: str,
    *,
    plot_pre_df: pd.DataFrame,
    plot_post_df: pd.DataFrame,
    test_name: str,
    ref_name: str,
    ref_ws_col: str,
    ref_wd_col: str,
    plot_cfg: PlotConfig,
    toggle_name: str | None = None,
) -> None:
    pre_label = f"{toggle_name} OFF" if toggle_name else "pre upgrade"
    post_label = f"{toggle_name} ON" if toggle_name else "post upgrade"

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.scatter(
        plot_pre_df[ref_wd_col],
        plot_pre_df[test_wd_col],
        s=2,
        alpha=0.5,
        label=pre_label,
    )
    plt.scatter(
        plot_post_df[ref_wd_col],
        plot_post_df[test_wd_col],
        s=2,
        alpha=0.5,
        label=post_label,
    )
    plt.xlabel(f"{ref_wd_col} [deg]")
    plt.ylabel(f"{test_wd_col} [deg]")
    plt.grid()
    plt.legend(loc="best", markerscale=2)

    plt.subplot(2, 1, 2)
    plt.scatter(
        plot_pre_df[ref_wd_col],
        plot_pre_df[signal_name],
        s=2,
        alpha=0.5,
        label=pre_label,
    )
    plt.scatter(
        plot_post_df[ref_wd_col],
        plot_post_df[signal_name],
        s=2,
        alpha=0.5,
        label=post_label,
    )
    plt.xlabel(f"{ref_wd_col} [deg]")
    plt.ylabel(f"{signal_name.replace('_', ' ')} [deg]")
    plt.grid()

    plot_title = (
        f"{test_name} ref {ref_name} yaw direction scatter"
        if signal_name == "yaw_offset"
        else f"{test_name} ref {ref_name} {signal_name.replace('_', ' ')} scatter"
    )
    plt.suptitle(plot_title)
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        (plot_cfg.plots_dir / test_name / "yaw_direction").mkdir(exist_ok=True)
        plt.savefig(plot_cfg.plots_dir / test_name / "yaw_direction" / f"{plot_title}.png")
    plt.close()

    ws_bin_width = 2
    wd_bin_width = 5

    ws_bin_edges = np.arange(
        0, max(plot_pre_df[ref_ws_col].max(), plot_post_df[ref_ws_col].max()) + ws_bin_width, ws_bin_width
    )
    plot_pre_df["ws_bins"] = pd.cut(plot_pre_df[ref_ws_col], bins=ws_bin_edges, retbins=False)
    plot_pre_df["ws_bin_centre"] = [x.mid for x in plot_pre_df["ws_bins"]]
    plot_post_df["ws_bins"] = pd.cut(plot_post_df[ref_ws_col], bins=ws_bin_edges, retbins=False)
    plot_post_df["ws_bin_centre"] = [x.mid for x in plot_post_df["ws_bins"]]

    def _get_mid(x: float | pd.Interval) -> float:
        if isinstance(x, pd.Interval):
            return x.mid
        return x

    ws_bin_edges = np.arange(
        0, max(plot_pre_df[ref_ws_col].max(), plot_post_df[ref_ws_col].max()) + ws_bin_width, ws_bin_width
    )
    plot_pre_df["ws_bins"] = pd.cut(plot_pre_df[ref_ws_col], bins=ws_bin_edges, retbins=False)
    plot_pre_df["ws_bin_centre"] = plot_pre_df["ws_bins"].apply(_get_mid)
    plot_post_df["ws_bins"] = pd.cut(plot_post_df[ref_ws_col], bins=ws_bin_edges, retbins=False)
    plot_post_df["ws_bin_centre"] = plot_post_df["ws_bins"].apply(_get_mid)

    wd_bin_edges = np.arange(0, plot_pre_df[ref_wd_col].max() + wd_bin_width, wd_bin_width)
    plot_pre_df["wd_bins"] = pd.cut(plot_pre_df[ref_wd_col], bins=wd_bin_edges, retbins=False)
    plot_pre_df["wd_bin_centre"] = plot_pre_df["wd_bins"].apply(_get_mid)
    plot_post_df["wd_bins"] = pd.cut(plot_post_df[ref_wd_col], bins=wd_bin_edges, retbins=False)
    plot_post_df["wd_bin_centre"] = plot_post_df["wd_bins"].apply(_get_mid)

    for i, label in enumerate([pre_label, post_label]):
        plot_df = plot_pre_df if i == 0 else plot_post_df
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        sns.heatmap(
            plot_df.pivot_table(
                index="ws_bin_centre",
                columns="wd_bin_centre",
                values=signal_name,
                aggfunc=lambda x: x.count() / 6,
                observed=False,
            ).iloc[::-1],
            annot=True,
            cmap="gray_r",
            fmt=".1f",
            linewidths=0.5,
            cbar_kws={"label": "hours of data"},
        )
        plt.xlabel("wind direction bin centre [deg]")
        plt.ylabel("wind speed bin centre [m/s]")

        plt.subplot(2, 1, 2)
        sns.heatmap(
            plot_df.pivot_table(
                index="ws_bin_centre", columns="wd_bin_centre", values=signal_name, observed=False
            ).iloc[::-1],
            annot=True,
            cmap="YlGnBu",
            fmt=".1f",
            linewidths=0.5,
            vmin=0,
            vmax=20,
            cbar_kws={"label": f"{signal_name.replace('_', ' ')} [deg]"},
        )

        plt.xlabel("wind direction bin centre [deg]")
        plt.ylabel("wind speed bin centre [m/s]")

        signal_descr = (
            f"{ref_name} minus {test_name} yaw direction"
            if signal_name == "yaw_offset"
            else f"{test_name} ref {ref_name} {signal_name.replace('_', ' ')}"
        )
        plot_title = f"{signal_descr} vs ws and wd {label}"
        plt.suptitle(plot_title)
        plt.tight_layout()
        if plot_cfg.show_plots:
            plt.show()
        if plot_cfg.save_plots:
            (plot_cfg.plots_dir / test_name / "yaw_direction").mkdir(exist_ok=True)
            plt.savefig(plot_cfg.plots_dir / test_name / "yaw_direction" / f"{plot_title}.png")
        plt.close()


def plot_yaw_direction_pre_post(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    *,
    test_name: str,
    ref_name: str,
    ref_ws_col: str,
    ref_wd_col: str,
    plot_cfg: PlotConfig,
    toggle_name: str | None = None,
) -> None:
    test_wd_col = "test_YawAngleMean"
    plot_pre_df = pre_df.dropna(subset=[test_wd_col, ref_ws_col, ref_wd_col]).copy()
    plot_post_df = post_df.dropna(subset=[test_wd_col, ref_ws_col, ref_wd_col]).copy()

    plot_pre_df["yaw_offset"] = circ_diff(plot_pre_df[ref_wd_col], plot_pre_df[test_wd_col])
    plot_post_df["yaw_offset"] = circ_diff(plot_post_df[ref_wd_col], plot_post_df[test_wd_col])

    plot_yaw_direction_pre_post_per_signal(
        signal_name="yaw_offset",
        test_wd_col=test_wd_col,
        plot_pre_df=plot_pre_df,
        plot_post_df=plot_post_df,
        test_name=test_name,
        ref_name=ref_name,
        ref_ws_col=ref_ws_col,
        ref_wd_col=ref_wd_col,
        plot_cfg=plot_cfg,
        toggle_name=toggle_name,
    )
    if "test_yaw_offset_command" in plot_pre_df.columns:
        plot_pre_df = plot_pre_df.rename(columns={"test_yaw_offset_command": "yaw_offset_command"})
        plot_post_df = plot_post_df.rename(columns={"test_yaw_offset_command": "yaw_offset_command"})
        plot_yaw_direction_pre_post_per_signal(
            signal_name="yaw_offset_command",
            test_wd_col=test_wd_col,
            plot_pre_df=plot_pre_df,
            plot_post_df=plot_post_df,
            test_name=test_name,
            ref_name=ref_name,
            ref_ws_col=ref_ws_col,
            ref_wd_col=ref_wd_col,
            plot_cfg=plot_cfg,
            toggle_name=toggle_name,
        )
