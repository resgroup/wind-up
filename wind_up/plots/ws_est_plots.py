import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_up.models import PlotConfig


def plot_ws_est_gain_xs_one_ttype(
    pc_low_high: pd.DataFrame,
    ttype: str,
    rated_power_kw: float,
    x0: float,
    x1: float,
    x2: float,
    x3: float,
    plot_cfg: PlotConfig,
) -> None:
    plt.figure()
    plt.plot(pc_low_high["x_mean"], pc_low_high["y_low"], label="q low")
    plt.plot(pc_low_high["x_mean"], pc_low_high["y_high"], label="q high")
    plt.vlines(x0, 0, rated_power_kw, label="ws0", color="#C30")
    plt.vlines(x1, 0, rated_power_kw, label="ws1", color="#D20")
    plt.vlines(x2, 0, rated_power_kw, label="ws2", color="#E10")
    plt.vlines(x3, 0, rated_power_kw, label="ws3", color="#F00")
    plot_title = f"{ttype} ws_est_gain xs"
    plt.title(plot_title)
    plt.xlabel("mean wind speed [m/s]")
    plt.ylabel("mean PositiveActivePower [kW]")
    plt.grid()
    plt.legend()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / ttype / f"{plot_title}.png")
    plt.close()


def plot_ws_est_one_ttype_or_wtg(  # noqa C901 PLR0915
    df: pd.DataFrame,
    ttype_or_wtg: str,
    pc_transposed: pd.DataFrame,
    plot_cfg: PlotConfig,
) -> None:
    plt.figure()
    plt.plot(pc_transposed.index, pc_transposed["ws_cp_corrected"])
    plot_title = f"{ttype_or_wtg} transposed power curve"
    plt.title(plot_title)
    plt.xlabel("pw_clipped bin centre [kW]")
    plt.ylabel("ws_cp_corrected [m/s]")
    plt.grid()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        tdir = plot_cfg.plots_dir / ttype_or_wtg
        tdir.mkdir(exist_ok=True, parents=True)
        plt.savefig(tdir / f"{plot_title}.png")
    plt.close()

    plt.figure()
    plt.scatter(df["ws_cp_corrected"], df["pw_clipped"], s=1, c=df["ws_est_gain"])
    plt.xlabel("ws_cp_corrected [m/s]")
    plt.ylabel("pw_clipped [kW]")
    plot_title = f"{ttype_or_wtg} power curve colored by ws_est_gain"
    plt.title(plot_title)
    plt.colorbar()
    plt.grid()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / ttype_or_wtg / f"{plot_title}.png")
    plt.close()

    plt.figure()
    plt.scatter(df["ws_est_blend"], df["pw_clipped"], s=1)
    plt.xlabel("ws_est_blend [m/s]")
    plt.ylabel("pw_clipped [kW]")
    plot_title = f"{ttype_or_wtg} wind speed estimated from power"
    plt.title(plot_title)
    plt.grid()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / ttype_or_wtg / f"{plot_title}.png")
    plt.close()

    plt.figure()
    plt.scatter(df["WindSpeedMean"], df["ws_est_blend"], s=1, c=df["ws_est_gain"])
    plt.xlabel("WindSpeedMean [m/s]")
    plt.ylabel("ws_est_blend [m/s]")
    plot_title = f"{ttype_or_wtg} wind speed estimated vs original colored by ws_est_gain"
    plt.title(plot_title)
    plt.grid()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / ttype_or_wtg / f"{plot_title}.png")
    plt.close()

    plt.figure()
    try:
        wsbins = np.arange(0, max(df["WindSpeedMean"].max(), df["ws_est_blend"].max()) + 0.5, 0.5)
    except ValueError:
        wsbins = np.arange(0, 25.5, 0.5)
    plt.hist(
        df["WindSpeedMean"],
        bins=wsbins.tolist(),
        label="WindSpeedMean",
    )
    plt.hist(
        df["ws_est_blend"],
        bins=wsbins.tolist(),
        alpha=0.5,
        label="ws_est_blend",
    )
    plt.xlabel("WindSpeedMean, ws_est_blend [m/s]")
    plt.ylabel("count")
    plot_title = f"{ttype_or_wtg} turbines wind speed histograms"
    plt.title(plot_title)
    plt.legend()
    plt.grid()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / ttype_or_wtg / f"{plot_title}.png")
    plt.close()
