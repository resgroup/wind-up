import matplotlib.pyplot as plt
import pandas as pd

from wind_up.constants import (
    REANALYSIS_WD_COL,
    REANALYSIS_WS_COL,
    SCATTER_ALPHA,
    SCATTER_S,
    TIMESTAMP_COL,
)
from wind_up.models import PlotConfig


def plot_find_best_shift_and_corr(
    wf_ws_df: pd.DataFrame,
    *,
    reanalysis_df: pd.DataFrame,
    shifts: list[int],
    corrs: list[float],
    wf_name: str,
    datastream_id: str,
    best_corr: float,
    best_s: int,
    timebase_s: int,
    plot_cfg: PlotConfig,
) -> None:
    plt.figure()
    plt.plot(shifts, corrs, marker=".")
    plot_title = f"correlation vs shift for {datastream_id}"
    plt.title(plot_title)
    plt.xlabel(f"shift [{timebase_s // 60:.0f}min rows]")
    plt.ylabel("correlation with wind farm mean wind speed")
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / f"{plot_title}.png")
    plt.close()

    plot_df = wf_ws_df.merge(reanalysis_df[REANALYSIS_WS_COL].shift(best_s), left_index=True, right_index=True)
    plt.figure()
    plt.scatter(plot_df[REANALYSIS_WS_COL], plot_df["WindSpeedMean"], s=SCATTER_S, alpha=SCATTER_ALPHA)
    plot_title = f"{wf_name} vs {datastream_id}"
    plt.suptitle(plot_title, fontsize=14)
    plt.title(f"corr= {best_corr:.3f}, shift = {best_s}", fontsize=10)
    plt.xlabel("reanalysis wind speed [m/s]")
    plt.ylabel("wind farm mean wind speed [m/s]")
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / f"{plot_title}.png")
    plt.close()


def plot_wf_and_reanalysis_sample_timeseries(wf_df: pd.DataFrame, plot_cfg: PlotConfig) -> None:
    max_timestamp = (
        wf_df.dropna(subset=["WindSpeedMean", "YawAngleMean", REANALYSIS_WS_COL, REANALYSIS_WD_COL])
        .index.get_level_values(TIMESTAMP_COL)
        .max()
    )
    one_month_before_max = max_timestamp - pd.Timedelta(days=365.25 / 12)
    filt_df = wf_df[wf_df.index.get_level_values(TIMESTAMP_COL) >= one_month_before_max]
    plt.figure(figsize=(10, 6))
    wtg_names = filt_df.index.unique(level="TurbineName")
    for wtg_name in wtg_names:
        plt.plot(filt_df.loc[wtg_name, "WindSpeedMean"], label=wtg_name)
    plt.plot(filt_df.loc[wtg_name, REANALYSIS_WS_COL], label="reanalysis", linewidth=3)
    plt.legend(loc="upper left", ncol=max(1, round(len(wtg_names) // 5)), fontsize="small")
    plot_title = "turbine and reanalysis wind speeds"
    plt.title(plot_title)
    plt.xticks(rotation=90)
    plt.xlabel("datetime")
    plt.ylabel("wind speed [m/s]")
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / f"{plot_title}.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    wtg_names = filt_df.index.unique(level="TurbineName")
    for wtg_name in wtg_names:
        plt.plot(filt_df.loc[wtg_name, "YawAngleMean"], label=wtg_name)
    plt.plot(filt_df.loc[wtg_name, REANALYSIS_WD_COL], label="reanalysis", linewidth=3)
    plt.legend(loc="lower left", ncol=max(1, round(len(wtg_names) // 5)), fontsize="small")
    plot_title = "turbine and reanalysis wind directions"
    plt.title(plot_title)
    plt.xticks(rotation=90)
    plt.xlabel("datetime")
    plt.ylabel("wind direction [deg]")
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / f"{plot_title}.png")
    plt.close()
