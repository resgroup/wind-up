import pandas as pd
from matplotlib import pyplot as plt

from wind_up.constants import RAW_POWER_COL, REANALYSIS_WD_COL, WINDFARM_YAWDIR_COL
from wind_up.models import PlotConfig, WindUpConfig


def plot_diff_to_north_ref_wd(
    wtg_df: pd.DataFrame,
    *,
    wtg_name: str,
    north_ref_wd_col: str,
    loop_count: int,
    plot_cfg: PlotConfig,
) -> None:
    plt.figure()
    plt.plot(wtg_df.index, wtg_df[f"yaw_diff_to_{north_ref_wd_col}"], label=f"yaw_diff_to_{north_ref_wd_col}")
    plt.plot(wtg_df.index, wtg_df[f"filt_diff_to_{north_ref_wd_col}"], label=f"filt_diff_to_{north_ref_wd_col}")
    plt.plot(
        wtg_df.index,
        wtg_df[f"short_rolling_diff_to_{north_ref_wd_col}"],
        label=f"short_rolling_diff_to_{north_ref_wd_col}",
    )
    plt.plot(
        wtg_df.index,
        wtg_df[f"long_rolling_diff_to_{north_ref_wd_col}"],
        label=f"long_rolling_diff_to_{north_ref_wd_col}",
    )
    plt.grid()
    plt.legend()
    title = f"{wtg_name} diff to {north_ref_wd_col} loop_count={loop_count}"
    plt.title(title)
    plt.xlabel("datetime")
    plt.ylabel(f"yaw angle diff to {north_ref_wd_col} [deg]")
    plt.tight_layout()
    plt.savefig(plot_cfg.plots_dir / wtg_name / f"{title}.png")
    plt.close()


def plot_yaw_diff_vs_power(wtg_df: pd.DataFrame, *, wtg_name: str, north_ref_wd_col: str, plot_cfg: PlotConfig) -> None:
    plt.figure()
    plt.scatter(wtg_df[RAW_POWER_COL], wtg_df[f"yaw_diff_to_{north_ref_wd_col}"], s=1, alpha=0.2)
    plt.grid()
    plt.xlabel(RAW_POWER_COL)
    plt.ylabel(f"yaw_diff_to_{north_ref_wd_col}")
    title = f"{wtg_name} yaw_diff_to_{north_ref_wd_col} vs {RAW_POWER_COL}"
    plt.tight_layout()
    (plot_cfg.plots_dir / wtg_name).mkdir(exist_ok=True)
    plt.savefig(plot_cfg.plots_dir / wtg_name / f"{title}.png")
    plt.close()


def plot_wf_yawdir_and_reanalysis_timeseries(wf_df: pd.DataFrame, *, cfg: WindUpConfig, plot_cfg: PlotConfig) -> None:
    days_to_plot = 60
    plot_df = wf_df.loc[cfg.asset.wtgs[0].name].copy()
    last_reanalysis_dt = plot_df[REANALYSIS_WD_COL].dropna().index.max()
    plot_start_dt = last_reanalysis_dt - pd.Timedelta(days=days_to_plot)
    plot_df = plot_df[plot_start_dt:]

    plt.figure()
    plt.plot(plot_df.index, plot_df[WINDFARM_YAWDIR_COL], label=WINDFARM_YAWDIR_COL, marker=".", linestyle="None")
    plt.plot(plot_df.index, plot_df[REANALYSIS_WD_COL], label=REANALYSIS_WD_COL)
    plt.xlabel("timestamp")
    plt.ylabel("direction [deg]")
    title = "wf_yawdir and reanalysis timeseries"
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_cfg.plots_dir / f"{title}.png")
    plt.close()
