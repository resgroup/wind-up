import matplotlib.pyplot as plt
import pandas as pd

from wind_up.constants import RAW_POWER_COL, RAW_WINDSPEED_COL
from wind_up.models import PlotConfig, WindUpConfig


def plot_detrend_data_cov(
    *,
    cfg: WindUpConfig,
    test_name: str,
    ref_name: str,
    test_df: pd.DataFrame,
    test_ws_col: str,
    ref_df: pd.DataFrame,
    ref_ws_col: str,
    ref_wd_col: str,
    detrend_df: pd.DataFrame,
    plot_cfg: PlotConfig,
) -> None:
    plt.figure(figsize=(12, 8))
    window_hrs = 14 * 24
    rows_per_hour = 3600 / cfg.timebase_s
    expected_rows = rows_per_hour * window_hrs
    xlims = [
        min(cfg.lt_first_dt_utc_start, test_df.index.min()) - pd.Timedelta(f"{window_hrs}h"),
        cfg.analysis_last_dt_utc_start + pd.Timedelta(f"{window_hrs}h"),
    ]
    plt.subplot(3, 1, 1)
    plot_df_raw = (
        test_df.resample(f"{window_hrs}h")["test_" + RAW_POWER_COL, "test_" + RAW_WINDSPEED_COL].count()
        / expected_rows
        * 100
    )
    plot_df = test_df.resample(f"{window_hrs}h")[test_ws_col].count().to_frame(name=test_ws_col) / expected_rows * 100
    plt.plot(plot_df_raw.index, plot_df_raw["test_" + RAW_POWER_COL], label="ActivePowerMean before filter")
    plt.plot(plot_df_raw.index, plot_df_raw["test_" + RAW_WINDSPEED_COL], label="WindSpeedMean before filter")
    plt.plot(plot_df.index, plot_df[test_ws_col], label=test_ws_col)
    plt.xlim(xlims)
    plt.ylim([0, 105])
    plt.ylabel("data coverage [%]")
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plot_df = ref_df.resample(f"{window_hrs}h")[ref_ws_col, ref_wd_col].count() / expected_rows * 100
    plt.plot(plot_df.index, plot_df[ref_ws_col], label=ref_ws_col)
    plt.plot(plot_df.index, plot_df[ref_wd_col], label=ref_wd_col)
    plt.xlim(xlims)
    plt.ylim([0, 105])
    plt.ylabel("data coverage [%]")
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plot_df = (
        detrend_df.resample(f"{window_hrs}h")[test_ws_col].count().to_frame(name=test_ws_col) / expected_rows * 100
    )
    plt.plot(plot_df.index, plot_df[test_ws_col], label=f"detrend {test_ws_col}")
    plt.xlim(xlims)
    plt.ylim([0, 105])
    plt.ylabel("data coverage [%]")
    plt.grid()
    plt.legend()
    plot_title = f"test={test_name} ref={ref_name} detrend data coverage"
    plt.suptitle(plot_title)
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / ref_name / f"{plot_title}.png")
    plt.close()


def plot_pre_post_data_cov(
    *,
    cfg: WindUpConfig,
    test_name: str,
    ref_name: str,
    test_df: pd.DataFrame,
    test_pw_col: str,
    test_ws_col: str,
    ref_df: pd.DataFrame,
    ref_pw_col: str,
    ref_ws_col: str,
    ref_wd_col: str,
    detrend_ws_col: str,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    plot_cfg: PlotConfig,
) -> None:
    plt.figure(figsize=(12, 8))
    window_hrs = 14 * 24
    rows_per_hour = 3600 / cfg.timebase_s
    expected_rows = rows_per_hour * window_hrs
    xlims = [
        min(cfg.lt_first_dt_utc_start, test_df.index.min()) - pd.Timedelta(f"{window_hrs}h"),
        cfg.analysis_last_dt_utc_start + pd.Timedelta(f"{window_hrs}h"),
    ]
    plt.subplot(3, 1, 1)
    plot_df_raw = (
        test_df.resample(f"{window_hrs}h")["test_" + RAW_POWER_COL, "test_" + RAW_WINDSPEED_COL].count()
        / expected_rows
        * 100
    )
    plot_df = test_df.resample(f"{window_hrs}h")[test_pw_col, test_ws_col].count() / expected_rows * 100
    plt.plot(plot_df_raw.index, plot_df_raw["test_" + RAW_POWER_COL], label="ActivePowerMean before filter")
    plt.plot(plot_df_raw.index, plot_df_raw["test_" + RAW_WINDSPEED_COL], label="WindSpeedMean before filter")
    plt.plot(plot_df.index, plot_df[test_pw_col], label=test_pw_col)
    plt.plot(plot_df.index, plot_df[test_ws_col], label=test_ws_col)
    plt.xlim(xlims)
    plt.ylim([0, 105])
    plt.ylabel("data coverage [%]")
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plot_df = ref_df.resample(f"{window_hrs}h")[ref_pw_col, ref_ws_col, ref_wd_col].count() / expected_rows * 100
    plt.plot(plot_df.index, plot_df[ref_pw_col], label=ref_pw_col)
    plt.plot(plot_df.index, plot_df[ref_ws_col], label=ref_ws_col)
    plt.plot(plot_df.index, plot_df[ref_wd_col], label=ref_wd_col)
    plt.xlim(xlims)
    plt.ylim([0, 105])
    plt.ylabel("data coverage [%]")
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plot_pre_df = pre_df.resample(f"{window_hrs}h")[test_pw_col, detrend_ws_col].count() / expected_rows * 100
    plot_post_df = post_df.resample(f"{window_hrs}h")[test_pw_col, detrend_ws_col].count() / expected_rows * 100
    plt.plot(plot_pre_df.index, plot_pre_df[test_pw_col], label=f"pre {test_pw_col}")
    plt.plot(plot_pre_df.index, plot_pre_df[detrend_ws_col], label=f"pre {detrend_ws_col}")
    plt.plot(plot_post_df.index, plot_post_df[test_pw_col], label=f"post {test_pw_col}")
    plt.plot(plot_post_df.index, plot_post_df[detrend_ws_col], label=f"post {detrend_ws_col}")
    plt.xlim(xlims)
    plt.ylim([0, 105])
    plt.ylabel("data coverage [%]")
    plt.grid()
    plt.legend()
    plot_title = f"test={test_name} ref={ref_name} pre and post data coverage"
    plt.suptitle(plot_title)
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / test_name / ref_name / f"{plot_title}.png")
    plt.close()
