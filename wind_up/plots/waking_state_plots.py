import matplotlib.pyplot as plt
import pandas as pd

from wind_up.constants import RAW_POWER_COL, RAW_WINDSPEED_COL, SCATTER_ALPHA, SCATTER_MARKERSCALE, SCATTER_S
from wind_up.models import PlotConfig


def plot_waking_state_one_ttype_or_wtg(wf_df: pd.DataFrame, ttype_or_wtg: str, plot_cfg: PlotConfig) -> None:
    waking_df = wf_df[wf_df["waking"]]
    not_waking_df = wf_df[wf_df["not_waking"]]
    unknown_df = wf_df[wf_df["unknown_waking"]]
    plt.figure()
    plt.scatter(
        waking_df[RAW_WINDSPEED_COL],
        waking_df[RAW_POWER_COL],
        label="waking",
        s=SCATTER_S,
        alpha=SCATTER_ALPHA,
    )
    plt.scatter(
        not_waking_df[RAW_WINDSPEED_COL],
        not_waking_df[RAW_POWER_COL],
        label="not waking",
        s=SCATTER_S,
        alpha=SCATTER_ALPHA,
    )
    plt.scatter(
        unknown_df[RAW_WINDSPEED_COL],
        unknown_df[RAW_POWER_COL],
        label="unknown or partial waking",
        s=SCATTER_S,
        alpha=SCATTER_ALPHA,
    )
    plt.xlabel(f"{RAW_WINDSPEED_COL} [m/s]")
    plt.ylabel(f"{RAW_POWER_COL} [kW]")
    plot_title = f"{ttype_or_wtg} power curve by waking state"
    plt.title(plot_title)
    plt.grid()
    plt.legend(loc="lower right", markerscale=SCATTER_MARKERSCALE)
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        plt.savefig(plot_cfg.plots_dir / ttype_or_wtg / f"{plot_title}.png")
    plt.close()
