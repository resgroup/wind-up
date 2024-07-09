import matplotlib.pyplot as plt
import pandas as pd

from wind_up.constants import RAW_POWER_COL, RAW_WINDSPEED_COL, SCATTER_ALPHA, SCATTER_MARKERSCALE, SCATTER_S
from wind_up.models import PlotConfig, WindUpConfig


def plot_pc_one_ttype(pc: pd.DataFrame, ttype: str, plot_cfg: PlotConfig) -> None:
    plt.plot(pc["x_mean"], pc["y_mean"], marker=".", label="raw")
    plt.plot(pc["WindSpeedMean"], pc["pw_clipped"], marker=".", label="cooked", linestyle=":")
    plot_title = f"{ttype} mean power curve"
    plt.title(plot_title)
    plt.xlabel("WindSpeedMean [m/s]")
    plt.ylabel("pw_clipped [kW]")
    plt.legend()
    plt.grid()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        tdir = plot_cfg.plots_dir / ttype
        tdir.mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_cfg.plots_dir / ttype / f"{plot_title}.png")
    plt.close()


def plot_pc_per_ttype(cfg: WindUpConfig, pc_per_ttype: dict, plot_cfg: PlotConfig) -> None:
    for ttype in cfg.list_unique_turbine_types():
        ttype_str = ttype.turbine_type
        plot_pc_one_ttype(pc=pc_per_ttype[ttype_str], ttype=ttype_str, plot_cfg=plot_cfg)


def plot_removed_data_one_ttype_or_wtg(
    t_df: pd.DataFrame,
    pc_df: pd.DataFrame,
    ttype_or_wtg: str,
    plot_cfg: PlotConfig,
) -> None:
    if RAW_WINDSPEED_COL in t_df.columns and RAW_POWER_COL in t_df.columns:
        plot_df = t_df[t_df["ActivePowerMean"].isna()]
        plt.figure()
        plt.scatter(
            plot_df[RAW_WINDSPEED_COL],
            plot_df[RAW_POWER_COL],
            s=SCATTER_S,
            alpha=SCATTER_ALPHA,
            label="removed data",
        )
        plt.plot(pc_df["WindSpeedMean"], pc_df["pw_clipped"], label="SCADA power curve", linestyle=":", color="C1")
        plot_title = f"{ttype_or_wtg} REMOVED DATA"
        plt.title(plot_title)
        plt.xlabel(f"{RAW_WINDSPEED_COL} [m/s]")
        plt.ylabel(f"{RAW_POWER_COL} [kW]")
        plt.legend(loc="best", markerscale=SCATTER_MARKERSCALE)
        plt.grid()
        plt.tight_layout()
        if plot_cfg.show_plots:
            plt.show()
        if plot_cfg.save_plots:
            plt.savefig(plot_cfg.plots_dir / ttype_or_wtg / f"{plot_title}.png")
        plt.close()
    else:
        msg = f"cannot plot removed data without columns {RAW_WINDSPEED_COL} and {RAW_POWER_COL}"
        raise ValueError(msg)


def plot_removed_data_per_ttype_and_wtg(
    cfg: WindUpConfig,
    wf_df: pd.DataFrame,
    pc_per_ttype: dict,
    plot_cfg: PlotConfig,
) -> None:
    for ttype in cfg.list_unique_turbine_types():
        ttype_str = ttype.turbine_type
        wtg_names = cfg.list_turbine_ids_of_type(ttype)
        df_ttype = wf_df.loc[wtg_names]
        plot_removed_data_one_ttype_or_wtg(
            t_df=df_ttype,
            pc_df=pc_per_ttype[ttype_str],
            ttype_or_wtg=ttype_str,
            plot_cfg=plot_cfg,
        )
        if not plot_cfg.skip_per_turbine_plots:
            for wtg_name in wtg_names:
                plot_removed_data_one_ttype_or_wtg(
                    t_df=df_ttype.loc[[wtg_name]],
                    pc_df=pc_per_ttype[ttype_str],
                    ttype_or_wtg=wtg_name,
                    plot_cfg=plot_cfg,
                )
