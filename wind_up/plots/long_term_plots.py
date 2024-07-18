import pandas as pd
from matplotlib import pyplot as plt

from wind_up.models import PlotConfig


def plot_lt_ws(
    lt_df: pd.DataFrame,
    *,
    turbine_or_wf_name: str,
    title_end: str,
    plot_cfg: PlotConfig,
    one_turbine: bool = False,
) -> None:
    plt.figure()
    plt.plot(lt_df["bin_mid"], lt_df["hours_per_year"], marker="s", label="hours per year")
    plt.plot(
        lt_df["bin_mid"],
        lt_df["mwh_per_year_per_turbine"],
        marker="s",
        label=f"MWh per year{'' if one_turbine else ' per turbine'}",
    )
    title_end = f" {title_end}" if len(title_end) > 0 else ""
    plot_title = f"{turbine_or_wf_name} long term distribution{title_end}"
    plt.title(plot_title)
    plt.xlabel("wind speed bin middle [m/s]")
    plt.ylabel("hours per year, MWh per year")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        if one_turbine:
            (plot_cfg.plots_dir / turbine_or_wf_name).mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_cfg.plots_dir / turbine_or_wf_name / f"{plot_title}.png")
        else:
            plt.savefig(plot_cfg.plots_dir / f"{plot_title}.png")
    plt.close()


def plot_lt_ws_raw_filt(
    *,
    lt_df_raw: pd.DataFrame,
    lt_df_filt: pd.DataFrame,
    wtg_or_wf_name: str,
    plot_cfg: PlotConfig,
    one_turbine: bool,
) -> None:
    plt.figure()
    plt.plot(lt_df_raw["bin_mid"], lt_df_raw["hours_per_year"], marker="s", label="before filter")
    plt.plot(lt_df_filt["bin_mid"], lt_df_filt["hours_per_year"], marker="s", label="after filter")
    plot_title = f"{wtg_or_wf_name} long term hours distribution"
    plt.title(plot_title)
    plt.xlabel("wind speed bin middle [m/s]")
    plt.ylabel("hours per year")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        if one_turbine:
            plt.savefig(plot_cfg.plots_dir / wtg_or_wf_name / f"{plot_title}.png")
        else:
            plt.savefig(plot_cfg.plots_dir / f"{plot_title}.png")
    plt.close()

    plt.figure()
    plt.plot(lt_df_raw["bin_mid"], lt_df_raw["mwh_per_year_per_turbine"], marker="s", label="before filter")
    plt.plot(lt_df_filt["bin_mid"], lt_df_filt["mwh_per_year_per_turbine"], marker="s", label="after filter")
    plot_title = f"{wtg_or_wf_name} long term MWh per year{one_turbine * ' per turbine'}"
    plt.title(plot_title)
    plt.xlabel("wind speed bin middle [m/s]")
    plt.ylabel("MWh per year")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        if one_turbine:
            plt.savefig(plot_cfg.plots_dir / wtg_or_wf_name / f"{plot_title}.png")
        else:
            plt.savefig(plot_cfg.plots_dir / f"{plot_title}.png")
    plt.close()
