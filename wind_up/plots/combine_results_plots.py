import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt

from wind_up.models import PlotConfig

mpl.use("Agg")


def plot_combine_results(trdf: pd.DataFrame, tdf: pd.DataFrame, plot_cfg: PlotConfig) -> None:
    show_plots = plot_cfg.show_plots
    save_plots = plot_cfg.save_plots

    plt.figure(figsize=(10, 6))
    labels = [
        x + "-" + y
        for x, y in zip(
            [x[-1] for x in trdf["test_wtg"].str.split("_")],
            [x[-1] for x in trdf["ref"].str.split("_")],
            strict=True,
        )
    ]
    values = trdf["uplift_frc"] * 100
    yerrs = trdf["unc_one_sigma_frc"] * 100
    plt.bar(labels, values, yerr=yerrs, capsize=3)
    plt.xlabel("turbine")
    plt.ylabel("uplift [%] and 1-sigma uncertainty")
    plot_title = "uplift by test-ref pair"
    plt.title("uplift by test-ref pair")
    plt.grid(axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if show_plots:
        plt.show()
    if save_plots:
        plt.savefig(plot_cfg.plots_dir / f"{plot_title}.png")

    plt.figure()
    labels = tdf["test_wtg"]
    values = tdf["p50_uplift"] * 100
    yerrs = tdf["sigma"] * 100
    plt.bar(labels, values, yerr=yerrs, capsize=3)
    plt.xlabel("turbine")
    plt.ylabel("uplift [%] and 1-sigma uncertainty")
    plot_title = "combined uplift"
    plt.title(plot_title)
    plt.grid(axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if show_plots:
        plt.show()
    if save_plots:
        plt.savefig(plot_cfg.plots_dir / f"{plot_title}.png")
