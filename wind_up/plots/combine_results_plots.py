import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

from wind_up.backporting import strict_zip
from wind_up.models import PlotConfig


def plot_combined_results(tdf: pd.DataFrame, *, plot_cfg: PlotConfig, confidence: float = 0.9) -> None:
    show_plots = plot_cfg.show_plots
    save_plots = plot_cfg.save_plots
    z_score = norm.ppf((1 + confidence) / 2)

    grouped_results = tdf.index.name == "role"

    plt.figure()
    labels = tdf.index.to_list() if grouped_results else tdf["test_wtg"]
    values = tdf["p50_uplift"] * 100
    yerrs = tdf["sigma"] * 100 * z_score
    plt.bar(labels, values, yerr=yerrs, capsize=3)
    plt.xlabel("turbine group" if grouped_results else "turbine")
    plt.ylabel("uplift [%]")
    plot_title = f"combined uplift and {confidence * 100:.0f}% CI"
    plt.title(plot_title)
    plt.grid(axis="y")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    if show_plots:
        plt.show()
    if save_plots:
        plt.savefig(plot_cfg.plots_dir / f"{plot_title}.png")
    plt.close()


def plot_testref_and_combined_results(
    *, trdf: pd.DataFrame, tdf: pd.DataFrame, plot_cfg: PlotConfig, confidence: float = 0.9
) -> None:
    show_plots = plot_cfg.show_plots
    save_plots = plot_cfg.save_plots
    z_score = norm.ppf((1 + confidence) / 2)

    plt.figure(figsize=(10, 6))
    labels = [
        x + "-" + y
        for x, y in strict_zip(
            [x[-1] for x in trdf["test_wtg"].str.split("_")],
            [x[-1] for x in trdf["ref"].str.split("_")],
        )
    ]
    values = trdf["uplift_frc"] * 100
    yerrs = trdf["unc_one_sigma_frc"] * 100 * z_score
    plt.bar(labels, values, yerr=yerrs, capsize=3)
    plt.xlabel("turbine")
    plt.ylabel("uplift [%]")
    plot_title = f"uplift and {confidence * 100:.0f}% CI by test-ref pair"
    plt.title("uplift by test-ref pair")
    plt.grid(axis="y")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    if show_plots:
        plt.show()
    if save_plots:
        plt.savefig(plot_cfg.plots_dir / f"{plot_title}.png")
    plt.close()

    plot_combined_results(tdf, plot_cfg=plot_cfg, confidence=confidence)
