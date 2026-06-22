"""Verification plots for synthetic datasets.

Three panels for the test turbine, all sharing the wind-speed x-axis:

1. *original* power curve (power vs wind speed);
2. *synthetic* power curve, on the same power y-axis as the original so the injected
   upgrade is directly comparable;
3. the per-record **kW change** (synthetic minus original) vs wind speed for the
   treated records, which makes the injected uplift shape easy to read.

Treated (post-upgrade) records are highlighted in the first two panels so you can
confirm the injection lands where expected and leaves the baseline rows untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from benchmarking.synthetic.ground_truth import changed_record_mask
from wind_up.constants import DataColumns

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from matplotlib.figure import Figure

_BASELINE_STYLE = {"s": 6, "alpha": 0.4, "color": "tab:blue", "label": "baseline rows"}
_TREATED_STYLE = {"s": 6, "alpha": 0.5, "color": "tab:red", "label": "treated rows"}


def plot_power_curve_comparison(
    synthetic_df: pd.DataFrame,
    original_df: pd.DataFrame,
    *,
    test_wtg: str,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """Plot the test turbine's original vs synthetic power curve plus the kW change.

    The original and synthetic power-curve panels share x (wind speed) and y (power)
    limits and gridlines; a third panel shows the synthetic-minus-original power change
    against wind speed for the treated records. Records the upgrade actually changed
    (NaN-safe) are highlighted in the first two panels.

    :param synthetic_df: wind-up-format synthetic SCADA (all turbines)
    :param original_df: the untouched original SCADA (all turbines)
    :param test_wtg: turbine to plot
    :param save_path: if given, the figure is written here (PNG)
    :param title: optional overall figure title
    :return: the matplotlib Figure
    """
    original = original_df[original_df[DataColumns.turbine_name] == test_wtg]
    synthetic = synthetic_df[synthetic_df[DataColumns.turbine_name] == test_wtg]

    ws = original[DataColumns.wind_speed_mean].to_numpy(dtype=float)
    original_power = original[DataColumns.active_power_mean].to_numpy(dtype=float)
    synthetic_power = synthetic[DataColumns.active_power_mean].to_numpy(dtype=float)

    # Treated = records genuinely modified by the upgrade (NaN downtime rows excluded).
    treated = changed_record_mask(synthetic_power, original_power)

    fig, (ax_orig, ax_syn, ax_delta) = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
    ax_syn.sharey(ax_orig)  # tie the two power-curve y-axes; the kW-change panel is its own

    for ax, power, panel_title in (
        (ax_orig, original_power, "Original"),
        (ax_syn, synthetic_power, "Synthetic"),
    ):
        ax.scatter(ws[~treated], power[~treated], **_BASELINE_STYLE)
        ax.scatter(ws[treated], power[treated], **_TREATED_STYLE)
        ax.set_title(panel_title)
        ax.set_xlabel("Wind speed [m/s]")
        ax.grid(visible=True, alpha=0.3)
        ax.legend(loc="lower right", markerscale=2)
    ax_orig.set_ylabel("Active power [kW]")

    delta = synthetic_power - original_power
    finite_treated = treated & np.isfinite(delta)
    ax_delta.scatter(ws[finite_treated], delta[finite_treated], s=6, alpha=0.5, color="tab:red")
    ax_delta.axhline(0.0, color="k", linewidth=0.8)
    ax_delta.set_title("Injected change (synthetic - original)")
    ax_delta.set_xlabel("Wind speed [m/s]")
    ax_delta.set_ylabel("Power change [kW]")
    ax_delta.grid(visible=True, alpha=0.3)

    fig.suptitle(title if title is not None else f"{test_wtg} power curve: original vs synthetic")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
