"""Campaign-length curves for comparing methods.

One line per method showing the combined ``score`` against campaign length, with a shaded
bias +/- spread band so accuracy and precision are both visible. All quantities are in
uplift-fraction units, so they share one y-axis. Consumes the leaderboard summary frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from matplotlib.figure import Figure


def plot_campaign_curves(
    summary_df: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """Plot combined score vs campaign length, one line per method, with a bias +/- spread band.

    :param summary_df: a leaderboard summary (columns ``method``, ``campaign_months``, ``bias``,
        ``spread``, ``score``)
    :param save_path: if given, the figure is written here (PNG)
    :param title: optional figure title
    :return: the matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in summary_df.sort_values("campaign_months").groupby("method"):
        months = group["campaign_months"].to_numpy()
        score = group["score"].to_numpy()
        bias = group["bias"].to_numpy()
        spread = group["spread"].to_numpy()
        (line,) = ax.plot(months, score, marker="o", label=str(method))
        ax.fill_between(months, bias - spread, bias + spread, alpha=0.15, color=line.get_color())

    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("Campaign length [months]")
    ax.set_ylabel("Uplift error [fraction]")
    ax.set_title(title if title is not None else "P50 accuracy/precision vs campaign length")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
