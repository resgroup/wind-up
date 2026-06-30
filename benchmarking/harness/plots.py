"""Campaign-length curves for comparing methods.

Three stacked panels sharing one campaign-length x-axis:

1. **Uplift recovery** — the true (injected) uplift plus each method's mean recovered uplift,
   so over- vs under-estimation is visible and the true uplift's variation with campaign
   length is shown.
2. **Bias +/- spread** — one ``bias`` line per method with a shaded ``bias +/- spread`` band
   (accuracy and precision together).
3. **Score** — the combined ``score`` (RMSE) per method.

All quantities are converted from uplift fractions to percentage points (a 0.01 fraction is
plotted as 1 pp). Each method keeps one colour across all panels. Consumes the leaderboard
summary frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from matplotlib.figure import Figure

# Uplift quantities are stored as fractions; plot them as percentage points.
_FRACTION_TO_PP = 100.0


def plot_campaign_curves(
    summary_df: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """Plot uplift recovery, bias/spread and score vs campaign length, per method.

    :param summary_df: a leaderboard summary (columns ``method``, ``campaign_months``, ``bias``,
        ``spread``, ``score``, and optionally ``mean_estimate`` / ``mean_truth`` for the top panel)
    :param save_path: if given, the figure is written here (PNG)
    :param title: optional title for the top panel
    :return: the matplotlib Figure
    """
    methods = sorted(summary_df["method"].unique())
    colors = {method: f"C{i}" for i, method in enumerate(methods)}

    fig, (ax_uplift, ax_band, ax_score) = plt.subplots(3, 1, sharex=True, figsize=(8, 11))

    # Top panel: the true uplift (method-independent) and each method's mean recovered uplift.
    if "mean_truth" in summary_df.columns:
        truth = summary_df.groupby("campaign_months")["mean_truth"].mean().sort_index()
        ax_uplift.plot(
            truth.index.to_numpy(), truth.to_numpy() * _FRACTION_TO_PP, "--", marker="s", color="k", label="true uplift"
        )

    for method in methods:
        group = summary_df[summary_df["method"] == method].sort_values("campaign_months")
        months = group["campaign_months"].to_numpy()
        color = colors[method]
        bias = group["bias"].to_numpy() * _FRACTION_TO_PP
        spread = group["spread"].to_numpy() * _FRACTION_TO_PP

        if "mean_estimate" in summary_df.columns:
            estimate = group["mean_estimate"].to_numpy() * _FRACTION_TO_PP
            ax_uplift.plot(months, estimate, marker="o", color=color, label=method)
            ax_uplift.fill_between(months, estimate - spread, estimate + spread, alpha=0.15, color=color)
        ax_band.plot(months, bias, marker="o", color=color, label=method)
        ax_band.fill_between(months, bias - spread, bias + spread, alpha=0.15, color=color)
        ax_score.plot(months, group["score"].to_numpy() * _FRACTION_TO_PP, marker="o", color=color, label=method)

    ax_uplift.set_ylabel("Measured uplift [pp]")
    ax_uplift.set_title(title if title is not None else "P50 uplift recovery vs campaign length")
    ax_uplift.grid(visible=True, alpha=0.3)
    ax_uplift.legend()

    ax_band.axhline(0.0, color="k", linewidth=0.8)
    ax_band.set_ylabel("Bias +/- spread [pp]")
    ax_band.grid(visible=True, alpha=0.3)

    ax_score.set_xlabel("Campaign length [months]")
    ax_score.set_ylabel("Score / RMSE [pp]")
    ax_score.grid(visible=True, alpha=0.3)
    ax_score.set_xlim(left=0.0)
    _set_score_ylim(ax_score, summary_df["score"].to_numpy() * _FRACTION_TO_PP)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def _set_score_ylim(ax: plt.Axes, scores_pp: np.ndarray) -> None:
    """Anchor the score y-axis at min(0, lowest point) minus a small margin.

    Score is a non-negative RMSE, but a hard floor of 0 clips data points sitting exactly at 0
    (e.g. an oracle). Drop the floor by a small margin so those points stay visible.
    """
    lo = float(np.nanmin(scores_pp))
    hi = float(np.nanmax(scores_pp))
    span = hi - lo
    margin = 0.05 * span if span > 0 else max(abs(hi), 1.0) * 0.05
    ax.set_ylim(bottom=min(0.0, lo) - margin)


def plot_conditional_uplift(
    summary_df: pd.DataFrame,
    *,
    condition: str,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """Plot mean recovered vs true uplift across bins of one condition, with a bias±spread band."""
    df = summary_df[summary_df["condition"] == condition].copy()
    df["_left"] = df["condition_bin"].str.extract(r"\(([-0-9.]+),").astype(float)
    df = df.sort_values("_left")
    order = df.drop_duplicates("condition_bin")["condition_bin"].tolist()
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(9, 5))
    truth = df.drop_duplicates("condition_bin").set_index("condition_bin").reindex(order)["mean_truth"]
    ax.plot(x, truth.to_numpy() * _FRACTION_TO_PP, "--", marker="s", color="k", label="true uplift")
    for i, method in enumerate(sorted(df["method"].unique())):
        m = df[df["method"] == method].set_index("condition_bin").reindex(order)
        est = m["mean_estimate"].to_numpy() * _FRACTION_TO_PP
        ax.plot(x, est, marker="o", color=f"C{i}", label=method)
        if "spread" in m:
            sp = m["spread"].to_numpy() * _FRACTION_TO_PP
            ax.fill_between(x, est - sp, est + sp, color=f"C{i}", alpha=0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_xlabel(condition)
    ax.set_ylabel("uplift [pp]")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    return fig
