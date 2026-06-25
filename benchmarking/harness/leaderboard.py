"""Leaderboard: summarise tidy scoring results into a side-by-side comparison.

Groups the tidy results by ``(method, profile, campaign_months)`` and reduces each group's
signed errors to bias, spread and the combined RMSE score via :func:`metrics.summarize_errors`.
A pure consumer of the results table; lower ``score`` is better.
"""

from __future__ import annotations

import pandas as pd

from benchmarking.harness.metrics import summarize_errors

_GROUP_KEYS = ["method", "profile", "campaign_months"]


def leaderboard(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise scoring results into per-(method, profile, campaign-length) bias/spread/score.

    Only overall-uplift rows (``condition == "overall"``) are summarised; per-condition rows are
    excluded. Returns one row per group with ``bias``, ``spread``, ``score``, the mean recovered
    and true uplift (``mean_estimate`` / ``mean_truth``, when those columns are present in the
    input), ``n_replicates``, and the group's wall time (``wall_time_s_sum`` total and
    ``wall_time_s_mean`` per run, when ``wall_time_s`` is present), sorted by method, profile then
    campaign length.
    """
    overall = results_df[results_df["condition"] == "overall"] if "condition" in results_df else results_df

    records = []
    for keys, group in overall.groupby(_GROUP_KEYS, sort=True):
        summary = summarize_errors(group["signed_error"].to_numpy())
        records.append(
            {
                **dict(zip(_GROUP_KEYS, keys, strict=True)),
                "bias": summary.bias,
                "spread": summary.spread,
                "score": summary.score,
                "mean_estimate": float(group["estimate"].mean()) if "estimate" in group else float("nan"),
                "mean_truth": float(group["truth"].mean()) if "truth" in group else float("nan"),
                "n_replicates": summary.n,
                "wall_time_s_sum": float(group["wall_time_s"].sum()) if "wall_time_s" in group else float("nan"),
                "wall_time_s_mean": float(group["wall_time_s"].mean()) if "wall_time_s" in group else float("nan"),
            }
        )
    columns = [
        *_GROUP_KEYS,
        "bias",
        "spread",
        "score",
        "mean_estimate",
        "mean_truth",
        "n_replicates",
        "wall_time_s_sum",
        "wall_time_s_mean",
    ]
    return pd.DataFrame(records, columns=columns)
