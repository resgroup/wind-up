"""Leaderboard: summarise tidy scoring results into a side-by-side comparison.

Groups the tidy results by ``(method, profile, <campaign-length column>)`` and reduces each group's
signed errors to bias, spread and the combined RMSE score via :func:`metrics.summarize_errors`.
A pure consumer of the results table; lower ``score`` is better.

The campaign-length column is ``campaign_months`` by default and ``campaign_weeks`` for a weeks-grid
study (see :class:`~benchmarking.harness.replicates.StudyConfig`); pass ``length_col`` to match the
results being summarised.
"""

from __future__ import annotations

import pandas as pd

from benchmarking.harness.metrics import summarize_errors

DEFAULT_LENGTH_COL = "campaign_months"


def _group_keys(length_col: str) -> list[str]:
    return ["method", "profile", length_col]


def _condition_group_keys(length_col: str) -> list[str]:
    return ["method", "profile", length_col, "condition", "condition_bin"]


def leaderboard(results_df: pd.DataFrame, *, length_col: str = DEFAULT_LENGTH_COL) -> pd.DataFrame:
    """Summarise scoring results into per-(method, profile, campaign-length) bias/spread/score.

    Only overall-uplift rows (``condition == "overall"``) are summarised; per-condition rows are
    excluded. Returns one row per group with ``bias``, ``spread``, ``score``, the mean recovered
    and true uplift (``mean_estimate`` / ``mean_truth``, when those columns are present in the
    input), ``n_replicates``, and the group's wall time (``wall_time_s_sum`` total and
    ``wall_time_s_mean`` per run, when ``wall_time_s`` is present), sorted by method, profile then
    campaign length.

    :param length_col: the campaign-length column in ``results_df`` (``campaign_months`` /
        ``campaign_weeks``)
    """
    group_keys = _group_keys(length_col)
    overall = results_df[results_df["condition"] == "overall"] if "condition" in results_df else results_df

    records = []
    for keys, group in overall.groupby(group_keys, sort=True):
        summary = summarize_errors(group["signed_error"].to_numpy())
        records.append(
            {
                **dict(zip(group_keys, keys, strict=True)),
                "bias": summary.bias,
                "spread": summary.spread,
                "score": summary.score,
                "mean_estimate": float(group["estimate"].mean()) if "estimate" in group else float("nan"),
                "mean_truth": float(group["truth"].mean()) if "truth" in group else float("nan"),
                "n_replicates": summary.n,
                "wall_time_s_sum": float(group["wall_time_s"].sum(min_count=1))
                if "wall_time_s" in group
                else float("nan"),
                "wall_time_s_mean": float(group["wall_time_s"].mean()) if "wall_time_s" in group else float("nan"),
            }
        )
    columns = [
        *group_keys,
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


def conditional_leaderboard(results_df: pd.DataFrame, *, length_col: str = DEFAULT_LENGTH_COL) -> pd.DataFrame:
    """Per-(method, profile, campaign, condition, bin) bias/spread/score over the conditional rows.

    Every per-condition axis (``ws``, ``ti``, ``power``, …) is summarised; only overall rows
    (``condition == "overall"``) are excluded. Returns one row per group with ``bias``,
    ``spread``, ``score``, the mean recovered and true uplift (``mean_estimate`` /
    ``mean_truth``, when those columns are present in the input), and ``n_replicates``,
    sorted by method, profile, campaign length, condition, then condition_bin.

    :param length_col: the campaign-length column in ``results_df`` (``campaign_months`` /
        ``campaign_weeks``)
    """
    group_keys = _condition_group_keys(length_col)
    cond = results_df[results_df["condition"] != "overall"] if "condition" in results_df else results_df.iloc[:0]

    records = []
    for keys, group in cond.groupby(group_keys, sort=True):
        summary = summarize_errors(group["signed_error"].to_numpy())
        records.append(
            {
                **dict(zip(group_keys, keys, strict=True)),
                "bias": summary.bias,
                "spread": summary.spread,
                "score": summary.score,
                "mean_estimate": float(group["estimate"].mean()) if "estimate" in group else float("nan"),
                "mean_truth": float(group["truth"].mean()) if "truth" in group else float("nan"),
                "n_replicates": summary.n,
            }
        )
    columns = [*group_keys, "bias", "spread", "score", "mean_estimate", "mean_truth", "n_replicates"]
    return pd.DataFrame(records, columns=columns)
