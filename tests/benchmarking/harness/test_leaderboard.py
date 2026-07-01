"""Tests for the leaderboard summary."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from benchmarking.harness.leaderboard import conditional_leaderboard, leaderboard


def _results(rows: list[dict]) -> pd.DataFrame:
    base = {"method": "m", "profile": "p", "condition": "overall"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_one_summary_row_per_method_profile_campaign() -> None:
    results = _results(
        [
            {"campaign_months": 3, "signed_error": 1.0},
            {"campaign_months": 3, "signed_error": 3.0},
            {"campaign_months": 6, "signed_error": 0.5},
        ]
    )
    summary = leaderboard(results)
    assert len(summary) == 2
    assert set(summary["campaign_months"]) == {3, 6}


def test_bias_spread_score_and_n_match_metrics() -> None:
    results = _results(
        [
            {"campaign_months": 3, "signed_error": 1.0},
            {"campaign_months": 3, "signed_error": 3.0},
        ]
    )
    row = leaderboard(results).set_index("campaign_months").loc[3]
    assert row["bias"] == pytest.approx(2.0)
    assert row["spread"] == pytest.approx(1.0)
    assert row["score"] == pytest.approx(math.sqrt((1.0**2 + 3.0**2) / 2))
    assert row["n_replicates"] == 2


def test_only_overall_condition_rows_are_summarised() -> None:
    results = _results(
        [
            {"campaign_months": 3, "signed_error": 2.0},
            {"campaign_months": 3, "signed_error": 100.0, "condition": "(4.0, 5.0]"},
        ]
    )
    row = leaderboard(results).set_index("campaign_months").loc[3]
    assert row["bias"] == pytest.approx(2.0)  # the per-condition row is excluded
    assert row["n_replicates"] == 1


def test_mean_estimate_and_truth_are_averaged() -> None:
    results = _results(
        [
            {"campaign_months": 3, "signed_error": 0.01, "estimate": 0.06, "truth": 0.05},
            {"campaign_months": 3, "signed_error": -0.01, "estimate": 0.04, "truth": 0.05},
        ]
    )
    row = leaderboard(results).set_index("campaign_months").loc[3]
    assert row["mean_estimate"] == pytest.approx(0.05)
    assert row["mean_truth"] == pytest.approx(0.05)


def test_wall_time_is_summed_and_averaged_per_group() -> None:
    results = _results(
        [
            {"campaign_months": 3, "signed_error": 0.0, "wall_time_s": 1.5},
            {"campaign_months": 3, "signed_error": 0.0, "wall_time_s": 2.5},
            {"campaign_months": 6, "signed_error": 0.0, "wall_time_s": 4.0},
        ]
    )
    summary = leaderboard(results).set_index("campaign_months")
    # total compute for the group, and the typical per-run cost
    assert summary.loc[3, "wall_time_s_sum"] == pytest.approx(4.0)
    assert summary.loc[3, "wall_time_s_mean"] == pytest.approx(2.0)
    assert summary.loc[6, "wall_time_s_sum"] == pytest.approx(4.0)
    assert summary.loc[6, "wall_time_s_mean"] == pytest.approx(4.0)


def test_methods_are_compared_side_by_side() -> None:
    results = pd.DataFrame(
        [
            {"method": "a", "profile": "p", "condition": "overall", "campaign_months": 6, "signed_error": 0.0},
            {"method": "b", "profile": "p", "condition": "overall", "campaign_months": 6, "signed_error": 0.1},
        ]
    )
    summary = leaderboard(results)
    assert set(summary["method"]) == {"a", "b"}
    by_method = summary.set_index("method")
    assert by_method.loc["a", "score"] == pytest.approx(0.0)
    assert by_method.loc["b", "score"] == pytest.approx(0.1)


def test_conditional_leaderboard_groups_by_condition_bin() -> None:
    df = pd.DataFrame(
        {
            "method": "m",
            "profile": "p",
            "campaign_months": 6,
            "condition": ["ws", "ws", "ws", "ws"],
            "condition_bin": ["(6.0, 8.0]", "(6.0, 8.0]", "(8.0, 10.0]", "(8.0, 10.0]"],
            "estimate": [0.11, 0.09, 0.05, 0.05],
            "truth": [0.10, 0.10, 0.05, 0.05],
            "signed_error": [0.01, -0.01, 0.0, 0.0],
        }
    )
    lb = conditional_leaderboard(df)
    assert set(lb.columns) >= {
        "method",
        "profile",
        "campaign_months",
        "condition",
        "condition_bin",
        "bias",
        "spread",
        "score",
    }
    row = lb[lb["condition_bin"] == "(6.0, 8.0]"].iloc[0]
    assert row["bias"] == 0.0
    assert row["spread"] == pytest.approx(0.01)


def test_conditional_leaderboard_ignores_overall_rows() -> None:
    df = pd.DataFrame(
        {
            "method": "m",
            "profile": "p",
            "campaign_months": 6,
            "condition": ["overall"],
            "condition_bin": ["overall"],
            "estimate": [0.1],
            "truth": [0.1],
            "signed_error": [0.0],
        }
    )
    assert conditional_leaderboard(df).empty
