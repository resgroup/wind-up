"""Tests for the leaderboard summary."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from benchmarking.harness.leaderboard import leaderboard


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
