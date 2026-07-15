"""Tests for the toggle-methods regression harness (profile selection, benchmark record/diff)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.study_toggle_methods_compare import (
    _UNCHANGED_ATOL,
    CAMPAIGN_WEEKS,
    COMPARE_METHODS,
    TOGGLE_PROFILES,
    _load_baseline,
    _select_profiles,
    compare_to_benchmark,
    methods_leaderboard,
    plot_results,
    record_baseline,
    toggle_study,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_study_is_toggle_over_the_weeks_grid() -> None:
    study = toggle_study()
    assert study.mode == "toggle"
    assert study.campaign_lengths == CAMPAIGN_WEEKS == [1, 2, 4, 8]
    assert study.campaign_length_col == "campaign_weeks"
    assert study.toggle_period is not None  # toggle mode requires it


def test_profiles_are_a_placebo_plus_a_symmetric_pair() -> None:
    # the symmetric +/-2% pair is what lets a sign error show up as an asymmetry between them
    assert sorted(TOGGLE_PROFILES) == ["cp_0pct", "cp_minus_2pct", "cp_plus_2pct"]
    deltas = {name: profile[0].delta for name, profile in TOGGLE_PROFILES.items()}
    assert deltas == {"cp_0pct": 0.0, "cp_plus_2pct": 0.02, "cp_minus_2pct": -0.02}


def test_select_profiles_none_returns_all() -> None:
    assert _select_profiles(None) == TOGGLE_PROFILES


def test_select_profiles_restricts_to_the_named_subset() -> None:
    assert list(_select_profiles(["cp_0pct"])) == ["cp_0pct"]


def test_select_profiles_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        _select_profiles(["cp_plus_99pct"])


def _results(*, bias_shift: float = 0.0) -> pd.DataFrame:
    """Tidy overall rows for both methods over the weeks grid; truth 0, error = ``bias_shift``."""
    rows = []
    for method in COMPARE_METHODS:
        for weeks in CAMPAIGN_WEEKS:
            for replicate in range(2):
                rows.append(  # noqa: PERF401
                    {
                        "method": method,
                        "profile": "cp_0pct",
                        "campaign_weeks": weeks,
                        "replicate": replicate,
                        "condition": "overall",
                        "condition_bin": "overall",
                        "estimate": bias_shift,
                        "truth": 0.0,
                        "signed_error": bias_shift,
                        "wall_time_s": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_leaderboard_has_one_cell_per_method_and_campaign_length() -> None:
    lb = methods_leaderboard(_results())
    assert len(lb) == len(COMPARE_METHODS) * len(CAMPAIGN_WEEKS)
    assert set(lb["method"]) == set(COMPARE_METHODS)
    assert sorted(lb["campaign_weeks"].unique()) == CAMPAIGN_WEEKS


def test_baseline_round_trips(tmp_path: Path) -> None:
    lb = methods_leaderboard(_results())
    path = tmp_path / "baseline.json"
    record_baseline(lb, study=toggle_study(), path=path)

    loaded = _load_baseline(path)
    assert loaded is not None
    cells, provenance = loaded
    assert provenance["campaign_weeks"] == CAMPAIGN_WEEKS
    assert provenance["methods"] == sorted(COMPARE_METHODS)
    assert provenance["seed"] == toggle_study().seed
    assert len(cells) == len(lb)


def test_load_baseline_returns_none_when_absent(tmp_path: Path) -> None:
    assert _load_baseline(tmp_path / "nope.json") is None


def test_load_baseline_returns_none_for_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"schema": "something_older", "cells": []}))
    assert _load_baseline(path) is None


def test_unchanged_method_diffs_within_the_band(tmp_path: Path) -> None:
    # the property the whole script rests on: an unchanged method reproduces its cells to numerical
    # noise. Use a bias with many significant digits, so record_baseline's round(8) actually bites --
    # a round number here would pass even if the rounding were broken, which is the trap the first
    # version of this test fell into.
    lb = methods_leaderboard(_results(bias_shift=0.0123456789012345))
    path = tmp_path / "baseline.json"
    record_baseline(lb, study=toggle_study(), path=path)

    merged = compare_to_benchmark(lb, baseline_path=path, comparison_dir=tmp_path / "comparison")
    assert not merged.empty
    for col in ("d_bias", "d_spread", "d_score"):
        for method, group in merged.groupby("method"):
            band = _UNCHANGED_ATOL[str(method)]
            assert (group[col].abs() < band).all(), f"{col} must be within {method}'s band when unchanged"


def test_round_trip_leaves_a_nonzero_residual_from_baseline_rounding(tmp_path: Path) -> None:
    # documents *why* the band is not zero: the stored baseline is rounded, so even an identical
    # re-run differs in the last digit. This is the smaller of the two reasons (power_model's
    # nondeterminism is the larger, and cannot be exercised in a unit test).
    lb = methods_leaderboard(_results(bias_shift=0.0123456789012345))
    path = tmp_path / "baseline.json"
    record_baseline(lb, study=toggle_study(), path=path)

    merged = compare_to_benchmark(lb, baseline_path=path, comparison_dir=tmp_path / "comparison")
    assert (merged["d_bias"].abs() > 0).any(), "round(8) should leave a residual; if not, the band can tighten"


def test_moved_method_shows_a_nonzero_bias_delta(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    record_baseline(methods_leaderboard(_results(bias_shift=0.0)), study=toggle_study(), path=path)

    moved = methods_leaderboard(_results(bias_shift=0.01))
    merged = compare_to_benchmark(moved, baseline_path=path, comparison_dir=tmp_path / "comparison")
    assert np.allclose(merged["d_bias"].to_numpy(), 0.01)
    assert (merged["d_bias"].abs() > max(_UNCHANGED_ATOL.values())).all()  # 1 pp is far outside any band


def test_comparison_csv_is_written(tmp_path: Path) -> None:
    lb = methods_leaderboard(_results())
    path = tmp_path / "baseline.json"
    record_baseline(lb, study=toggle_study(), path=path)
    comparison_dir = tmp_path / "comparison"

    compare_to_benchmark(lb, baseline_path=path, comparison_dir=comparison_dir)
    assert (comparison_dir / "benchmark_comparison.csv").exists()


def test_compare_without_a_baseline_is_a_noop(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        merged = compare_to_benchmark(
            methods_leaderboard(_results()), baseline_path=tmp_path / "absent.json", comparison_dir=tmp_path / "cmp"
        )
    assert merged.empty
    assert "No benchmark recorded" in caplog.text


def test_unchanged_verdict_is_logged_per_method(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    lb = methods_leaderboard(_results())
    path = tmp_path / "baseline.json"
    record_baseline(lb, study=toggle_study(), path=path)

    with caplog.at_level(logging.INFO):
        compare_to_benchmark(lb, baseline_path=path, comparison_dir=tmp_path / "comparison")
    for method in COMPARE_METHODS:
        assert f"{method}: UNCHANGED" in caplog.text


def test_moved_verdict_warns_and_names_the_cells(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "baseline.json"
    record_baseline(methods_leaderboard(_results()), study=toggle_study(), path=path)

    with caplog.at_level(logging.WARNING):
        compare_to_benchmark(
            methods_leaderboard(_results(bias_shift=0.01)), baseline_path=path, comparison_dir=tmp_path / "comparison"
        )
    assert "MOVED" in caplog.text


def test_plot_results_writes_one_curve_per_profile(tmp_path: Path) -> None:
    lb = methods_leaderboard(_results())
    plot_results(lb, tmp_path)
    assert (tmp_path / "campaign_curves_cp_0pct.png").exists()
