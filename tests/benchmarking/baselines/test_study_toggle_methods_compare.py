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


_POWER_BINS = ["(-230.0, 230.0]", "(230.0, 690.0]"]


def _results(*, bias_shift: float = 0.0) -> pd.DataFrame:
    """Tidy rows for both methods over the weeks grid: the headline plus per-power-bin rows.

    Truth is 0 and every estimate is ``bias_shift``, so each cell's bias is ``bias_shift`` exactly.
    """
    rows = []
    cells = [("overall", "overall"), *[("power", b) for b in _POWER_BINS]]
    for method in COMPARE_METHODS:
        for weeks in CAMPAIGN_WEEKS:
            for replicate in range(2):
                for condition, condition_bin in cells:
                    rows.append(
                        {
                            "method": method,
                            "profile": "cp_0pct",
                            "campaign_weeks": weeks,
                            "replicate": replicate,
                            "condition": condition,
                            "condition_bin": condition_bin,
                            "estimate": bias_shift,
                            "truth": 0.0,
                            "signed_error": bias_shift,
                            "wall_time_s": 1.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_leaderboard_records_the_headline_and_the_power_bins() -> None:
    # the point of recording per-bin cells: a change can leave the headline untouched and wreck a bin,
    # so the benchmark must carry both.
    lb = methods_leaderboard(_results())
    per_method_cells = len(CAMPAIGN_WEEKS) * (1 + len(_POWER_BINS))  # overall + each power bin
    assert len(lb) == len(COMPARE_METHODS) * per_method_cells
    assert set(lb["method"]) == set(COMPARE_METHODS)
    assert sorted(lb["campaign_weeks"].unique()) == CAMPAIGN_WEEKS
    assert set(lb["condition"]) == {"overall", "power"}


def test_leaderboard_records_power_bins_for_both_methods() -> None:
    lb = methods_leaderboard(_results())
    power = lb[lb["condition"] == "power"]
    for method in COMPARE_METHODS:
        bins = set(power[power["method"] == method]["condition_bin"])
        assert bins == set(_POWER_BINS), f"{method} must contribute per-power-bin cells"


def test_wall_time_is_recorded_on_the_headline_rows_only() -> None:
    # wall time is per estimate, not per bin, and it is what makes "this change made the method 2x
    # slower" visible. Stacking the conditional rows must not silently drop it (it did once).
    lb = methods_leaderboard(_results())
    headline = lb[lb["condition"] == "overall"]
    per_bin = lb[lb["condition"] == "power"]
    for col in ("wall_time_s_sum", "wall_time_s_mean"):
        assert col in lb.columns
        assert headline[col].notna().all(), f"{col} must be recorded on the headline rows"
        assert per_bin[col].isna().all(), f"{col} is meaningless per bin and must be NaN there"


def test_wall_time_is_not_diffed(tmp_path: Path) -> None:
    # it is machine- and load-dependent, so diffing it would trip the unchanged verdict every run
    lb = methods_leaderboard(_results())
    path = tmp_path / "baseline.json"
    record_baseline(lb, study=toggle_study(), path=path, git_commit="abc1234")

    slower = lb.assign(wall_time_s_mean=lb["wall_time_s_mean"] * 10)
    merged = compare_to_benchmark(slower, baseline_path=path, comparison_dir=tmp_path / "comparison")
    assert "d_wall_time_s_mean" not in merged.columns
    for col in ("d_bias", "d_spread", "d_score"):
        assert (merged[col].abs() < max(_UNCHANGED_ATOL.values())).all()


def test_baseline_round_trips(tmp_path: Path) -> None:
    lb = methods_leaderboard(_results())
    path = tmp_path / "baseline.json"
    record_baseline(lb, study=toggle_study(), path=path, git_commit="abc1234")

    loaded = _load_baseline(path)
    assert loaded is not None
    cells, provenance = loaded
    assert provenance["campaign_weeks"] == CAMPAIGN_WEEKS
    assert provenance["methods"] == sorted(COMPARE_METHODS)
    assert provenance["seed"] == toggle_study().seed
    assert len(cells) == len(lb)


def test_baseline_records_the_commit_it_was_given_not_the_current_head(tmp_path: Path) -> None:
    # the commit must be captured before the sweep and threaded in: a ~15-min run can straddle a
    # commit, and reading HEAD at write time would stamp a commit whose code never produced these
    # numbers (which is exactly what happened once).
    path = tmp_path / "baseline.json"
    record_baseline(methods_leaderboard(_results()), study=toggle_study(), path=path, git_commit="deadbee")

    loaded = _load_baseline(path)
    assert loaded is not None
    assert loaded[1]["git_commit"] == "deadbee"


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
    record_baseline(lb, study=toggle_study(), path=path, git_commit="abc1234")

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
    record_baseline(lb, study=toggle_study(), path=path, git_commit="abc1234")

    merged = compare_to_benchmark(lb, baseline_path=path, comparison_dir=tmp_path / "comparison")
    assert (merged["d_bias"].abs() > 0).any(), "round(8) should leave a residual; if not, the band can tighten"


def test_moved_method_shows_a_nonzero_bias_delta(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    record_baseline(
        methods_leaderboard(_results(bias_shift=0.0)), study=toggle_study(), path=path, git_commit="abc1234"
    )

    moved = methods_leaderboard(_results(bias_shift=0.01))
    merged = compare_to_benchmark(moved, baseline_path=path, comparison_dir=tmp_path / "comparison")
    assert np.allclose(merged["d_bias"].to_numpy(), 0.01)
    assert (merged["d_bias"].abs() > max(_UNCHANGED_ATOL.values())).all()  # 1 pp is far outside any band


def test_comparison_csv_is_written(tmp_path: Path) -> None:
    lb = methods_leaderboard(_results())
    path = tmp_path / "baseline.json"
    record_baseline(lb, study=toggle_study(), path=path, git_commit="abc1234")
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
    record_baseline(lb, study=toggle_study(), path=path, git_commit="abc1234")

    with caplog.at_level(logging.INFO):
        compare_to_benchmark(lb, baseline_path=path, comparison_dir=tmp_path / "comparison")
    for method in COMPARE_METHODS:
        assert f"{method}: UNCHANGED" in caplog.text


def test_moved_verdict_warns_and_names_the_cells(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "baseline.json"
    record_baseline(methods_leaderboard(_results()), study=toggle_study(), path=path, git_commit="abc1234")

    with caplog.at_level(logging.WARNING):
        compare_to_benchmark(
            methods_leaderboard(_results(bias_shift=0.01)), baseline_path=path, comparison_dir=tmp_path / "comparison"
        )
    assert "MOVED" in caplog.text


def test_plot_results_writes_one_curve_per_profile(tmp_path: Path) -> None:
    lb = methods_leaderboard(_results())
    plot_results(lb, tmp_path)
    assert (tmp_path / "campaign_curves_cp_0pct.png").exists()
