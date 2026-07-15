"""Tests for the toggle-methods regression harness (profile selection, benchmark record/diff).

The benchmark is split into a portable baseline plus one per platform (F30), so these also cover the
routing, the portability invariant and the missing-file paths.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.study_toggle_methods_compare import (
    CAMPAIGN_WEEKS,
    COMPARE_METHODS,
    TOGGLE_PROFILES,
    MethodReproducibility,
    _load_baseline,
    _portable_methods,
    _reproducibility,
    _select_profiles,
    baseline_paths,
    compare_to_benchmark,
    load_merged_baseline,
    methods_leaderboard,
    plot_results,
    record_baselines,
    toggle_study,
)

if TYPE_CHECKING:
    from pathlib import Path

_BANDS = {m: _reproducibility(m).band for m in COMPARE_METHODS}


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


def _results(*, bias_shift: float = 0.0, methods: list[str] | None = None) -> pd.DataFrame:
    """Tidy rows for both methods over the weeks grid: the headline plus per-power-bin rows.

    Truth is 0 and every estimate is ``bias_shift``, so each cell's bias is ``bias_shift`` exactly.
    """
    rows = []
    cells = [("overall", "overall"), *[("power", b) for b in _POWER_BINS]]
    for method in methods or COMPARE_METHODS:
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


def _record(lb: pd.DataFrame, directory: Path, *, git_commit: str = "abc1234") -> None:
    record_baselines(lb, study=toggle_study(), git_commit=git_commit, baseline_dir=directory)


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
    _record(lb, tmp_path)

    slower = lb.assign(wall_time_s_mean=lb["wall_time_s_mean"] * 10)
    merged = compare_to_benchmark(slower, comparison_dir=tmp_path / "comparison", baseline_dir=tmp_path)
    assert "d_wall_time_s_mean" not in merged.columns
    for col in ("d_bias", "d_spread", "d_score"):
        assert (merged[col].abs() < max(_BANDS.values())).all()


class TestReproducibilityRegistry:
    def test_toggle_specialist_is_portable_and_power_model_is_not(self) -> None:
        assert _reproducibility("toggle_specialist") == MethodReproducibility(band=1e-7, portable=True)
        assert _reproducibility("power_model").portable is False

    def test_an_unclassified_method_is_assumed_machine_specific(self) -> None:
        """The safe side: wrongly calling one portable is a permanent failure on the other machine."""
        assert _reproducibility("some_new_method").portable is False

    def test_portable_methods_is_derived_from_the_registry(self) -> None:
        assert _portable_methods() == {"toggle_specialist"}


class TestBaselinePaths:
    def test_routes_to_the_current_platform(self, tmp_path: Path) -> None:
        portable, platform = baseline_paths(tmp_path, platform="win32")
        assert portable.name.endswith("_portable.json")
        assert platform.name.endswith("_win32.json")

    def test_platform_defaults_to_this_machine(self, tmp_path: Path) -> None:
        _, platform = baseline_paths(tmp_path)
        assert platform.name.endswith(f"_{sys.platform}.json")


class TestRecordSplitsByPortability:
    def test_writes_a_portable_and_a_platform_file(self, tmp_path: Path) -> None:
        _record(methods_leaderboard(_results()), tmp_path)
        portable_path, platform_path = baseline_paths(tmp_path)
        assert portable_path.exists()
        assert platform_path.exists()

    def test_portable_file_holds_only_portable_methods(self, tmp_path: Path) -> None:
        _record(methods_leaderboard(_results()), tmp_path)
        portable_path, platform_path = baseline_paths(tmp_path)
        assert json.loads(portable_path.read_text())["methods"] == ["toggle_specialist"]
        assert json.loads(platform_path.read_text())["methods"] == ["power_model"]

    def test_never_writes_another_platforms_file(self, tmp_path: Path) -> None:
        """The property that keeps two laptops from conflicting in git."""
        _, other = baseline_paths(tmp_path, platform="some_other_os")
        _record(methods_leaderboard(_results()), tmp_path)
        assert not other.exists()

    def test_records_a_machine_fingerprint(self, tmp_path: Path) -> None:
        _record(methods_leaderboard(_results()), tmp_path)
        _, platform_path = baseline_paths(tmp_path)
        prov = json.loads(platform_path.read_text())
        assert prov["platform"]
        assert prov["cpu_count"]
        assert prov["python_version"]


class TestPortabilityInvariant:
    def test_identical_portable_cells_are_not_rewritten(self, tmp_path: Path) -> None:
        """No churn, no conflict; the re-record doubles as proof portability holds."""
        lb = methods_leaderboard(_results(bias_shift=0.0123456789012345))
        _record(lb, tmp_path, git_commit="aaa")
        portable_path, _ = baseline_paths(tmp_path)
        before = portable_path.read_bytes()

        _record(lb, tmp_path, git_commit="bbb")  # later commit, same portable numbers
        assert portable_path.read_bytes() == before

    def test_differing_portable_cells_at_the_same_commit_raise(self, tmp_path: Path) -> None:
        """Same code, different machine, different numbers => portability broke."""
        _record(methods_leaderboard(_results(bias_shift=0.0)), tmp_path, git_commit="samecommit")
        with pytest.raises(ValueError, match="portable baseline"):
            _record(methods_leaderboard(_results(bias_shift=0.05)), tmp_path, git_commit="samecommit")

    def test_differing_portable_cells_at_a_different_commit_are_recorded(self, tmp_path: Path) -> None:
        """The accepted-change path. Refusing here would block every legitimate re-record."""
        _record(methods_leaderboard(_results(bias_shift=0.0)), tmp_path, git_commit="oldcommit")
        _record(methods_leaderboard(_results(bias_shift=0.05)), tmp_path, git_commit="newcommit")

        portable_path, _ = baseline_paths(tmp_path)
        doc = json.loads(portable_path.read_text())
        assert doc["git_commit"] == "newcommit"
        assert np.allclose([c["bias"] for c in doc["cells"]], 0.05)

    def test_machine_specific_cells_may_move_freely_at_the_same_commit(self, tmp_path: Path) -> None:
        """power_model is expected to differ across machines; only portable methods are policed."""
        _record(methods_leaderboard(_results(bias_shift=0.0)), tmp_path, git_commit="samecommit")
        moved_pm = pd.concat(
            [
                _results(bias_shift=0.0, methods=["toggle_specialist"]),
                _results(bias_shift=0.05, methods=["power_model"]),
            ]
        )
        _record(methods_leaderboard(moved_pm), tmp_path, git_commit="samecommit")  # must not raise


class TestLoadMerged:
    def test_merges_portable_and_platform_cells(self, tmp_path: Path) -> None:
        lb = methods_leaderboard(_results())
        _record(lb, tmp_path)
        loaded = load_merged_baseline(tmp_path)
        assert loaded is not None
        cells, prov = loaded
        assert set(cells["method"]) == set(COMPARE_METHODS)
        assert len(cells) == len(lb)
        assert len(prov) == 2  # one provenance block per file

    def test_missing_platform_file_still_diffs_the_portable_half(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A fresh machine gets its toggle_specialist check without recording anything."""
        _record(methods_leaderboard(_results()), tmp_path)
        _, platform_path = baseline_paths(tmp_path)
        platform_path.unlink()

        with caplog.at_level(logging.WARNING):
            loaded = load_merged_baseline(tmp_path)
        assert loaded is not None
        assert set(loaded[0]["method"]) == {"toggle_specialist"}
        assert "No usable benchmark" in caplog.text

    def test_missing_portable_file_still_diffs_the_platform_half(self, tmp_path: Path) -> None:
        _record(methods_leaderboard(_results()), tmp_path)
        portable_path, _ = baseline_paths(tmp_path)
        portable_path.unlink()

        loaded = load_merged_baseline(tmp_path)
        assert loaded is not None
        assert set(loaded[0]["method"]) == {"power_model"}

    def test_none_when_nothing_is_recorded(self, tmp_path: Path) -> None:
        assert load_merged_baseline(tmp_path) is None

    def test_a_foreign_fingerprint_warns_but_does_not_fail(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The hole that caused a wrong 'stale baseline' conclusion: the file could not say where it came from."""
        _record(methods_leaderboard(_results()), tmp_path)
        _, platform_path = baseline_paths(tmp_path)
        doc = json.loads(platform_path.read_text())
        doc["cpu_count"] = (doc["cpu_count"] or 0) + 99
        platform_path.write_text(json.dumps(doc))

        with caplog.at_level(logging.WARNING):
            loaded = load_merged_baseline(tmp_path)
        assert loaded is not None  # a warning, never fatal
        assert "recorded on a machine unlike this one" in caplog.text

    def test_null_fingerprint_fields_make_no_claim(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """The migrated v2 file cannot recover the recording machine's cpu_count; null must not warn."""
        _record(methods_leaderboard(_results()), tmp_path)
        _, platform_path = baseline_paths(tmp_path)
        doc = json.loads(platform_path.read_text())
        doc["cpu_count"] = None
        doc["lightgbm_version"] = None
        platform_path.write_text(json.dumps(doc))

        with caplog.at_level(logging.WARNING):
            load_merged_baseline(tmp_path)
        assert "recorded on a machine unlike this one" not in caplog.text


class TestBaselineRoundTrip:
    def test_round_trips(self, tmp_path: Path) -> None:
        lb = methods_leaderboard(_results())
        _record(lb, tmp_path)
        portable_path, _ = baseline_paths(tmp_path)

        loaded = _load_baseline(portable_path)
        assert loaded is not None
        cells, provenance = loaded
        assert provenance["campaign_weeks"] == CAMPAIGN_WEEKS
        assert provenance["seed"] == toggle_study().seed
        assert len(cells) == len(lb[lb["method"] == "toggle_specialist"])

    def test_records_the_commit_it_was_given_not_the_current_head(self, tmp_path: Path) -> None:
        # a ~15-min run can straddle a commit, so reading HEAD at write time would stamp a commit
        # whose code never produced these numbers (which is exactly what happened once).
        _record(methods_leaderboard(_results()), tmp_path, git_commit="deadbee")
        _, platform_path = baseline_paths(tmp_path)
        loaded = _load_baseline(platform_path)
        assert loaded is not None
        assert loaded[1]["git_commit"] == "deadbee"

    def test_load_baseline_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert _load_baseline(tmp_path / "nope.json") is None

    def test_load_baseline_returns_none_for_wrong_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps({"schema": "something_older", "cells": []}))
        assert _load_baseline(path) is None


def test_unchanged_method_diffs_within_the_band(tmp_path: Path) -> None:
    # the property the whole script rests on: an unchanged method reproduces its cells to numerical
    # noise. Use a bias with many significant digits, so the round(8) actually bites -- a round number
    # here would pass even if the rounding were broken, which is the trap the first version fell into.
    lb = methods_leaderboard(_results(bias_shift=0.0123456789012345))
    _record(lb, tmp_path)

    merged = compare_to_benchmark(lb, comparison_dir=tmp_path / "comparison", baseline_dir=tmp_path)
    assert not merged.empty
    for col in ("d_bias", "d_spread", "d_score"):
        for method, group in merged.groupby("method"):
            band = _BANDS[str(method)]
            assert (group[col].abs() < band).all(), f"{col} must be within {method}'s band when unchanged"


def test_round_trip_leaves_a_nonzero_residual_from_baseline_rounding(tmp_path: Path) -> None:
    # documents *why* the band is not zero: the stored baseline is rounded, so even an identical
    # re-run differs in the last digit.
    lb = methods_leaderboard(_results(bias_shift=0.0123456789012345))
    _record(lb, tmp_path)

    merged = compare_to_benchmark(lb, comparison_dir=tmp_path / "comparison", baseline_dir=tmp_path)
    assert (merged["d_bias"].abs() > 0).any(), "round(8) should leave a residual; if not, the band can tighten"


def test_moved_method_shows_a_nonzero_bias_delta(tmp_path: Path) -> None:
    _record(methods_leaderboard(_results(bias_shift=0.0)), tmp_path)

    moved = methods_leaderboard(_results(bias_shift=0.01))
    merged = compare_to_benchmark(moved, comparison_dir=tmp_path / "comparison", baseline_dir=tmp_path)
    assert np.allclose(merged["d_bias"].to_numpy(), 0.01)
    assert (merged["d_bias"].abs() > max(_BANDS.values())).all()  # 1 pp is far outside any band


def test_comparison_csv_is_written(tmp_path: Path) -> None:
    lb = methods_leaderboard(_results())
    _record(lb, tmp_path)
    comparison_dir = tmp_path / "comparison"

    compare_to_benchmark(lb, comparison_dir=comparison_dir, baseline_dir=tmp_path)
    assert (comparison_dir / "benchmark_comparison.csv").exists()


def test_compare_without_a_baseline_is_a_noop(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        merged = compare_to_benchmark(
            methods_leaderboard(_results()), comparison_dir=tmp_path / "cmp", baseline_dir=tmp_path
        )
    assert merged.empty
    assert "No benchmark recorded" in caplog.text


def test_unchanged_verdict_is_logged_per_method(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    lb = methods_leaderboard(_results())
    _record(lb, tmp_path)
    with caplog.at_level(logging.INFO):
        compare_to_benchmark(lb, comparison_dir=tmp_path / "comparison", baseline_dir=tmp_path)
    for method in COMPARE_METHODS:
        assert f"{method}: UNCHANGED" in caplog.text


def test_plots_are_written_per_profile(tmp_path: Path) -> None:
    plot_results(methods_leaderboard(_results()), tmp_path)
    assert (tmp_path / "campaign_curves_cp_0pct.png").exists()
