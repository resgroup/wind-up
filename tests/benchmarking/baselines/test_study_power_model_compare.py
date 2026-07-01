"""Benchmark reshaping in study_power_model_compare."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from benchmarking.baselines.study_power_model_compare import (
    _BASELINE_SCHEMA,
    _MATERIAL_PP,
    _conditional_plot_subset,
    _load_baseline_cells,
    _overlay_frame,
    _select_profiles,
    _tally,
    conditional_before_after,
    conditional_before_after_table,
    power_model_leaderboard,
    record_baseline,
)
from benchmarking.harness import StudyConfig


def test_power_model_leaderboard_includes_overall_and_conditional_cells() -> None:
    fresh = pd.DataFrame(
        {
            "method": "power_model",
            "profile": "ws_dependent_cp",
            "campaign_months": 6,
            "replicate": [0, 1, 0, 1],
            "condition": ["overall", "overall", "ws", "ws"],
            "condition_bin": ["overall", "overall", "(6.0, 8.0]", "(6.0, 8.0]"],
            "estimate": [0.05, 0.05, 0.08, 0.06],
            "truth": [0.05, 0.05, 0.07, 0.07],
            "signed_error": [0.0, 0.0, 0.01, -0.01],
        }
    )
    lb = power_model_leaderboard(fresh)
    keys = set(zip(lb["condition"], lb["condition_bin"], strict=True))
    assert ("overall", "overall") in keys
    assert ("ws", "(6.0, 8.0]") in keys
    assert {"profile", "campaign_months", "condition", "condition_bin", "bias", "spread", "score"} <= set(lb.columns)


def test_conditional_plot_subset_collapses_to_longest_campaign() -> None:
    # conditional_leaderboard keeps one row per (campaign_months, condition_bin); the plot needs a
    # single row per bin, so mixing campaign lengths must be collapsed (regression: a multi-campaign
    # subset raised "cannot reindex on an axis with duplicate labels" in plot_conditional_uplift).
    cond_lb = pd.DataFrame(
        {
            "method": "power_model",
            "profile": "ws_dependent_cp",
            "campaign_months": [3, 3, 12, 12],
            "condition": "ws",
            "condition_bin": ["(4.0, 6.0]", "(6.0, 8.0]", "(4.0, 6.0]", "(6.0, 8.0]"],
            "mean_estimate": [0.09, 0.05, 0.10, 0.05],
            "mean_truth": [0.10, 0.05, 0.10, 0.05],
            "bias": [-0.01, 0.0, 0.0, 0.0],
            "spread": [0.01, 0.005, 0.008, 0.004],
        }
    )
    subset = _conditional_plot_subset(cond_lb, "ws_dependent_cp", "ws")
    assert list(subset["campaign_months"].unique()) == [12]
    assert not subset["condition_bin"].duplicated().any()


def test_conditional_plot_subset_empty_for_absent_profile() -> None:
    cond_lb = pd.DataFrame(
        {
            "profile": ["ws_dependent_cp"],
            "condition": ["ws"],
            "campaign_months": [6],
            "condition_bin": ["(6.0, 8.0]"],
        }
    )
    assert _conditional_plot_subset(cond_lb, "other_profile", "ws").empty


def _fresh_cond_lb() -> pd.DataFrame:
    """Fresh conditional leaderboard: covered profile, two campaigns (short must be dropped)."""
    return pd.DataFrame(
        {
            "method": "power_model",
            "profile": "ws_dependent_cp",
            "condition": "ws",
            "campaign_months": [3, 3, 12, 12],
            "condition_bin": ["(4.0, 6.0]", "(6.0, 8.0]", "(4.0, 6.0]", "(6.0, 8.0]"],
            # longest-campaign (12mo) rows are the ones the table must keep:
            "mean_truth": [0.10, 0.05, 0.10, 0.05],
            "mean_estimate": [0.09, 0.05, 0.105, 0.08],
            "bias": [-0.01, 0.0, 0.005, 0.03],
            "spread": [0.01, 0.005, 0.008, 0.004],
        }
    )


def _baseline_cells() -> pd.DataFrame:
    """Pre-change benchmark cells for the same (profile, campaign, condition, bins)."""
    return pd.DataFrame(
        {
            "profile": "ws_dependent_cp",
            "campaign_months": 12,
            "condition": "ws",
            "condition_bin": ["(4.0, 6.0]", "(6.0, 8.0]"],
            "bias": [0.02, 0.03],
            "spread": [0.009, 0.005],
            "score": [0.02, 0.03],
        }
    )


def test_conditional_before_after_reconstructs_est_before_and_verdict() -> None:
    table = conditional_before_after_table(_fresh_cond_lb(), _baseline_cells(), material_pp=_MATERIAL_PP)

    # Only the longest campaign (12mo) survives.
    assert set(table["campaign_months"].unique()) == {12}
    rows = table.set_index("condition_bin")

    # est_before = (mean_truth + benchmark bias), reported in pp.
    # bin (4.0, 6.0]: truth 0.10 + bias_before 0.02 = 0.12 -> 12.0 pp
    assert rows.loc["(4.0, 6.0]", "est_before"] == pytest.approx(12.0)
    # est_after is the fresh mean_estimate: 0.105 -> 10.5 pp
    assert rows.loc["(4.0, 6.0]", "est_after"] == pytest.approx(10.5)
    # |bias| moved 2.0 pp -> 0.5 pp, so d_abs_bias = -1.5 pp -> "better".
    assert rows.loc["(4.0, 6.0]", "d_abs_bias"] == pytest.approx(-1.5)
    assert rows.loc["(4.0, 6.0]", "verdict"] == "better"

    # bin (6.0, 8.0]: |bias| unchanged (3.0 pp -> 3.0 pp) -> neutral "~".
    assert rows.loc["(6.0, 8.0]", "d_abs_bias"] == pytest.approx(0.0)
    assert rows.loc["(6.0, 8.0]", "verdict"] == "~"


def test_conditional_before_after_verdict_band_edges() -> None:
    # Three bins whose |bias| change is just inside, exactly at, and just outside the 0.1 pp band.
    fresh = pd.DataFrame(
        {
            "method": "power_model",
            "profile": "ti_dependent_cp",
            "condition": "ti",
            "campaign_months": 12,
            "condition_bin": ["inside", "at", "outside"],
            "mean_truth": [0.0, 0.0, 0.0],
            # fresh |bias| in pp: 4.9, 4.9, 4.89 (via mean_estimate); benchmark |bias| = 5.0 pp below.
            "mean_estimate": [0.049, 0.049, 0.0489],
            "bias": [0.049, 0.049, 0.0489],
            "spread": [0.0, 0.0, 0.0],
        }
    )
    baseline = pd.DataFrame(
        {
            "profile": "ti_dependent_cp",
            "campaign_months": 12,
            "condition": "ti",
            "condition_bin": ["inside", "at", "outside"],
            "bias": [0.05, 0.05, 0.05],  # |bias| 5.0 pp
            "spread": [0.0, 0.0, 0.0],
            "score": [0.05, 0.05, 0.05],
        }
    )
    rows = conditional_before_after_table(fresh, baseline, material_pp=0.1).set_index("condition_bin")
    # inside: d_abs_bias = 4.9 - 5.0 = -0.1 pp exactly -> not beyond a strict band -> "~".
    assert rows.loc["inside", "d_abs_bias"] == pytest.approx(-0.1)
    assert rows.loc["inside", "verdict"] == "~"
    # outside: d_abs_bias = 4.89 - 5.0 = -0.11 pp -> beyond -0.1 -> "better".
    assert rows.loc["outside", "d_abs_bias"] == pytest.approx(-0.11)
    assert rows.loc["outside", "verdict"] == "better"


def test_conditional_before_after_keeps_only_covered_profiles() -> None:
    fresh = pd.concat(
        [
            _fresh_cond_lb(),
            _fresh_cond_lb().assign(profile="cp_plus_10pct"),  # not a covered profile
        ],
        ignore_index=True,
    )
    baseline = pd.concat(
        [
            _baseline_cells(),
            _baseline_cells().assign(profile="cp_plus_10pct"),
        ],
        ignore_index=True,
    )
    table = conditional_before_after_table(fresh, baseline, material_pp=_MATERIAL_PP)
    assert set(table["profile"].unique()) == {"ws_dependent_cp"}


def test_overlay_frame_reconstructs_benchmark_series() -> None:
    frame = _overlay_frame(_fresh_cond_lb(), _baseline_cells())
    assert set(frame["method"].unique()) == {"power_model (benchmark)", "power_model (current)"}
    bench = frame[frame["method"] == "power_model (benchmark)"].set_index("condition_bin")
    curr = frame[frame["method"] == "power_model (current)"].set_index("condition_bin")
    # benchmark estimate reconstructed in fraction: truth 0.10 + bias_before 0.02 = 0.12.
    assert bench.loc["(4.0, 6.0]", "mean_estimate"] == pytest.approx(0.12)
    assert bench.loc["(4.0, 6.0]", "spread"] == pytest.approx(0.009)  # benchmark's own spread
    # current estimate is the fresh mean_estimate (fraction), for plot_conditional_uplift to scale.
    assert curr.loc["(4.0, 6.0]", "mean_estimate"] == pytest.approx(0.105)
    assert {"condition", "condition_bin", "mean_truth", "mean_estimate", "spread"} <= set(frame.columns)


def _write_baseline(path: Path) -> None:
    """A v2 benchmark JSON with covered-profile conditional cells for the orchestrator test."""
    doc = {
        "schema": _BASELINE_SCHEMA,
        "modes": {
            "prepost": {
                "recorded_utc": "2025-01-01T00:00:00Z",
                "git_commit": "abc1234",
                "n_replicates": 2,
                "seed": 0,
                "campaign_months": [12],
                "profiles": ["ws_dependent_cp"],
                "cells": [
                    {
                        "profile": "ws_dependent_cp",
                        "campaign_months": 12,
                        "condition": "ws",
                        "condition_bin": "(4.0, 6.0]",
                        "bias": 0.02,
                        "spread": 0.009,
                        "score": 0.02,
                    },
                    {
                        "profile": "ws_dependent_cp",
                        "campaign_months": 12,
                        "condition": "ws",
                        "condition_bin": "(6.0, 8.0]",
                        "bias": 0.03,
                        "spread": 0.005,
                        "score": 0.03,
                    },
                ],
            }
        },
    }
    path.write_text(json.dumps(doc))


def _fresh_results() -> pd.DataFrame:
    """Tidy fresh scoring results (raw rows) for one covered profile, two ws bins, two replicates."""
    rows = []
    for rep in (0, 1):
        for cbin, truth, est in (("(4.0, 6.0]", 0.10, 0.105), ("(6.0, 8.0]", 0.05, 0.08)):
            rows.append(
                {
                    "method": "power_model",
                    "profile": "ws_dependent_cp",
                    "campaign_months": 12,
                    "replicate": rep,
                    "condition": "ws",
                    "condition_bin": cbin,
                    "estimate": est,
                    "truth": truth,
                    "signed_error": est - truth,
                }
            )
    return pd.DataFrame(rows)


def test_conditional_before_after_writes_csv_and_plots(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    _write_baseline(baseline_path)
    comparison_dir = tmp_path / "comparison"
    comparison_dir.mkdir()

    conditional_before_after("prepost", _fresh_results(), baseline_path, comparison_dir)

    assert (comparison_dir / "conditional_benchmark_comparison_prepost.csv").exists()
    assert (comparison_dir / "conditional_before_after_ws_dependent_cp_ws.png").exists()


def test_conditional_before_after_no_baseline_is_noop(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparison"
    comparison_dir.mkdir()
    # No baseline file -> warn and return without writing anything (mirrors compare_to_benchmark).
    conditional_before_after("prepost", _fresh_results(), tmp_path / "missing.json", comparison_dir)
    assert not list(comparison_dir.iterdir())


def test_select_profiles_none_returns_all() -> None:
    selected = _select_profiles(None)
    assert "cp_0pct" in selected
    assert len(selected) == 7  # the full overnight set


def test_select_profiles_restricts_and_preserves_order() -> None:
    selected = _select_profiles(["cp_0pct"])
    assert list(selected) == ["cp_0pct"]


def test_select_profiles_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        _select_profiles(["cp_0pct", "not_a_profile"])


def test_tally_uses_pp_band_in_fractional_form() -> None:
    # _tally works on fractional deltas; the 0.1 pp band is 1e-3 in fraction.
    threshold = _MATERIAL_PP / 100.0
    # inside the band (0.9e-3) -> neutral; outside (1.1e-3) -> worse/better.
    delta = pd.Series([-0.0009, 0.0009, 0.0011, -0.0011])
    result = _tally(delta, n_cells=4, threshold=threshold)
    assert result == "1 better / 1 worse (of 4)"


def _minimal_study() -> StudyConfig:
    return StudyConfig(
        mode="prepost",
        turbine_subset=["WT01"],
        treatment_start_range=(pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-06-01", tz="UTC")),
        min_pre_months=6,
        campaign_months=[6],
        n_replicates=1,
        seed=0,
    )


def _minimal_leaderboard() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "profile": ["test_profile"],
            "campaign_months": [6],
            "condition": ["overall"],
            "condition_bin": ["overall"],
            "bias": [0.01],
            "spread": [0.02],
            "score": [0.03],
        }
    )


def test_load_baseline_cells_returns_none_for_wrong_schema(tmp_path: Path) -> None:
    """_load_baseline_cells must return None (not crash) when the file has an old/mismatched schema."""
    path = tmp_path / "baseline.json"
    doc = {
        "schema": "power_model_compare_baseline_v1",
        "modes": {
            "prepost": {
                "recorded_utc": "2025-01-01T00:00:00Z",
                "git_commit": "abc1234",
                "n_replicates": 1,
                "seed": 0,
                "campaign_months": [6],
                "profiles": ["test_profile"],
                "cells": [{"profile": "test_profile", "campaign_months": 6, "bias": 0.01}],
            }
        },
    }
    path.write_text(json.dumps(doc))

    result = _load_baseline_cells("prepost", path)

    assert result is None


def test_record_baseline_drops_stale_modes_on_schema_bump(tmp_path: Path) -> None:
    """When on-disk schema != _BASELINE_SCHEMA, old-schema mode cells must be dropped (not inherited)."""
    path = tmp_path / "baseline.json"
    # Simulate a v1 file that has a toggle entry with v1-shaped cells (no condition/condition_bin).
    old_doc = {
        "schema": "power_model_compare_baseline_v1",
        "modes": {
            "toggle": {
                "recorded_utc": "2025-01-01T00:00:00Z",
                "git_commit": "abc1234",
                "n_replicates": 1,
                "seed": 0,
                "campaign_months": [6],
                "profiles": ["test_profile"],
                "cells": [
                    {"profile": "test_profile", "campaign_months": 6, "bias": 0.01, "spread": 0.02, "score": 0.03}
                ],
            }
        },
    }
    path.write_text(json.dumps(old_doc))

    lb = _minimal_leaderboard()
    study = _minimal_study()
    record_baseline({"prepost": lb}, {"prepost": study}, path)

    written = json.loads(path.read_text())
    assert written["schema"] == _BASELINE_SCHEMA
    # The stale v1 toggle entry must be gone — a later toggle compare must not KeyError on missing columns.
    assert "toggle" not in written.get("modes", {}), "stale v1 toggle cells must be dropped after schema bump"
    # Equivalently, _load_baseline_cells for toggle returns None.
    assert _load_baseline_cells("toggle", path) is None


def test_record_baseline_preserves_sibling_modes_when_schemas_match(tmp_path: Path) -> None:
    """When on-disk schema matches _BASELINE_SCHEMA, sibling modes must be preserved (incremental update)."""
    path = tmp_path / "baseline.json"
    # Write a v2 file that already has a toggle entry.
    existing_toggle_cells = [
        {
            "profile": "test_profile",
            "campaign_months": 6,
            "condition": "overall",
            "condition_bin": "overall",
            "bias": 0.05,
            "spread": 0.06,
            "score": 0.07,
        }
    ]
    existing_doc = {
        "schema": _BASELINE_SCHEMA,
        "modes": {
            "toggle": {
                "recorded_utc": "2025-01-01T00:00:00Z",
                "git_commit": "abc1234",
                "n_replicates": 1,
                "seed": 0,
                "campaign_months": [6],
                "profiles": ["test_profile"],
                "cells": existing_toggle_cells,
            }
        },
    }
    path.write_text(json.dumps(existing_doc))

    lb = _minimal_leaderboard()
    study = _minimal_study()
    record_baseline({"prepost": lb}, {"prepost": study}, path)

    written = json.loads(path.read_text())
    assert written["schema"] == _BASELINE_SCHEMA
    # The sibling toggle mode must still be present (incremental update must not drop it).
    assert "toggle" in written.get("modes", {}), "matching-schema sibling mode must be preserved"
    assert "prepost" in written.get("modes", {}), "newly recorded prepost mode must be present"
    toggle_result = _load_baseline_cells("toggle", path)
    assert toggle_result is not None


def test_record_baseline_stamps_current_schema_over_old_file(tmp_path: Path) -> None:
    """record_baseline must overwrite 'schema' with _BASELINE_SCHEMA even when an old-schema file exists."""
    path = tmp_path / "baseline.json"
    # Write a v1 stub to simulate the on-disk old file.
    old_doc = {
        "schema": "power_model_compare_baseline_v1",
        "modes": {},
    }
    path.write_text(json.dumps(old_doc))

    lb = _minimal_leaderboard()
    study = _minimal_study()
    record_baseline({"prepost": lb}, {"prepost": study}, path)

    written = json.loads(path.read_text())
    assert written["schema"] == _BASELINE_SCHEMA

    # The round-trip via _load_baseline_cells must now succeed (not return None).
    loaded = _load_baseline_cells("prepost", path)
    assert loaded is not None
    cells_df, prov = loaded
    assert isinstance(cells_df, pd.DataFrame)
    assert prov["n_replicates"] == 1
