"""Benchmark reshaping in study_power_model_compare."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

from benchmarking.baselines.study_power_model_compare import (
    _BASELINE_SCHEMA,
    _load_baseline_cells,
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
