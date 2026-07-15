"""Tests for the toggle-specialist uncertainty study driver.

The sweep itself is far too slow to run here (64 replicates of real SCADA), so these cover the
pieces that decide whether a run's *output* is right: the block-length variants, the recovery of
block length from a variant name, and the plotting's behaviour on an arbitrary ``--block-hours``
grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.study_toggle_specialist_uncertainty import (
    _block_hours_of,
    _focus_block_hours,
    build_methods,
    calibration_tables,
    plot_results,
)
from benchmarking.baselines.toggle_specialist import DEFAULT_BLOCK_HOURS

if TYPE_CHECKING:
    from pathlib import Path


def _cases(block_hours: list[float]) -> pd.DataFrame:
    """A minimal scored-cases frame with a headline and one power bin per block length."""
    rng = np.random.default_rng(0)
    rows = []
    for bl in block_hours:
        for replicate in range(8):
            for weeks in (1, 4):
                for condition, bin_label, n_up in (("overall", "overall", 5000), ("power", "(0.0, 460.0]", 40)):
                    err = float(rng.normal(0, 0.01))
                    rows.append(
                        {
                            "method": f"toggle_specialist_bl{bl:g}",
                            "profile": "cp_0pct",
                            "replicate": replicate,
                            "test_wtg": "T01",
                            "campaign_weeks": weeks,
                            "condition": condition,
                            "condition_bin": bin_label,
                            "estimate": err,
                            "truth": 0.0,
                            "signed_error": err,
                            "sigma": 0.01,
                            "sigma_robust": 0.01,
                            "n_upgraded_records": n_up,
                            "n_baseline_records": n_up,
                            "n_blocks": 10,
                            "frac_resamples_finite": 1.0,
                            "block_hours": bl,
                        }
                    )
    return pd.DataFrame(rows)


class TestBuildMethods:
    def test_one_variant_per_block_length_each_named_for_it(self) -> None:
        methods = build_methods([6.0, 48.0])
        assert [m.name for m in methods] == ["toggle_specialist_bl6", "toggle_specialist_bl48"]
        assert [m.block_hours for m in methods] == [6.0, 48.0]

    def test_variants_differ_only_in_block_length(self) -> None:
        """They must produce identical uplifts; only sigma may differ."""
        a, b = build_methods([6.0, 48.0])
        assert a.conditions == b.conditions
        assert a.rated_power_kw == b.rated_power_kw
        assert a.n_resamples == b.n_resamples
        assert a.bootstrap_seed == b.bootstrap_seed

    def test_block_hours_round_trips_through_the_variant_name(self) -> None:
        for block_hours in (1.0, 6.0, 48.0, 96.0):
            (method,) = build_methods([block_hours])
            assert _block_hours_of(method.name) == block_hours


class TestFocusBlockHours:
    def test_prefers_the_method_default_when_it_was_swept(self) -> None:
        assert _focus_block_hours(_cases([6.0, DEFAULT_BLOCK_HOURS, 96.0])) == DEFAULT_BLOCK_HOURS

    def test_falls_back_to_the_nearest_swept_length(self) -> None:
        """``--block-hours`` is a free grid, so the default need not be in it.

        The grid is built from the default rather than hardcoded, so this keeps testing the
        fallback rather than the default's current value.
        """
        near, far = DEFAULT_BLOCK_HOURS * 2, DEFAULT_BLOCK_HOURS * 8
        assert _focus_block_hours(_cases([near, far])) == near

    def test_single_length_grid(self) -> None:
        assert _focus_block_hours(_cases([3.0])) == 3.0


class TestPlotResults:
    def test_all_four_plots_are_written_for_a_grid_containing_the_default(self, tmp_path: Path) -> None:
        cases = _cases([DEFAULT_BLOCK_HOURS, DEFAULT_BLOCK_HOURS * 4])
        plot_results(cases, calibration_tables(cases, n_independent=8), tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "coverage_by_campaign_length.png",
            "coverage_vs_record_count.png",
            "error_vs_sigma.png",
            "sigma_vs_block_length.png",
        ]

    def test_all_four_plots_are_written_when_the_default_was_not_swept(self, tmp_path: Path) -> None:
        """Regression: the per-case plots hardcoded 48h, so a grid without it silently lost them.

        ``coverage_vs_record_count`` was skipped entirely and ``error_vs_sigma`` was drawn from an
        empty frame (an all-NaN axis limit) — a run that looked successful but produced junk.
        """
        cases = _cases([DEFAULT_BLOCK_HOURS * 16, DEFAULT_BLOCK_HOURS * 32])
        plot_results(cases, calibration_tables(cases, n_independent=8), tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "coverage_by_campaign_length.png",
            "coverage_vs_record_count.png",
            "error_vs_sigma.png",
            "sigma_vs_block_length.png",
        ]


class TestCalibrationTables:
    def test_reports_every_read_keyed_by_block_length(self) -> None:
        tables = calibration_tables(_cases([6.0, 48.0]), n_independent=8)
        assert set(tables) == {
            "headline_by_block",
            "headline_by_block_and_length",
            "headline_by_block_and_profile",
            "per_bin_by_block",
            "per_bin_by_block_and_bin",
        }
        assert sorted(tables["headline_by_block"]["block_hours"]) == [6.0, 48.0]

    def test_headline_and_per_bin_are_scored_separately(self) -> None:
        """They fail for different reasons, so pooling them would hide both."""
        tables = calibration_tables(_cases([6.0]), n_independent=8)
        assert tables["headline_by_block"]["n"].iloc[0] == 16  # 8 replicates x 2 campaign lengths
        assert tables["per_bin_by_block"]["n"].iloc[0] == 16

    def test_the_coverage_standard_error_is_quoted_on_independent_draws(self) -> None:
        """Rows are not evidence: profiles share windows, so SE must come from the replicate count."""
        tables = calibration_tables(_cases([6.0]), n_independent=64)
        assert tables["headline_by_block"]["coverage_se_independent"].iloc[0] == pytest.approx(0.058, abs=0.001)
