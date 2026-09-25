"""Tests for the uncertainty-calibration metrics.

Built on constructed z's with known properties, so a target the metric should hit is known exactly
rather than inferred from a method's behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.harness.calibration import (
    TARGET_COVERAGE_1SIGMA,
    calibration_summary,
    coverage_standard_error,
    summarize_calibration,
)


def _errors_from_z(z: np.ndarray, sigma: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Signed errors and sigmas whose ratio is exactly ``z``."""
    return z * sigma, np.full(len(z), sigma)


class TestCalibrationSummary:
    def test_a_perfectly_calibrated_sigma_hits_the_target(self) -> None:
        z = np.random.default_rng(0).standard_normal(20000)
        summary = calibration_summary(*_errors_from_z(z))
        assert summary.coverage_1sigma == pytest.approx(TARGET_COVERAGE_1SIGMA, abs=0.01)
        assert summary.z_spread == pytest.approx(1.0, abs=0.02)
        assert summary.z_robust == pytest.approx(1.0, abs=0.03)

    def test_a_sigma_that_is_too_small_under_covers(self) -> None:
        z = np.random.default_rng(0).standard_normal(20000) * 2.0  # errors twice as wide as claimed
        summary = calibration_summary(*_errors_from_z(z))
        assert summary.coverage_1sigma < 0.45
        assert summary.z_spread == pytest.approx(2.0, abs=0.05)

    def test_a_sigma_that_is_too_large_over_covers_and_shows_the_width(self) -> None:
        z = np.random.default_rng(0).standard_normal(20000) * 0.5
        summary = calibration_summary(*_errors_from_z(z, sigma=0.02))
        assert summary.coverage_1sigma > 0.9
        # Over-covering is bought with width: mean_sigma is what exposes it.
        assert summary.mean_sigma == pytest.approx(0.02)
        assert summary.rms_error == pytest.approx(0.01, rel=0.05)

    def test_bias_alone_can_break_coverage(self) -> None:
        """A method with no scatter but a 2-sigma bias covers nothing, which is the honest read."""
        errors = np.full(500, 0.02)
        sigma = np.full(500, 0.01)
        summary = calibration_summary(errors, sigma)
        assert summary.coverage_1sigma == 0.0
        assert summary.z_spread == pytest.approx(0.0)

    def test_z_spread_blows_up_on_a_tail_while_z_robust_holds(self) -> None:
        """The pair's whole purpose: disagreement localises the miscalibration to the tails."""
        z = np.random.default_rng(0).standard_normal(2000)
        z[0] = 300.0
        summary = calibration_summary(*_errors_from_z(z))
        assert summary.z_spread > 5
        assert summary.z_robust == pytest.approx(1.0, abs=0.1)


class TestUnusable:
    def test_a_non_positive_sigma_is_excluded_but_counted(self) -> None:
        errors = np.array([0.01, 0.01, 0.01, 0.01])
        sigma = np.array([0.01, 0.0, -1.0, np.nan])
        summary = calibration_summary(errors, sigma)
        assert summary.n == 1
        assert summary.n_unusable == 3

    def test_a_non_finite_error_is_ignored_entirely(self) -> None:
        """No estimate means nothing to calibrate: it is not an uncertainty failure."""
        summary = calibration_summary(np.array([0.01, np.nan]), np.array([0.01, np.nan]))
        assert summary.n == 1
        assert summary.n_unusable == 0

    def test_all_unusable_gives_a_nan_summary(self) -> None:
        summary = calibration_summary(np.array([0.01, 0.02]), np.array([np.nan, 0.0]))
        assert summary.n == 0
        assert summary.n_unusable == 2
        assert np.isnan(summary.coverage_1sigma)

    def test_empty_input_gives_a_nan_summary(self) -> None:
        summary = calibration_summary(np.array([]), np.array([]))
        assert summary.n == 0
        assert np.isnan(summary.z_spread)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            calibration_summary(np.array([0.01]), np.array([0.01, 0.02]))


class TestSummarizeCalibration:
    def _frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        good = rng.standard_normal(4000)
        bad = rng.standard_normal(4000) * 3.0
        return pd.DataFrame(
            {
                "group": ["good"] * 4000 + ["bad"] * 4000,
                "signed_error": np.concatenate([good, bad]) * 0.01,
                "sigma": 0.01,
            }
        )

    def test_groups_are_summarised_independently(self) -> None:
        table = summarize_calibration(self._frame(), group_keys=["group"]).set_index("group")
        assert table.loc["good", "coverage_1sigma"] == pytest.approx(TARGET_COVERAGE_1SIGMA, abs=0.03)
        assert table.loc["bad", "coverage_1sigma"] < 0.35

    def test_missing_columns_raise(self) -> None:
        with pytest.raises(ValueError, match="missing column"):
            summarize_calibration(pd.DataFrame({"group": ["a"]}), group_keys=["group"])

    def test_multiple_group_keys(self) -> None:
        frame = self._frame().assign(other=lambda d: np.where(d.index < 2000, "x", "y"))
        table = summarize_calibration(frame, group_keys=["group", "other"])
        assert len(table) == 3  # good/x, good/y, bad/y
        assert list(table.columns[:2]) == ["group", "other"]


class TestCoverageStandardError:
    def test_shrinks_as_the_root_of_n(self) -> None:
        assert coverage_standard_error(64) == pytest.approx(coverage_standard_error(16) / 2, rel=1e-6)

    def test_at_sixty_four_independent_draws(self) -> None:
        assert coverage_standard_error(64) == pytest.approx(0.058, abs=0.001)

    def test_non_positive_n_is_nan(self) -> None:
        assert np.isnan(coverage_standard_error(0))
