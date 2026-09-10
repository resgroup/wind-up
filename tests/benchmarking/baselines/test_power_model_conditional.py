"""Unit tests for the pure conditional-decomposition helpers (imputation + energy-identity re-level)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarking.baselines.power_model.conditional import impute_uncovered_bins, relevel_conditional


class TestImputeUncoveredBins:
    def test_ws_bfills_low_holes_from_the_nearest_covered_bin_above(self) -> None:
        one_plus_u = np.array([np.nan, np.nan, 1.08, 1.06, np.nan])
        measured = np.array([False, False, True, True, False])
        out = impute_uncovered_bins(one_plus_u, condition="ws", measured=measured, one_plus_overall=1.03)
        # low holes take the nearest covered bin above; trailing hole -> 1.0 (0 uplift at rated)
        assert out.tolist() == pytest.approx([1.08, 1.08, 1.08, 1.06, 1.0])

    def test_ws_all_uncovered_top_fills_to_zero_uplift(self) -> None:
        one_plus_u = np.array([np.nan, np.nan])
        measured = np.array([False, False])
        out = impute_uncovered_bins(one_plus_u, condition="ws", measured=measured, one_plus_overall=1.03)
        assert out.tolist() == pytest.approx([1.0, 1.0])

    def test_ti_fills_uncovered_at_overall(self) -> None:
        one_plus_u = np.array([1.07, np.nan, np.nan])
        measured = np.array([True, False, False])
        out = impute_uncovered_bins(one_plus_u, condition="ti", measured=measured, one_plus_overall=1.03)
        assert out.tolist() == pytest.approx([1.07, 1.03, 1.03])

    def test_measured_bins_pass_through_even_if_shape_is_finite_elsewhere(self) -> None:
        one_plus_u = np.array([1.20, 1.05, 1.10])
        measured = np.array([True, True, True])
        out = impute_uncovered_bins(one_plus_u, condition="ws", measured=measured, one_plus_overall=1.03)
        assert out.tolist() == pytest.approx([1.20, 1.05, 1.10])

    def test_power_behaves_like_ws_bfill_then_zero_at_rated(self) -> None:
        # power is monotone-saturating like ws: low holes bfill from the covered bin above, and the
        # trailing (near-rated) hole takes 1.0 (0 uplift at rated).
        one_plus_u = np.array([np.nan, 1.08, 1.06, np.nan])
        measured = np.array([False, True, True, False])
        out = impute_uncovered_bins(one_plus_u, condition="power", measured=measured, one_plus_overall=1.03)
        assert out.tolist() == pytest.approx([1.08, 1.08, 1.06, 1.0])

    def test_unknown_condition_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown condition"):
            impute_uncovered_bins(np.array([1.1]), condition="wd", measured=np.array([True]), one_plus_overall=1.0)

    def test_measured_bin_with_nan_shape_is_not_trusted(self) -> None:
        # a bin flagged measured but carrying a NaN shape (shouldn't happen, but be defensive) is filled
        one_plus_u = np.array([1.05, np.nan, 1.02])
        measured = np.array([True, True, True])
        out = impute_uncovered_bins(one_plus_u, condition="ti", measured=measured, one_plus_overall=1.04)
        assert out.tolist() == pytest.approx([1.05, 1.04, 1.02])


def _agg(sum_actual: np.ndarray, one_plus_u: np.ndarray) -> float:
    """Energy-weighted aggregation of a per-bin (1+u): ratio-of-sums Σactual / Σ(actual/(1+u))."""
    a = np.asarray(sum_actual, float)
    u = np.asarray(one_plus_u, float)
    return float(a.sum() / (a / u).sum())


class TestRelevelConditionalPinned:
    def test_measured_and_pinned_imputed_aggregate_to_overall(self) -> None:
        sum_actual = np.array([1000.0, 2000.0, 500.0])
        one_plus_u = np.array([1.10, 0.95, 1.02])  # last is an imputed pin
        measured = np.array([True, True, False])
        final = relevel_conditional(sum_actual, one_plus_u, measured=measured, one_plus_overall=1.05)
        assert final[2] == pytest.approx(1.02)  # imputed bin unchanged (pinned)
        assert _agg(sum_actual, final) == pytest.approx(1.05)  # whole thing aggregates to overall

    def test_all_measured_reduces_to_a_single_scale(self) -> None:
        sum_actual = np.array([1000.0, 1000.0])
        one_plus_u = np.array([1.1, 1.1])
        measured = np.array([True, True])
        final = relevel_conditional(sum_actual, one_plus_u, measured=measured, one_plus_overall=1.1)
        assert final == pytest.approx(one_plus_u)

    def test_no_measured_bins_falls_back_to_overall_everywhere(self) -> None:
        sum_actual = np.array([1000.0, 500.0])
        one_plus_u = np.array([1.5, 0.7])  # imputed pins only
        measured = np.array([False, False])
        final = relevel_conditional(sum_actual, one_plus_u, measured=measured, one_plus_overall=1.04)
        assert final.tolist() == pytest.approx([1.04, 1.04])

    def test_nonpositive_denominator_falls_back_to_overall(self) -> None:
        # imputed counterfactual energy already exceeds the headline total -> cannot solve λ>0
        sum_actual = np.array([1000.0, 1000.0])
        one_plus_u = np.array([1.10, 0.20])  # bin 1 imputed with a huge implied cf
        measured = np.array([True, False])
        final = relevel_conditional(sum_actual, one_plus_u, measured=measured, one_plus_overall=1.5)
        assert final.tolist() == pytest.approx([1.5, 1.5])
