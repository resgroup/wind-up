"""Tests for the R3 reference-validity screen: the ranking, the threshold and the pass loop.

The screen looks for references that are clear outliers from what is normal *for this farm over
this analysis period*. Common drift is invisible to it by design: turbines that all decline
together still read ~0 on each other, which is correct, since the test turbine would have declined
too.

The rule and the loop are pure -- the loop takes an ``estimate_one`` callable -- so they are
tested here against real functions rather than a fitted model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarking.baselines.power_model.screening import (
    ScreenResult,
    max_screenable,
    rank_by_deviation,
    screen_references,
    worst_outlier,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TestRankByDeviation:
    def test_references_are_ranked_by_distance_from_the_median(self) -> None:
        ranked = rank_by_deviation({"T15": 0.03, "T10": -0.015, "T08": -0.015})
        assert [name for name, _ in ranked] == ["T15", "T08", "T10"]  # the two zero-deviation refs tie on name

    def test_the_deviation_is_measured_from_the_median_not_from_zero(self) -> None:
        """Screening against a pool that still holds the bad reference offsets every good one."""
        ranked = dict(rank_by_deviation({"T15": 0.03, "T10": -0.015, "T08": -0.015}))
        assert ranked["T15"] == pytest.approx(0.045)
        assert ranked["T10"] == pytest.approx(0.0)

    def test_a_pack_that_drifts_together_ranks_flat(self) -> None:
        """All turbines losing power at the same rate measure ~0 on each other, which is correct."""
        ranked = dict(rank_by_deviation({"T15": -0.01, "T10": -0.01, "T08": -0.01}))
        assert all(dev == pytest.approx(0.0) for dev in ranked.values())

    def test_ties_break_on_name_so_the_order_is_deterministic(self) -> None:
        ranked = rank_by_deviation({"T15": 0.03, "T10": 0.03, "T08": 0.0, "T04": 0.0, "T02": 0.0})
        assert [name for name, _ in ranked][:2] == ["T10", "T15"]

    def test_a_non_finite_estimate_ranks_worst(self) -> None:
        """A reference the screen could not estimate is not thereby shown to be good."""
        ranked = rank_by_deviation({"T15": float("nan"), "T10": 0.03, "T08": 0.0, "T04": 0.0})
        assert ranked[0][0] == "T15"


class TestWorstOutlier:
    def test_the_furthest_reference_is_returned_when_it_clears_the_floor(self) -> None:
        assert worst_outlier({"T15": 0.03, "T10": -0.015, "T08": -0.015}, floor=0.01) == "T15"

    def test_nothing_is_returned_when_the_pack_agrees(self) -> None:
        assert worst_outlier({"T15": 0.001, "T10": 0.001, "T08": 0.001}, floor=0.01) is None

    def test_a_deviation_below_the_floor_is_left_alone(self) -> None:
        assert worst_outlier({"T15": 0.004, "T10": 0.0, "T08": 0.0}, floor=0.01) is None

    def test_a_deviation_at_the_floor_is_returned(self) -> None:
        assert worst_outlier({"T15": 0.01, "T10": 0.0, "T08": 0.0}, floor=0.01) == "T15"

    def test_only_one_reference_is_returned_even_when_several_clear_the_floor(self) -> None:
        """A bad reference infects every other estimate, so the pool is re-judged after each drop."""
        assert worst_outlier({"T15": 0.02, "T10": 0.05, "T08": 0.0, "T04": 0.0, "T02": 0.0}, floor=0.01) == "T10"


class TestMaxScreenable:
    @pytest.mark.parametrize(("pool", "expected"), [(3, 1), (4, 1), (5, 2), (7, 3), (21, 10)])
    def test_the_majority_must_survive(self, pool: int, expected: int) -> None:
        assert max_screenable(pool) == expected


def _estimator(truth: dict[str, float]) -> Callable[[str, list[str]], float]:
    """An estimate_one reading each turbine's own offset minus the mean of its reference pool.

    Reproduces the real signature: a bad reference reads its own change, and every reference
    screened against a pool containing it is dragged the other way.
    """

    def estimate_one(target: str, refs: list[str]) -> float:
        return truth[target] - sum(truth[r] for r in refs) / len(refs)

    return estimate_one


class TestScreenReferences:
    def test_a_clean_pool_screens_nobody(self) -> None:
        result = screen_references(
            ["T15", "T10", "T08"], estimate_one=_estimator(dict.fromkeys(["T15", "T10", "T08"], 0.0)), floor=0.01
        )
        assert result.screened == ()

    def test_a_pool_drifting_together_screens_nobody(self) -> None:
        """The screen removes outliers from normal, and a shared decline is normal."""
        result = screen_references(
            ["T15", "T10", "T08"], estimate_one=_estimator(dict.fromkeys(["T15", "T10", "T08"], -0.02)), floor=0.01
        )
        assert result.screened == ()

    def test_one_bad_reference_in_three_is_found(self) -> None:
        result = screen_references(
            ["T15", "T10", "T08"], estimate_one=_estimator({"T15": 0.03, "T10": 0.0, "T08": 0.0}), floor=0.01
        )
        assert result.screened == ("T15",)

    def test_two_bad_references_in_five_are_found_one_pass_at_a_time(self) -> None:
        truth = {"T15": 0.03, "T10": 0.03, "T08": 0.0, "T04": 0.0, "T02": 0.0}
        result = screen_references(["T15", "T10", "T08", "T04", "T02"], estimate_one=_estimator(truth), floor=0.01)
        assert set(result.screened) == {"T15", "T10"}
        assert len(result.screened) == 2

    def test_a_degrading_reference_is_found_as_readily_as_an_improving_one(self) -> None:
        result = screen_references(
            ["T15", "T10", "T08"], estimate_one=_estimator({"T15": -0.03, "T10": 0.0, "T08": 0.0}), floor=0.01
        )
        assert result.screened == ("T15",)

    def test_only_the_worst_reference_is_dropped_per_pass(self) -> None:
        """Dropping the worst changes what every remaining reference reads, so re-screen before the next."""
        truth = {"T15": 0.03, "T10": 0.03, "T08": 0.0, "T04": 0.0, "T02": 0.0}
        result = screen_references(["T15", "T10", "T08", "T04", "T02"], estimate_one=_estimator(truth), floor=0.01)
        first_pass = result.passes[result.passes["pass"] == 1]
        assert int(first_pass["dropped"].sum()) == 1

    def test_a_pool_too_small_to_vote_is_left_alone(self) -> None:
        """Two references cannot form a majority, so there is nothing to rule them out with."""
        result = screen_references(["T15", "T10"], estimate_one=_estimator({"T15": 0.03, "T10": 0.0}), floor=0.01)
        assert result.screened == ()
        assert not result.screenable

    def test_a_poisoned_majority_raises_rather_than_estimating(self) -> None:
        """More bad than good means a farm-wide problem that has probably reached the test turbines."""
        truth = {"T15": 0.03, "T10": -0.03, "T08": 0.05, "T04": 0.0, "T02": 0.0}
        with pytest.raises(ValueError, match="majority"):
            screen_references(["T15", "T10", "T08", "T04", "T02"], estimate_one=_estimator(truth), floor=0.01)

    def test_each_pass_is_recorded_for_inspection(self) -> None:
        result = screen_references(
            ["T15", "T10", "T08"], estimate_one=_estimator({"T15": 0.03, "T10": 0.0, "T08": 0.0}), floor=0.01
        )
        assert set(result.passes.columns) >= {"pass", "turbine", "estimate", "deviation", "dropped"}
        assert result.passes["pass"].nunique() == 2

    def test_a_screened_reference_is_not_re_estimated_afterwards(self) -> None:
        result = screen_references(
            ["T15", "T10", "T08"], estimate_one=_estimator({"T15": 0.03, "T10": 0.0, "T08": 0.0}), floor=0.01
        )
        final = result.passes[result.passes["pass"] == result.passes["pass"].max()]
        assert "T15" not in set(final["turbine"])

    def test_the_result_is_a_screen_result(self) -> None:
        result = screen_references(
            ["T15", "T10", "T08"], estimate_one=_estimator(dict.fromkeys(["T15", "T10", "T08"], 0.0)), floor=0.01
        )
        assert isinstance(result, ScreenResult)
        assert result.screenable
