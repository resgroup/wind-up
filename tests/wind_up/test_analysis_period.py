"""Tests for choosing each test turbine's analysis span and power references."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.wind_up.layouts import line_layout
from wind_up.analysis_period import MONTH, AnalysisPlan, PlanSettings, intervals_duration, plan_analysis
from wind_up.layout import Layout

ROTOR_D = 100.0


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def line(n: int, *, spacing_d: float = 3.0) -> Layout:
    """T0..T{n-1} on a line, T0 at the west end, ``spacing_d`` rotor diameters apart."""
    return Layout.from_frame(line_layout([i * spacing_d * ROTOR_D for i in range(n)], rotor_diameter_m=ROTOR_D))


EXTENT = (utc("2017-01-01"), utc("2022-01-01"))
TEST_WORKS = (utc("2020-01-01"), utc("2020-01-08"))


def plan(layout: Layout, *, works: dict | None = None, **overrides: object) -> AnalysisPlan:
    """Plan T0, worked over ``TEST_WORKS``, with every other turbine a candidate and data over ``EXTENT``."""
    names = [str(n) for n in layout.frame["name"]]
    fields: dict[str, object] = {
        "turbine": "T0",
        "works": {"T0": [TEST_WORKS], **(works or {})},
        "exclusions": [],
        "extents": dict.fromkeys(names, EXTENT),
        "candidates": [n for n in names if n != "T0"],
    }
    fields.update(overrides)
    return plan_analysis(layout, **fields)  # type: ignore[arg-type]


class TestTheSpan:
    def test_with_no_other_works_both_sides_reach_their_caps(self) -> None:
        p = plan(line(6))
        assert p.start == TEST_WORKS[0] - 24 * MONTH
        assert p.end == TEST_WORKS[1] + 12 * MONTH
        assert p.pre == 24 * MONTH
        assert p.post == 12 * MONTH
        assert p.pool_rule_met
        assert p.pool_rule_reason == ""

    def test_the_span_ends_before_the_nearest_references_works(self) -> None:
        # a longer post would lose T1, the nearest, and with it the pool rule
        p = plan(line(6), works={"T1": [(utc("2020-06-01"), utc("2020-06-05"))]})
        assert p.end == utc("2020-06-01")
        assert p.power_references[0] == "T1"

    def test_the_span_starts_after_the_nearest_references_works(self) -> None:
        p = plan(line(6), works={"T1": [(utc("2018-11-01"), utc("2018-12-01"))]})
        assert p.start == utc("2018-12-01")
        assert p.end == TEST_WORKS[1] + 12 * MONTH

    def test_a_longer_short_side_wins_over_a_nearer_reference(self) -> None:
        # dropping T2 keeps the pool rule (T1, T3, T4 eligible), so the full post is kept without it
        p = plan(line(6), works={"T2": [(utc("2020-03-01"), utc("2020-03-05"))]})
        assert p.end == TEST_WORKS[1] + 12 * MONTH
        assert p.power_references == ("T1", "T3", "T4", "T5")
        assert p.waking_only["T2"] == "works 2020-03-01..2020-03-04 overlap the span"

    def test_pre_is_as_long_as_the_data_allows_up_to_its_cap(self) -> None:
        extents = dict.fromkeys([f"T{i}" for i in range(6)], (utc("2019-06-01"), EXTENT[1]))
        assert plan(line(6), extents=extents).start == utc("2019-06-01")

    def test_among_equal_lengths_the_tighter_span_keeps_the_nearer_references(self) -> None:
        # ending at the data end instead of the post cap would lose T3 for no extra counted post
        p = plan(line(7), works={"T3": [(utc("2021-03-01"), utc("2021-03-05"))]})
        assert p.end == TEST_WORKS[1] + 12 * MONTH
        assert p.power_references == ("T1", "T2", "T3", "T4")

    def test_a_full_set_of_references_beats_fewer_nearer_ones(self) -> None:
        # ending at the data end drops T5 and so tightens the worst distance, but leaves k unfilled
        p = plan(line(6), works={"T5": [(utc("2021-03-01"), utc("2021-03-05"))]}, settings=PlanSettings(k=5))
        assert p.power_references == ("T1", "T2", "T3", "T4", "T5")

    def test_lengths_subtract_the_test_turbines_exclusions(self) -> None:
        exclusions = [("T0", utc("2020-02-01"), utc("2020-03-01")), (None, utc("2019-06-01"), utc("2019-06-11"))]
        p = plan(line(6), exclusions=exclusions)
        assert p.post == (p.end - TEST_WORKS[1]) - pd.Timedelta(days=29)
        assert p.pre == (TEST_WORKS[0] - p.start) - pd.Timedelta(days=10)

    def test_the_caps_are_reached_in_usable_time_without_overshooting(self) -> None:
        exclusions = [("T0", utc("2020-02-01"), utc("2020-03-01")), (None, utc("2019-06-01"), utc("2019-06-11"))]
        p = plan(line(6), exclusions=exclusions)
        assert p.post == 12 * MONTH
        assert p.end == TEST_WORKS[1] + 12 * MONTH + pd.Timedelta(days=29)
        assert p.pre == 24 * MONTH
        assert p.start == TEST_WORKS[0] - 24 * MONTH - pd.Timedelta(days=10)

    def test_another_turbines_exclusion_does_not_shorten_the_sides(self) -> None:
        p = plan(line(6), exclusions=[("T3", utc("2020-02-01"), utc("2020-03-01"))])
        assert p.post == p.end - TEST_WORKS[1]


class TestEligibility:
    def test_a_turbine_without_data_over_the_span_is_ineligible(self) -> None:
        extents = {f"T{i}": EXTENT for i in range(6)} | {"T3": (utc("2020-06-01"), EXTENT[1])}
        p = plan(line(6), extents=extents)
        assert "T3" not in p.power_references
        assert p.waking_only["T3"] == "no data over the whole span"

    def test_another_analysed_turbine_changed_inside_the_span_is_ineligible(self) -> None:
        works = {"T1": [(utc("2020-04-01"), utc("2020-04-03"))]}
        p = plan(line(6), works=works, settings=PlanSettings(min_side=12 * MONTH))
        assert "T1" not in p.power_references

    def test_another_analysed_turbine_changed_before_the_span_is_eligible(self) -> None:
        p = plan(line(6), works={"T1": [(utc("2017-04-01"), utc("2017-04-03"))]})
        assert "T1" in p.power_references

    def test_a_shared_changeover_makes_another_analysed_turbine_ineligible(self) -> None:
        change = utc("2020-01-08")
        works = {"T0": [(change, change)], "T1": [(change, change)]}
        p = plan(line(6), works=works, settings=PlanSettings(min_side=MONTH))
        assert "T1" not in p.power_references
        assert p.pre == p.works[0] - p.start

    def test_a_turbine_not_offered_is_waking_only(self) -> None:
        p = plan(line(6), candidates=["T2", "T3", "T4", "T5"])
        assert p.waking_only["T1"] == "not offered as a reference"

    def test_a_far_turbine_is_waking_only(self) -> None:
        p = plan(line(5, spacing_d=6.0))
        assert p.waking_only["T4"] == "beyond 20 D (24.0 D)"


class TestReferences:
    def test_the_k_nearest_eligible_are_power_references(self) -> None:
        p = plan(line(8), settings=PlanSettings(k=3))
        assert p.power_references == ("T1", "T2", "T3")
        assert p.reserves == ("T4", "T5", "T6")
        assert p.waking_only["T4"] == "reserve: beyond the nearest 3"
        assert p.waking_only["T7"] == "beyond 20 D (21.0 D)"

    def test_reading_pools_leave_out_the_test_turbine(self) -> None:
        p = plan(line(6))
        assert p.reading_pools["T1"] == ("T2", "T3", "T4", "T5")
        assert "T0" not in p.reading_pools["T3"]

    def test_distances_are_in_the_test_turbines_rotor_diameters(self) -> None:
        assert plan(line(4)).distances_d == pytest.approx({"T1": 3.0, "T2": 6.0, "T3": 9.0}, rel=1e-6)


class TestFallback:
    def test_the_pool_rule_breaks_gracefully_with_a_reason(self) -> None:
        works = {w: [(utc("2020-03-01"), utc("2020-03-05"))] for w in ("T1", "T2")}
        p = plan(line(6), works=works, settings=PlanSettings(min_side=6 * MONTH))
        assert not p.pool_rule_met
        assert "nearest T1 is not eligible (works 2020-03-01..2020-03-04 overlap the span)" in p.pool_rule_reason
        assert "only 2 of the 4 nearest are eligible within 20 D: T3, T4" in p.pool_rule_reason

    def test_no_span_with_both_sides_long_enough_raises(self) -> None:
        extents = dict.fromkeys([f"T{i}" for i in range(6)], (utc("2019-12-01"), utc("2020-02-01")))
        with pytest.raises(ValueError, match="3 months"):
            plan(line(6), extents=extents)

    def test_no_eligible_reference_at_all_raises(self) -> None:
        with pytest.raises(ValueError, match="power reference"):
            plan(line(6), candidates=[])

    def test_a_shorter_span_with_three_references_beats_a_longer_one_with_fewer(self) -> None:
        # T1 and T2 have no data, so the pool rule fails over every span; a full post keeps only T5, T6
        extents = {f"T{i}": EXTENT for i in range(7)} | {t: (utc("2021-06-01"), EXTENT[1]) for t in ("T1", "T2")}
        works = {t: [(utc("2020-06-01"), utc("2020-06-05"))] for t in ("T3", "T4")}
        p = plan(line(7), extents=extents, works=works)
        assert not p.pool_rule_met
        assert p.end == utc("2020-06-01")
        assert p.power_references == ("T3", "T4", "T5", "T6")

    def test_no_span_with_three_power_references_raises(self) -> None:
        extents = {f"T{i}": EXTENT for i in range(4)} | {"T3": (utc("2021-06-01"), EXTENT[1])}
        with pytest.raises(ValueError, match="no span with both sides at least 3 months has 3 power references"):
            plan(line(4), extents=extents)

    def test_references_that_start_later_move_the_span_start(self) -> None:
        # only T1 and T2 have data from T0's first record; waiting for T3..T5 buys the references
        late = (utc("2019-06-01"), EXTENT[1])
        extents = {f"T{i}": EXTENT for i in range(3)} | {f"T{i}": late for i in range(3, 6)}
        p = plan(line(6), extents=extents)
        assert p.start == late[0]
        assert p.power_references == ("T1", "T2", "T3", "T4")
        assert p.pool_rule_met

    def test_a_reference_ending_inside_the_post_ends_the_span_when_it_is_needed(self) -> None:
        extents = {f"T{i}": EXTENT for i in range(4)} | {"T3": (EXTENT[0], utc("2020-06-01"))}
        p = plan(line(4), extents=extents)
        assert p.end == utc("2020-06-01")
        assert p.power_references == ("T1", "T2", "T3")

    def test_a_reference_ending_inside_the_post_does_not_shorten_it_when_others_suffice(self) -> None:
        extents = {f"T{i}": EXTENT for i in range(6)} | {"T3": (EXTENT[0], utc("2020-06-01"))}
        p = plan(line(6), extents=extents)
        assert p.end == TEST_WORKS[1] + 12 * MONTH
        assert p.power_references == ("T1", "T2", "T4", "T5")
        assert p.waking_only["T3"] == "no data over the whole span"

    def test_a_short_minimum_side_is_reported_in_days(self) -> None:
        extents = {f"T{i}": EXTENT for i in range(4)} | {"T3": (utc("2021-06-01"), EXTENT[1])}
        with pytest.raises(ValueError, match="no span with both sides at least 14 days has 3 power references"):
            plan(line(4), extents=extents, settings=PlanSettings(min_side=pd.Timedelta(days=14)))

    def test_the_minimum_number_of_references_is_a_setting(self) -> None:
        extents = {f"T{i}": EXTENT for i in range(4)} | {"T3": (utc("2021-06-01"), EXTENT[1])}
        p = plan(line(4), extents=extents, settings=PlanSettings(min_references=2))
        assert p.power_references == ("T1", "T2")


class TestADeclaredSpan:
    SPAN = (utc("2019-06-01"), utc("2020-06-01"))

    def test_it_is_used_as_declared(self) -> None:
        p = plan(line(6), span=self.SPAN)
        assert (p.start, p.end) == self.SPAN
        assert p.declared

    def test_a_declared_reference_is_forced_in_despite_its_works(self) -> None:
        works = {"T1": [(utc("2020-03-01"), utc("2020-03-05"))]}
        p = plan(line(6), span=self.SPAN, works=works, forced=["T1"])
        assert p.forced == ("T1",)
        assert p.power_references == ("T2", "T3", "T4", "T5", "T1")

    def test_forcing_a_reference_needs_its_data(self) -> None:
        extents = {f"T{i}": EXTENT for i in range(6)} | {"T1": (utc("2020-01-01"), EXTENT[1])}
        p = plan(line(6), span=self.SPAN, extents=extents, forced=["T1"])
        assert p.forced == ()

    def test_forcing_does_nothing_without_a_declared_span(self) -> None:
        works = {"T1": [(utc("2020-03-01"), utc("2020-03-05"))]}
        assert plan(line(6), works=works, forced=["T1"]).forced == ()


def test_intervals_duration_merges_overlaps_and_clips() -> None:
    windows = [(utc("2020-01-01"), utc("2020-01-05")), (utc("2020-01-03"), utc("2020-01-10"))]
    assert intervals_duration(windows, start=utc("2020-01-02"), end=utc("2020-01-08")) == pd.Timedelta(days=6)


def test_the_plan_table_lists_every_other_turbine_once() -> None:
    table = plan(line(8), settings=PlanSettings(k=3)).table()
    assert list(table["turbine"]) == ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    assert list(table["role"]) == ["power_reference"] * 3 + ["reserve"] * 3 + ["waking_only"]


def test_the_summary_is_plain_data() -> None:
    summary = plan(line(6)).summary()
    assert summary["power_references"] == ["T1", "T2", "T3", "T4"]
    assert summary["post_days"] == pytest.approx(365.25)
    assert summary["pool_rule_met"] is True
