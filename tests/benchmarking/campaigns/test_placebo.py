"""Tests for the two declared placebo campaigns."""

from __future__ import annotations

import pandas as pd
import pytest

from benchmarking.campaigns import CampaignRunner, per_turbine_table
from benchmarking.campaigns.placebo import (
    MIN_SCREENABLE_REFERENCES,
    PLACEBO_CAMPAIGN_START,
    PLACEBO_INSTANCE_KEEP_AS_REFERENCE,
    PLACEBO_INSTANCE_LAST_CLEAN,
    PLACEBO_INSTANCE_LAST_GOOD_END,
    PLACEBO_TEST_CANDIDATES,
    PLACEBO_TURBINES,
    PLACEBO_UPGRADED,
    PLACEBO_WTG_NUMBERS,
    placebo_analysis_period,
    placebo_campaign,
    placebo_instance,
)
from benchmarking.harness import MethodInput, MethodOutput
from benchmarking.synthetic import HOT_COLUMNS, SensorGainStep, ToggleSchedule

TOLERANCE = 1e-9
# A small slice of the farm, so the fixtures stay cheap; the production default is all 21 turbines.
TEST_TURBINES = ("T07", "T11")
# A line of turbines 400 m apart, so "the two nearest" is unambiguous.
LINE_COORDS = {f"T{i:02d}": (57.5 + i * 0.0036, -3.25) for i in range(1, 22)}
TEST_PARTICIPANTS = ("T07", "T11", "T01", "T02", "T03")


class ZeroMethod:
    """Reports exactly zero uplift."""

    name = "zero"

    def estimate(self, mi: MethodInput) -> MethodOutput:  # noqa: ARG002
        """Return a zero P50."""
        return MethodOutput(p50_overall=0.0)


def fixture_scada(mode: str) -> pd.DataFrame:
    """A tiny stand-in for the Hill of Towie download: flat hourly power over ``mode``'s period."""
    index = pd.date_range(*placebo_analysis_period(mode), freq="1h", tz="UTC", inclusive="left")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    HOT_COLUMNS.turbine: wtg,
                    HOT_COLUMNS.active_power: 900.0,
                    HOT_COLUMNS.active_power_min: 850.0,
                    HOT_COLUMNS.wind_speed: 8.0,
                    HOT_COLUMNS.wind_speed_sd: 0.8,
                    HOT_COLUMNS.gen_rpm: 1400.0,
                    HOT_COLUMNS.availability: 3600.0,
                },
                index=index,
            )
            for wtg in TEST_PARTICIPANTS
        ]
    )


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_placebo_injects_nothing(mode: str) -> None:
    declared = placebo_campaign(mode, upgraded=TEST_TURBINES, turbines=TEST_PARTICIPANTS)
    assert declared.upgrades == []
    dataset = declared.generate(fixture_scada(mode))
    pd.testing.assert_frame_equal(dataset.synthetic_df, dataset.original_df)


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_placebo_spec_mode_matches(mode: str) -> None:
    assert placebo_campaign(mode).spec().mode == mode


@pytest.mark.parametrize(("mode", "months"), [("prepost", 12), ("toggle", 6)])
def test_each_mode_gets_its_campaign_length_on_a_full_year_of_baseline(mode: str, months: int) -> None:
    start, end = placebo_analysis_period(mode)
    assert start < PLACEBO_CAMPAIGN_START < end
    assert pd.Timestamp("2018-01-01", tz="UTC") == PLACEBO_CAMPAIGN_START
    assert end == PLACEBO_CAMPAIGN_START + pd.DateOffset(months=months)
    assert start == PLACEBO_CAMPAIGN_START - pd.DateOffset(months=12)


def test_prepost_baseline_and_treated_periods_hold_the_same_seasons() -> None:
    # the point of the 12-month prepost campaign: an unconditioned method cannot then confuse
    # a seasonal difference between the two periods with an effect
    start, end = placebo_analysis_period("prepost")
    assert (end - PLACEBO_CAMPAIGN_START) == (PLACEBO_CAMPAIGN_START - start)
    assert start.month == PLACEBO_CAMPAIGN_START.month == end.month


def test_the_toggle_blocks_are_fifty_minutes() -> None:
    schedule = placebo_campaign("toggle", upgraded=TEST_TURBINES, turbines=TEST_PARTICIPANTS).upgrade_timing
    assert schedule.period / 2 == pd.Timedelta(minutes=50)
    assert schedule.start == PLACEBO_CAMPAIGN_START


def test_turbine_names_are_built_from_the_turbine_numbers() -> None:
    assert PLACEBO_TURBINES[0] == "T01"
    assert PLACEBO_TURBINES[-1] == "T21"
    assert len(PLACEBO_TURBINES) == len(PLACEBO_WTG_NUMBERS)


def test_the_test_turbines_are_drawn_from_the_eligible_candidates() -> None:
    assert set(PLACEBO_UPGRADED) <= set(PLACEBO_TEST_CANDIDATES)
    assert set(PLACEBO_UPGRADED) <= set(PLACEBO_TURBINES)


def test_references_are_every_participating_turbine_that_is_not_a_test_turbine() -> None:
    spec = placebo_campaign("prepost").spec()
    assert set(spec.candidate_references) == set(PLACEBO_TURBINES) - set(PLACEBO_UPGRADED)
    assert not set(spec.candidate_references) & set(spec.upgraded_turbines)


def test_toggle_placebo_declares_a_schedule() -> None:
    assert isinstance(placebo_campaign("toggle").upgrade_timing, ToggleSchedule)


def test_an_excluded_turbine_is_carried_onto_the_spec() -> None:
    spec = placebo_campaign("prepost", excluded=["T21"]).spec()
    assert spec.excluded_turbines == ["T21"]
    assert not set(spec.upgraded_turbines) & {"T21"}


def test_an_excluded_turbine_is_not_also_offered_as_a_reference() -> None:
    # the spec would otherwise declare contradictory roles for the same turbine
    spec = placebo_campaign("prepost", excluded=["T21"]).spec()
    assert "T21" not in spec.candidate_references
    assert not set(spec.candidate_references) & set(spec.excluded_turbines)


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_placebo_runs_end_to_end_to_zero(mode: str) -> None:
    declared = placebo_campaign(mode, upgraded=TEST_TURBINES, turbines=TEST_PARTICIPANTS)
    dataset = declared.generate(fixture_scada(mode))
    result = CampaignRunner(declared.spec(), dataset, build_methods=lambda _wtg: [ZeroMethod()]).run()
    assert per_turbine_table(result)["signed_error"].abs().max() < TOLERANCE
    assert abs(result.farm_uplifts["zero"].uplift) < TOLERANCE
    assert abs(result.truth_farm_uplift) < TOLERANCE


def test_placebo_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        placebo_campaign("sideways")


def test_no_faults_are_injected_by_default() -> None:
    assert placebo_campaign("prepost").faults == []


def test_declared_faults_reach_the_campaign() -> None:
    """R2 runs the placebo with a sensor fault injected, so the declaration has to carry one."""
    fault = SensorGainStep(turbine="T01", at=PLACEBO_CAMPAIGN_START, gain=1.5)
    assert placebo_campaign("prepost", faults=[fault]).faults == [fault]


# --- the randomised instance, so a handover does not identify itself ---------------------------


def instance(mode: str = "prepost", *, seed: int, turbines: tuple[str, ...] | None = None):  # noqa: ANN201
    """A randomised instance on the straight-line fixture geometry."""
    names = turbines if turbines is not None else tuple(LINE_COORDS)
    return placebo_instance(mode, seed=seed, turbines=names, coords={w: LINE_COORDS[w] for w in names})


def two_nearest(turbine: str, among: list[str]) -> list[str]:
    """The two turbines of ``among`` closest to ``turbine`` on the fixture line."""
    others = [w for w in among if w != turbine]
    return sorted(others, key=lambda w: abs(LINE_COORDS[w][0] - LINE_COORDS[turbine][0]))[:2]


class TestGeographicSpacing:
    def test_every_test_turbine_keeps_its_two_nearest_as_references(self) -> None:
        # a real campaign is designed so each test turbine has nearby references to compare against
        for seed in range(25):
            campaign = instance(seed=seed)
            participating = [*campaign.upgraded_turbines, *campaign.candidate_references]
            for wtg in campaign.upgraded_turbines:
                nearest = two_nearest(wtg, participating)
                assert set(nearest) <= set(campaign.candidate_references), (
                    f"seed {seed}: {wtg}'s nearest {nearest} are not both references"
                )

    def test_no_two_test_turbines_are_immediate_neighbours(self) -> None:
        for seed in range(25):
            campaign = instance(seed=seed)
            chosen = set(campaign.upgraded_turbines)
            participating = [*campaign.upgraded_turbines, *campaign.candidate_references]
            for wtg in chosen:
                assert not (set(two_nearest(wtg, participating)) & chosen)

    def test_it_still_upgrades_more_than_one_turbine(self) -> None:
        for seed in range(25):
            assert len(instance(seed=seed).upgraded_turbines) >= 2


class TestARandomisedInstance:
    def test_the_same_seed_gives_the_same_campaign(self) -> None:
        first, second = instance(seed=7), instance(seed=7)
        assert first.upgraded_turbines == second.upgraded_turbines
        assert first.analysis_period == second.analysis_period

    def test_different_seeds_give_different_campaigns(self) -> None:
        drawn = {
            (
                tuple(instance(seed=s).upgraded_turbines),
                instance(seed=s).analysis_period,
            )
            for s in range(12)
        }
        assert len(drawn) > 1

    def test_it_does_not_hand_back_the_checked_in_default(self) -> None:
        # a populated YAML matching the defaults would identify the campaign on sight
        matches = [
            s
            for s in range(12)
            if tuple(instance(seed=s).upgraded_turbines) == tuple(PLACEBO_UPGRADED)
            and instance(seed=s).upgrade_timing == PLACEBO_CAMPAIGN_START
        ]
        assert not matches

    def test_it_upgrades_several_turbines_of_the_farm(self) -> None:
        campaign = instance(seed=3)
        assert 1 < len(campaign.upgraded_turbines) < len(PLACEBO_TURBINES)
        assert set(campaign.upgraded_turbines) <= set(PLACEBO_TURBINES)

    def test_references_always_outnumber_the_upgraded_turbines(self) -> None:
        # reference count is the biggest lever on accuracy, and the screen needs a pool to judge
        for seed in range(20):
            for farm in (PLACEBO_TURBINES, PLACEBO_TURBINES[:9]):
                campaign = instance(seed=seed, turbines=farm)
                assert len(campaign.candidate_references) > len(campaign.upgraded_turbines)

    def test_a_small_farm_still_leaves_a_pool_the_screen_can_judge(self) -> None:
        # a screen with fewer than three references cannot form a majority and stops
        for seed in range(20):
            campaign = instance(seed=seed, turbines=PLACEBO_TURBINES[:9])
            assert len(campaign.candidate_references) >= MIN_SCREENABLE_REFERENCES + 1

    def test_every_other_turbine_is_offered_as_a_reference(self) -> None:
        campaign = instance(seed=3)
        assert set(campaign.candidate_references) == set(PLACEBO_TURBINES) - set(campaign.upgraded_turbines)

    def test_the_site_s_known_bad_turbine_is_never_a_test_turbine(self) -> None:
        for seed in range(20):
            assert set(instance(seed=seed).upgraded_turbines).isdisjoint(PLACEBO_INSTANCE_KEEP_AS_REFERENCE)

    def test_the_window_stays_clear_of_the_site_s_real_upgrades(self) -> None:
        # a real install inside the window would put a genuine change in a campaign whose truth is 0
        for seed in range(20):
            for mode in ("prepost", "toggle"):
                start, end = instance(mode, seed=seed).analysis_period
                assert start >= pd.Timestamp("2017-01-01", tz="UTC")
                assert end <= PLACEBO_INSTANCE_LAST_CLEAN

    def test_the_prepost_window_avoids_the_thin_baseline_year(self) -> None:
        # a treated period reaching into 2020 rests on a 2019-only baseline, which reads far worse
        for seed in range(30):
            _, end = instance(seed=seed).analysis_period
            assert end <= PLACEBO_INSTANCE_LAST_GOOD_END

    def test_prepost_keeps_a_full_year_each_side(self) -> None:
        start, end = instance(seed=5).analysis_period
        changeover = instance(seed=5).upgrade_timing
        assert changeover - start == pd.Timedelta(days=365) or (changeover - start).days in (365, 366)
        assert (end - changeover).days in (365, 366)

    def test_toggle_instances_declare_a_schedule(self) -> None:
        assert isinstance(instance("toggle", seed=5).upgrade_timing, ToggleSchedule)

    def test_it_injects_nothing(self) -> None:
        assert instance(seed=5).upgrades == []
        assert instance(seed=5).faults == []
