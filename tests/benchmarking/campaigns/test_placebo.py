"""Tests for the two declared placebo campaigns."""

from __future__ import annotations

import functools

import pandas as pd
import pytest

from benchmarking.campaigns import CampaignRunner, per_turbine_table
from benchmarking.campaigns.placebo import (
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
    placebo_layout,
)
from benchmarking.harness import MethodInput, MethodOutput
from benchmarking.synthetic import HOT_COLUMNS, SensorGainStep, ToggleSchedule
from tests.conftest import TEST_DATA_FLD
from wind_up.campaign_design import check_design, design_campaign

TOLERANCE = 1e-9
# A small slice of the farm, so the fixtures stay cheap; the production default is all 21 turbines.
TEST_TURBINES = ("T07", "T11")
# A line of turbines 400 m apart (about 5 rotor diameters).
LINE_COORDS = {f"T{i:02d}": (57.5 + i * 0.0036, -3.25) for i in range(1, 22)}
# The real Hill of Towie layout.
HOT_METADATA = pd.read_csv(TEST_DATA_FLD / "hot" / "scada" / "Hill_of_Towie_turbine_metadata.csv")
HOT_COORDS = {str(r["Turbine Name"]): (float(r["Latitude"]), float(r["Longitude"])) for _, r in HOT_METADATA.iterrows()}
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


@functools.cache
def instance(mode: str = "prepost", *, seed: int, turbines: tuple[str, ...] | None = None):  # noqa: ANN201
    """A randomised instance on the straight-line fixture geometry.

    Cached: the design solve behind an instance is the expensive part, and the tests below read the
    same seeds repeatedly. Treat the result as read-only.
    """
    names = turbines if turbines is not None else tuple(LINE_COORDS)
    return placebo_instance(mode, seed=seed, turbines=names, coords={w: LINE_COORDS[w] for w in names})


@functools.cache
def hot_instance(mode: str = "prepost", *, seed: int):  # noqa: ANN201
    """The same, on the real Hill of Towie layout. Read-only, as above."""
    return placebo_instance(mode, seed=seed, coords=HOT_COORDS)


def complies(campaign, coords: dict[str, tuple[float, float]]) -> bool:  # noqa: ANN001
    """Whether ``campaign``'s test turbines pass the campaign-design check on ``coords``."""
    names = [*campaign.upgraded_turbines, *campaign.candidate_references]
    report = check_design(
        placebo_layout({w: coords[w] for w in names}),
        test_turbines=campaign.upgraded_turbines,
        reference_only=[w for w in PLACEBO_INSTANCE_KEEP_AS_REFERENCE if w in names],
    )
    return report.compliant


class TestTheInstanceIsACompliantCampaignDesign:
    def test_every_instance_complies(self) -> None:
        for seed in range(10):
            assert complies(instance(seed=seed), LINE_COORDS), f"seed {seed}"

    def test_every_instance_on_the_real_layout_complies(self) -> None:
        for seed in range(10):
            campaign = hot_instance(seed=seed)
            assert complies(campaign, HOT_COORDS), f"seed {seed}"

    def test_every_instance_tests_one_turbine_fewer_than_a_compliant_design_allows(self) -> None:
        # at the maximum the real layout has only two compliant designs; one fewer has sixty
        most = design_campaign(
            placebo_layout(HOT_COORDS), reference_only=PLACEBO_INSTANCE_KEEP_AS_REFERENCE
        ).max_test_turbines
        for seed in range(5):
            assert len(hot_instance(seed=seed).upgraded_turbines) == most - 1

    def test_instances_on_the_real_layout_vary_their_test_turbines(self) -> None:
        drawn = {tuple(hot_instance(seed=s).upgraded_turbines) for s in range(10)}
        assert len(drawn) >= 5

    def test_a_farm_that_supports_one_test_turbine_still_gets_one(self) -> None:
        cluster = ("T01", "T02", "T03", "T05")
        campaign = placebo_instance("prepost", seed=0, coords=HOT_COORDS, turbines=cluster)
        assert len(campaign.upgraded_turbines) == 1

    def test_the_clustered_draw_of_the_first_dry_runs_does_not_comply(self) -> None:
        # T02/T04/T05 a mutual triangle and T13/T14 an adjacent pair: what an unconstrained draw gave
        report = check_design(placebo_layout(HOT_COORDS), test_turbines=["T02", "T04", "T05", "T13", "T14"])
        assert not report.compliant
        assert any(p.startswith("T05") for p in report.problems)
        assert any("T13" in p and "T14" in p for p in report.problems)

    def test_it_still_upgrades_more_than_one_turbine(self) -> None:
        for seed in range(10):
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
            assert len(campaign.candidate_references) >= 4

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
