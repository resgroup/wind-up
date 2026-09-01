"""Tests for the two declared placebo campaigns."""

from __future__ import annotations

import pandas as pd
import pytest

from benchmarking.campaigns import CampaignRunner, per_turbine_table
from benchmarking.campaigns.placebo import (
    PLACEBO_CAMPAIGN_START,
    PLACEBO_TEST_CANDIDATES,
    PLACEBO_TURBINES,
    PLACEBO_UPGRADED,
    PLACEBO_WTG_NUMBERS,
    placebo_analysis_period,
    placebo_campaign,
)
from benchmarking.harness import MethodInput, MethodOutput
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule

TOLERANCE = 1e-9
# A small slice of the farm, so the fixtures stay cheap; the production default is all 21 turbines.
TEST_TURBINES = ("T07", "T11")
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
