"""Tests for the R3 reference fixture's declaration and its impact arithmetic.

The fixture's actual runs are a driver (they need the Hill of Towie download and the power
model); what is unit-tested here is that each arm is declared as intended and that the impact
tables measure movement against the right clean cell.
"""

from __future__ import annotations

import pandas as pd
import pytest

from benchmarking.campaigns.placebo import PLACEBO_CAMPAIGN_START
from benchmarking.campaigns.reference_fixture import (
    BAD_REFERENCE,
    DOWN_DELTA,
    FIXTURE_TEST_WTG,
    REFERENCES_3,
    REFERENCES_5,
    SECOND_BAD_REFERENCE,
    UP_DELTA,
    analysis_period,
    change_time,
    conditional_impact_table,
    fixture_arms,
    fixture_campaign,
    impact_table,
    references,
)

MODES = ["prepost", "toggle"]


class TestFixtureArms:
    def test_the_matrix_is_five_arms(self) -> None:
        """Three on the 3-reference pool (clean, +3%, -3%) and two on the 5-reference pool."""
        assert len(fixture_arms("prepost")) == 5

    def test_arm_names_are_unique(self) -> None:
        names = [a.name for a in fixture_arms("prepost")]
        assert len(names) == len(set(names))

    def test_each_pool_size_has_exactly_one_clean_arm(self) -> None:
        """A 3-reference and a 5-reference estimate differ in their own right, so each needs a baseline."""
        for pool in (3, 5):
            clean = [a for a in fixture_arms("prepost") if a.pool == pool and not a.changes]
            assert len(clean) == 1, pool

    def test_the_three_reference_pool_is_the_r2_set(self) -> None:
        assert REFERENCES_3 == ("T15", "T10", "T08")

    def test_the_five_reference_pool_extends_the_three(self) -> None:
        assert REFERENCES_5[: len(REFERENCES_3)] == REFERENCES_3
        assert len(REFERENCES_5) == 5

    def test_t05_is_in_neither_pool(self) -> None:
        """T05 is T06's nearest neighbour but carries real northing steps over 2017-2018."""
        assert "T05" not in REFERENCES_5

    def test_both_signs_are_measured_on_the_three_reference_pool(self) -> None:
        deltas = {c.delta for a in fixture_arms("prepost") if a.pool == 3 for c in a.changes}
        assert deltas == {UP_DELTA, DOWN_DELTA}

    def test_the_degradation_is_the_realistic_case_so_the_larger_pool_uses_it(self) -> None:
        """A reference is far likelier to pick up a problem of its own than an unannounced upgrade."""
        assert DOWN_DELTA < 0 < UP_DELTA
        deltas = {c.delta for a in fixture_arms("prepost") if a.pool == 5 for c in a.changes}
        assert deltas == {DOWN_DELTA}

    def test_the_five_reference_pool_carries_two_bad_references(self) -> None:
        """Two of five keeps the good references in the majority, which is what the fix relies on."""
        arm = next(a for a in fixture_arms("prepost") if a.pool == 5 and a.changes)
        assert {c.turbine for c in arm.changes} == {BAD_REFERENCE, SECOND_BAD_REFERENCE}

    def test_the_bad_reference_is_the_nearest_one(self) -> None:
        """A reference-side change lands on the most influential turbine."""
        assert REFERENCES_3[0] == BAD_REFERENCE

    def test_a_prepost_change_lands_at_the_changeover(self) -> None:
        """The maximally confounded case: the reference changes exactly where the contrast is measured."""
        assert change_time("prepost") == PLACEBO_CAMPAIGN_START
        for arm in fixture_arms("prepost"):
            for change in arm.changes:
                assert change.at == PLACEBO_CAMPAIGN_START, arm.name

    def test_a_toggle_change_lands_inside_the_test_period(self) -> None:
        """A change predating a toggle test is common-mode across its blocks; only a mid-test one bites."""
        _, end = analysis_period("toggle")
        at = change_time("toggle")
        assert PLACEBO_CAMPAIGN_START < at < end
        for arm in fixture_arms("toggle"):
            for change in arm.changes:
                assert change.at == at, arm.name

    def test_no_change_targets_the_test_turbine(self) -> None:
        for arm in fixture_arms("prepost"):
            for change in arm.changes:
                assert change.turbine != FIXTURE_TEST_WTG, arm.name

    def test_every_change_targets_a_reference_of_its_own_pool(self) -> None:
        for arm in fixture_arms("prepost"):
            for change in arm.changes:
                assert change.turbine in references(arm.pool), arm.name


class TestReferences:
    def test_a_pool_size_returns_that_many_references(self) -> None:
        assert len(references(3)) == 3
        assert len(references(5)) == 5

    def test_an_unknown_pool_size_raises(self) -> None:
        with pytest.raises(ValueError, match="pool size"):
            references(4)


class TestFixtureCampaign:
    @pytest.mark.parametrize("mode", MODES)
    def test_the_test_turbine_is_never_a_reference(self, mode: str) -> None:
        campaign = fixture_campaign(mode, arm=fixture_arms("prepost")[0])
        assert campaign.upgraded_turbines == [FIXTURE_TEST_WTG]
        assert FIXTURE_TEST_WTG not in campaign.candidate_references

    @pytest.mark.parametrize("mode", MODES)
    def test_the_pool_size_sets_the_candidate_references(self, mode: str) -> None:
        for pool in (3, 5):
            arm = next(a for a in fixture_arms("prepost") if a.pool == pool)
            campaign = fixture_campaign(mode, arm=arm)
            assert tuple(campaign.candidate_references) == references(pool), pool

    @pytest.mark.parametrize("mode", MODES)
    def test_no_upgrade_is_injected_so_the_truth_is_exactly_zero(self, mode: str) -> None:
        """A placebo bed: any movement in the estimate is unambiguously the change's doing."""
        for arm in fixture_arms("prepost"):
            assert fixture_campaign(mode, arm=arm).upgrades == [], arm.name

    @pytest.mark.parametrize("mode", MODES)
    def test_the_arms_changes_reach_the_campaign_as_faults(self, mode: str) -> None:
        arm = next(a for a in fixture_arms("prepost") if a.changes)
        assert fixture_campaign(mode, arm=arm).faults == list(arm.changes)

    @pytest.mark.parametrize("mode", MODES)
    def test_a_clean_arm_injects_nothing(self, mode: str) -> None:
        arm = next(a for a in fixture_arms("prepost") if not a.changes)
        assert fixture_campaign(mode, arm=arm).faults == []

    def test_the_change_never_reaches_the_public_spec(self) -> None:
        arm = next(a for a in fixture_arms("prepost") if a.changes)
        spec = fixture_campaign("prepost", arm=arm).spec()
        assert not hasattr(spec, "faults")

    def test_northing_is_discovered_rather_than_supplied(self) -> None:
        campaign = fixture_campaign("prepost", arm=fixture_arms("prepost")[0])
        assert campaign.north_offsets is None

    def test_an_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            fixture_campaign("sideways", arm=fixture_arms("prepost")[0])

    def test_an_unknown_mode_has_no_change_time(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            change_time("sideways")  # type: ignore[arg-type]


def _rows(estimates: dict[tuple[str, int], float]) -> pd.DataFrame:
    """One mode, one method, from ``{(arm, pool): estimate_fraction}``."""
    return pd.DataFrame(
        [
            {
                "mode": "prepost",
                "method": "power_model",
                "arm": arm,
                "pool": pool,
                "faulted": not arm.endswith("clean"),
                "estimate": estimate,
                "truth": 0.0,
                "signed_error": estimate,
            }
            for (arm, pool), estimate in estimates.items()
        ]
    )


class TestImpactTable:
    def test_movement_is_measured_against_the_clean_cell(self) -> None:
        impact = impact_table(_rows({("3ref_clean", 3): 0.003, ("3ref_T15_up3", 3): -0.009}))
        row = impact.iloc[0]
        assert row["arm"] == "3ref_T15_up3"
        assert row["moved_pp"] == pytest.approx(-1.2)

    def test_the_clean_cell_is_not_itself_a_row(self) -> None:
        impact = impact_table(_rows({("3ref_clean", 3): 0.003, ("3ref_T15_up3", 3): -0.009}))
        assert "3ref_clean" not in set(impact["arm"])

    def test_a_five_reference_arm_is_compared_with_the_five_reference_clean_cell(self) -> None:
        """A 3-reference and a 5-reference estimate differ in their own right, change or no change."""
        impact = impact_table(
            _rows(
                {
                    ("3ref_clean", 3): 0.003,
                    ("5ref_clean", 5): 0.008,
                    ("5ref_T15_T10_up3", 5): 0.002,
                }
            )
        )
        five = impact[impact["pool"] == 5].iloc[0]
        assert five["moved_pp"] == pytest.approx(-0.6)

    def test_movement_keeps_its_sign(self) -> None:
        """An improving reference drags the estimate down; a degrading one lifts it."""
        impact = impact_table(_rows({("3ref_clean", 3): 0.003, ("3ref_T15_down3", 3): 0.015}))
        assert impact.iloc[0]["moved_pp"] == pytest.approx(1.2)

    def test_movement_below_the_threshold_is_flagged_immaterial(self) -> None:
        impact = impact_table(_rows({("3ref_clean", 3): 0.003, ("3ref_T15_up3", 3): 0.004}))
        assert not bool(impact.iloc[0]["material"])

    def test_movement_above_the_threshold_is_flagged_material(self) -> None:
        impact = impact_table(_rows({("3ref_clean", 3): 0.003, ("3ref_T15_up3", 3): -0.009}))
        assert bool(impact.iloc[0]["material"])

    def test_a_group_with_no_clean_cell_is_skipped_rather_than_half_judged(self) -> None:
        assert impact_table(_rows({("3ref_T15_up3", 3): 0.009})).empty

    def test_an_empty_table_is_returned_unchanged(self) -> None:
        assert impact_table(pd.DataFrame()).empty


def _conditional_rows(estimates: dict[tuple[str, str], float]) -> pd.DataFrame:
    """One mode/method on the 3-reference pool, from ``{(arm, condition_bin): p50_uplift}``."""
    return pd.DataFrame(
        [
            {
                "mode": "prepost",
                "method": "power_model",
                "arm": arm,
                "pool": 3,
                "faulted": not arm.endswith("clean"),
                "condition": "ws",
                "condition_bin": condition_bin,
                "p50_uplift": value,
            }
            for (arm, condition_bin), value in estimates.items()
        ]
    )


class TestConditionalImpactTable:
    def test_each_bin_moves_against_its_own_clean_value(self) -> None:
        impact = conditional_impact_table(
            _conditional_rows(
                {
                    ("3ref_clean", "4-6"): 0.002,
                    ("3ref_clean", "8-10"): 0.005,
                    ("3ref_T15_up3", "4-6"): -0.008,
                    ("3ref_T15_up3", "8-10"): 0.006,
                }
            )
        )
        moved = impact.set_index("condition_bin")["moved_pp"]
        assert moved["4-6"] == pytest.approx(-1.0)
        assert moved["8-10"] == pytest.approx(0.1)

    def test_a_bin_the_clean_run_never_produced_is_dropped(self) -> None:
        impact = conditional_impact_table(
            _conditional_rows({("3ref_clean", "4-6"): 0.002, ("3ref_T15_up3", "20-22"): 0.050})
        )
        assert impact.empty
