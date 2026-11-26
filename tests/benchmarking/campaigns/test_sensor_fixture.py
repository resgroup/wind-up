"""Tests for the R2 sensor fixture's declaration and its impact arithmetic.

The fixture's actual runs are a driver (they need the Hill of Towie download and the power
model); what is unit-tested here is that each arm is declared as intended and that the impact
tables measure movement against the right clean cell.
"""

from __future__ import annotations

import pandas as pd
import pytest

from benchmarking.campaigns.sensor_fixture import (
    EXPOSED_STAT_COLS,
    FAULT_REFERENCE,
    FAULT_TARGETS,
    FIXTURE_REFERENCES,
    FIXTURE_TEST_WTG,
    GAINS,
    analysis_period,
    conditional_impact_table,
    fault_time,
    fixture_arms,
    fixture_campaign,
    impact_table,
)
from benchmarking.synthetic import HOT_COLUMNS, SensorGainDrift, SensorGainStep

MODES = ["prepost", "toggle"]


class TestFixtureArms:
    @pytest.mark.parametrize("mode", MODES)
    def test_the_matrix_is_fourteen_arms_per_mode(self, mode: str) -> None:
        """1 clean + 2 shapes x 2 gains x 2 targets, then 1 exposed clean + 2 gains x 2 targets."""
        assert len(fixture_arms(mode)) == 14

    @pytest.mark.parametrize("mode", MODES)
    def test_exactly_one_clean_arm_per_exposure(self, mode: str) -> None:
        arms = fixture_arms(mode)
        assert len([a for a in arms if a.fault is None and not a.exposed]) == 1
        assert len([a for a in arms if a.fault is None and a.exposed]) == 1

    @pytest.mark.parametrize("mode", MODES)
    def test_arm_names_are_unique(self, mode: str) -> None:
        names = [a.name for a in fixture_arms(mode)]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("mode", MODES)
    def test_every_faulted_arm_carries_exactly_one_fault(self, mode: str) -> None:
        for arm in fixture_arms(mode):
            if arm.fault is not None:
                assert arm.fault.turbine in FAULT_TARGETS, arm.name

    @pytest.mark.parametrize("mode", MODES)
    def test_faults_land_on_the_test_turbine_and_the_nearest_reference(self, mode: str) -> None:
        """The two targets probe different paths: conditional axes vs the ERA5 lag sync."""
        targets = {a.fault.turbine for a in fixture_arms(mode) if a.fault is not None}
        assert targets == {FIXTURE_TEST_WTG, FAULT_REFERENCE}

    @pytest.mark.parametrize("mode", MODES)
    def test_both_shapes_are_measured_in_the_default_configuration(self, mode: str) -> None:
        shapes = {type(a.fault) for a in fixture_arms(mode) if a.fault is not None and not a.exposed}
        assert shapes == {SensorGainStep, SensorGainDrift}

    @pytest.mark.parametrize("mode", MODES)
    def test_the_exposed_arm_covers_steps_only(self, mode: str) -> None:
        """The exposed configuration exists to price the feature exclusion, not to be characterised."""
        shapes = {type(a.fault) for a in fixture_arms(mode) if a.fault is not None and a.exposed}
        assert shapes == {SensorGainStep}

    @pytest.mark.parametrize("mode", MODES)
    def test_the_gains_are_the_worst_case_pair(self, mode: str) -> None:
        gains = {a.fault.gain for a in fixture_arms(mode) if a.fault is not None}
        assert gains == set(GAINS)
        assert gains == {1.5, 0.5}

    @pytest.mark.parametrize("mode", MODES)
    def test_every_fault_scales_the_anemometer_channels_together(self, mode: str) -> None:
        for arm in fixture_arms(mode):
            if arm.fault is not None:
                assert arm.fault.roles == ("wind_speed", "wind_speed_sd"), arm.name

    @pytest.mark.parametrize("mode", MODES)
    def test_steps_land_at_the_moment_the_contrast_is_measured_across(self, mode: str) -> None:
        for arm in fixture_arms(mode):
            if isinstance(arm.fault, SensorGainStep):
                assert arm.fault.at == fault_time(mode), arm.name


def test_the_exposed_arm_feeds_exactly_the_anemometer_channels() -> None:
    """The exposed arm has to name the channels the standing rule excludes, or it prices nothing."""
    assert (HOT_COLUMNS.wind_speed, HOT_COLUMNS.wind_speed_sd) == EXPOSED_STAT_COLS


class TestFixtureCampaign:
    @pytest.mark.parametrize("mode", MODES)
    def test_the_test_turbine_is_never_a_reference(self, mode: str) -> None:
        campaign = fixture_campaign(mode, arm=fixture_arms(mode)[0])
        assert campaign.upgraded_turbines == [FIXTURE_TEST_WTG]
        assert set(campaign.candidate_references) == set(FIXTURE_REFERENCES)

    def test_t05_is_not_in_the_fixture(self) -> None:
        """T05 is T06's nearest neighbour but carries real northing steps over 2017-2018."""
        campaign = fixture_campaign("prepost", arm=fixture_arms("prepost")[0])
        assert "T05" not in campaign.turbines

    @pytest.mark.parametrize("mode", MODES)
    def test_no_upgrade_is_injected_so_the_truth_is_exactly_zero(self, mode: str) -> None:
        """A placebo bed: any movement in the estimate is unambiguously the fault's doing."""
        for arm in fixture_arms(mode):
            assert fixture_campaign(mode, arm=arm).upgrades == [], arm.name

    @pytest.mark.parametrize("mode", MODES)
    def test_the_arms_fault_reaches_the_campaign(self, mode: str) -> None:
        arm = next(a for a in fixture_arms(mode) if a.fault is not None)
        assert fixture_campaign(mode, arm=arm).faults == [arm.fault]

    @pytest.mark.parametrize("mode", MODES)
    def test_a_clean_arm_injects_no_fault(self, mode: str) -> None:
        arm = next(a for a in fixture_arms(mode) if a.fault is None)
        assert fixture_campaign(mode, arm=arm).faults == []

    def test_northing_is_discovered_rather_than_supplied(self) -> None:
        campaign = fixture_campaign("prepost", arm=fixture_arms("prepost")[0])
        assert campaign.north_offsets is None

    def test_the_fault_never_reaches_the_public_spec(self) -> None:
        arm = next(a for a in fixture_arms("prepost") if a.fault is not None)
        spec = fixture_campaign("prepost", arm=arm).spec()
        assert not hasattr(spec, "faults")

    def test_prepost_steps_at_the_changeover(self) -> None:
        start, _ = analysis_period("prepost")
        assert fault_time("prepost") > start

    def test_toggle_steps_mid_campaign(self) -> None:
        _, end = analysis_period("toggle")
        assert fault_time("prepost") < fault_time("toggle") < end

    def test_an_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            fixture_arms("sideways")


def _rows(estimates: dict[tuple[str, bool], float]) -> pd.DataFrame:
    """One mode, one method, from ``{(arm, exposed): estimate_fraction}``."""
    return pd.DataFrame(
        [
            {
                "mode": "prepost",
                "method": "power_model",
                "arm": arm,
                "exposed": exposed,
                "faulted": not arm.endswith("clean"),
                "estimate": estimate,
                "truth": 0.0,
                "signed_error": estimate,
            }
            for (arm, exposed), estimate in estimates.items()
        ]
    )


class TestImpactTable:
    def test_movement_is_measured_against_the_clean_cell(self) -> None:
        impact = impact_table(_rows({("clean", False): 0.003, ("step_x1.5_T06", False): 0.009}))
        row = impact.iloc[0]
        assert row["arm"] == "step_x1.5_T06"
        assert row["moved_pp"] == pytest.approx(0.6)

    def test_the_clean_cell_is_not_itself_a_row(self) -> None:
        impact = impact_table(_rows({("clean", False): 0.003, ("step_x1.5_T06", False): 0.009}))
        assert "clean" not in set(impact["arm"])

    def test_movement_keeps_its_sign(self) -> None:
        """A fault that drags the estimate down is as interesting as one that lifts it."""
        impact = impact_table(_rows({("clean", False): 0.003, ("step_x0.5_T06", False): -0.002}))
        assert impact.iloc[0]["moved_pp"] == pytest.approx(-0.5)

    def test_the_exposed_arm_is_compared_with_the_exposed_clean_cell(self) -> None:
        """Otherwise the exposed arm's own configuration bias would be counted as fault movement."""
        impact = impact_table(
            _rows(
                {
                    ("clean", False): 0.003,
                    ("exposed_clean", True): 0.010,
                    ("exposed_step_x1.5_T06", True): 0.013,
                }
            )
        )
        exposed = impact[impact["exposed"]].iloc[0]
        assert exposed["moved_pp"] == pytest.approx(0.3)

    def test_movement_below_the_threshold_is_flagged_immaterial(self) -> None:
        impact = impact_table(_rows({("clean", False): 0.003, ("step_x1.5_T06", False): 0.004}))
        assert not bool(impact.iloc[0]["material"])

    def test_movement_above_the_threshold_is_flagged_material(self) -> None:
        impact = impact_table(_rows({("clean", False): 0.003, ("step_x1.5_T06", False): 0.010}))
        assert bool(impact.iloc[0]["material"])

    def test_a_group_with_no_clean_cell_is_skipped_rather_than_half_judged(self) -> None:
        assert impact_table(_rows({("step_x1.5_T06", False): 0.009})).empty


def _conditional_rows(estimates: dict[tuple[str, str], float]) -> pd.DataFrame:
    """One mode/method, from ``{(arm, condition_bin): p50_uplift}`` on the ws condition."""
    return pd.DataFrame(
        [
            {
                "mode": "prepost",
                "method": "power_model",
                "arm": arm,
                "exposed": False,
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
                    ("clean", "4-6"): 0.002,
                    ("clean", "8-10"): 0.005,
                    ("step_x1.5_T06", "4-6"): 0.012,
                    ("step_x1.5_T06", "8-10"): 0.006,
                }
            )
        )
        moved = impact.set_index("condition_bin")["moved_pp"]
        assert moved["4-6"] == pytest.approx(1.0)
        assert moved["8-10"] == pytest.approx(0.1)

    def test_a_bin_the_clean_run_never_produced_is_dropped(self) -> None:
        """A gain fault re-bins rows, so a faulted run can report bins the clean one did not."""
        impact = conditional_impact_table(
            _conditional_rows({("clean", "4-6"): 0.002, ("step_x1.5_T06", "20-22"): 0.050})
        )
        assert impact.empty
