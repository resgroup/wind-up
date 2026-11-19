"""Tests for deriving the facts and each plan's analysis from a campaign declaration."""

from __future__ import annotations

import logging

import pandas as pd
import pytest  # noqa: TC002 - caplog fixtures are runtime types

from benchmarking.campaigns.context import UPGRADE, campaign_facts, plan_analysis, upgrade_of
from benchmarking.campaigns.plans import plans_for
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule
from tests.benchmarking.campaigns.timeline_fixtures import hourly_scada, staggered_spec, utc
from wind_up.record_selection import ChangedInterval
from wind_up.subject import subject_records


def _measured(turbine: str = "T0", scada: pd.DataFrame | None = None, **overrides: object) -> tuple:
    spec = staggered_spec(**overrides)
    frame = hourly_scada() if scada is None else scada
    plan = plans_for(spec, frame, columns=HOT_COLUMNS).plans[turbine]
    facts = campaign_facts(spec, frame, columns=HOT_COLUMNS)
    analysis = plan_analysis(spec, plan)
    records = subject_records(
        facts,
        analysis,
        subject=turbine,
        span=analysis.period,  # type: ignore[arg-type]
        n_references=len(plan.power_references),
    )
    return spec, plan, records


class TestTheFacts:
    def test_the_upgraded_turbines_share_one_upgrade(self) -> None:
        facts = campaign_facts(staggered_spec(), hourly_scada(), columns=HOT_COLUMNS)
        upgrade = facts.upgrade(UPGRADE)
        assert list(upgrade.turbines) == ["T0", "T6"]
        assert upgrade.changeover("T0") == (utc("2019-06-01"), utc("2019-06-06"))

    def test_another_works_window_is_an_upgrade_of_its_own(self) -> None:
        works = {
            "T0": [(utc("2019-06-01"), utc("2019-06-06"))],
            "T6": [(utc("2019-09-01"), utc("2019-09-04"))],
            "T5": [(utc("2018-03-01"), utc("2018-03-05"))],
        }
        facts = campaign_facts(staggered_spec(works=works), hourly_scada(), columns=HOT_COLUMNS)
        (other,) = [u for u in facts.upgrades if u.name != UPGRADE]
        assert list(other.turbines) == ["T5"]
        assert other.changeover("T5") == (utc("2018-03-01"), utc("2018-03-05"))

    def test_turbines_with_different_reaches_get_an_upgrade_each(self) -> None:
        window = (utc("2019-06-01"), utc("2019-06-06"))
        spec = staggered_spec(works_reach_d={"T0": {window: 4.0}})
        assert upgrade_of(spec, "T0") != upgrade_of(spec, "T6")
        facts = campaign_facts(spec, hourly_scada(), columns=HOT_COLUMNS)
        assert facts.upgrade(upgrade_of(spec, "T0")).reach_d == 4.0

    def test_a_shared_changeover_is_the_upgrades_date(self) -> None:
        spec = staggered_spec(upgrade_timing=utc("2019-07-01"), works={})
        assert campaign_facts(spec, hourly_scada(), columns=HOT_COLUMNS).upgrade(UPGRADE).changeover("T0") == (
            utc("2019-07-01"),
            utc("2019-07-01"),
        )

    def test_a_toggle_schedule_states_its_cycle(self) -> None:
        schedule = ToggleSchedule(period=pd.Timedelta(hours=2), start=utc("2019-01-01"))
        spec = staggered_spec(upgrade_timing=schedule, works={}, analysis_period=(utc("2018-06-01"), utc("2020-01-01")))
        upgrade = campaign_facts(spec, hourly_scada(), columns=HOT_COLUMNS).upgrade(UPGRADE)
        assert upgrade.cycle == pd.Timedelta(hours=2)

    def test_reanalysis_is_carried_at_the_centroid(self) -> None:
        era5 = pd.DataFrame({"wind_speed_100m": 8.0}, index=pd.date_range("2018-01-01", periods=3, freq="1h", tz="UTC"))
        facts = campaign_facts(staggered_spec(), hourly_scada(), columns=HOT_COLUMNS, era5_hourly_df=era5)
        assert facts.reanalysis[0].records is era5

    def test_records_of_a_turbine_the_layout_does_not_hold_are_left_out(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            facts = campaign_facts(
                staggered_spec(), hourly_scada([*[f"T{i}" for i in range(7)], "T9"]), columns=HOT_COLUMNS
            )
        assert "T9" not in set(facts.records[HOT_COLUMNS.turbine])
        assert "T9" in caplog.text


class TestAPlansMeasurement:
    def test_the_power_references_are_the_candidates_and_the_reserves_wait(self) -> None:
        _, plan, records = _measured()
        assert records.candidate_references == list(plan.power_references)
        assert records.reserve_references == list(plan.reserves)
        assert records.timing == plan.works[1]
        assert records.mode == "prepost"

    def test_every_other_turbine_contributes_its_wake(self) -> None:
        _, plan, records = _measured()
        assert set(records.wake_contributors) == {f"T{i}" for i in range(7)} - {"T0", *plan.power_references}

    def test_an_excluded_turbine_contributes_its_wake_and_is_never_a_reference(self) -> None:
        _, _, records = _measured(excluded_turbines=["T1"])
        assert "T1" not in records.candidate_references
        assert "T1" not in records.reserve_references
        assert "T1" in records.wake_contributors

    def test_a_change_near_the_changeover_counts_over_the_whole_span(self) -> None:
        _, plan, records = _measured()
        assert plan.start < utc("2019-09-01") < plan.end
        assert records.changed_turbines == {"T0": (ChangedInterval(None, None),), "T6": (ChangedInterval(None, None),)}

    def test_a_change_far_from_the_changeover_counts_on_its_far_side(self) -> None:
        works = {
            "T0": [(utc("2019-06-01"), utc("2019-06-06"))],
            "T6": [(utc("2019-09-01"), utc("2019-09-04"))],
            "T5": [(utc("2018-03-01"), utc("2018-03-05"))],
        }
        _, plan, records = _measured(works=works)
        assert plan.start < utc("2018-03-01")
        assert records.changed_turbines["T5"] == (ChangedInterval(None, utc("2018-03-05")),)

    def test_the_subjects_installation_window_is_carried(self) -> None:
        _, plan, records = _measured()
        assert records.works == plan.works

    def test_the_records_are_the_plans_span(self) -> None:
        _, plan, records = _measured()
        assert records.records.index.min() >= plan.start
        assert records.records.index.max() < plan.end

    def test_turbines_without_data_are_dropped_from_every_list(self) -> None:
        spec, plan, _ = _measured()
        missing = plan.power_references[0]
        full = hourly_scada()
        facts = campaign_facts(spec, full[full["TurbineName"] != missing], columns=HOT_COLUMNS)
        analysis = plan_analysis(spec, plan)
        records = subject_records(facts, analysis, subject="T0", span=analysis.period)  # type: ignore[arg-type]
        assert missing not in records.candidate_references
        assert missing not in records.wake_contributors

    def test_validity_masks_works_and_leaves_exclusions_to_the_labels(self) -> None:
        _, _, records = _measured()
        valid = records.valid_for_uplift
        excluded = (valid.index >= utc("2019-02-01")) & (valid.index < utc("2019-02-05"))
        assert valid.loc[excluded, "T4"].all()
        works = (valid.index >= utc("2019-06-01")) & (valid.index < utc("2019-06-06"))
        assert not valid.loc[works, "T0"].any()
