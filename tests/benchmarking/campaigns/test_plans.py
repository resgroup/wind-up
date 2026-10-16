"""Tests for resolving an analysis plan per upgraded turbine from a campaign and its SCADA."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd
import pytest  # noqa: TC002 - caplog fixtures are runtime types

from benchmarking.campaigns.plans import changed_works, data_extents, extent_around_works, other_works, plans_for
from benchmarking.synthetic import HOT_COLUMNS
from tests.benchmarking.campaigns.timeline_fixtures import DATA_END, hourly_scada, staggered_spec, utc
from wind_up.analysis_period import PlanSettings


def resolve(spec=None, scada=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
    return plans_for(
        staggered_spec() if spec is None else spec,
        hourly_scada() if scada is None else scada,
        columns=HOT_COLUMNS,
        **kwargs,
    )


def plans(spec=None, scada=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
    return resolve(spec, scada, **kwargs).plans


def without_pre(scada: pd.DataFrame, turbine: str) -> pd.DataFrame:
    """Blank ``turbine``'s power until a week before its works, so no span gives it a pre side."""
    works_start = staggered_spec().works[turbine][0][0]
    blank = (scada[HOT_COLUMNS.turbine] == turbine) & (scada.index < works_start - pd.Timedelta(days=7))
    scada.loc[blank, HOT_COLUMNS.active_power] = np.nan
    return scada


def test_extents_run_from_the_first_to_after_the_last_finite_power() -> None:
    scada = hourly_scada(["T0", "T3"])
    late = (scada[HOT_COLUMNS.turbine] == "T3") & (scada.index < utc("2019-01-01"))
    scada.loc[late, HOT_COLUMNS.active_power] = np.nan
    extents = data_extents(scada, turbine_col=HOT_COLUMNS.turbine, power_col=HOT_COLUMNS.active_power)
    assert extents["T3"] == (utc("2019-01-01"), DATA_END)
    assert extents["T0"][0] == utc("2018-01-01")


def test_every_upgraded_turbine_gets_a_plan() -> None:
    resolved = plans()
    assert set(resolved) == {"T0", "T6"}
    assert resolved["T0"].works == (utc("2019-06-01"), utc("2019-06-06"))


def test_a_shared_changeover_is_a_works_window_of_no_length() -> None:
    change = utc("2019-06-06")
    spec = staggered_spec(upgrade_timing=change, works={})
    assert plans(spec)["T0"].works == (change, change)


def test_defaulted_references_include_the_other_analysed_turbines() -> None:
    # T6 is worked after T0's span would end, so it is eligible for T0 when references were not declared
    spec = staggered_spec(
        works={"T0": [(utc("2019-06-01"), utc("2019-06-06"))], "T6": [(utc("2020-09-01"), utc("2020-09-04"))]}
    )
    resolved = plans(spec, settings=PlanSettings(k=6, side_cap=pd.Timedelta(days=150)))
    assert "T6" in resolved["T0"].power_references


def test_declared_references_restrict_the_candidates() -> None:
    spec = staggered_spec(candidate_references=["T2", "T3", "T4", "T5"], references_declared=True)
    assert plans(spec)["T0"].waking_only["T1"] == "not offered as a reference"
    assert "T6" not in plans(spec)["T0"].power_references


def test_excluded_turbines_are_never_power_references() -> None:
    spec = staggered_spec(excluded_turbines=["T1"], candidate_references=["T1", "T2", "T3", "T4", "T5"])
    assert "T1" not in plans(spec)["T0"].power_references


def test_declared_references_are_forced_in_only_under_a_declared_span(caplog: pytest.LogCaptureFixture) -> None:
    works = {**staggered_spec().works, "T1": [(utc("2019-10-01"), utc("2019-10-03"))]}
    span = (utc("2018-06-01"), utc("2020-06-01"))
    declared = staggered_spec(works=works, candidate_references=["T1", "T2", "T3", "T4"], references_declared=True)
    with caplog.at_level(logging.WARNING):
        forced = plans(dataclasses.replace(declared, analysis_period={"T0": span}))["T0"]
    assert forced.forced == ("T1",)
    assert "T1" in caplog.text
    assert plans(declared)["T0"].forced == ()


def test_the_settings_reach_the_selector() -> None:
    assert len(plans(settings=PlanSettings(k=3))["T0"].power_references) == 3


def test_a_turbine_the_selector_cannot_plan_is_reported_and_the_others_planned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        resolved = resolve(scada=without_pre(hourly_scada(), "T6"))
    assert set(resolved.plans) == {"T0"}
    assert set(resolved.unplanned) == {"T6"}
    assert resolved.unplanned["T6"].startswith("T6: no span gives both sides at least 3 months")
    assert "T6" in caplog.text


def test_a_campaign_whose_turbines_all_plan_reports_none_unplanned() -> None:
    assert resolve().unplanned == {}


def test_a_toggle_campaign_is_planned_from_the_start_of_toggling() -> None:
    from benchmarking.synthetic import ToggleSchedule  # noqa: PLC0415

    start = utc("2019-06-01")
    schedule = ToggleSchedule(period=pd.Timedelta(hours=8), start=start)
    span = (utc("2019-01-01"), utc("2020-01-01"))
    spec = staggered_spec(upgrade_timing=schedule, works={}, analysis_period=span)
    plan = plans(spec)["T0"]
    assert plan.works == (start, start)
    assert plan.before == (span[0], start)
    assert plan.after == (start, span[1])
    assert plan.declared


def test_a_declared_span_with_a_shared_changeover_is_used_as_given() -> None:
    change = utc("2019-06-06")
    span = (utc("2018-06-01"), utc("2020-01-01"))
    plan = plans(staggered_spec(upgrade_timing=change, works={}, analysis_period=span))["T0"]
    assert plan.declared
    assert (plan.start, plan.end) == span
    assert len(plan.power_references) == 4
    assert plan.reserves


def test_a_possible_change_inside_the_periods_is_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    plan = plans()["T0"]
    reference = plan.power_references[0]
    inside = plan.start + pd.Timedelta(days=30)
    outside = plan.end + pd.Timedelta(days=30)
    with caplog.at_level(logging.WARNING):
        resolved = resolve(possible_changes={"T0": [inside, outside], reference: [outside, inside], "T6": [inside]})
    assert resolved.warnings["T0"] == {
        "T0": f"possible undeclared change on {inside:%Y-%m-%d}",
        reference: f"possible undeclared change on {inside:%Y-%m-%d}",
    }
    assert "T6" not in resolved.warnings["T0"]
    assert reference in caplog.text


def test_a_possible_change_inside_the_test_turbines_works_is_no_warning() -> None:
    plan = plans()["T0"]
    during_works = plan.works[0] + (plan.works[1] - plan.works[0]) / 2
    assert resolve(possible_changes={"T0": [during_works]}).warnings["T0"] == {}


CHANGEOVER = utc("2019-06-03")
T0_UPGRADE = (utc("2019-06-01"), utc("2019-06-06"))
T0_EARLIER = (utc("2018-09-01"), utc("2018-09-03"))
T0_LATER = (utc("2019-11-01"), utc("2019-11-02"))


def shared(*windows: tuple[pd.Timestamp, pd.Timestamp], **overrides: object) -> object:
    """The line campaign under a shared changeover, T0 declared with ``windows``."""
    return staggered_spec(upgrade_timing=CHANGEOVER, works={"T0": list(windows)}, **overrides)


def test_a_test_turbines_other_works_are_those_beside_its_upgrade() -> None:
    spec = shared(T0_UPGRADE, T0_LATER)
    assert other_works(spec, "T0") == [T0_LATER]
    assert other_works(spec, "T6") == []
    assert other_works(staggered_spec(), "T0") == []


def test_the_extent_is_cut_at_the_nearest_other_works_on_each_side() -> None:
    extent = (utc("2018-01-01"), utc("2021-01-01"))
    others = [T0_EARLIER, T0_LATER, (utc("2018-03-01"), utc("2018-03-02"))]
    assert extent_around_works(extent, upgrade=T0_UPGRADE, others=others) == (T0_EARLIER[1], T0_LATER[0])


def test_neither_period_holds_another_of_the_test_turbines_works() -> None:
    plan = plans(shared(T0_UPGRADE, T0_EARLIER, T0_LATER))["T0"]
    assert plan.start >= T0_EARLIER[1]
    assert plan.end <= T0_LATER[0]


def test_another_change_of_a_test_turbine_does_not_shorten_its_use_as_a_reference() -> None:
    assert resolve(shared(T0_UPGRADE, T0_EARLIER)).extents["T0"][0] == utc("2018-01-01")


def test_other_works_inside_a_declared_span_are_a_warning() -> None:
    span = (utc("2018-06-01"), utc("2020-01-01"))
    resolved = resolve(shared(T0_UPGRADE, T0_EARLIER, analysis_period=span))
    assert (resolved.plans["T0"].start, resolved.plans["T0"].end) == span
    assert "declared works 2018-09-01..2018-09-03 inside the declared span" in resolved.warnings["T0"]["T0"]


def test_the_changed_upwind_rule_sees_every_declared_window_and_each_upgrade() -> None:
    works = changed_works(shared(T0_EARLIER))
    assert works["T0"] == [T0_EARLIER, (CHANGEOVER, CHANGEOVER)]
    assert works["T6"] == [(CHANGEOVER, CHANGEOVER)]
    assert changed_works(staggered_spec()) == staggered_spec().works


def test_with_the_wake_unchanged_the_upgrade_is_left_out() -> None:
    assert changed_works(shared(T0_EARLIER, wake_unchanged=True)) == {"T0": [T0_EARLIER]}
