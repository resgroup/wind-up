"""Tests for resolving an analysis plan per upgraded turbine from a campaign and its SCADA."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd
import pytest  # noqa: TC002 - caplog fixtures are runtime types

from benchmarking.campaigns.plans import data_extents, plans_for
from benchmarking.synthetic import HOT_COLUMNS
from tests.benchmarking.campaigns.timeline_fixtures import DATA_END, hourly_scada, staggered_spec, utc
from wind_up.analysis_period import PlanSettings


def plans(spec=None, scada=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
    return plans_for(
        staggered_spec() if spec is None else spec,
        hourly_scada() if scada is None else scada,
        columns=HOT_COLUMNS,
        **kwargs,
    )


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
