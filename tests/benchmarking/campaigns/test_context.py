"""Tests for deriving a method-facing context from a campaign declaration and its analysis plan."""

from __future__ import annotations

import dataclasses
import logging

import pandas as pd
import pytest  # noqa: TC002 - caplog fixtures are runtime types

from benchmarking.campaigns.context import context_for_plan
from benchmarking.campaigns.plans import plans_for
from benchmarking.harness.context import CampaignContext
from benchmarking.synthetic import HOT_COLUMNS
from tests.benchmarking.campaigns.timeline_fixtures import hourly_scada, staggered_spec, utc


def test_the_context_carries_only_the_documented_answers() -> None:
    # Guards the truth boundary: a field added here reaches every method, so it must be deliberate.
    # coords is the layout the analyst declares in turbines.csv, not an answer: diagnostics draw
    # it, no estimate reads it. The reserves and reading pools come from the analysis plan, itself
    # derived from the layout, the works table and the data extents.
    assert {f.name for f in dataclasses.fields(CampaignContext)} == {
        "test_wtg",
        "timing",
        "turbine_col",
        "candidate_references",
        "wake_contributors",
        "valid_for_uplift",
        "coords",
        "reserve_references",
        "changed_turbines",
        "works",
    }


class TestAContextFromAPlan:
    @staticmethod
    def build(turbine: str = "T0", scada: pd.DataFrame | None = None, **overrides: object) -> tuple:
        spec = staggered_spec(**overrides)
        frame = hourly_scada() if scada is None else scada
        plan = plans_for(spec, frame, columns=HOT_COLUMNS).plans[turbine]
        return spec, plan, context_for_plan(spec, plan, scada_df=frame)

    def test_the_power_references_are_the_candidates(self) -> None:
        _, plan, context = self.build()
        assert context.candidate_references == list(plan.power_references)
        assert context.timing == plan.works[1]
        assert context.mode == "prepost"

    def test_every_other_present_turbine_contributes_its_wake(self) -> None:
        _, plan, context = self.build()
        assert set(context.wake_contributors) == {f"T{i}" for i in range(7)} - {"T0", *plan.power_references}

    def test_an_excluded_turbine_contributes_its_wake_and_is_never_a_reference(self) -> None:
        _, _, context = self.build(excluded_turbines=["T1"])
        assert "T1" not in context.candidate_references
        assert "T1" not in context.reserve_references
        assert "T1" in context.wake_contributors

    def test_a_turbine_the_campaign_does_not_list_contributes_its_wake(self) -> None:
        _, _, context = self.build(scada=hourly_scada([*[f"T{i}" for i in range(7)], "T9"]))
        assert "T9" in context.wake_contributors
        assert "T9" not in context.candidate_references

    def test_a_change_near_the_changeover_counts_over_the_whole_span(self) -> None:
        _, plan, context = self.build()
        assert plan.start < utc("2019-09-01") < plan.end
        assert context.changed_turbines == {"T0": ((None, None),), "T6": ((None, None),)}

    def test_a_change_far_from_the_changeover_counts_on_its_far_side(self) -> None:
        works = {
            "T0": [(utc("2019-06-01"), utc("2019-06-06"))],
            "T6": [(utc("2019-09-01"), utc("2019-09-04"))],
            "T5": [(utc("2018-03-01"), utc("2018-03-05"))],
        }
        _, plan, context = self.build(works=works)
        assert plan.start < utc("2018-03-01")
        assert context.changed_turbines["T5"] == ((None, utc("2018-03-05")),)

    def test_works_outside_the_span_do_not_make_a_turbine_changed(self) -> None:
        _, _, context = self.build(
            works={"T0": [(utc("2019-06-01"), utc("2019-06-06"))], "T6": [(utc("2017-01-01"), utc("2017-01-02"))]}
        )
        assert context.changed_turbines == {"T0": ((None, None),)}

    def test_the_test_turbines_works_window_is_carried(self) -> None:
        _, plan, context = self.build()
        assert context.works == plan.works
        assert context.works[1] == context.timing

    def test_reserves_are_carried(self) -> None:
        _, plan, context = self.build()
        assert context.reserve_references == list(plan.reserves)

    def test_turbines_without_data_are_dropped_from_every_list(self, caplog: pytest.LogCaptureFixture) -> None:
        spec, plan, _ = self.build()
        missing = plan.power_references[0]
        full = hourly_scada()
        with caplog.at_level(logging.WARNING):
            context = context_for_plan(spec, plan, scada_df=full[full["TurbineName"] != missing])
        assert missing not in context.candidate_references
        assert missing not in context.wake_contributors
        assert missing in caplog.text

    def test_validity_covers_every_turbine_present(self) -> None:
        _, _, context = self.build(scada=hourly_scada([*[f"T{i}" for i in range(7)], "T9"]))
        assert list(context.valid_for_uplift.columns) == [f"T{i}" for i in range(7)] + ["T9"]

    def test_validity_comes_from_the_specs_usable_mask(self) -> None:
        _, _, context = self.build()
        held = context.valid_for_uplift["T4"]
        inside = (held.index >= pd.Timestamp("2019-02-01", tz="UTC")) & (
            held.index < pd.Timestamp("2019-02-05", tz="UTC")
        )
        assert not held[inside].any()
