"""Tests for deriving a method-facing context from a campaign declaration."""

from __future__ import annotations

import dataclasses
import logging

import pandas as pd
import pytest  # noqa: TC002 - caplog fixtures are runtime types

from benchmarking.campaigns.context import context_for, context_for_plan
from benchmarking.campaigns.declaration import CampaignSpec, layout_from_coords
from benchmarking.campaigns.plans import plans_for
from benchmarking.harness.context import CampaignContext
from benchmarking.synthetic import HOT_COLUMNS
from tests.benchmarking.campaigns.timeline_fixtures import hourly_scada, staggered_spec

_TURBINE_COL = "TurbineName"
_INDEX = pd.date_range("2020-01-01", periods=4, freq="10min", tz="UTC")
_START = pd.Timestamp("2020-01-01", tz="UTC")


def _scada(turbines: tuple[str, ...] = ("T1", "T2", "T3", "T4")) -> pd.DataFrame:
    frames = [pd.DataFrame({_TURBINE_COL: t, "ActivePowerMean": 1.0}, index=_INDEX) for t in turbines]
    return pd.concat(frames).sort_index()


def _spec(**overrides: object) -> CampaignSpec:
    kwargs: dict = {
        "upgraded_turbines": ["T1", "T2"],
        "upgrade_timing": pd.Timestamp("2020-01-01 00:20", tz="UTC"),
        "candidate_references": ["T3", "T4"],
        "excluded_turbines": [],
        "layout": layout_from_coords({f"T{i}": (57.5 + 0.01 * i, -3.25) for i in range(1, 5)}, rotor_diameter_m=82.0),
        "north_offsets": [],
        "rated_power_kw": 2300.0,
        "analysis_period": (_START, _START + pd.Timedelta(days=1)),
        "turbine_col": _TURBINE_COL,
    }
    kwargs.update(overrides)
    return CampaignSpec(**kwargs)


class TestCandidateReferences:
    def test_come_from_the_declaration_not_the_frame(self) -> None:
        # T2 is upgraded and present in the frame, but the declaration does not offer it.
        context = context_for(_spec(), turbine="T1", scada_df=_scada())
        assert context.candidate_references == ["T3", "T4"]

    def test_a_declared_reference_absent_from_the_frame_is_dropped(self) -> None:
        context = context_for(_spec(candidate_references=["T3", "T4", "T9"]), turbine="T1", scada_df=_scada())
        assert context.candidate_references == ["T3", "T4"]

    def test_dropping_a_declared_reference_is_announced(self, caplog: pytest.LogCaptureFixture) -> None:
        """A quietly smaller pool is a weaker estimate, so the gap between declared and delivered is said."""
        with caplog.at_level(logging.WARNING):
            context_for(_spec(candidate_references=["T3", "T4", "T9"]), turbine="T1", scada_df=_scada())
        assert "T9" in caplog.text

    def test_a_fully_delivered_pool_says_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            context_for(_spec(), turbine="T1", scada_df=_scada())
        assert "carries no rows" not in caplog.text


class TestValidForUplift:
    def test_covers_every_declared_turbine_present_not_just_the_references(self) -> None:
        # A co-analysed turbine (v0's estimate_multi passes other upgraded turbines via
        # select(also=...)) must be covered, or its rows would bypass declared validity.
        valid = context_for(_spec(), turbine="T1", scada_df=_scada()).valid_for_uplift
        assert list(valid.columns) == ["T1", "T2", "T3", "T4"]
        assert valid.index.equals(_INDEX)
        assert valid.to_numpy().all()

    def test_covers_every_turbine_with_data_declared_or_not(self) -> None:
        spec = _spec(excluded_turbines=["T4"])
        valid = context_for(spec, turbine="T1", scada_df=_scada(("T1", "T2", "T3", "T4", "T9"))).valid_for_uplift
        assert list(valid.columns) == ["T1", "T2", "T3", "T4", "T9"]
        assert valid.to_numpy().all()


class TestWakeContributors:
    """Every turbine with data can wake this one, so each that is not a reference stays in its frame."""

    def test_include_the_other_upgraded_turbines(self) -> None:
        assert context_for(_spec(), turbine="T1", scada_df=_scada()).wake_contributors == ["T2"]

    def test_leave_out_the_turbine_being_estimated(self) -> None:
        assert context_for(_spec(), turbine="T2", scada_df=_scada()).wake_contributors == ["T1"]

    def test_leave_out_an_upgraded_turbine_the_frame_has_no_rows_for(self) -> None:
        context = context_for(_spec(), turbine="T1", scada_df=_scada(("T1", "T3", "T4")))
        assert context.wake_contributors == []

    def test_include_an_excluded_turbine_and_never_offer_it_as_a_reference(self) -> None:
        context = context_for(_spec(excluded_turbines=["T4"]), turbine="T1", scada_df=_scada())
        assert context.candidate_references == ["T3"]
        assert context.wake_contributors == ["T2", "T4"]

    def test_include_a_turbine_the_campaign_does_not_list(self) -> None:
        context = context_for(_spec(), turbine="T1", scada_df=_scada(("T1", "T2", "T3", "T4", "T9")))
        assert context.wake_contributors == ["T2", "T9"]

    def test_leave_out_an_upgraded_turbine_the_campaign_also_offers_as_a_reference(self) -> None:
        spec = _spec(upgraded_turbines=["T1", "T2", "T3", "T4"], candidate_references=["T3", "T4"])
        context = context_for(spec, turbine="T1", scada_df=_scada())
        assert context.wake_contributors == ["T2"]
        assert context.candidate_references == ["T3", "T4"]


class TestTiming:
    def test_is_the_turbines_own_timing_and_drives_mode(self) -> None:
        context = context_for(_spec(), turbine="T1", scada_df=_scada())
        assert context.timing == pd.Timestamp("2020-01-01 00:20", tz="UTC")
        assert context.mode == "prepost"


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
        "reading_pools",
        "reading_pool_size",
    }


class TestAContextFromAPlan:
    @staticmethod
    def build(turbine: str = "T0", scada: pd.DataFrame | None = None) -> tuple:
        spec = staggered_spec()
        frame = hourly_scada() if scada is None else scada
        plan = plans_for(spec, frame, columns=HOT_COLUMNS).plans[turbine]
        return spec, plan, context_for_plan(spec, plan, scada_df=frame)

    def test_the_power_references_are_the_candidates(self) -> None:
        _, plan, context = self.build()
        assert context.candidate_references == list(plan.power_references)
        assert context.timing == plan.works[1]

    def test_every_other_present_turbine_contributes_its_wake(self) -> None:
        _, plan, context = self.build()
        assert set(context.wake_contributors) == {f"T{i}" for i in range(7)} - {"T0", *plan.power_references}

    def test_reserves_and_reading_pools_are_carried(self) -> None:
        _, plan, context = self.build()
        assert context.reserve_references == list(plan.reserves)
        assert context.reading_pools == {r: list(p) for r, p in plan.reading_pools.items()}
        assert context.reading_pool_size == plan.k

    def test_turbines_without_data_are_dropped_from_every_list(self) -> None:
        spec, plan, _ = self.build()
        missing = plan.power_references[0]
        full = hourly_scada()
        context = context_for_plan(spec, plan, scada_df=full[full["TurbineName"] != missing])
        assert missing not in context.candidate_references
        assert missing not in context.wake_contributors
        assert all(missing not in pool for pool in (context.reading_pools or {}).values())

    def test_validity_comes_from_the_specs_usable_mask(self) -> None:
        spec, _, context = self.build()
        held = context.valid_for_uplift["T4"]
        inside = (held.index >= pd.Timestamp("2019-02-01", tz="UTC")) & (
            held.index < pd.Timestamp("2019-02-05", tz="UTC")
        )
        assert not held[inside].any()
        assert held[~inside].all()
        assert spec.usable_mask("T0", pd.DatetimeIndex(held.index)).sum() == context.valid_for_uplift["T0"].sum()
