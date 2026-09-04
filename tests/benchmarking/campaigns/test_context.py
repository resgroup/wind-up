"""Tests for deriving a method-facing context from a campaign declaration."""

from __future__ import annotations

import dataclasses

import pandas as pd

from benchmarking.campaigns.context import context_for
from benchmarking.campaigns.declaration import CampaignSpec
from benchmarking.harness.context import CampaignContext

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
        "coords": {},
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


class TestValidForUplift:
    def test_covers_every_declared_turbine_present_not_just_the_references(self) -> None:
        # A co-analysed turbine (v0's estimate_multi passes other upgraded turbines via
        # select(also=...)) must be covered, or its rows would bypass declared validity.
        valid = context_for(_spec(), turbine="T1", scada_df=_scada()).valid_for_uplift
        assert list(valid.columns) == ["T1", "T2", "T3", "T4"]
        assert valid.index.equals(_INDEX)
        assert valid.to_numpy().all()

    def test_an_excluded_turbine_is_never_valid(self) -> None:
        spec = _spec(candidate_references=["T3", "T4"], excluded_turbines=["T4"])
        valid = context_for(spec, turbine="T1", scada_df=_scada()).valid_for_uplift
        assert not valid["T4"].any()
        assert valid["T3"].all()


class TestTiming:
    def test_is_the_turbines_own_timing_and_drives_mode(self) -> None:
        context = context_for(_spec(), turbine="T1", scada_df=_scada())
        assert context.timing == pd.Timestamp("2020-01-01 00:20", tz="UTC")
        assert context.mode == "prepost"


def test_the_context_carries_only_the_documented_answers() -> None:
    # Guards the truth boundary: a field added here reaches every method, so it must be deliberate.
    assert {f.name for f in dataclasses.fields(CampaignContext)} == {
        "test_wtg",
        "timing",
        "turbine_col",
        "candidate_references",
        "valid_for_uplift",
    }
