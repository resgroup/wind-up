"""Tests for the campaign context: the per-test-turbine facts a method may consult."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from benchmarking.harness.context import CampaignContext
from benchmarking.synthetic import ToggleSchedule

_TURBINE_COL = "TurbineName"
_INDEX = pd.date_range("2020-01-01", periods=4, freq="10min", tz="UTC")
_CHANGEOVER = pd.Timestamp("2020-01-01 00:20", tz="UTC")


def _scada(turbines: tuple[str, ...] = ("T1", "T2", "T3")) -> pd.DataFrame:
    frames = [pd.DataFrame({_TURBINE_COL: t, "ActivePowerMean": 1.0}, index=_INDEX) for t in turbines]
    return pd.concat(frames).sort_index()


def _context(**overrides: object) -> CampaignContext:
    kwargs: dict = {
        "scada_df": _scada(),
        "test_wtg": "T1",
        "timing": _CHANGEOVER,
        "turbine_col": _TURBINE_COL,
    }
    kwargs.update(overrides)
    return CampaignContext.from_frame(**kwargs)


class TestFromFrame:
    def test_candidate_references_are_every_other_turbine_in_the_frame(self) -> None:
        assert _context().candidate_references == ["T2", "T3"]

    def test_valid_for_uplift_is_all_true_over_the_frames_timestamps(self) -> None:
        valid = _context().valid_for_uplift
        assert list(valid.columns) == ["T1", "T2", "T3"]
        assert valid.index.equals(_INDEX)
        assert valid.to_numpy().all()

    def test_a_turbine_absent_from_the_frame_is_not_a_candidate(self) -> None:
        assert _context(scada_df=_scada(("T1", "T2"))).candidate_references == ["T2"]

    def test_there_are_no_wake_contributors(self) -> None:
        # Every other turbine is a reference, so none is left to contribute its wake alone.
        assert _context().wake_contributors == []


def _with_wake_contributor() -> CampaignContext:
    """T1 under test, T2 its only reference, T3 another changed turbine that wakes them."""
    context = _context()
    return dataclasses.replace(context, candidate_references=["T2"], wake_contributors=["T3"])


class TestWakeContributors:
    def test_select_keeps_their_rows(self) -> None:
        selected = _with_wake_contributor().select(_scada())
        assert sorted(selected[_TURBINE_COL].unique()) == ["T1", "T2", "T3"]

    def test_select_drops_the_rows_they_may_not_contribute(self) -> None:
        context = _with_wake_contributor()
        valid = context.valid_for_uplift.copy()
        valid.loc[_INDEX[:2], "T3"] = False
        selected = dataclasses.replace(context, valid_for_uplift=valid).select(_scada())
        assert selected[selected[_TURBINE_COL] == "T3"].index.equals(_INDEX[2:])

    def test_they_are_never_references(self) -> None:
        assert _with_wake_contributor().references_among(["T1", "T2", "T3"]) == ["T2"]

    def test_a_candidate_reference_cannot_also_be_one(self) -> None:
        with pytest.raises(ValueError, match="T2"):
            dataclasses.replace(_context(), wake_contributors=["T2"])

    def test_the_test_turbine_cannot_be_one(self) -> None:
        with pytest.raises(ValueError, match="T1"):
            dataclasses.replace(_context(), candidate_references=["T2"], wake_contributors=["T1"])


class TestMode:
    def test_a_timestamp_is_prepost(self) -> None:
        assert _context().mode == "prepost"

    def test_a_schedule_is_toggle(self) -> None:
        assert _context(timing=ToggleSchedule(period=pd.Timedelta(minutes=20))).mode == "toggle"

    def test_an_explicit_toggle_frame_is_toggle(self) -> None:
        toggle_df = pd.DataFrame({"toggle_on": True, "toggle_off": False}, index=_INDEX)
        assert _context(timing=toggle_df).mode == "toggle"


class TestValidOver:
    def test_narrows_to_the_requested_timestamps(self) -> None:
        narrowed = _context().valid_over(_INDEX[1:3])
        assert narrowed.index.equals(_INDEX[1:3])
        assert list(narrowed.columns) == ["T1", "T2", "T3"]

    def test_raises_on_a_timestamp_the_context_does_not_cover(self) -> None:
        outside = pd.DatetimeIndex([pd.Timestamp("2021-06-01", tz="UTC")])
        with pytest.raises(ValueError, match="2021-06-01"):
            _context().valid_over(outside)


class TestReferencesAmong:
    def test_keeps_the_callers_column_order(self) -> None:
        # Methods keep their own reference ordering; the context supplies membership only.
        assert _context().references_among(["T3", "T2", "T1"]) == ["T3", "T2"]

    def test_drops_a_column_the_campaign_does_not_offer(self) -> None:
        context = _context()
        object.__setattr__(context, "candidate_references", ["T2"])
        assert context.references_among(["T2", "T3"]) == ["T2"]

    def test_drops_a_candidate_absent_from_the_columns(self) -> None:
        assert _context().references_among(["T2"]) == ["T2"]


class TestMaskInvalid:
    def test_is_a_no_op_when_everything_is_valid(self) -> None:
        wide = pd.DataFrame(1.0, index=_INDEX, columns=["T1", "T2", "T3"])
        pd.testing.assert_frame_equal(_context().mask_invalid(wide), wide)

    def test_nans_the_cells_a_turbine_may_not_contribute(self) -> None:
        context = _context()
        valid = context.valid_for_uplift.copy()
        valid.loc[_INDEX[:2], "T2"] = False
        object.__setattr__(context, "valid_for_uplift", valid)
        masked = context.mask_invalid(pd.DataFrame(1.0, index=_INDEX, columns=["T1", "T2", "T3"]))
        assert masked["T2"].isna().tolist() == [True, True, False, False]
        assert masked["T1"].notna().all()
        assert masked["T3"].notna().all()


class TestSelect:
    def test_is_a_no_op_when_everything_is_offered_and_valid(self) -> None:
        scada = _scada()
        pd.testing.assert_frame_equal(_context().select(scada), scada)

    def test_drops_a_turbine_the_campaign_does_not_offer(self) -> None:
        context = _context()
        object.__setattr__(context, "candidate_references", ["T2"])
        selected = context.select(_scada())
        assert sorted(selected[_TURBINE_COL].unique()) == ["T1", "T2"]

    def test_drops_the_rows_a_turbine_may_not_contribute(self) -> None:
        context = _context()
        valid = context.valid_for_uplift.copy()
        valid.loc[_INDEX[:2], "T2"] = False
        object.__setattr__(context, "valid_for_uplift", valid)
        selected = context.select(_scada())
        t2 = selected[selected[_TURBINE_COL] == "T2"]
        assert t2.index.equals(_INDEX[2:])
        assert len(selected[selected[_TURBINE_COL] == "T3"]) == len(_INDEX)

    def test_also_keeps_named_turbines_that_are_not_candidates(self) -> None:
        # A method co-analysing several test turbines keeps them all, though only one is `test_wtg`.
        context = _context()
        object.__setattr__(context, "candidate_references", ["T2"])
        selected = context.select(_scada(), also=["T3"])
        assert sorted(selected[_TURBINE_COL].unique()) == ["T1", "T2", "T3"]

    def test_raises_for_an_also_turbine_present_but_not_covered_by_validity(self) -> None:
        # Silently keeping every row of an uncovered turbine would bypass declared validity.
        context = _context()
        object.__setattr__(context, "candidate_references", ["T2"])
        object.__setattr__(context, "valid_for_uplift", context.valid_for_uplift[["T1", "T2"]])
        with pytest.raises(ValueError, match="T3"):
            context.select(_scada(), also=["T3"])
