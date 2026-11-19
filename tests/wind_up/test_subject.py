"""Tests for a subject's records: what one subject's measurement may read."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.synthetic import ToggleSchedule
from tests.wind_up import facts_fixtures as ff
from wind_up.analysis import AnalysisOutput, AnalysisSpec
from wind_up.labels import Label
from wind_up.method import MethodConfigurationError
from wind_up.operating_state import (
    MISSING,
    NORMAL_OPERATION,
    STATE_COL,
    STATE_COLUMNS,
    VALID_NORTHING_COL,
    VALID_UPLIFT_COL,
    WAKING_STATE_COL,
)
from wind_up.subject import (
    SubjectMeasurement,
    SubjectRecords,
    answer_analyses,
    measurement_of,
    subject_records,
    with_states,
)

if TYPE_CHECKING:
    from wind_up.facts import WindFarmFacts

_TURBINE_COL = "TurbineName"
_INDEX = pd.date_range("2020-01-01", periods=4, freq="10min", tz="UTC")
_CHANGEOVER = pd.Timestamp("2020-01-01 00:20", tz="UTC")


def _scada(turbines: tuple[str, ...] = ("T1", "T2", "T3")) -> pd.DataFrame:
    frames = [pd.DataFrame({_TURBINE_COL: t, "ActivePowerMean": 1.0}, index=_INDEX) for t in turbines]
    return pd.concat(frames).sort_index()


def _records(**overrides: object) -> SubjectRecords:
    kwargs: dict = {
        "records": _scada(),
        "subject": "T1",
        "timing": _CHANGEOVER,
        "turbine_col": _TURBINE_COL,
    }
    kwargs.update(overrides)
    return SubjectRecords.from_frame(**kwargs)


class TestFromFrame:
    def test_candidate_references_are_every_other_turbine_in_the_frame(self) -> None:
        assert _records().candidate_references == ["T2", "T3"]

    def test_valid_for_uplift_is_all_true_over_the_frames_timestamps(self) -> None:
        valid = _records().valid_for_uplift
        assert list(valid.columns) == ["T1", "T2", "T3"]
        assert valid.index.equals(_INDEX)
        assert valid.to_numpy().all()

    def test_a_turbine_absent_from_the_frame_is_not_a_candidate(self) -> None:
        assert _records(records=_scada(("T1", "T2"))).candidate_references == ["T2"]

    def test_there_are_no_wake_contributors(self) -> None:
        # Every other turbine is a reference, so none is left to contribute its wake alone.
        assert _records().wake_contributors == []


def _with_wake_contributor() -> SubjectRecords:
    """T1 under test, T2 its only reference, T3 another changed turbine that wakes them."""
    records_ = _records()
    return dataclasses.replace(records_, candidate_references=["T2"], wake_contributors=["T3"])


class TestWakeContributors:
    def test_select_keeps_their_rows(self) -> None:
        selected = _with_wake_contributor().select(_scada())
        assert sorted(selected[_TURBINE_COL].unique()) == ["T1", "T2", "T3"]

    def test_select_drops_the_rows_they_may_not_contribute(self) -> None:
        records_ = _with_wake_contributor()
        valid = records_.valid_for_uplift.copy()
        valid.loc[_INDEX[:2], "T3"] = False
        selected = dataclasses.replace(records_, valid_for_uplift=valid).select(_scada())
        assert selected[selected[_TURBINE_COL] == "T3"].index.equals(_INDEX[2:])

    def test_they_are_never_references(self) -> None:
        assert _with_wake_contributor().references_among(["T1", "T2", "T3"]) == ["T2"]

    def test_a_candidate_reference_cannot_also_be_one(self) -> None:
        with pytest.raises(ValueError, match="T2"):
            dataclasses.replace(_records(), wake_contributors=["T2"])

    def test_the_test_turbine_cannot_be_one(self) -> None:
        with pytest.raises(ValueError, match="T1"):
            dataclasses.replace(_records(), candidate_references=["T2"], wake_contributors=["T1"])


class TestMode:
    def test_a_timestamp_is_prepost(self) -> None:
        assert _records().mode == "prepost"

    def test_a_schedule_is_toggle(self) -> None:
        assert _records(timing=ToggleSchedule(period=pd.Timedelta(minutes=20))).mode == "toggle"

    def test_an_explicit_toggle_frame_is_toggle(self) -> None:
        toggle_df = pd.DataFrame({"toggle_on": True, "toggle_off": False}, index=_INDEX)
        assert _records(timing=toggle_df).mode == "toggle"


class TestValidOver:
    def test_narrows_to_the_requested_timestamps(self) -> None:
        narrowed = _records().valid_over(_INDEX[1:3])
        assert narrowed.index.equals(_INDEX[1:3])
        assert list(narrowed.columns) == ["T1", "T2", "T3"]

    def test_raises_on_a_timestamp_the_records__does_not_cover(self) -> None:
        outside = pd.DatetimeIndex([pd.Timestamp("2021-06-01", tz="UTC")])
        with pytest.raises(ValueError, match="2021-06-01"):
            _records().valid_over(outside)


class TestReferencesAmong:
    def test_keeps_the_callers_column_order(self) -> None:
        # Methods keep their own reference ordering; the records_ supplies membership only.
        assert _records().references_among(["T3", "T2", "T1"]) == ["T3", "T2"]

    def test_drops_a_column_the_campaign_does_not_offer(self) -> None:
        records_ = _records()
        object.__setattr__(records_, "candidate_references", ["T2"])
        assert records_.references_among(["T2", "T3"]) == ["T2"]

    def test_drops_a_candidate_absent_from_the_columns(self) -> None:
        assert _records().references_among(["T2"]) == ["T2"]


class TestMaskInvalid:
    def test_is_a_no_op_when_everything_is_valid(self) -> None:
        wide = pd.DataFrame(1.0, index=_INDEX, columns=["T1", "T2", "T3"])
        pd.testing.assert_frame_equal(_records().mask_invalid(wide), wide)

    def test_nans_the_cells_a_turbine_may_not_contribute(self) -> None:
        records_ = _records()
        valid = records_.valid_for_uplift.copy()
        valid.loc[_INDEX[:2], "T2"] = False
        object.__setattr__(records_, "valid_for_uplift", valid)
        masked = records_.mask_invalid(pd.DataFrame(1.0, index=_INDEX, columns=["T1", "T2", "T3"]))
        assert masked["T2"].isna().tolist() == [True, True, False, False]
        assert masked["T1"].notna().all()
        assert masked["T3"].notna().all()


class TestSelect:
    def test_is_a_no_op_when_everything_is_offered_and_valid(self) -> None:
        scada = _scada()
        pd.testing.assert_frame_equal(_records().select(scada), scada)

    def test_drops_a_turbine_the_campaign_does_not_offer(self) -> None:
        records_ = _records()
        object.__setattr__(records_, "candidate_references", ["T2"])
        selected = records_.select(_scada())
        assert sorted(selected[_TURBINE_COL].unique()) == ["T1", "T2"]

    def test_drops_the_rows_a_turbine_may_not_contribute(self) -> None:
        records_ = _records()
        valid = records_.valid_for_uplift.copy()
        valid.loc[_INDEX[:2], "T2"] = False
        object.__setattr__(records_, "valid_for_uplift", valid)
        selected = records_.select(_scada())
        t2 = selected[selected[_TURBINE_COL] == "T2"]
        assert t2.index.equals(_INDEX[2:])
        assert len(selected[selected[_TURBINE_COL] == "T3"]) == len(_INDEX)

    def test_also_keeps_named_turbines_that_are_not_candidates(self) -> None:
        # A method co-analysing several test turbines keeps them all, though only one is `test_wtg`.
        records_ = _records()
        object.__setattr__(records_, "candidate_references", ["T2"])
        selected = records_.select(_scada(), also=["T3"])
        assert sorted(selected[_TURBINE_COL].unique()) == ["T1", "T2", "T3"]

    def test_raises_for_an_also_turbine_present_but_not_covered_by_validity(self) -> None:
        # Silently keeping every row of an uncovered turbine would bypass declared validity.
        records_ = _records()
        object.__setattr__(records_, "candidate_references", ["T2"])
        object.__setattr__(records_, "valid_for_uplift", records_.valid_for_uplift[["T1", "T2"]])
        with pytest.raises(ValueError, match="T3"):
            records_.select(_scada(), also=["T3"])


def test_a_reserve_must_be_a_wake_contributor() -> None:
    with pytest.raises(ValueError, match="reserve"):
        SubjectRecords(
            records=pd.DataFrame(),
            subject="A",
            timing=pd.Timestamp("2020-01-01", tz="UTC"),
            turbine_col="t",
            candidate_references=["B"],
            wake_contributors=[],
            reserve_references=["C"],
            valid_for_uplift=pd.DataFrame(),
        )


def test_the_plan_fields_default_to_none() -> None:
    records_ = SubjectRecords(
        records=pd.DataFrame(),
        subject="A",
        timing=pd.Timestamp("2020-01-01", tz="UTC"),
        turbine_col="t",
        candidate_references=["B"],
        wake_contributors=[],
        valid_for_uplift=pd.DataFrame(),
    )
    assert records_.reserve_references == []


# --- a subject's records drawn from the facts, and a group's output -------------------------------


def _subject_facts(**fields: object) -> WindFarmFacts:
    return ff.facts(**fields)


class TestSubjectRecordsFromTheFacts:
    @staticmethod
    def _records(analysis: AnalysisSpec | None = None, **kwargs: object) -> SubjectRecords:
        facts = _subject_facts()
        analysis = analysis or AnalysisSpec(name="a", upgrade="blades")
        kwargs.setdefault("span", (ff.at(0), None))
        return subject_records(facts, analysis, subject="T1", **kwargs)  # type: ignore[arg-type]

    def test_every_unchanged_turbine_is_a_candidate_by_default(self) -> None:
        records = self._records()
        assert records.candidate_references == ["T2", "T3", "T4"]
        assert records.wake_contributors == []
        assert records.reserve_references == []

    def test_n_references_keeps_the_rest_as_reserves(self) -> None:
        analysis = AnalysisSpec(name="a", upgrade="blades", references=["T3", "T2", "T4"])
        records = self._records(analysis, n_references=1)
        assert records.candidate_references == ["T3"]
        assert records.reserve_references == ["T2", "T4"]
        assert records.wake_contributors == ["T2", "T4"]

    def test_the_span_cuts_the_records(self) -> None:
        records = self._records(span=(ff.at(1), ff.at(4)))
        assert records.records.index.min() == ff.at(1)
        assert records.records.index.max() < ff.at(4)

    def test_the_timing_is_the_end_of_the_changeover(self) -> None:
        records = self._records()
        assert records.timing == ff.at(3)
        assert records.works == (ff.at(3), ff.at(3))
        assert records.mode == "prepost"

    def test_records_inside_an_installation_window_are_not_valid(self) -> None:
        window = (ff.at(2), ff.at(3))
        facts = _subject_facts(upgrades=[ff.permanent(installation_windows={"T1": [window]})])
        records = subject_records(facts, AnalysisSpec(name="a", upgrade="blades"), subject="T1", span=(ff.at(0), None))
        valid = records.valid_for_uplift["T1"]
        assert not valid[(valid.index >= window[0]) & (valid.index < window[1])].any()
        assert valid[valid.index >= window[1]].all()

    def test_a_given_validity_is_used(self) -> None:
        index = pd.date_range(ff.at(0), periods=36, freq=ff.TIMEBASE)
        given = pd.DataFrame(data=False, index=index, columns=list(ff.NAMES))
        assert not self._records(valid_for_uplift=given).valid_for_uplift.to_numpy().any()

    def test_the_coordinates_come_from_the_turbine_table(self) -> None:
        assert set(self._records().coords or {}) == set(ff.NAMES)

    def test_given_the_record_labels_each_record_carries_its_operating_state(self) -> None:
        facts = _subject_facts()
        labels = facts.record_labels()
        records = self._records(record_labels=labels).records
        assert set(STATE_COLUMNS) <= set(records.columns)
        assert records[VALID_UPLIFT_COL].dtype == bool
        assert (records[STATE_COL] == NORMAL_OPERATION).all()


class TestWithStates:
    @staticmethod
    def _labels(records: pd.DataFrame) -> pd.DataFrame:
        labels = records[[_TURBINE_COL]].assign(
            **{STATE_COL: MISSING, VALID_NORTHING_COL: False, WAKING_STATE_COL: "missing", VALID_UPLIFT_COL: False}
        )
        labels.loc[labels[_TURBINE_COL] == "T2", [STATE_COL, VALID_UPLIFT_COL]] = [NORMAL_OPERATION, True]
        return labels

    def test_each_record_takes_its_own_turbines_label(self) -> None:
        scada = _scada()
        out = with_states(scada, record_labels=self._labels(scada), turbine_col=_TURBINE_COL)
        assert out.loc[out[_TURBINE_COL] == "T2", VALID_UPLIFT_COL].all()
        assert not out.loc[out[_TURBINE_COL] != "T2", VALID_UPLIFT_COL].any()

    def test_columns_of_the_same_name_are_replaced(self) -> None:
        scada = _scada().assign(**{STATE_COL: NORMAL_OPERATION, VALID_UPLIFT_COL: True})
        out = with_states(scada, record_labels=self._labels(_scada()), turbine_col=_TURBINE_COL)
        assert (out.loc[out[_TURBINE_COL] == "T1", STATE_COL] == MISSING).all()

    def test_a_record_without_a_label_raises(self) -> None:
        scada = _scada()
        labels = self._labels(scada)
        with pytest.raises(ValueError, match="no label"):
            with_states(scada, record_labels=labels[labels[_TURBINE_COL] != "T3"], turbine_col=_TURBINE_COL)


class _Fixed:
    name = "fixed"

    def __init__(self, uplift: float, *, failing: tuple[str, ...] = ()) -> None:
        self.uplift = uplift
        self.failing = failing

    def measure(self, records: SubjectRecords) -> SubjectMeasurement:
        if records.subject in self.failing:
            msg = "too few hours"
            raise ValueError(msg)
        rows = records.records[records.records[ff.COLS.turbine] == records.subject]
        labeled = rows.assign(used=True, segment=np.where(rows.index >= ff.at(3), "upgraded", "baseline"))
        frame = pd.DataFrame([{"turbine": "T4", "uplift": 0.0}])
        return SubjectMeasurement(
            p50_overall=self.uplift, sigma_overall=0.01, labeled_rows=labeled, reference_uplifts=frame
        )


class TestAnsweringAnalyses:
    @staticmethod
    def _answer(method: _Fixed, analysis: AnalysisSpec) -> AnalysisOutput:
        facts = _subject_facts(upgrades=[ff.permanent(turbines=("T1", "T2"))])
        (out,) = answer_analyses(facts, [analysis], name=method.name, measure=method.measure)
        return out

    def test_each_measurement_reads_labelled_records(self) -> None:
        seen: list[SubjectRecords] = []
        method = _Fixed(0.03)

        def measure(records: SubjectRecords) -> SubjectMeasurement:
            seen.append(records)
            return method.measure(records)

        facts = _subject_facts(upgrades=[ff.permanent(turbines=("T1", "T2"))])
        answer_analyses(facts, [AnalysisSpec(name="a", upgrade="blades")], name="fixed", measure=measure)
        assert len(seen) == 2
        assert all(VALID_UPLIFT_COL in records.records.columns for records in seen)

    def test_one_subject_gives_its_uplift_and_sigma(self) -> None:
        out = self._answer(_Fixed(0.03), AnalysisSpec(name="a", upgrade="blades", subjects=["T1"], report="campaign"))
        assert out.campaign_uplift == 0.03
        assert out.campaign_sigma == 0.01
        assert out.warnings == ()

    def test_a_group_gives_the_energy_weighted_uplift_and_no_sigma(self) -> None:
        out = self._answer(_Fixed(0.03), AnalysisSpec(name="a", upgrade="blades", report="campaign"))
        assert out.campaign_uplift == pytest.approx(0.03)
        assert np.isnan(out.campaign_sigma)
        assert [w.severity for w in out.warnings] == ["low"]

    def test_an_unmeasured_subject_is_nan_with_its_reason(self) -> None:
        out = self._answer(_Fixed(0.03, failing=("T1",)), AnalysisSpec(name="a", upgrade="blades", report="campaign"))
        assert measurement_of(out, "T1") == "too few hours"
        measured = measurement_of(out, "T2")
        assert isinstance(measured, SubjectMeasurement)
        assert measured.p50_overall == 0.03
        assert list(measured.reference_uplifts["turbine"]) == ["T4"]  # type: ignore[index]

    def test_no_subject_measured_is_a_high_warning(self) -> None:
        out = self._answer(_Fixed(0.03, failing=("T1",)), AnalysisSpec(name="a", upgrade="blades", subjects=["T1"]))
        assert np.isnan(out.campaign_uplift)
        assert any("measured none" in w.message for w in out.warnings)
        assert any("no AEP" in w.message for w in out.warnings)

    def test_only_the_aep_asked_for_gives_no_campaign_uplift(self) -> None:
        out = self._answer(_Fixed(0.03), AnalysisSpec(name="a", upgrade="blades", subjects=["T1"], report="aep"))
        assert np.isnan(out.campaign_uplift)

    def test_a_measured_nan_is_a_measurement(self) -> None:
        out = self._answer(_Fixed(float("nan")), AnalysisSpec(name="a", upgrade="blades", subjects=["T1"]))
        measured = measurement_of(out, "T1")
        assert isinstance(measured, SubjectMeasurement)
        assert np.isnan(measured.p50_overall)
        assert measured.sigma_overall == 0.01

    def test_a_turbine_not_a_subject_raises(self) -> None:
        out = self._answer(_Fixed(0.03), AnalysisSpec(name="a", upgrade="blades", subjects=["T1"]))
        with pytest.raises(KeyError, match="T3"):
            measurement_of(out, "T3")

    def test_labels_are_refused(self) -> None:
        frame = ff.records().assign(flag=False)
        facts = _subject_facts(frame=frame, labels=[Label(name="x", column="flag")])
        analysis = AnalysisSpec(name="a", upgrade="blades", subjects=["T1"], exclude_labels=["x"])
        with pytest.raises(MethodConfigurationError, match="labels"):
            answer_analyses(facts, [analysis], name="fixed", measure=_Fixed(0.0).measure)  # type: ignore[arg-type]

    def test_a_configuration_error_fails_the_call(self) -> None:
        def measure(_records: SubjectRecords) -> SubjectMeasurement:
            msg = "bad option"
            raise MethodConfigurationError(msg)

        facts = _subject_facts()
        with pytest.raises(MethodConfigurationError):
            answer_analyses(facts, [AnalysisSpec(name="a", upgrade="blades")], name="m", measure=measure)  # type: ignore[arg-type]


class _Unlabelled(_Fixed):
    def measure(self, records: SubjectRecords) -> SubjectMeasurement:  # noqa: ARG002
        return SubjectMeasurement(p50_overall=self.uplift)


def test_a_subject_without_labeled_rows_has_no_energy_and_a_group_of_them_no_uplift() -> None:
    out = TestAnsweringAnalyses._answer(_Unlabelled(0.03), AnalysisSpec(name="a", upgrade="blades"))  # noqa: SLF001
    assert out.per_turbine["actual_energy"].isna().all()  # type: ignore[index]
    assert np.isnan(out.campaign_uplift)


def test_mask_invalid_leaves_a_frame_of_other_turbines_alone() -> None:
    wide = pd.DataFrame(1.0, index=_INDEX, columns=["X1"])
    pd.testing.assert_frame_equal(_records().mask_invalid(wide), wide)


def test_select_skips_a_turbine_without_records() -> None:
    records_ = dataclasses.replace(_records(), candidate_references=["T2", "T3"])
    selected = records_.select(_scada(("T1", "T2")))
    assert sorted(selected[_TURBINE_COL].unique()) == ["T1", "T2"]
