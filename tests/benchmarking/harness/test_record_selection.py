"""Tests for the shared record selection of method step 11 i and iii."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest  # noqa: TC002 - caplog fixtures are runtime types

from benchmarking.harness.operating_state import VALID_UPLIFT_COL
from benchmarking.harness.record_selection import (
    CHANGED_UPWIND_OF_REFERENCE,
    CHANGED_UPWIND_OF_SUBJECT,
    EXCLUDED,
    REFERENCE_INVALID,
    SUBJECT_INVALID,
    UPWIND_UNKNOWN,
    USABLE,
    changed_times,
    loss_reasons,
    pair_reasons,
    selected,
)
from benchmarking.harness.waking import UPWIND_COL

if TYPE_CHECKING:
    from benchmarking.harness.record_selection import ChangedTimes

INDEX = pd.date_range("2020-01-01", periods=8, freq="10min", tz="UTC")
TURBINES = ("T01", "T02", "T03")


def frame(
    *,
    valid: dict[str, list[bool]] | None = None,
    upwind: dict[str, list[str]] | None = None,
    drop: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """Long SCADA of three turbines, every record valid with nothing upwind unless overridden."""
    parts = []
    for turbine in TURBINES:
        part = pd.DataFrame(
            {
                "TurbineName": turbine,
                VALID_UPLIFT_COL: (valid or {}).get(turbine, [True] * len(INDEX)),
                UPWIND_COL: (upwind or {}).get(turbine, ["none"] * len(INDEX)),
            },
            index=INDEX,
        )
        rows = (drop or {}).get(turbine, [])
        parts.append(part.drop(index=INDEX[rows]))
    return pd.concat(parts)


def always(*turbines: str) -> ChangedTimes:
    """Each turbine changed over the whole span."""
    return dict.fromkeys(turbines, ((None, None),))


def usable_frame(**false_at: list[int]) -> pd.DataFrame:
    """Timestamps x turbines, all True except the given positions."""
    usable = pd.DataFrame(data=True, index=INDEX, columns=list(TURBINES))
    for turbine, rows in false_at.items():
        usable.iloc[rows, usable.columns.get_loc(turbine)] = False
    return usable


def reasons(scada: pd.DataFrame, *, changed: ChangedTimes | None = None, **usable: list[int]) -> pd.Series:
    """The loss reasons with T01 the subject and T02, T03 the references."""
    return loss_reasons(
        scada,
        subject="T01",
        references=["T02", "T03"],
        changed=always("T01") if changed is None else changed,
        turbine_col="TurbineName",
        usable=usable_frame(**usable),
        index=INDEX,
    )


class TestLossReasons:
    def test_a_clean_frame_is_all_usable(self) -> None:
        assert (reasons(frame()) == USABLE).all()

    def test_each_reason_is_named(self) -> None:
        scada = frame(
            valid={"T01": [True, False, *[True] * 6], "T03": [True, True, False, *[True] * 5]},
            upwind={
                "T01": ["none"] * 3 + ["T03", "T02"] + ["none"] * 3,
                "T02": ["none"] * 5 + ["T01"] + ["none"] * 2,
                "T03": ["none"] * 6 + ["unknown", "none"],
            },
        )
        got = reasons(scada, changed=always("T01", "T03"), T02=[0])
        assert list(got) == [
            EXCLUDED,
            SUBJECT_INVALID,
            REFERENCE_INVALID,
            CHANGED_UPWIND_OF_SUBJECT,
            USABLE,
            CHANGED_UPWIND_OF_REFERENCE,
            UPWIND_UNKNOWN,
            USABLE,
        ]

    def test_the_first_reason_wins(self) -> None:
        scada = frame(valid={"T01": [False] * 8, "T02": [False] * 8}, upwind={"T01": ["unknown"] * 8})
        got = reasons(scada, T03=[0])
        assert got.iloc[0] == EXCLUDED
        assert (got.iloc[1:] == SUBJECT_INVALID).all()

    def test_a_reference_without_a_row_is_invalid(self) -> None:
        got = reasons(frame(drop={"T02": [4]}))
        assert got.iloc[4] == REFERENCE_INVALID

    def test_a_subject_without_a_row_is_invalid(self) -> None:
        got = reasons(frame(drop={"T01": [2]}))
        assert got.iloc[2] == SUBJECT_INVALID

    def test_unknown_upwind_drops_the_record(self) -> None:
        got = reasons(frame(upwind={"T02": ["unknown"] * 8}))
        assert (got == UPWIND_UNKNOWN).all()

    def test_an_unchanged_turbine_upwind_keeps_the_record(self) -> None:
        got = reasons(frame(upwind={"T01": ["T02; T03"] * 8}))
        assert (got == USABLE).all()

    def test_the_reasons_are_categorical_in_priority_order(self) -> None:
        got = reasons(frame())
        assert list(got.cat.categories) == [
            EXCLUDED,
            SUBJECT_INVALID,
            REFERENCE_INVALID,
            CHANGED_UPWIND_OF_SUBJECT,
            CHANGED_UPWIND_OF_REFERENCE,
            UPWIND_UNKNOWN,
            USABLE,
        ]

    def test_selected_is_the_usable_records(self) -> None:
        scada = frame(valid={"T01": [True, False, *[True] * 6]})
        got = selected(
            scada,
            subject="T01",
            references=["T02", "T03"],
            changed=always("T01"),
            turbine_col="TurbineName",
            usable=usable_frame(),
            index=INDEX,
        )
        assert got.dtype == bool
        assert list(got) == [True, False, *[True] * 6]


class TestMissingUpwind:
    def test_the_upwind_rules_are_skipped_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        scada = frame(upwind={"T01": ["unknown"] * 8}).drop(columns=[UPWIND_COL])
        with caplog.at_level(logging.WARNING):
            got = reasons(scada)
        assert (got == USABLE).all()
        assert UPWIND_COL in caplog.text


class TestPairReasons:
    def test_one_column_per_candidate(self) -> None:
        scada = frame(valid={"T02": [False, *[True] * 7]}, upwind={"T03": ["T01"] * 8})
        got = pair_reasons(
            scada,
            subject="T01",
            candidates=["T02", "T03"],
            changed=always("T01"),
            turbine_col="TurbineName",
            usable=usable_frame(),
            index=INDEX,
        )
        assert list(got.columns) == ["T02", "T03"]
        assert got["T02"].iloc[0] == REFERENCE_INVALID
        assert (got["T02"].iloc[1:] == USABLE).all()
        assert (got["T03"] == CHANGED_UPWIND_OF_REFERENCE).all()

    def test_the_joint_mask_is_the_and_of_the_pair_masks(self) -> None:
        rng = np.random.default_rng(0)
        valid = {t: list(rng.random(8) > 0.2) for t in TURBINES}
        upwind = {
            t: list(rng.choice(["none", "T01", "T02", "unknown"], size=8, p=[0.6, 0.15, 0.15, 0.1])) for t in TURBINES
        }
        scada = frame(valid=valid, upwind=upwind)
        common = {
            "changed": always("T01"),
            "turbine_col": "TurbineName",
            "usable": usable_frame(T03=[5]),
            "index": INDEX,
        }
        pairs = pair_reasons(scada, subject="T01", candidates=["T02", "T03"], **common)
        joint = selected(scada, subject="T01", references=["T02", "T03"], **common)
        assert list(joint) == list(((pairs == USABLE).all(axis=1)).to_numpy())


class TestChangedTimesUpwind:
    def test_a_turbine_is_upwind_as_changed_only_inside_its_times(self) -> None:
        changed = {"T01": ((None, None),), "T03": ((INDEX[5], None),)}
        got = reasons(frame(upwind={"T01": ["T03"] * 8}), changed=changed)
        assert list(got) == [USABLE] * 5 + [CHANGED_UPWIND_OF_SUBJECT] * 3

    def test_an_early_change_counts_before_its_end(self) -> None:
        changed = {"T01": ((None, None),), "T03": ((None, INDEX[2]),)}
        got = reasons(frame(upwind={"T02": ["T03"] * 8}), changed=changed)
        assert list(got) == [CHANGED_UPWIND_OF_REFERENCE] * 2 + [USABLE] * 6

    def test_any_changed_name_in_the_label_counts(self) -> None:
        changed = {"T01": ((None, None),), "T03": ((INDEX[6], None),), "T02": ((None, INDEX[1]),)}
        got = reasons(frame(upwind={"T01": ["T02; T03"] * 8}), changed=changed)
        assert list(got) == [CHANGED_UPWIND_OF_SUBJECT] + [USABLE] * 5 + [CHANGED_UPWIND_OF_SUBJECT] * 2


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


class TestChangedTimes:
    START = day("2020-01-01")
    END = day("2022-01-01")
    CHANGEOVER = (day("2021-01-01"), day("2021-01-08"))

    def times(self, works: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]) -> ChangedTimes:
        return changed_times(test_wtg="T01", works=works, changeover=self.CHANGEOVER, start=self.START, end=self.END)

    def test_the_test_turbine_is_changed_over_the_whole_span(self) -> None:
        assert self.times({"T01": [self.CHANGEOVER]}) == {"T01": ((None, None),)}

    def test_a_change_near_the_changeover_is_changed_over_the_whole_span(self) -> None:
        got = self.times({"T02": [(day("2021-06-01"), day("2021-06-03"))]})
        assert got["T02"] == ((None, None),)

    def test_a_change_far_after_the_changeover_counts_from_its_start(self) -> None:
        got = self.times({"T02": [(day("2021-09-01"), day("2021-09-03"))]})
        assert got["T02"] == ((day("2021-09-01"), None),)

    def test_a_change_far_before_the_changeover_counts_until_its_end(self) -> None:
        got = self.times({"T02": [(day("2020-03-01"), day("2020-03-03"))]})
        assert got["T02"] == ((None, day("2020-03-03")),)

    def test_six_months_from_the_changeover_is_far(self) -> None:
        start = self.CHANGEOVER[1] + pd.Timedelta(days=183)
        got = self.times({"T02": [(start, start + pd.Timedelta(days=1))]})
        assert got["T02"] == ((start, None),)

    def test_just_under_six_months_is_near(self) -> None:
        end = self.CHANGEOVER[0] - pd.Timedelta(days=182)
        got = self.times({"T02": [(end - pd.Timedelta(days=1), end)]})
        assert got["T02"] == ((None, None),)

    def test_the_distance_is_a_setting(self) -> None:
        got = changed_times(
            test_wtg="T01",
            works={"T02": [(day("2021-06-01"), day("2021-06-03"))]},
            changeover=self.CHANGEOVER,
            start=self.START,
            end=self.END,
            near=pd.Timedelta(days=30),
        )
        assert got["T02"] == ((day("2021-06-01"), None),)

    def test_each_window_counts(self) -> None:
        works = {"T02": [(day("2020-03-01"), day("2020-03-03")), (day("2021-09-01"), day("2021-09-03"))]}
        assert self.times(works)["T02"] == ((None, day("2020-03-03")), (day("2021-09-01"), None))

    def test_works_outside_the_span_mean_not_changed(self) -> None:
        works = {
            "T02": [(day("2019-06-01"), day("2019-06-03"))],
            "T03": [(self.END, self.END + pd.Timedelta(days=1))],
        }
        assert self.times(works) == {"T01": ((None, None),)}

    def test_a_change_of_no_length_inside_the_span_counts(self) -> None:
        moment = day("2021-09-01")
        assert self.times({"T02": [(moment, moment)]})["T02"] == ((moment, None),)
