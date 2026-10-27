"""Which records a subject turbine's uplift may be measured on, and why each other record is lost.

A record is used when the subject and every power reference are inside the usable frame and valid for
uplift, no turbine changed at that time is upwind of any of them, none of their upwind columns is
unknown, and none of their waking scenarios has a part waking or missing turbine upwind.
Each lost record takes the first reason, in :data:`LOSS_REASONS` order, that applies to it.

:func:`rare_configurations` is the rule over the whole reference set: a waking configuration (the
scenarios of the subject and its references together) with too few hours in either period is dropped.
A measurement labels the records it drops this way :data:`RARE_CONFIGURATION`, and a toggle record
without a record of the other state nearby :data:`UNPAIRED`. :func:`record_losses` sums the hours
lost to each reason in each period.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

from benchmarking.harness.operating_state import PART_WAKING, VALID_UPLIFT_COL
from benchmarking.harness.waking import UNKNOWN, UPWIND_COL, WAKING_SCENARIO_COL
from wind_up.layout import NAME_COL, ROTOR_DIAMETER_COL
from wind_up.wake_reach import MAX_REACH_D

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from wind_up.layout import Layout

logger = logging.getLogger(__name__)

EXCLUDED = "excluded"
SUBJECT_INVALID = "subject invalid"
REFERENCE_INVALID = "reference invalid"
CHANGED_UPWIND_OF_SUBJECT = "changed upwind of subject"
CHANGED_UPWIND_OF_REFERENCE = "changed upwind of reference"
UPWIND_UNKNOWN = "upwind unknown"
PART_WAKING_UPWIND_OF_SUBJECT = "part waking or missing upwind of subject"
PART_WAKING_UPWIND_OF_REFERENCE = "part waking or missing upwind of reference"
RARE_CONFIGURATION = "rare waking configuration"
UNPAIRED = "unpaired toggle record"
USABLE = "usable"
LOSS_REASONS = (
    EXCLUDED,
    SUBJECT_INVALID,
    REFERENCE_INVALID,
    CHANGED_UPWIND_OF_SUBJECT,
    CHANGED_UPWIND_OF_REFERENCE,
    UPWIND_UNKNOWN,
    PART_WAKING_UPWIND_OF_SUBJECT,
    PART_WAKING_UPWIND_OF_REFERENCE,
)
PAIR_CATEGORIES = (*LOSS_REASONS, USABLE)
CATEGORIES = (*LOSS_REASONS, RARE_CONFIGURATION, UNPAIRED, USABLE)

_SEPARATOR = "; "

NEAR_CHANGEOVER = pd.Timedelta(days=183)
MIN_CONFIGURATION_HOURS = 24.0


class ChangedInterval(NamedTuple):
    """A half-open ``[start, end)`` stretch of time a turbine counts as changed; ``None`` leaves a side unbounded.

    ``reaches`` names the turbines it counts as changed upwind of; ``None`` is every turbine.
    """

    start: pd.Timestamp | None
    end: pd.Timestamp | None
    reaches: frozenset[str] | None = None


ChangedTimes = dict[str, tuple[ChangedInterval, ...]]
"""Each changed turbine and the times it counts as changed."""

WorksWindow = tuple[pd.Timestamp, pd.Timestamp, float]
"""A works window ``[start, end)`` and the reach of its wake change, in rotor diameters."""


def changed_times(
    *,
    test_wtg: str,
    works: Mapping[str, Sequence[WorksWindow]],
    changeover: tuple[pd.Timestamp, pd.Timestamp],
    start: pd.Timestamp,
    end: pd.Timestamp,
    near: pd.Timedelta = NEAR_CHANGEOVER,
    test_reach_d: float = MAX_REACH_D,
    layout: Layout | None = None,
) -> ChangedTimes:
    """Return the times each turbine counts as changed over the span ``[start, end)``.

    The test turbine counts as changed throughout. Every other turbine is changed by each works
    window overlapping the span: throughout, when the window is less than ``near`` from
    ``changeover``; otherwise from the window's start when it follows the changeover, or until its
    end when it precedes it. A works window of no length counts when it falls inside the span.

    A change counts only upwind of the turbines within its reach, in the changed turbine's rotor
    diameters; a reach of :data:`~wind_up.wake_reach.MAX_REACH_D` or more counts upwind of every
    turbine, and a change that reaches no turbine is left out.

    :param test_wtg: the test turbine
    :param works: every turbine's works windows, each with its reach
    :param changeover: the test turbine's works window
    :param start: the span's start
    :param end: the span's end, exclusive
    :param near: how close to the changeover a change counts as changed throughout
    :param test_reach_d: the reach of the test turbine's change
    :param layout: the farm layout, needed for a reach under the maximum
    """
    c0, c1 = changeover
    changed: ChangedTimes = {}
    reaches = _reaches(layout, waker=test_wtg, reach_d=test_reach_d)
    if reaches is None or reaches:
        changed[test_wtg] = (ChangedInterval(None, None, reaches),)
    for turbine, windows in works.items():
        if turbine == test_wtg:
            continue
        times: list[ChangedInterval] = []
        for w0, w1, reach_d in windows:
            if not (w0 < end and (w1 > start or w0 == w1 >= start)):
                continue
            reaches = _reaches(layout, waker=turbine, reach_d=reach_d)
            if reaches is not None and not reaches:
                continue
            if w0 - c1 >= near:
                times.append(ChangedInterval(w0, None, reaches))
            elif c0 - w1 >= near:
                times.append(ChangedInterval(None, w1, reaches))
            else:
                times.append(ChangedInterval(None, None, reaches))
        if times:
            changed[turbine] = tuple(times)
    return changed


def _reaches(layout: Layout | None, *, waker: str, reach_d: float) -> frozenset[str] | None:
    """Return the turbines within ``reach_d`` of ``waker`` in its rotor diameters, ``None`` for every turbine."""
    if reach_d >= MAX_REACH_D:
        return None
    if layout is None:
        msg = f"a reach of {reach_d} D for {waker} needs the farm layout"
        raise ValueError(msg)
    row = layout.index_of(waker)
    distance_d = layout.distance_m[row] / float(layout.frame[ROTOR_DIAMETER_COL].iloc[row])
    names = layout.frame[NAME_COL].to_numpy()
    within = (distance_d <= reach_d) & (np.arange(len(names)) != row)
    return frozenset(str(n) for n in names[within])


def changed_at(
    changed: ChangedTimes, *, index: pd.DatetimeIndex, waked: str | None = None
) -> dict[str, npt.NDArray[np.bool_]]:
    """Return, per changed turbine, whether it counts as changed at each timestamp of ``index``.

    With ``waked``, only the intervals that reach that turbine count.
    """
    out: dict[str, npt.NDArray[np.bool_]] = {}
    for turbine, times in changed.items():
        mask = np.zeros(len(index), dtype=bool)
        for t0, t1, reaches in times:
            if waked is not None and reaches is not None and waked not in reaches:
                continue
            inside = np.ones(len(index), dtype=bool)
            if t0 is not None:
                inside &= np.asarray(index >= t0)
            if t1 is not None:
                inside &= np.asarray(index < t1)
            mask |= inside
        out[turbine] = mask
    return out


def loss_reasons(
    scada: pd.DataFrame,
    *,
    subject: str,
    references: Sequence[str],
    changed: ChangedTimes,
    turbine_col: str,
    usable: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Return, per timestamp of ``index``, the first reason the record is lost, or :data:`USABLE`.

    :param scada: long SCADA carrying :data:`VALID_UPLIFT_COL` and, for the upwind rules, :data:`UPWIND_COL`
    :param subject: the turbine whose uplift is measured
    :param references: its power references
    :param changed: the times each turbine counts as changed, from :func:`changed_times`
    :param turbine_col: the turbine-identifier column
    :param usable: timestamps x turbines, False inside an exclusion or works window
    :param index: the timestamps to label
    """
    flags = _Flags(
        scada, turbines=[subject, *references], changed=changed, turbine_col=turbine_col, usable=usable, index=index
    )
    return flags.reasons(subject=subject, references=references)


def selected(
    scada: pd.DataFrame,
    *,
    subject: str,
    references: Sequence[str],
    changed: ChangedTimes,
    turbine_col: str,
    usable: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> npt.NDArray[np.bool_]:
    """Return, per timestamp of ``index``, whether the record is used. See :func:`loss_reasons`."""
    reasons = loss_reasons(
        scada,
        subject=subject,
        references=references,
        changed=changed,
        turbine_col=turbine_col,
        usable=usable,
        index=index,
    )
    return np.asarray(reasons == USABLE, dtype=bool)


def pair_reasons(
    scada: pd.DataFrame,
    *,
    subject: str,
    candidates: Sequence[str],
    changed: ChangedTimes,
    turbine_col: str,
    usable: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return :func:`loss_reasons` for ``subject`` with each candidate alone as its reference, one column each."""
    flags = _Flags(
        scada, turbines=[subject, *candidates], changed=changed, turbine_col=turbine_col, usable=usable, index=index
    )
    return pd.DataFrame({c: flags.reasons(subject=subject, references=[c]) for c in candidates}, index=index)


class _Flags:
    """Each turbine's per-timestamp loss conditions over one index."""

    def __init__(
        self,
        scada: pd.DataFrame,
        *,
        turbines: Sequence[str],
        changed: ChangedTimes,
        turbine_col: str,
        usable: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> None:
        self.index = index
        has_upwind = UPWIND_COL in scada.columns
        if not has_upwind:
            logger.warning(
                "%s: no %r column (method step 5), so records are not dropped for a changed or unknown turbine upwind",
                turbines[0],
                UPWIND_COL,
            )
        has_scenario = WAKING_SCENARIO_COL in scada.columns
        if not has_scenario:
            logger.warning(
                "%s: no %r column (method step 5), so records are not dropped for a part waking or missing turbine "
                "upwind",
                turbines[0],
                WAKING_SCENARIO_COL,
            )
        names = scada[turbine_col].astype(str).to_numpy()
        n = len(index)
        self.excluded: dict[str, npt.NDArray[np.bool_]] = {}
        self.invalid: dict[str, npt.NDArray[np.bool_]] = {}
        self.changed_upwind: dict[str, npt.NDArray[np.bool_]] = {}
        self.unknown: dict[str, npt.NDArray[np.bool_]] = {}
        self.part_waking: dict[str, npt.NDArray[np.bool_]] = {}
        for turbine in dict.fromkeys(turbines):
            in_frame = usable[turbine].reindex(index, fill_value=False) if turbine in usable.columns else None
            if in_frame is None:
                msg = f"{turbine!r} has no column in the usable frame (covers {list(usable.columns)})"
                raise ValueError(msg)
            self.excluded[turbine] = ~in_frame.to_numpy(dtype=bool)
            rows = scada[names == turbine]
            rows = rows[~rows.index.duplicated()]
            valid = rows[VALID_UPLIFT_COL].astype(bool).reindex(index, fill_value=False)
            self.invalid[turbine] = ~valid.to_numpy(dtype=bool)
            self.part_waking[turbine] = (
                _scenario_unusable(rows[WAKING_SCENARIO_COL].reindex(index))
                if has_scenario
                else np.zeros(n, dtype=bool)
            )
            if not has_upwind:
                self.changed_upwind[turbine] = np.zeros(n, dtype=bool)
                self.unknown[turbine] = np.zeros(n, dtype=bool)
                continue
            at = changed_at(changed, index=index, waked=turbine)
            upwind = rows[UPWIND_COL].reindex(index)
            # A record the turbine has no row for is already invalid; unknown covers a row without a label.
            text = upwind.fillna(UNKNOWN).astype(str)
            self.unknown[turbine] = (text == UNKNOWN).to_numpy()
            codes, labels = pd.factorize(text)
            hit = np.zeros(n, dtype=bool)
            for code, label in enumerate(labels):
                upwind_changed = [name for name in str(label).split(_SEPARATOR) if name in at]
                if upwind_changed:
                    at_label = codes == code
                    hit[at_label] = _any(at, upwind_changed, n=n)[at_label]
            self.changed_upwind[turbine] = hit

    def reasons(self, *, subject: str, references: Sequence[str]) -> pd.Series:
        """Return the first reason each record is lost with ``references`` as the power references."""
        refs = list(references)
        both = [subject, *refs]
        conditions = [
            _any(self.excluded, both, n=len(self.index)),
            self.invalid[subject],
            _any(self.invalid, refs, n=len(self.index)),
            self.changed_upwind[subject],
            _any(self.changed_upwind, refs, n=len(self.index)),
            _any(self.unknown, both, n=len(self.index)),
            self.part_waking[subject],
            _any(self.part_waking, refs, n=len(self.index)),
        ]
        labels = np.select(conditions, list(LOSS_REASONS), default=USABLE)
        return pd.Series(pd.Categorical(labels, categories=list(CATEGORIES)), index=self.index)


def _scenario_unusable(scenario: pd.Series) -> npt.NDArray[np.bool_]:
    """Return where a waking scenario is unknown or names a part waking turbine."""
    text = scenario.fillna(UNKNOWN).astype(str)
    return ((text == UNKNOWN) | text.str.contains(PART_WAKING, regex=False)).to_numpy()


def rare_configurations(
    scenarios: pd.DataFrame,
    *,
    before: npt.NDArray[np.bool_],
    after: npt.NDArray[np.bool_],
    hours_per_record: float,
    min_hours: float = MIN_CONFIGURATION_HOURS,
) -> npt.NDArray[np.bool_]:
    """Return, per row of ``scenarios``, whether its waking configuration is rare.

    A configuration is the row's scenarios taken together. It is rare when it has under ``min_hours``
    of rows in ``before`` or in ``after``. Rows outside both periods are rare when their
    configuration is.

    :param scenarios: one waking scenario column per turbine, the subject and its power references
    :param before: the rows of the before period
    :param after: the rows of the after period
    :param hours_per_record: the timebase, in hours
    :param min_hours: the fewest hours a configuration needs in each period
    """
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(scenarios.astype(str)))
    n_codes = int(codes.max()) + 1 if len(codes) else 0
    hours_before = np.bincount(codes[before], minlength=n_codes) * hours_per_record
    hours_after = np.bincount(codes[after], minlength=n_codes) * hours_per_record
    rare = (hours_before < min_hours) | (hours_after < min_hours)
    return rare[codes]


def _any(flags: Mapping[str, npt.NDArray[np.bool_]], turbines: Sequence[str], *, n: int) -> npt.NDArray[np.bool_]:
    """Return where any of ``turbines`` is flagged."""
    out = np.zeros(n, dtype=bool)
    for turbine in turbines:
        out |= flags[turbine]
    return out


def record_losses(
    reasons: pd.Series,
    *,
    before: npt.NDArray[np.bool_],
    after: npt.NDArray[np.bool_],
    hours_per_record: float,
) -> pd.DataFrame:
    """Return the hours of each reason in :data:`CATEGORIES` in the before and after periods, and outside both.

    :param reasons: per record, its loss reason or :data:`USABLE`
    :param before: the records of the before period
    :param after: the records of the after period
    :param hours_per_record: the timebase, in hours
    """
    labels = reasons.astype(str).to_numpy()
    outside = ~(np.asarray(before, dtype=bool) | np.asarray(after, dtype=bool))
    return pd.DataFrame(
        {
            "reason": list(CATEGORIES),
            "hours_before": [float(np.count_nonzero((labels == r) & before)) * hours_per_record for r in CATEGORIES],
            "hours_after": [float(np.count_nonzero((labels == r) & after)) * hours_per_record for r in CATEGORIES],
            "hours_outside": [float(np.count_nonzero((labels == r) & outside)) * hours_per_record for r in CATEGORIES],
        }
    )
