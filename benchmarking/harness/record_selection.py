"""Which records a subject turbine's uplift may be measured on, and why each other record is lost.

A record is used when the subject and every power reference are inside the usable frame and valid for
uplift, no turbine changed at that time is upwind of any of them, and none of their upwind columns is
unknown.
Each lost record takes the first reason, in :data:`LOSS_REASONS` order, that applies to it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.harness.operating_state import VALID_UPLIFT_COL
from benchmarking.harness.waking import UNKNOWN, UPWIND_COL

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

logger = logging.getLogger(__name__)

EXCLUDED = "excluded"
SUBJECT_INVALID = "subject invalid"
REFERENCE_INVALID = "reference invalid"
CHANGED_UPWIND_OF_SUBJECT = "changed upwind of subject"
CHANGED_UPWIND_OF_REFERENCE = "changed upwind of reference"
UPWIND_UNKNOWN = "upwind unknown"
USABLE = "usable"
LOSS_REASONS = (
    EXCLUDED,
    SUBJECT_INVALID,
    REFERENCE_INVALID,
    CHANGED_UPWIND_OF_SUBJECT,
    CHANGED_UPWIND_OF_REFERENCE,
    UPWIND_UNKNOWN,
)
CATEGORIES = (*LOSS_REASONS, USABLE)

_SEPARATOR = "; "

NEAR_CHANGEOVER = pd.Timedelta(days=183)

Interval = tuple[pd.Timestamp | None, pd.Timestamp | None]
"""A half-open ``[start, end)`` stretch of time; ``None`` leaves that side unbounded."""

ChangedTimes = dict[str, tuple[Interval, ...]]
"""Each changed turbine and the times it counts as changed."""


def changed_times(
    *,
    test_wtg: str,
    works: Mapping[str, Sequence[tuple[pd.Timestamp, pd.Timestamp]]],
    changeover: tuple[pd.Timestamp, pd.Timestamp],
    start: pd.Timestamp,
    end: pd.Timestamp,
    near: pd.Timedelta = NEAR_CHANGEOVER,
) -> ChangedTimes:
    """Return the times each turbine counts as changed over the span ``[start, end)``.

    The test turbine counts as changed throughout. Every other turbine is changed by each works window
    overlapping the span: throughout, when the window is less than ``near`` from ``changeover``;
    otherwise from the window's start when it follows the changeover, or until its end when it
    precedes it. A works window of no length counts when it falls inside the span.

    :param test_wtg: the test turbine
    :param works: every turbine's works windows
    :param changeover: the test turbine's works window
    :param start: the span's start
    :param end: the span's end, exclusive
    :param near: how close to the changeover a change counts as changed throughout
    """
    c0, c1 = changeover
    changed: ChangedTimes = {test_wtg: ((None, None),)}
    for turbine, windows in works.items():
        if turbine == test_wtg:
            continue
        times: list[Interval] = []
        for w0, w1 in windows:
            if not (w0 < end and (w1 > start or w0 == w1 >= start)):
                continue
            if w0 - c1 >= near:
                times.append((w0, None))
            elif c0 - w1 >= near:
                times.append((None, w1))
            else:
                times.append((None, None))
        if times:
            changed[turbine] = tuple(times)
    return changed


def changed_at(changed: ChangedTimes, *, index: pd.DatetimeIndex) -> dict[str, npt.NDArray[np.bool_]]:
    """Return, per changed turbine, whether it counts as changed at each timestamp of ``index``."""
    out = {}
    for turbine, times in changed.items():
        mask = np.zeros(len(index), dtype=bool)
        for t0, t1 in times:
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
        names = scada[turbine_col].astype(str).to_numpy()
        n = len(index)
        at = changed_at(changed, index=index)
        self.excluded: dict[str, npt.NDArray[np.bool_]] = {}
        self.invalid: dict[str, npt.NDArray[np.bool_]] = {}
        self.changed_upwind: dict[str, npt.NDArray[np.bool_]] = {}
        self.unknown: dict[str, npt.NDArray[np.bool_]] = {}
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
            if not has_upwind:
                self.changed_upwind[turbine] = np.zeros(n, dtype=bool)
                self.unknown[turbine] = np.zeros(n, dtype=bool)
                continue
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
        ]
        labels = np.select(conditions, list(LOSS_REASONS), default=USABLE)
        return pd.Series(pd.Categorical(labels, categories=list(CATEGORIES)), index=self.index)


def _any(flags: Mapping[str, npt.NDArray[np.bool_]], turbines: Sequence[str], *, n: int) -> npt.NDArray[np.bool_]:
    """Return where any of ``turbines`` is flagged."""
    out = np.zeros(n, dtype=bool)
    for turbine in turbines:
        out |= flags[turbine]
    return out
