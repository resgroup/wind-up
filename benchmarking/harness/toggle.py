"""The single, shared interpretation of a toggle campaign, consumed by every benchmarking method.

A toggle campaign can be described two ways: a **periodic** ``ToggleSchedule`` (the harness's
convenience form) or an explicit, possibly **irregular** ``toggle_df`` (per-timestamp boolean
``toggle_on``/``toggle_off``, the form released wind_up already uses and a real shuffled campaign
needs). Prepost is described by a changeover ``pd.Timestamp``.

Whatever the form, :func:`resolve_toggle` turns it into the three row-sets a method legitimately
needs, so ``naive_ratio``, ``power_model`` and ``v0_binned`` all agree on which rows are what:

- ``upgraded`` — the "on" rows (the segment whose uplift is estimated).
- ``campaign_baseline`` — the strict "off" rows (``toggle_off``). The on/off energy comparison and
  the power model's covariate matching use this.
- ``training_baseline`` — the lenient baseline ``(index < first upgraded) U toggle_off``: all
  pre-campaign rows plus the off-blocks. Only the counterfactual **fit** uses this (more
  upgrade-invariant data helps the model); it mirrors released wind_up's detrend-data selection.

Rows that are neither ``upgraded`` nor a ``training_baseline`` row (e.g. a third perturbation state,
noise/ice rows, transition bins) are **excluded from every segment** — the "neither" case a binary
``treated``/``~treated`` split cannot represent.

:func:`pair_within` is the toggle pairing filter of method step 11 iv: a record of one state is kept
only when a record of the other state lies within a gap, by default one toggle cycle
(:func:`toggle_cycle`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

from benchmarking.synthetic import ToggleSchedule, treated_mask

if TYPE_CHECKING:
    import numpy.typing as npt

_TOGGLE_COLUMNS = ("toggle_on", "toggle_off")


class PairedRows(NamedTuple):
    """The rows of each segment that have a row of the other segment within the pairing gap."""

    baseline: npt.NDArray[np.bool_]
    upgraded: npt.NDArray[np.bool_]


def pair_within(
    index: pd.DatetimeIndex,
    *,
    baseline: npt.NDArray[np.bool_],
    upgraded: npt.NDArray[np.bool_],
    max_gap: pd.Timedelta,
) -> PairedRows:
    """Keep each flagged row only if the other segment has a flagged row within ``max_gap``, inclusive."""
    gap = gap_to_other_segment(index, baseline=baseline, upgraded=upgraded)
    within = gap <= max_gap.total_seconds()
    return PairedRows(baseline=baseline & within, upgraded=upgraded & within)


def gap_to_other_segment(
    index: pd.DatetimeIndex, *, baseline: npt.NDArray[np.bool_], upgraded: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Seconds from each flagged row to the nearest flagged row of the other segment; NaN if unflagged."""
    times = index.as_unit("ns").asi8
    gap = np.full(len(index), np.nan)
    gap[baseline] = _nearest_gap_s(times[baseline], others=times[upgraded])
    gap[upgraded] = _nearest_gap_s(times[upgraded], others=times[baseline])
    return gap


def _nearest_gap_s(times: npt.NDArray[np.int64], *, others: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    if not len(others):
        return np.full(len(times), np.inf)
    others = np.sort(others)
    position = np.searchsorted(others, times)
    after = others[np.minimum(position, len(others) - 1)]
    before = others[np.maximum(position - 1, 0)]
    return np.minimum(np.abs(after - times), np.abs(times - before)) / 1e9


def toggle_cycle(upgrade_timing: ToggleSchedule | pd.DataFrame, index: pd.DatetimeIndex) -> pd.Timedelta:
    """Return the length of one on and off cycle of a toggle campaign.

    A ``ToggleSchedule`` states it. For an explicit ``toggle_df`` it is twice the median length of
    the runs of consecutive same-state rows within ``index``, each run measured from its first
    timestamp to the start of the row after it.
    """
    if isinstance(upgrade_timing, ToggleSchedule):
        return upgrade_timing.period
    unique = pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(index))).sort_values()
    rows = resolve_toggle(upgrade_timing, unique)
    in_campaign = rows.upgraded | rows.campaign_baseline
    times = unique[in_campaign]
    state = rows.upgraded[in_campaign]
    if len(times) < 2:  # noqa: PLR2004
        msg = "toggle_df has fewer than two on or off rows within the index, so no toggle cycle can be inferred"
        raise ValueError(msg)
    change = np.flatnonzero(state[1:] != state[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(times)]))
    step = pd.Timedelta(np.median(np.diff(unique.to_numpy())))
    lengths = [(times[e - 1] - times[s]) + step for s, e in zip(starts, ends, strict=True)]
    return 2 * pd.Timedelta(np.median(np.asarray(lengths, dtype="timedelta64[ns]")))


@dataclass(frozen=True)
class ToggleRowSets:
    """The three boolean row-sets a method selects over, aligned to the index passed to resolve.

    :param upgraded: the "on" rows (uplift is estimated over these)
    :param campaign_baseline: the strict "off" rows (on/off comparison + matching)
    :param training_baseline: the lenient baseline (pre-campaign U off) for counterfactual fitting
    """

    upgraded: npt.NDArray[np.bool_]
    campaign_baseline: npt.NDArray[np.bool_]
    training_baseline: npt.NDArray[np.bool_]


def is_toggle(upgrade_timing: object) -> bool:
    """Return whether ``upgrade_timing`` describes a toggle campaign (schedule or explicit frame).

    A ``pd.Timestamp`` (prepost changeover) is not a toggle; a ``ToggleSchedule`` or a
    ``toggle_df`` (``pd.DataFrame`` with ``toggle_on``/``toggle_off``) is.
    """
    return isinstance(upgrade_timing, (ToggleSchedule, pd.DataFrame))


def build_toggle_df(index: pd.DatetimeIndex, schedule: ToggleSchedule) -> pd.DataFrame:
    """Build the canonical three-valued ``toggle_df`` from a periodic ``ToggleSchedule``.

    Before ``schedule.start`` both flags are False (no campaign signal yet); from the start onward
    exactly one is True per record (``toggle_on`` = upgraded, ``toggle_off`` = the interleaved
    off-blocks). Indexed by the unique timestamps of ``index``.
    """
    unique = pd.DatetimeIndex(pd.unique(index)).sort_values()
    toggle_on = np.asarray(treated_mask(unique, schedule))
    after_start = np.asarray(unique >= schedule.start) if schedule.start is not None else np.ones(len(unique), bool)
    toggle_off = ~toggle_on & after_start
    return pd.DataFrame({"toggle_on": toggle_on, "toggle_off": toggle_off}, index=unique)


def _reindex_flag(toggle_df: pd.DataFrame, column: str, index: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
    """Reindex one boolean ``toggle_df`` column onto ``index`` (repeats allowed), missing → False."""
    return toggle_df[column].reindex(index, fill_value=False).to_numpy(dtype=bool)


def _validate_toggle_df(toggle_df: pd.DataFrame) -> None:
    """Validate a ``toggle_df`` before it is reindexed onto an analysis index, with clear errors.

    A frame (downstream-supplied or built here) must carry both flag columns and hold one row per
    timestamp: a duplicated index would otherwise make the reindex in :func:`resolve_toggle` fail
    with pandas' opaque "cannot reindex on an axis with duplicate labels".
    """
    missing = [c for c in _TOGGLE_COLUMNS if c not in toggle_df.columns]
    if missing:
        msg = f"toggle_df must have columns {_TOGGLE_COLUMNS}; missing {missing}"
        raise ValueError(msg)
    if not toggle_df.index.is_unique:
        dupes = toggle_df.index[toggle_df.index.duplicated()].unique().tolist()
        msg = f"toggle_df index must be unique (one row per timestamp); duplicated timestamps: {dupes}"
        raise ValueError(msg)


def resolve_toggle(
    upgrade_timing: pd.Timestamp | ToggleSchedule | pd.DataFrame, index: pd.DatetimeIndex
) -> ToggleRowSets:
    """Resolve any campaign description into the three row-sets, aligned to ``index``.

    ``index`` may be a repeated (long-frame) index; the labels broadcast to every row. A prepost
    ``pd.Timestamp`` splits on the changeover (both baselines are the pre-changeover rows). A
    ``ToggleSchedule`` is first turned into a ``toggle_df`` via :func:`build_toggle_df`.
    """
    index = pd.DatetimeIndex(index)
    if not is_toggle(upgrade_timing):
        upgraded = np.asarray(index >= upgrade_timing)
        baseline = ~upgraded
        return ToggleRowSets(upgraded=upgraded, campaign_baseline=baseline, training_baseline=baseline)

    toggle_df = build_toggle_df(index, upgrade_timing) if isinstance(upgrade_timing, ToggleSchedule) else upgrade_timing
    _validate_toggle_df(toggle_df)

    upgraded = _reindex_flag(toggle_df, "toggle_on", index)
    campaign_baseline = _reindex_flag(toggle_df, "toggle_off", index)
    if (upgraded & campaign_baseline).any():
        msg = "toggle_on and toggle_off cannot both be True for the same timestamp"
        raise ValueError(msg)

    if upgraded.any():
        first_upgraded = index[upgraded].min()
        pre_campaign = np.asarray(index < first_upgraded)
    else:
        pre_campaign = np.zeros(len(index), dtype=bool)
    training_baseline = pre_campaign | campaign_baseline
    return ToggleRowSets(upgraded=upgraded, campaign_baseline=campaign_baseline, training_baseline=training_baseline)


def toggle_upgrade_start(
    upgrade_timing: pd.Timestamp | ToggleSchedule | pd.DataFrame, index: pd.DatetimeIndex
) -> pd.Timestamp:
    """Return the campaign's upgrade-start timestamp, shared for run naming and time-decay weighting.

    Prepost: the changeover timestamp. A ``ToggleSchedule``: its ``start`` (or ``index`` min when
    open-ended). An explicit ``toggle_df``: the first upgraded row *within* ``index`` — resolved
    through :func:`resolve_toggle` so a malformed frame fails with the same clear error as the rest
    of the harness, and the start is aligned to the analysis window rather than trusting on-rows the
    analysis never sees — falling back to ``index`` min when the campaign never turns on inside it.
    """
    if isinstance(upgrade_timing, ToggleSchedule):
        return upgrade_timing.start if upgrade_timing.start is not None else pd.DatetimeIndex(index).min()
    if isinstance(upgrade_timing, pd.DataFrame):
        upgraded = resolve_toggle(upgrade_timing, index).upgraded
        index = pd.DatetimeIndex(index)
        return index[upgraded].min() if upgraded.any() else index.min()
    return pd.Timestamp(upgrade_timing)
