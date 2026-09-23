"""Estimate and apply north-calibration corrections for a direction signal.

:func:`estimate_north_table` compares a direction signal with a reference and returns a table of
``(timestamp, north_offset)`` describing the steps it found; :func:`apply_north_table` steps that
table onto the raw signal. Offsets are absolute -- relative to the raw field, never to an
already-corrected one -- so a supplied table and an estimated one are directly comparable.

:func:`north_farm` runs the farm workflow: anchor every device to reanalysis, build a farm
consensus direction from the results and north every device to that, then, where a layout and power
are supplied, nudge each device to its wake nadirs (:mod:`wind_up.wake_nadir`).

The estimator works on any direction field. Only :func:`yaw_usable` is turbine-specific.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from wind_up.circular_math import circ_diff, circ_median
from wind_up.geodesy import geodesic_matrices
from wind_up.layout import NAME_COL
from wind_up.wake_nadir import wake_nadir_offsets

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import numpy.typing as npt

    from wind_up.layout import Layout

logger = logging.getLogger(__name__)

TIMESTAMP_COL = "timestamp"
NORTH_OFFSET_COL = "north_offset"

# Yaw is read only above this fraction of rated power.
YAW_OK_POWER_FRACTION = 0.05
# Above this many aggregation bins the search warns; it still returns a correct result.
_BIN_COUNT_WARN = 3000
_DEFAULT_GRID = pd.Timedelta(days=1)
_DEFAULT_MIN_SEGMENT = pd.Timedelta(days=7)
# A segment needs a row either side of a candidate split for the split to mean anything.
_MIN_ROWS_TO_SPLIT = 2
# Direction sectors the residual is normalised over before the changepoint search.
_DEFAULT_VEER_SECTOR_DEG = 30.0
# A sector with fewer usable rows than this has no trustworthy level of its own.
_MIN_ROWS_PER_SECTOR = 50
# Steps larger than this are never ironed out as wander.
_MAX_TRANSIENT_STEP_DEG = 10.0


# The span either side of a changepoint at which ``min_step_deg`` applies unmodified; a shorter
# segment needs a larger step.
_DEFAULT_CONFIDENT_SEGMENT = pd.Timedelta(days=90)


@dataclass(frozen=True)
class NorthingSettings:
    """How the changepoint search is bounded, in physical units.

    :param changepoints_per_year: budget of changepoints per year of record, so a longer
        record is allowed more; the cap is ``max(min_changepoints, ceil(rate * years))``
    :param min_changepoints: floor on that budget, so a short record can still hold several
        corrections
    :param min_step_deg: the smallest step reported. A changepoint whose estimated step is
        below this is dropped and its segments merged.
    :param refine: pin each changepoint to native resolution after the search, instead of
        leaving it on a ``grid`` boundary
    :param grid: aggregation bin for the changepoint search
    :param min_segment: shortest allowed gap between changepoints
    :param veer_sector_deg: width of the direction sectors the residual is normalised over
        before the changepoint search, cancelling site veer (see :func:`veer_normalised`).
        ``None`` searches the raw residual.
    :param max_transient_step_deg: the largest step that may be ironed out as wander. Above it a
        step is treated as a recalibration however the record behaves afterwards, since real ones
        are sometimes reversed later. Also the ceiling on the support-scaled threshold, so a big
        enough step is credible however little record sits either side of it.
    :param confident_segment: the span either side of a changepoint at which ``min_step_deg``
        applies as written; with less record than that the required step grows as
        ``sqrt(confident_segment / span)``, since the level is veer-limited and veer averages out
        no faster than that.
    """

    changepoints_per_year: float = 12.0
    min_step_deg: float = 3.0
    refine: bool = True
    min_changepoints: int = 3
    grid: pd.Timedelta = _DEFAULT_GRID
    min_segment: pd.Timedelta = _DEFAULT_MIN_SEGMENT
    veer_sector_deg: float | None = _DEFAULT_VEER_SECTOR_DEG
    max_transient_step_deg: float = _MAX_TRANSIENT_STEP_DEG
    confident_segment: pd.Timedelta = _DEFAULT_CONFIDENT_SEGMENT


# Minimum step attributable to a turbine when northing against reanalysis rather than a farm
# consensus. See :func:`against_reanalysis`.
REANALYSIS_MIN_STEP_DEG = 10.0
# Minimum gap between changepoints when northing against reanalysis. Reanalysis cannot resolve
# recalibrations closer together than this, so a nearer pair is treated as reference wander and
# merged. See :func:`against_reanalysis`.
REANALYSIS_MIN_SEGMENT = pd.Timedelta(days=30)
# Minimum step the first pass may act on. See :func:`anchoring_only`.
ANCHORING_MIN_STEP_DEG = 30.0
# Minimum step taken out of the residual before the veer signature is measured.
# See :func:`_confident_steps`.
VEER_SIGNATURE_MIN_STEP_DEG = 10.0

DEFAULT_NORTHING = NorthingSettings()


def anchoring_only(settings: NorthingSettings) -> NorthingSettings:
    """Return ``settings`` reduced to what the first pass is for: anchoring, not changepoint work.

    Only steps of at least :data:`ANCHORING_MIN_STEP_DEG` are acted on; finer structure is left
    to the second pass, which works against the farm consensus.
    """
    return replace(settings, min_step_deg=ANCHORING_MIN_STEP_DEG)


def against_reanalysis(settings: NorthingSettings) -> NorthingSettings:
    """Return ``settings`` made safe for northing against reanalysis rather than a farm consensus.

    Raises ``min_step_deg`` to at least :data:`REANALYSIS_MIN_STEP_DEG`, so drift in the reanalysis
    reference is not attributed to the turbines as a small step change, and ``min_segment`` to at
    least :data:`REANALYSIS_MIN_SEGMENT`, so changepoints closer together than reanalysis can resolve
    are merged rather than read as a burst of recalibrations. Everything else is unchanged.
    """
    return replace(
        settings,
        min_step_deg=max(settings.min_step_deg, REANALYSIS_MIN_STEP_DEG),
        min_segment=max(settings.min_segment, REANALYSIS_MIN_SEGMENT),
    )


def yaw_usable(
    *,
    power: npt.NDArray[np.float64],
    downtime_s: npt.NDArray[np.float64],
    reference_deg: npt.NDArray[np.float64],
    rated_power: float,
    timebase_s: float,
) -> npt.NDArray[np.bool_]:
    """Rows where a turbine's yaw reading may be used for northing.

    The turbine must be generating (above :data:`YAW_OK_POWER_FRACTION` of rated), largely
    free of downtime within the record, and have a reference direction to compare against.
    """
    return np.asarray(
        np.isfinite(reference_deg)
        & np.isfinite(power)
        & (np.nan_to_num(power, nan=-1.0) > rated_power * YAW_OK_POWER_FRACTION)
        & (np.nan_to_num(downtime_s, nan=0.0) < timebase_s / 4),
        dtype=bool,
    )


def _table(timestamps: list[pd.Timestamp], offsets: list[float]) -> pd.DataFrame:
    """Build a north table from parallel timestamp and offset lists."""
    return pd.DataFrame({TIMESTAMP_COL: pd.DatetimeIndex(timestamps), NORTH_OFFSET_COL: offsets})


def _residual(
    direction_deg: npt.NDArray[np.float64],
    *,
    reference_deg: npt.NDArray[np.float64],
    usable: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Signed circular difference direction - reference (deg), NaN where unusable."""
    residual = np.asarray(circ_diff(direction_deg, reference_deg), dtype=float)
    keep = usable & np.isfinite(direction_deg) & np.isfinite(reference_deg)
    return np.where(keep, residual, np.nan)


def _de_stepped(
    residual: npt.NDArray[np.float64], *, index: pd.DatetimeIndex, edges: list[pd.Timestamp]
) -> npt.NDArray[np.float64]:
    """Return ``residual`` with each segment's own level removed, leaving the within-segment shape.

    Measuring the veer signature needs the step structure out of the way first: a sector's level
    would otherwise average across the steps, and uneven direction sampling between segments would
    distort the very steps being looked for.
    """
    out = residual.copy()
    for begin, finish in itertools.pairwise(edges):
        rows = np.asarray((index >= begin) & (index < finish))
        values = residual[rows]
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        out[rows] = np.asarray(circ_diff(values, circ_median(finite, range_360=False)), dtype=float)
    return out


def _confident_steps(
    changepoints: list[pd.Timestamp],
    *,
    start: pd.Timestamp,
    residual: npt.NDArray[np.float64],
    index: pd.DatetimeIndex,
    min_step_deg: float = VEER_SIGNATURE_MIN_STEP_DEG,
) -> list[pd.Timestamp]:
    """Return the changepoints whose step is large enough to be a real recalibration.

    What the veer signature may be measured around. A search over a strongly veering residual
    proposes splits that are the veer itself; de-stepping those would remove the signature.
    """
    if not changepoints:
        return []
    offsets = _segment_offsets(changepoints, start=start, residual=residual, index=index)
    steps = _steps(offsets)
    return [when for when, step in zip(changepoints, steps, strict=True) if step >= min_step_deg]


def veer_normalised(
    residual: npt.NDArray[np.float64],
    *,
    reference_deg: npt.NDArray[np.float64],
    sector_deg: float,
    de_stepped: npt.NDArray[np.float64] | None = None,
    min_rows_per_sector: int = _MIN_ROWS_PER_SECTOR,
) -> npt.NDArray[np.float64]:
    """Remove each direction sector's own long-run level from the residual.

    Subtracting each sector's whole-record median leaves a genuine north offset intact, since one
    shifts every sector alike. Sectors with too little data fall back to the overall level.

    Use this for detection only -- segment offsets are estimated from the raw residual, so the
    correction stays absolute.

    :param de_stepped: the residual with a first-pass estimate of the step structure removed. The
        sector levels are measured on it rather than on ``residual``. Defaults to ``residual``.
    """
    signature = _sector_signature(
        residual if de_stepped is None else de_stepped,
        reference_deg=reference_deg,
        sector_deg=sector_deg,
        min_rows_per_sector=min_rows_per_sector,
    )
    finite = np.isfinite(residual) & np.isfinite(signature)
    out = residual.copy()
    out[finite] = np.asarray(circ_diff(residual[finite], signature[finite]), dtype=float)
    return out


def _sector_signature(
    values_deg: npt.NDArray[np.float64],
    *,
    reference_deg: npt.NDArray[np.float64],
    sector_deg: float,
    min_rows_per_sector: int = _MIN_ROWS_PER_SECTOR,
) -> npt.NDArray[np.float64]:
    """Return the long-run level of ``values_deg`` in each row's direction sector, per row.

    This is the veer signature: how far this device sits from the reference when the wind comes
    from each direction. Sectors with too little data fall back to the overall level; rows with
    no usable direction get NaN.
    """
    finite = np.isfinite(values_deg) & np.isfinite(reference_deg)
    if not finite.any():
        return np.full(len(values_deg), np.nan)
    n_sectors = max(1, int(np.ceil(360.0 / sector_deg)))
    has_reference = np.isfinite(reference_deg)
    sector = np.zeros(len(values_deg), dtype=int)
    sector[has_reference] = (np.mod(reference_deg[has_reference], 360.0) // sector_deg).astype(int) % n_sectors

    overall = float(circ_median(values_deg[finite], range_360=False))
    level = np.full(n_sectors, overall)
    for s in range(n_sectors):
        rows = finite & (sector == s)
        if int(rows.sum()) >= min_rows_per_sector:
            level[s] = float(circ_median(values_deg[rows], range_360=False))
    out = np.full(len(values_deg), np.nan)
    out[has_reference] = level[sector[has_reference]]
    return out


def _bin_levels(
    residual: npt.NDArray[np.float64], *, bins: npt.NDArray[np.int64], n_bins: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-bin circular median of the residual (deg) and the count backing it.

    The median is taken about each bin's circular mean, which is what makes it well defined
    across the 0/360 wrap. Empty bins get level 0 and weight 0, so they cost nothing.
    """
    finite = np.isfinite(residual)
    bin_of = bins[finite]
    values = residual[finite]
    counts = np.bincount(bin_of, minlength=n_bins).astype(float)
    if len(values) == 0:
        return np.zeros(n_bins), counts

    rad = np.deg2rad(values)
    sin_sum = np.bincount(bin_of, weights=np.sin(rad), minlength=n_bins)
    cos_sum = np.bincount(bin_of, weights=np.cos(rad), minlength=n_bins)
    mean_deg = np.degrees(np.arctan2(sin_sum, cos_sum))

    centred = (values - mean_deg[bin_of] + 180.0) % 360.0 - 180.0
    median_centred = pd.Series(centred).groupby(bin_of).median().reindex(range(n_bins)).to_numpy(dtype=float)
    level = (np.nan_to_num(median_centred) + mean_deg + 180.0) % 360.0 - 180.0
    return np.where(counts > 0, level, 0.0), counts


def _segment_costs(level_deg: npt.NDArray[np.float64], weight: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Cost matrix ``C[i, j]`` of treating bins ``[i, j)`` as one constant-offset segment.

    The cost is ``W - R``: total weight minus the length of the weighted resultant vector,
    which is the loss the circular mean minimises and is zero for a perfectly coherent
    segment. Prefix sums make every entry O(1), so the whole matrix is one vectorised pass.
    """
    rad = np.deg2rad(level_deg)
    cum_w = np.concatenate([[0.0], np.cumsum(weight)])
    cum_cos = np.concatenate([[0.0], np.cumsum(weight * np.cos(rad))])
    cum_sin = np.concatenate([[0.0], np.cumsum(weight * np.sin(rad))])
    total_w = cum_w[None, :] - cum_w[:, None]
    resultant = np.hypot(cum_cos[None, :] - cum_cos[:, None], cum_sin[None, :] - cum_sin[:, None])
    return np.asarray(total_w - resultant)


def _best_breakpoints(cost: npt.NDArray[np.float64], *, max_k: int, min_span: int, penalty: float) -> list[int]:
    """Bin indices of the optimal changepoints, by exact dynamic programming.

    ``best[k][j]`` is the least cost of splitting bins ``[0, j)`` into ``k + 1`` segments;
    each ``k`` is solved from ``k - 1`` in one vectorised minimisation. The reported ``k`` is
    the one minimising ``best[k][n] + penalty * k``.
    """
    n = cost.shape[0] - 1
    span = np.arange(n + 1)[None, :] - np.arange(n + 1)[:, None]
    feasible = np.where(span >= min_span, cost, np.inf)

    best = np.full((max_k + 1, n + 1), np.inf)
    came_from = np.zeros((max_k + 1, n + 1), dtype=int)
    best[0] = feasible[0]
    for k in range(1, max_k + 1):
        total = best[k - 1][:, None] + feasible
        came_from[k] = np.argmin(total, axis=0)
        best[k] = total[came_from[k], np.arange(n + 1)]

    penalised = best[:, n] + penalty * np.arange(max_k + 1)
    if not np.isfinite(penalised).any():
        return []
    k = int(np.nanargmin(np.where(np.isfinite(penalised), penalised, np.nan)))

    breakpoints: list[int] = []
    j = n
    while k > 0:
        i = int(came_from[k][j])
        breakpoints.append(i)
        j, k = i, k - 1
    return sorted(breakpoints)


def _native_prefix_sums(
    residual: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Positions of finite residual rows and prefix sums of their cos/sin, for local scoring."""
    finite = np.flatnonzero(np.isfinite(residual))
    rad = np.deg2rad(residual[finite])
    cum_cos = np.concatenate([[0.0], np.cumsum(np.cos(rad))])
    cum_sin = np.concatenate([[0.0], np.cumsum(np.sin(rad))])
    return finite, cum_cos, cum_sin


def _local_cost(
    lo: npt.NDArray[np.int64] | int,
    hi: npt.NDArray[np.int64] | int,
    *,
    cum_cos: npt.NDArray[np.float64],
    cum_sin: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """``W - R`` over finite-row positions ``[lo, hi)``; the native-resolution segment cost."""
    weight = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    resultant = np.hypot(cum_cos[hi] - cum_cos[lo], cum_sin[hi] - cum_sin[lo])
    return np.asarray(weight - resultant)


def _refine(
    changepoints: list[pd.Timestamp],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    finite_times: pd.DatetimeIndex,
    cum_cos: npt.NDArray[np.float64],
    cum_sin: npt.NDArray[np.float64],
    settings: NorthingSettings,
) -> list[pd.Timestamp]:
    """Move each changepoint to the native timestamp that best splits its two neighbours.

    Searches within one grid bin either side, while keeping ``min_segment`` clear of the
    neighbouring changepoints.
    """
    refined = list(changepoints)
    # integer nanoseconds throughout, so tz-aware and tz-naive inputs compare alike
    times = finite_times.asi8
    for position, changepoint in enumerate(refined):
        previous = refined[position - 1] if position > 0 else start
        following = refined[position + 1] if position + 1 < len(refined) else end
        earliest = max(changepoint - settings.grid, previous + settings.min_segment)
        latest = min(changepoint + settings.grid, following - settings.min_segment)
        if earliest >= latest:
            continue
        span_lo = int(np.searchsorted(times, previous.value))
        span_hi = int(np.searchsorted(times, following.value))
        first = int(np.searchsorted(times, earliest.value))
        last = int(np.searchsorted(times, latest.value))
        if last <= first or span_hi - span_lo < _MIN_ROWS_TO_SPLIT:
            continue
        candidates = np.arange(max(first, span_lo + 1), min(last, span_hi - 1) + 1)
        if len(candidates) == 0:
            continue
        totals = _local_cost(span_lo, candidates, cum_cos=cum_cos, cum_sin=cum_sin) + _local_cost(
            candidates, span_hi, cum_cos=cum_cos, cum_sin=cum_sin
        )
        refined[position] = finite_times[int(candidates[int(np.argmin(totals))])]
    return refined


def _segment_offsets(
    changepoints: list[pd.Timestamp],
    *,
    start: pd.Timestamp,
    residual: npt.NDArray[np.float64],
    index: pd.DatetimeIndex,
) -> list[float]:
    """Return each segment's correcting offset: minus the circular median of its residual."""
    edges = [start, *changepoints, index.max() + pd.Timedelta(nanoseconds=1)]
    offsets = []
    for begin, finish in itertools.pairwise(edges):
        rows = residual[(index >= begin) & (index < finish)]
        rows = rows[np.isfinite(rows)]
        median = circ_median(rows, range_360=False) if len(rows) else 0.0
        offsets.append(0.0 if not np.isfinite(median) else -float(median))
    return offsets


def _weighted_level(offsets: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]) -> float:
    """Duration-weighted circular mean of a run of segment offsets (deg)."""
    rad = np.deg2rad(offsets)
    return float(np.degrees(np.arctan2(np.sum(weights * np.sin(rad)), np.sum(weights * np.cos(rad)))))


def _persistence(offsets: list[float], *, durations: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """How much each changepoint moves the long-run level, in degrees.

    A recalibration moves the level and leaves it moved. An excursion -- the level wandering away
    and back -- moves it only in between, so the record either side of any one of its changepoints
    sits at the same place.
    """
    values = np.asarray(offsets, dtype=float)
    return np.array(
        [
            abs(
                float(
                    circ_diff(
                        _weighted_level(values[k + 1 :], durations[k + 1 :]),
                        _weighted_level(values[: k + 1], durations[: k + 1]),
                    )
                )
            )
            for k in range(len(values) - 1)
        ]
    )


def _prune_while(
    changepoints: list[pd.Timestamp],
    offsets: list[float],
    *,
    start: pd.Timestamp,
    residual: npt.NDArray[np.float64],
    index: pd.DatetimeIndex,
    worst: Callable[[list[pd.Timestamp], list[float]], int | None],
) -> tuple[list[pd.Timestamp], list[float]]:
    """Drop whichever changepoint ``worst`` names, re-estimating offsets, until it names none.

    Offsets must be re-estimated after every merge: joining two segments changes the level of the
    result, which can in turn change which of the survivors looks weakest.
    """
    while changepoints:
        drop = worst(changepoints, offsets)
        if drop is None:
            return changepoints, offsets
        changepoints = [c for i, c in enumerate(changepoints) if i != drop]
        offsets = _segment_offsets(changepoints, start=start, residual=residual, index=index)
    return changepoints, offsets


def _steps(offsets: list[float]) -> npt.NDArray[np.float64]:
    """Return the size of the step at each changepoint, in degrees."""
    return np.abs(circ_diff(np.array(offsets[1:]), np.array(offsets[:-1])))


def _worst_transient(
    changepoints: list[pd.Timestamp],
    offsets: list[float],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_step_deg: float,
    max_transient_step_deg: float,
) -> int | None:
    """Return the least persistent small changepoint -- site veer wandering away and back.

    Steps larger than ``max_transient_step_deg`` are never named.
    """
    edges = [start, *changepoints, end]
    durations = np.array([max((b - a).total_seconds(), 1.0) for a, b in itertools.pairwise(edges)], dtype=float)
    persistence = _persistence(offsets, durations=durations)
    candidates = np.flatnonzero((_steps(offsets) < max_transient_step_deg) & (persistence < min_step_deg))
    if len(candidates) == 0:
        return None
    return int(candidates[np.argmin(persistence[candidates])])


def _worst_unsupported(
    changepoints: list[pd.Timestamp],
    offsets: list[float],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_step_deg: float,
    max_transient_step_deg: float,
    confident_segment: pd.Timedelta,
) -> int | None:
    """Return the changepoint whose step falls furthest short of what its record can support.

    A step smaller than ``min_step_deg`` is never reported. Near the start or end of a record --
    or squeezed between two other changepoints -- more is required.
    """
    required = _required_step(
        changepoints,
        start=start,
        end=end,
        min_step_deg=min_step_deg,
        max_transient_step_deg=max_transient_step_deg,
        confident_segment=confident_segment,
    )
    shortfall = required - _steps(offsets)
    weakest = int(np.argmax(shortfall))
    return weakest if shortfall[weakest] > 0 else None


def _required_step(
    changepoints: list[pd.Timestamp],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_step_deg: float,
    max_transient_step_deg: float,
    confident_segment: pd.Timedelta,
) -> npt.NDArray[np.float64]:
    """Return the step size each changepoint must reach, given the record supporting it.

    A segment's level is limited by site veer rather than by sampling noise, and veer averages out
    no faster than ``1/sqrt(span)``. So with less than ``confident_segment`` either side the
    required step grows accordingly, capped at ``max_transient_step_deg`` -- above which a step is
    credible however little record sits around it. The cap never falls below ``min_step_deg``:
    ``np.clip`` with its bounds inverted returns the upper one, which let a pass asking for 30 deg
    steps accept 10 deg ones.
    """
    edges = [start, *changepoints, end]
    spans = np.array([max((b - a) / confident_segment, 1e-9) for a, b in itertools.pairwise(edges)])
    support = np.minimum(spans[:-1], spans[1:])
    ceiling = max(min_step_deg, max_transient_step_deg)
    return np.clip(min_step_deg / np.sqrt(np.minimum(support, 1.0)), min_step_deg, ceiling)


def estimate_north_table(
    index: pd.DatetimeIndex,
    direction_deg: npt.NDArray[np.float64],
    *,
    reference_deg: npt.NDArray[np.float64],
    usable: npt.NDArray[np.bool_],
    settings: NorthingSettings = DEFAULT_NORTHING,
) -> pd.DataFrame:
    """Estimate a direction signal's north offsets over time.

    Compares ``direction_deg`` with ``reference_deg`` over the rows ``usable`` allows, finds
    the step changes in their circular difference, and returns the offset that corrects each
    resulting period. Offsets are absolute: adding one to the raw signal norths it.

    :param index: timestamps of every array; need not be sorted
    :param direction_deg: the signal to north, in degrees
    :param reference_deg: the direction to north it against (reanalysis, or a farm consensus)
    :param usable: rows whose comparison is meaningful -- see :func:`yaw_usable`. Also the
        place to exclude periods when the direction is deliberately offset, such as a turbine
        steering its wake.
    :param settings: how the search is bounded; the default suits a farm record
    :return: columns ``timestamp`` and ``north_offset``, one row per period, the first row at
        the start of ``index``. Always at least one row; all-zero when nothing is usable.
    """
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        msg = "cannot estimate a north table from an empty index"
        raise ValueError(msg)
    direction = np.asarray(direction_deg, dtype=float)
    reference = np.asarray(reference_deg, dtype=float)
    ok = np.asarray(usable, dtype=bool)
    if not len(direction) == len(reference) == len(ok) == len(index):
        msg = (
            f"index, direction_deg, reference_deg and usable must be the same length; got "
            f"{len(index)}, {len(direction)}, {len(reference)}, {len(ok)}"
        )
        raise ValueError(msg)

    if not index.is_monotonic_increasing:
        order = np.argsort(index.to_numpy())
        index, direction, reference, ok = index[order], direction[order], reference[order], ok[order]

    residual = _residual(direction, reference_deg=reference, usable=ok)
    start = index.min()
    if not np.isfinite(residual).any():
        logger.warning("no usable rows to north against; returning a zero offset")
        return _table([start], [0.0])

    bins = ((index - start) // settings.grid).to_numpy().astype(np.int64)
    n_bins = int(bins.max()) + 1
    if n_bins > _BIN_COUNT_WARN:
        logger.warning("northing over %d %s bins; consider a coarser grid", n_bins, settings.grid)
    years = (index.max() - start) / pd.Timedelta(days=365.25)
    max_k = max(settings.min_changepoints, math.ceil(settings.changepoints_per_year * max(years, 0.0)))
    min_span = max(1, math.ceil(settings.min_segment / settings.grid))
    end = index.max() + pd.Timedelta(nanoseconds=1)

    def detect(searched: npt.NDArray[np.float64]) -> list[pd.Timestamp]:
        """Return one residual's changepoint timestamps: aggregate, solve, then refine."""
        level, weight = _bin_levels(searched, bins=bins, n_bins=n_bins)
        if max_k <= 0 or n_bins <= min_span:
            return []
        occupied = int((weight > 0).sum())
        typical = float(weight.sum()) / max(occupied, 1)
        # A changepoint must pay for itself: the cost drop a ``min_step_deg`` step sustained
        # over ``min_segment`` of typical-density data would produce.
        penalty = typical * min_span * (1.0 - math.cos(math.radians(settings.min_step_deg) / 2.0))
        breaks = _best_breakpoints(_segment_costs(level, weight), max_k=max_k, min_span=min_span, penalty=penalty)
        found = [start + b * settings.grid for b in breaks if b > 0]
        if found and settings.refine:
            finite, cum_cos, cum_sin = _native_prefix_sums(searched)
            found = _refine(
                found,
                start=start,
                end=end,
                finite_times=index[finite],
                cum_cos=cum_cos,
                cum_sin=cum_sin,
                settings=settings,
            )
        return found

    if settings.veer_sector_deg is None:
        changepoints = detect(residual)
    else:
        sector_deg = settings.veer_sector_deg

        def normalised(de_stepped: npt.NDArray[np.float64] | None) -> npt.NDArray[np.float64]:
            return veer_normalised(
                residual,
                reference_deg=reference,
                sector_deg=sector_deg,
                de_stepped=de_stepped,
            )

        # Search the veer-normalised residual, measuring the sector signature twice: first assuming
        # no step structure, then around only the steps that search was confident of. Offsets come
        # from the raw residual either way, so the correction stays absolute.
        provisional = detect(normalised(None))
        confident = _confident_steps(provisional, start=start, residual=residual, index=index)
        changepoints = (
            detect(normalised(_de_stepped(residual, index=index, edges=[start, *confident, end])))
            if confident
            else provisional
        )

    offsets = _segment_offsets(changepoints, start=start, residual=residual, index=index)
    # First iron out excursions, then drop what the record cannot support. Order matters: a step
    # only looks unsupported once the excursion around it has gone.
    changepoints, offsets = _prune_while(
        changepoints,
        offsets,
        start=start,
        residual=residual,
        index=index,
        worst=partial(
            _worst_transient,
            start=start,
            end=end,
            min_step_deg=settings.min_step_deg,
            max_transient_step_deg=settings.max_transient_step_deg,
        ),
    )
    changepoints, offsets = _prune_while(
        changepoints,
        offsets,
        start=start,
        residual=residual,
        index=index,
        worst=partial(
            _worst_unsupported,
            start=start,
            end=end,
            min_step_deg=settings.min_step_deg,
            max_transient_step_deg=settings.max_transient_step_deg,
            confident_segment=settings.confident_segment,
        ),
    )
    return _table([start, *changepoints], offsets)


def apply_north_table(
    index: pd.DatetimeIndex,
    direction_deg: npt.NDArray[np.float64],
    *,
    north_table: pd.DataFrame,
) -> npt.NDArray[np.float64]:
    """North a direction signal: ``(direction + offset) % 360``, offsets step-applied.

    Each row of ``north_table`` holds from its timestamp until the next; rows before the first
    timestamp take the first offset. Takes a single array, so one table can north several
    fields of the same device. NaNs are preserved.
    """
    index = pd.DatetimeIndex(index)
    direction = np.asarray(direction_deg, dtype=float)
    table = north_table.sort_values(TIMESTAMP_COL)
    edges = pd.DatetimeIndex(table[TIMESTAMP_COL]).to_numpy()
    offsets = table[NORTH_OFFSET_COL].to_numpy(dtype=float)
    which = np.clip(np.searchsorted(edges, index.to_numpy(), side="right") - 1, 0, len(offsets) - 1)
    return np.where(np.isfinite(direction), (direction + offsets[which]) % 360.0, np.nan)


def _median_across(stack: npt.NDArray[np.float64], *, enough: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """Per-timestamp circular median down a devices x time stack, NaN where ``enough`` is False."""
    farm = np.full(stack.shape[1], np.nan)
    if not enough.any():
        return farm
    columns = stack[:, enough]
    rad = np.deg2rad(columns)
    counts = np.isfinite(columns).sum(axis=0)
    mean = np.degrees(
        np.arctan2(
            np.nansum(np.sin(rad), axis=0) / counts,
            np.nansum(np.cos(rad), axis=0) / counts,
        )
    )
    centred = (columns - mean + 180.0) % 360.0 - 180.0
    # every retained column has at least ``min_devices`` finite entries, so no all-NaN slice
    farm[enough] = (np.nanmedian(centred, axis=0) + mean) % 360.0
    return farm


def write_north_table_yaml(tables: Mapping[str, pd.DataFrame], *, path: Path) -> None:
    """Write per-device north tables as the YAML list ``north_offsets`` and v0 both read.

    The format matches v0's ``optimized_northing_corrections.yaml``, so the file can be hand
    edited and supplied back as a prior. Offsets are wrapped into [-180, 180) on the way out.

    :param tables: one absolute north table per device
    :param path: file to write
    """
    lines = [
        f"    - ['{device}', {pd.Timestamp(row.timestamp).strftime('%Y-%m-%d %H:%M:%S')}, "
        f"{(row.north_offset + 180.0) % 360.0 - 180.0}]"
        for device in sorted(tables)
        for row in tables[device].itertuples()
    ]
    path.write_text("\n".join(lines) + "\n")


_MIN_LATITUDE_DEG = -90.0
_MAX_LATITUDE_DEG = 90.0


def _usable_coordinate(point: tuple[float, float]) -> bool:
    """Whether ``(latitude, longitude)`` is finite with a latitude in ``[-90, 90]``."""
    latitude, longitude = point
    return math.isfinite(latitude) and math.isfinite(longitude) and _MIN_LATITUDE_DEG <= latitude <= _MAX_LATITUDE_DEG


def nearest_neighbours(coordinates: Mapping[str, tuple[float, float]], *, k: int) -> dict[str, tuple[str, ...]]:
    """Map each device to its ``k`` nearest others by geodesic distance.

    Distances are the WGS84 ellipsoidal geodesic (:func:`wind_up.geodesy.geodesic_matrices`) over
    each device's ``(latitude, longitude)``. ``k`` is capped at the number of other devices, so a
    farm smaller than ``k + 1`` simply lists everyone else. A device is never its own neighbour.

    ``k`` must be positive, and every coordinate must be finite with a latitude in ``[-90, 90]``:
    an out-of-range or non-finite point yields a non-finite distance that ``argsort`` would still
    order, so the nearest set is rejected rather than silently arbitrary.

    This is what :func:`north_farm` uses to turn turbine positions into each device's pass-2
    reference consensus.
    """
    if k < 1:
        msg = f"k must be a positive number of neighbours, got {k}"
        raise ValueError(msg)
    devices = sorted(coordinates)
    bad = sorted(d for d in devices if not _usable_coordinate(coordinates[d]))
    if bad:
        msg = f"coordinates for device(s) {bad} are not a finite (latitude in [-90, 90], longitude) pair"
        raise ValueError(msg)
    latitudes = [coordinates[d][0] for d in devices]
    longitudes = [coordinates[d][1] for d in devices]
    distance_m, _ = geodesic_matrices(latitudes=latitudes, longitudes=longitudes)
    limit = min(k, len(devices) - 1)
    out: dict[str, tuple[str, ...]] = {}
    for i, name in enumerate(devices):
        order = [j for j in np.argsort(distance_m[i], kind="stable") if j != i]
        out[name] = tuple(devices[j] for j in order[:limit])
    return out


def _neighbours_from_layout(layout: Layout, *, devices: list[str], k: int) -> dict[str, tuple[str, ...]]:
    """Map each device to its ``k`` nearest of the other ``devices`` by the layout's geodesic distance.

    Reuses ``layout.distance_m`` rather than recomputing. External turbines in the layout that are
    not being northed take no part. ``k`` is capped at the number of other devices, so a device is
    never its own neighbour and a small farm simply lists everyone else.
    """
    rows = {name: layout.index_of(name) for name in devices}
    limit = min(k, len(devices) - 1)
    out: dict[str, tuple[str, ...]] = {}
    for name in devices:
        ranked = sorted((float(layout.distance_m[rows[name], rows[o]]), o) for o in devices if o != name)
        out[name] = tuple(o for _, o in ranked[:limit])
    return out


def _validate_north_farm_inputs(
    devices: list[str],
    *,
    usable: Mapping[str, npt.NDArray[np.bool_]],
    power: Mapping[str, npt.NDArray[np.float64]] | None,
    layout: Layout | None,
) -> None:
    """Check every device has a usable mask, a power entry when ``power`` is given, and a layout row."""
    missing = sorted(set(devices) - set(usable))
    if missing:
        msg = f"usable is missing masks for device(s) {missing}"
        raise ValueError(msg)
    if power is not None:
        missing_power = sorted(set(devices) - set(power))
        if missing_power:
            msg = f"power is missing an entry for device(s) {missing_power}"
            raise ValueError(msg)
    if layout is not None:
        known = {n for n in layout.frame[NAME_COL].to_numpy() if n is not None}
        unknown = sorted(set(devices) - known)
        if unknown:
            msg = (
                f"layout has no row for device(s) {unknown}. "
                "Pass layout=None explicitly to north against the whole-farm consensus instead."
            )
            raise ValueError(msg)


def _pass_two_reference(
    layout: Layout | None, *, devices: list[str], neighbours: int, min_devices: int
) -> dict[str, tuple[str, ...]] | None:
    """Return each device's pass-2 neighbour set, or ``None`` for the whole-farm consensus.

    With a layout, a device is northed against its ``neighbours`` nearest turbines -- but only where
    the layout leaves every device at least ``min_devices`` of them; a farm too small for that falls
    back to the whole-farm consensus.
    """
    if layout is None:
        return None
    candidate = _neighbours_from_layout(layout, devices=devices, k=neighbours)
    thinnest = min(len(candidate[d]) for d in devices)
    if thinnest < min_devices:
        logger.warning(
            "layout gives each device only %d neighbour(s), below min_devices_for_farm_reference=%d; "
            "northing against the whole-farm consensus instead",
            thinnest,
            min_devices,
        )
        return None
    return candidate


def _farm_quorum(n_devices: int, *, floor: int) -> int:
    """Return how many devices must report for their median to stand for the farm's consensus."""
    return max(floor, n_devices // 2 + 1)


def _wake_nadir_pass(
    tables: dict[str, pd.DataFrame],
    *,
    layout: Layout | None,
    index: pd.DatetimeIndex,
    direction_deg: Mapping[str, npt.NDArray[np.float64]],
    power: Mapping[str, npt.NDArray[np.float64]] | None,
    wind_speed: Mapping[str, npt.NDArray[np.float64]] | None,
    usable: Mapping[str, npt.NDArray[np.bool_]],
    nadir_out: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Add pass 4's wake-nadir correction to each device's table, or return ``tables`` unchanged.

    Runs only with both a ``layout`` and ``power``. The correction is one absolute number per
    turbine; it shifts every offset in that turbine's table and never touches which rows are valid.
    When ``nadir_out`` is given it is filled with the per-device correction, for callers that plot
    or log the nudge; it is left untouched when pass 4 does not run.
    """
    if layout is None or power is None:
        return tables
    northed = {name: apply_north_table(index, direction_deg[name], north_table=tables[name]) for name in tables}
    deltas = wake_nadir_offsets(
        layout, index=index, northed_direction=northed, power=power, wind_speed=wind_speed, usable=usable
    )
    if nadir_out is not None:
        nadir_out.update(deltas)
    return {
        name: table.assign(**{NORTH_OFFSET_COL: table[NORTH_OFFSET_COL] + deltas[name]}) if deltas.get(name) else table
        for name, table in tables.items()
    }


def _farm_direction(
    northed: Mapping[str, npt.NDArray[np.float64]],
    *,
    usable: Mapping[str, npt.NDArray[np.bool_]],
    min_devices: int,
) -> npt.NDArray[np.float64]:
    """Per-timestamp circular median of the devices' northed directions, NaN where too few report.

    ``min_devices`` is a quorum, not a fixed floor: see :func:`north_farm`.
    """
    stack = np.vstack(
        [np.where(usable[name] & np.isfinite(values), values, np.nan) for name, values in northed.items()]
    )
    enough = np.isfinite(stack).sum(axis=0) >= min_devices
    return _median_across(stack, enough=enough)


def _consensus_references(
    northed: Mapping[str, npt.NDArray[np.float64]],
    *,
    usable: Mapping[str, npt.NDArray[np.bool_]],
    quorum: int,
    reference_neighbours: Mapping[str, Sequence[str]] | None,
    min_devices: int,
) -> dict[str, npt.NDArray[np.float64]]:
    """Return the direction each device is northed against in pass 2.

    Without ``reference_neighbours`` every device shares the one whole-farm consensus -- the
    historical behaviour, and the object is shared so the result is identical to computing it once.
    With it, each device is northed against the circular-median consensus of *its own* listed
    neighbours (itself excluded), so a far or miscalibrated turbine on the other side of the farm
    cannot pull its reference. A neighbour set forms a consensus by the same quorum rule as the
    whole farm: a strict majority of the set, floored at ``min_devices``.
    """
    if reference_neighbours is None:
        farm = _farm_direction(northed, usable=usable, min_devices=quorum)
        return dict.fromkeys(northed, farm)
    references: dict[str, npt.NDArray[np.float64]] = {}
    for name in northed:
        neighbours = [n for n in reference_neighbours[name] if n != name]
        references[name] = _farm_direction(
            {n: northed[n] for n in neighbours},
            usable={n: usable[n] for n in neighbours},
            min_devices=_farm_quorum(len(neighbours), floor=min_devices),
        )
    return references


def north_farm(
    index: pd.DatetimeIndex,
    *,
    direction_deg: Mapping[str, npt.NDArray[np.float64]],
    usable: Mapping[str, npt.NDArray[np.bool_]],
    reanalysis_deg: npt.NDArray[np.float64],
    layout: Layout | None,
    power: Mapping[str, npt.NDArray[np.float64]] | None = None,
    wind_speed: Mapping[str, npt.NDArray[np.float64]] | None = None,
    neighbours: int = 4,
    settings: NorthingSettings = DEFAULT_NORTHING,
    min_devices_for_farm_reference: int = 3,
    nadir_out: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """North a whole farm, returning one absolute table per device.

    Pass 1 is a constant bulk alignment: each device gets a single offset nulling its whole-record
    direction to ``reanalysis_deg``, with no changepoints. Pass 2 then builds a farm consensus
    direction from those aligned signals and norths each device's raw signal to it, finding every
    changepoint against that consensus. Pass 1 fixes the farm in absolute terms; pass 2 is the more
    precise, and does all the changepoint work.

    Pass 1 attributes no changepoints on purpose: reanalysis is short-term unreliable, so a pass-1
    changepoint lets that noise leak into the very consensus pass 2 trusts (an unusual weather spell
    moves every device's residual against reanalysis together, and correcting it writes the
    excursion into the consensus). A constant anchor cannot do that.

    Every device's arrays are positional on the shared ``index``, which is what lets the farm
    consensus be taken across devices at each timestamp.

    :param direction_deg: device name to its raw direction signal
    :param usable: device name to the rows usable for northing it
    :param reanalysis_deg: the absolute direction reference, on ``index``
    :param layout: the farm :class:`~wind_up.layout.Layout`. Pass 2 then norths each device against
        the consensus of its ``neighbours`` nearest turbines (by the layout's geodesic distance),
        which keeps a far or miscalibrated turbine out of its reference -- what a large, heterogeneous
        farm needs, since a turbine only shares wind with its neighbours. Every device in
        ``direction_deg`` must resolve to a layout row; external turbines in the layout are ignored.
        Pass ``layout=None`` -- explicitly -- to fall back to the one whole-farm consensus; that lets
        a distant or miscalibrated turbine into every device's reference, so choose it only when no
        positions are available.
    :param power: device name to its power signal on ``index``, for the pass-4 wake-nadir nudge.
        With a ``layout``, pass 4 adds one absolute correction per turbine on top of its changepoint
        table, from where each turbine's wake lands on its downstream neighbours. ``None`` (or no
        ``layout``) disables pass 4.
    :param wind_speed: device name to its nacelle wind speed on ``index``. When given, pass 4
        combines it with power as a second, independent deficit signal; otherwise power is used alone.
    :param neighbours: how many nearest turbines form each device's pass-2 consensus when ``layout``
        is given; capped at the farm size, and must leave every device at least
        ``min_devices_for_farm_reference`` neighbours or the call raises.
    :param min_devices_for_farm_reference: the floor on how many devices must report at a
        timestamp for the consensus to be defined there, and the minimum farm size. The effective
        requirement is the larger of this and a strict majority of the farm.
    :param nadir_out: when given, filled with each device's pass-4 correction (deg), for callers
        that plot or log the nudge; left untouched when pass 4 does not run.
    """
    devices = sorted(direction_deg)
    _validate_north_farm_inputs(devices, usable=usable, power=power, layout=layout)

    finite_reference = np.isfinite(np.asarray(reanalysis_deg, dtype=float))
    anchorable = {d: int((np.asarray(usable[d], dtype=bool) & finite_reference).sum()) for d in devices}
    if not any(anchorable.values()):
        msg = (
            "no device has a usable row where reanalysis_deg is finite, so pass 1 cannot anchor the farm. "
            "Pass 2 would still return plausible relative offsets, but a farm that is uniformly wrong "
            "looks self-consistent, so the result would be unanchored. Check that reanalysis_deg covers "
            "index and is not all NaN."
        )
        raise ValueError(msg)
    thin = sorted(d for d, n in anchorable.items() if n == 0)
    if len(devices) - len(thin) < min_devices_for_farm_reference:
        logger.warning(
            "only %d of %d devices have a usable row anchored to reanalysis (%s have none); the absolute "
            "anchor rests on few devices",
            len(devices) - len(thin),
            len(devices),
            thin,
        )

    # Pass 1 is a constant bulk alignment (no changepoints); pass 2's farm consensus does all the
    # changepoint work, at the caller's chosen threshold.
    anchoring = replace(settings, changepoints_per_year=0.0, min_changepoints=0)
    first_pass = {
        name: estimate_north_table(
            index,
            direction_deg[name],
            reference_deg=reanalysis_deg,
            usable=usable[name],
            settings=anchoring,
        )
        for name in devices
    }
    northed = {name: apply_north_table(index, direction_deg[name], north_table=first_pass[name]) for name in devices}
    wake_nadir = partial(
        _wake_nadir_pass,
        layout=layout,
        index=index,
        direction_deg=direction_deg,
        power=power,
        wind_speed=wind_speed,
        usable=usable,
        nadir_out=nadir_out,
    )

    # Whole-farm switch: below the floor there is no farm consensus to form, so pass 3 norths each
    # device against reanalysis directly. Unlike the pass-1 anchor it attributes changepoints, but at
    # a coarser step floor, since reanalysis drift must not be read as a small turbine step. Pass 1 is
    # pass 3's no-changepoint special case.
    if len(devices) < min_devices_for_farm_reference:
        logger.warning(
            "farm of %d device(s) is below min_devices_for_farm_reference=%d; northing against "
            "reanalysis with changepoints (pass 3)",
            len(devices),
            min_devices_for_farm_reference,
        )
        reanalysis_settings = against_reanalysis(settings)
        pass_three = {
            name: estimate_north_table(
                index,
                direction_deg[name],
                reference_deg=reanalysis_deg,
                usable=usable[name],
                settings=reanalysis_settings,
            )
            for name in devices
        }
        return wake_nadir(pass_three)

    reference_neighbours = _pass_two_reference(
        layout, devices=devices, neighbours=neighbours, min_devices=min_devices_for_farm_reference
    )
    quorum = _farm_quorum(len(devices), floor=min_devices_for_farm_reference)
    references = _consensus_references(
        northed,
        usable=usable,
        quorum=quorum,
        reference_neighbours=reference_neighbours,
        min_devices=min_devices_for_farm_reference,
    )
    if not any(np.isfinite(reference).any() for reference in references.values()):
        logger.warning("farm reference is empty; keeping the reanalysis-only north tables")
        return wake_nadir(first_pass)

    tables = {}
    for name in devices:
        reference = references[name]
        # The reference must be finite where this device can be northed against it, not merely finite
        # somewhere: pass 2's residual is taken over usable & finite(direction) & finite(reference)
        # (see _residual). With no such overlap -- e.g. a device and its neighbours reporting in
        # disjoint periods -- estimate_north_table returns a zero offset, so keep the pass-1 anchor.
        overlap = (
            np.asarray(usable[name], dtype=bool)
            & np.isfinite(np.asarray(direction_deg[name], dtype=float))
            & np.isfinite(reference)
        )
        if not overlap.any():
            logger.warning("no usable farm reference for device %s; keeping its reanalysis anchor", name)
            tables[name] = first_pass[name]
        else:
            tables[name] = estimate_north_table(
                index, direction_deg[name], reference_deg=reference, usable=usable[name], settings=settings
            )
    return wake_nadir(tables)
