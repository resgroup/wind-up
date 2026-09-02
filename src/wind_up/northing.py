"""Estimate and apply north-calibration corrections for a direction signal.

A turbine's reported yaw direction carries an unknown offset from true north that changes in
**steps** when the sensor is recalibrated or replaced. :func:`estimate_north_table` recovers
those steps by comparing the signal with a reference direction, and returns a table of
``(timestamp, north_offset)`` that :func:`apply_north_table` steps onto the raw signal.

Offsets are always **absolute** -- relative to the raw field, never to an already-corrected
one -- so a supplied table and an estimated one are directly comparable and repeated runs
compose.

:func:`north_farm` runs the two-pass farm workflow: north every device to reanalysis, build a
farm consensus direction from the results, then north every device to that. The second pass is
the more precise one; the first is what anchors the farm in absolute terms, without which a
farm that is uniformly wrong looks perfectly self-consistent.

The estimator works on any direction field. Only :func:`yaw_usable` is turbine-specific -- a
mast or LiDAR needs a wind-speed-based mask instead, which is not wired up yet.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from wind_up.circular_math import circ_diff, circ_median

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

logger = logging.getLogger(__name__)

TIMESTAMP_COL = "timestamp"
NORTH_OFFSET_COL = "north_offset"

# A turbine's yaw reading is only meaningful when it is generating; below this fraction of
# rated power it often points away from the wind.
YAW_OK_POWER_FRACTION = 0.05
# Above this many aggregation bins the cost matrix gets large (it is O(bins^2)); warn rather
# than fail, since the result is still correct.
_BIN_COUNT_WARN = 3000
# Search-shape defaults, as constants so the dataclass defaults are not function calls.
_DEFAULT_GRID = pd.Timedelta(days=1)
_DEFAULT_MIN_SEGMENT = pd.Timedelta(days=7)
# A segment needs a row either side of a candidate split for the split to mean anything.
_MIN_ROWS_TO_SPLIT = 2
# Direction sectors the residual is normalised over before the changepoint search.
_DEFAULT_VEER_SECTOR_DEG = 30.0
# A sector with fewer usable rows than this has no trustworthy level of its own.
_MIN_ROWS_PER_SECTOR = 50
# A step larger than this is a recalibration whatever else the record does, so it is never ironed
# out as wander -- real ones do sometimes reverse later.
_MAX_TRANSIENT_STEP_DEG = 10.0


# A consensus needs a strict majority of the farm reporting. Below that the median is over an
# unrepresentative few, whose own veer moves the reference rather than the farm's.
def _farm_quorum(n_devices: int, *, floor: int) -> int:
    """Return how many devices must report for their median to stand for the farm's consensus."""
    return max(floor, n_devices // 2 + 1)


# The span either side of a changepoint at which ``min_step_deg`` applies unmodified. With less
# record than this the level is veer-limited rather than sample-limited, so a bigger step is
# needed to tell a recalibration from the wander.
_DEFAULT_CONFIDENT_SEGMENT = pd.Timedelta(days=90)


@dataclass(frozen=True)
class NorthingSettings:
    """How the changepoint search is bounded, in physical units.

    There is one setting, not a menu: a low-effort tier was measured and dropped, because the
    search is a small part of the runtime (a whole farm-year differs by ~2 seconds) and a smaller
    changepoint budget cost real detections. Construct one of these only to tune deliberately.

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


# Reanalysis is a modelled, drift-prone direction: a shift in it looks exactly like a shift in
# every turbine at once, so only large steps may be attributed to a turbine against it. A farm
# consensus shares that common-mode error, so against one a residual step really is the
# turbine's. See :func:`against_reanalysis`.
REANALYSIS_MIN_STEP_DEG = 10.0
# The first pass may only act on a *gross* recalibration -- one large enough that leaving it
# uncorrected would drag the farm consensus the second pass depends on. Reanalysis' own
# direction-dependent bias moves every turbine together by up to ~20 degrees during a spell of
# unusual wind, so the bar sits above that.
ANCHORING_MIN_STEP_DEG = 30.0

DEFAULT_NORTHING = NorthingSettings()


def anchoring_only(settings: NorthingSettings) -> NorthingSettings:
    """Return ``settings`` reduced to what the first pass is for: anchoring, not changepoint work.

    The first pass exists to fix the farm in absolute terms against reanalysis. Reanalysis has its
    own direction-dependent bias, so a spell of unusual wind moves every turbine's residual
    against it together, by tens of degrees -- and acting on that writes the artefact into the
    corrected directions and from there into the farm consensus the second pass trusts.

    So only a **gross** step is acted on here (:data:`ANCHORING_MIN_STEP_DEG`): large enough that
    leaving it would drag the consensus, and larger than reanalysis' own excursions. Everything
    finer is left to the second pass, which works against the farm consensus and estimates from
    the **raw** direction, so nothing is lost by deferring it.
    """
    return replace(against_reanalysis(settings), min_step_deg=ANCHORING_MIN_STEP_DEG)


def against_reanalysis(settings: NorthingSettings) -> NorthingSettings:
    """Return ``settings`` made safe for northing against reanalysis rather than a farm consensus.

    Raises ``min_step_deg`` to at least :data:`REANALYSIS_MIN_STEP_DEG`, so drift in the
    reanalysis reference is not attributed to the turbines as a small step change. Everything
    else is unchanged.
    """
    if settings.min_step_deg >= REANALYSIS_MIN_STEP_DEG:
        return settings
    return replace(settings, min_step_deg=REANALYSIS_MIN_STEP_DEG)


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


def veer_normalised(
    residual: npt.NDArray[np.float64],
    *,
    reference_deg: npt.NDArray[np.float64],
    sector_deg: float,
    de_stepped: npt.NDArray[np.float64] | None = None,
    min_rows_per_sector: int = _MIN_ROWS_PER_SECTOR,
) -> npt.NDArray[np.float64]:
    """Remove each direction sector's own long-run level from the residual.

    Across a site the wind direction differs from turbine to turbine -- veer, varying with the
    bulk direction, stability and wind speed. A turbine's residual therefore has a level that
    depends on *which* directions the wind blew from, so a shift in the direction mix moves the
    level without anything at the turbine changing, and a changepoint search reads that as a step.

    Subtracting each sector's whole-record median removes it: a genuine north offset shifts every
    sector alike and so survives, while a change in the mix cannot move the level at all. Sectors
    with too little data fall back to the overall level.

    Use this for **detection only** -- segment offsets are estimated from the raw residual, so the
    correction stays absolute.

    :param de_stepped: the residual with a first-pass estimate of the step structure removed. The
        sector levels are measured on it rather than on ``residual``, so a large step cannot leak
        into the veer signature. Defaults to ``residual`` itself.
    """
    signature = sector_signature(
        residual if de_stepped is None else de_stepped,
        reference_deg=reference_deg,
        sector_deg=sector_deg,
        min_rows_per_sector=min_rows_per_sector,
    )
    finite = np.isfinite(residual) & np.isfinite(signature)
    out = residual.copy()
    out[finite] = np.asarray(circ_diff(residual[finite], signature[finite]), dtype=float)
    return out


def sector_signature(
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
    sector = np.zeros(len(values_deg), dtype=int)
    sector[finite] = (np.mod(reference_deg[finite], 360.0) // sector_deg).astype(int) % n_sectors

    overall = float(circ_median(values_deg[finite], range_360=False))
    level = np.full(n_sectors, overall)
    for s in range(n_sectors):
        rows = finite & (sector == s)
        if int(rows.sum()) >= min_rows_per_sector:
            level[s] = float(circ_median(values_deg[rows], range_360=False))
    out = np.full(len(values_deg), np.nan)
    out[np.isfinite(reference_deg)] = level[sector[np.isfinite(reference_deg)]]
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


def _prune_transient_steps(
    changepoints: list[pd.Timestamp],
    offsets: list[float],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    residual: npt.NDArray[np.float64],
    index: pd.DatetimeIndex,
    min_step_deg: float,
    max_transient_step_deg: float,
) -> tuple[list[pd.Timestamp], list[float]]:
    """Iron out small excursions -- site veer wandering away and back, rather than a recalibration.

    Repeatedly removes the least persistent changepoint while any **small** one fails to move the
    long-run level by ``min_step_deg``, re-estimating the offsets after each merge. Steps larger
    than ``max_transient_step_deg`` are never removed: a real recalibration is sometimes reversed
    later, and its size is the evidence that it happened.
    """
    while len(changepoints) > 0:
        edges = [start, *changepoints, end]
        durations = np.array([max((b - a).total_seconds(), 1.0) for a, b in itertools.pairwise(edges)], dtype=float)
        persistence = _persistence(offsets, durations=durations)
        steps = np.abs(circ_diff(np.array(offsets[1:]), np.array(offsets[:-1])))
        candidates = np.flatnonzero((steps < max_transient_step_deg) & (persistence < min_step_deg))
        if len(candidates) == 0:
            break
        weakest = int(candidates[np.argmin(persistence[candidates])])
        changepoints = [c for i, c in enumerate(changepoints) if i != weakest]
        offsets = _segment_offsets(changepoints, start=start, residual=residual, index=index)
    return changepoints, offsets


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
    credible however little record sits around it.
    """
    edges = [start, *changepoints, end]
    spans = np.array([max((b - a) / confident_segment, 1e-9) for a, b in itertools.pairwise(edges)])
    support = np.minimum(spans[:-1], spans[1:])
    return np.clip(min_step_deg / np.sqrt(np.minimum(support, 1.0)), min_step_deg, max_transient_step_deg)


def _prune_small_steps(
    changepoints: list[pd.Timestamp],
    offsets: list[float],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    residual: npt.NDArray[np.float64],
    index: pd.DatetimeIndex,
    min_step_deg: float,
    max_transient_step_deg: float,
    confident_segment: pd.Timedelta,
) -> tuple[list[pd.Timestamp], list[float]]:
    """Drop changepoints whose step is too small for the record supporting them.

    This is what makes ``min_step_deg`` mean what it says: a step smaller than it is never
    reported. Near the start or end of a record -- or squeezed between two other changepoints --
    more is required, because there is less data to tell a step from veer. Offsets are
    re-estimated after each merge, since merging two segments changes the level of the result.
    """
    while changepoints:
        steps = np.abs(circ_diff(np.array(offsets[1:]), np.array(offsets[:-1])))
        required = _required_step(
            changepoints,
            start=start,
            end=end,
            min_step_deg=min_step_deg,
            max_transient_step_deg=max_transient_step_deg,
            confident_segment=confident_segment,
        )
        shortfall = required - steps
        weakest = int(np.argmax(shortfall))
        if shortfall[weakest] <= 0:
            break
        changepoints = [c for i, c in enumerate(changepoints) if i != weakest]
        offsets = _segment_offsets(changepoints, start=start, residual=residual, index=index)
    return changepoints, offsets


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
    resulting period. Offsets are absolute: adding one to the **raw** signal norths it.

    :param index: timestamps of every array; need not be sorted
    :param direction_deg: the signal to north, in degrees
    :param reference_deg: the direction to north it against (reanalysis, or a farm consensus)
    :param usable: rows whose comparison is meaningful -- see :func:`yaw_usable`. Also the
        place to exclude periods when the direction is deliberately offset, such as a turbine
        steering its wake.
    :param settings: how the search is bounded; the default suits a farm record and there is no
        tier to choose between
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

    residual = _residual(direction, reference, ok)
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

    changepoints = detect(residual)
    if settings.veer_sector_deg is not None:
        # Search again in the veer-normalised residual, so a shift in the direction mix cannot look
        # like a step. The first pass exists only to take the step structure out of the way while
        # the veer signature is measured; offsets come from the raw residual either way, so the
        # correction stays absolute.
        changepoints = detect(
            veer_normalised(
                residual,
                reference_deg=reference,
                sector_deg=settings.veer_sector_deg,
                de_stepped=_de_stepped(residual, index=index, edges=[start, *changepoints, end]),
            )
        )

    offsets = _segment_offsets(changepoints, start=start, residual=residual, index=index)
    changepoints, offsets = _prune_transient_steps(
        changepoints,
        offsets,
        start=start,
        end=end,
        residual=residual,
        index=index,
        min_step_deg=settings.min_step_deg,
        max_transient_step_deg=settings.max_transient_step_deg,
    )
    changepoints, offsets = _prune_small_steps(
        changepoints,
        offsets,
        start=start,
        end=end,
        residual=residual,
        index=index,
        min_step_deg=settings.min_step_deg,
        max_transient_step_deg=settings.max_transient_step_deg,
        confident_segment=settings.confident_segment,
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
    timestamp take the first offset. Takes a single array so **one table can north several
    fields of the same device** -- derive the correction from yaw position, then apply it to
    yaw position and to a measured wind-direction channel. NaNs are preserved.
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


def _farm_direction(
    northed: Mapping[str, npt.NDArray[np.float64]],
    *,
    usable: Mapping[str, npt.NDArray[np.bool_]],
    min_devices: int,
) -> npt.NDArray[np.float64]:
    """Per-timestamp circular median of the devices' northed directions, NaN where too few report.

    ``min_devices`` is what keeps this trustworthy. Devices differ from the consensus by their own
    direction-dependent veer, so a median over only a few of them is not the farm's consensus --
    and when an outage coincides with an unusual wind direction, every device appears to step at
    once and back again. The guard is a quorum rather than a floor: see :func:`north_farm`.
    """
    stack = np.vstack(
        [np.where(usable[name] & np.isfinite(values), values, np.nan) for name, values in northed.items()]
    )
    enough = np.isfinite(stack).sum(axis=0) >= min_devices
    return _median_across(stack, enough=enough)


def north_farm(
    index: pd.DatetimeIndex,
    *,
    direction_deg: Mapping[str, npt.NDArray[np.float64]],
    usable: Mapping[str, npt.NDArray[np.bool_]],
    reanalysis_deg: npt.NDArray[np.float64],
    settings: NorthingSettings = DEFAULT_NORTHING,
    min_devices_for_farm_reference: int = 3,
) -> dict[str, pd.DataFrame]:
    """North a whole farm in two passes, returning one absolute table per device.

    Pass 1 norths each device to ``reanalysis_deg``; the northed directions give a farm
    consensus direction, and pass 2 norths each device's **raw** signal to that. Pass 2 is the
    more precise of the two, but pass 1 is what fixes the farm in absolute terms: a farm whose
    devices are all wrong by the same amount agrees with itself perfectly, so a farm-relative
    pass alone cannot see it.

    Every device's arrays are positional on the shared ``index``, which is what lets the farm
    consensus be taken across devices at each timestamp.

    :param direction_deg: device name to its raw direction signal
    :param usable: device name to the rows usable for northing it
    :param reanalysis_deg: the absolute direction reference, on ``index``
    :param min_devices_for_farm_reference: the floor on how many devices must report at a
        timestamp for the consensus to be defined there, and the minimum farm size. The effective
        requirement is the larger of this and a strict majority of the farm: a median over an
        unrepresentative few
        carries their veer rather than the farm's, which is what makes an outage look like every
        turbine stepping at once.
    """
    devices = sorted(direction_deg)
    if len(devices) < min_devices_for_farm_reference:
        msg = (
            f"north_farm needs at least min_devices_for_farm_reference={min_devices_for_farm_reference} "
            f"devices to form a farm reference, got {len(devices)}: {devices}"
        )
        raise ValueError(msg)
    missing = sorted(set(devices) - set(usable))
    if missing:
        msg = f"usable is missing masks for device(s) {missing}"
        raise ValueError(msg)

    # Pass 1's reference is reanalysis, so it may only attribute large steps; pass 2's farm
    # consensus is clean enough for the caller's chosen threshold.
    anchoring = anchoring_only(settings)
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
    quorum = _farm_quorum(len(devices), floor=min_devices_for_farm_reference)
    farm = _farm_direction(northed, usable=usable, min_devices=quorum)
    if not np.isfinite(farm).any():
        logger.warning("farm reference is empty; keeping the reanalysis-only north tables")
        return first_pass

    return {
        name: estimate_north_table(
            index, direction_deg[name], reference_deg=farm, usable=usable[name], settings=settings
        )
        for name in devices
    }
