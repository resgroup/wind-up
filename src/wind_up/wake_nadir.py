"""Pass-4 wake-nadir absolute nudge for :func:`wind_up.northing.north_farm`.

A turbine's wake lands on a downstream turbine at one wind-from direction fixed by geometry. The
downstream deficit, plotted against the upstream turbine's northed direction, dips at that direction;
where the measured dip sits away from the geometric nadir is the upstream turbine's residual northing
error. :func:`wake_nadir_offsets` measures that per turbine and returns one absolute correction each,
to add on top of that turbine's changepoint table.

The correction is one number per turbine. A turbine with no resolvable dip inherits the circular
median of its nearest resolved neighbours; one with neither gets zero.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from wind_up.circular_math import circ_diff, circ_median
from wind_up.layout import ROTOR_DIAMETER_COL

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    import pandas as pd

    from wind_up.layout import Layout

logger = logging.getLogger(__name__)

# Wake reaches this many rotor diameters downstream.
WAKE_CUTOFF_DIAMETERS = 10.0
# Half-width of the direction sector searched around the geometric nadir.
SECTOR_HALF_WIDTH_DEG = 15.0
# A 1-degree bin needs this many rows to contribute (about half an hour of 10-minute data).
MIN_BIN_ROWS = 3
# The sector needs this many populated 1-degree bins to constrain the dip.
MIN_POPULATED_BINS = 8
# The dip must fall at least this far below the sector's out-of-wake level.
MIN_DIP_DEPTH = 0.02
# A turbine with no dip of its own inherits from up to this many nearest resolved turbines.
MAX_INHERIT_NEIGHBOURS = 4
# Half-window of bins each side of the minimum used for the quadratic fit.
_FIT_HALF_WINDOW = 5


class _Nadir(NamedTuple):
    """A resolved wake nadir.

    ``delta`` is the correction (deg, to add to the offset); ``volume`` is the in-sector row count
    supporting it.
    """

    delta: float
    volume: float


def wake_nadir_offsets(
    layout: Layout,
    *,
    index: pd.DatetimeIndex,
    northed_direction: Mapping[str, npt.NDArray[np.float64]],
    power: Mapping[str, npt.NDArray[np.float64]],
    wind_speed: Mapping[str, npt.NDArray[np.float64]] | None = None,
    usable: Mapping[str, npt.NDArray[np.bool_]] | None = None,
    cutoff_diameters: float = WAKE_CUTOFF_DIAMETERS,
    sector_half_width_deg: float = SECTOR_HALF_WIDTH_DEG,
) -> dict[str, float]:
    """Return one absolute northing correction per turbine, from wake-nadir geometry.

    For each directed pair (upstream wakes downstream) within ``cutoff_diameters``, the downstream
    deficit versus the upstream turbine's northed direction dips at the geometric nadir; the offset
    of the measured dip is the upstream turbine's residual. Each turbine's pairs are combined by the
    circular median of their corrections, which resists the terrain-driven deflection that biases an
    individual pair. A turbine with no resolvable pair inherits the circular median of up to
    :data:`MAX_INHERIT_NEIGHBOURS` nearest resolved turbines, or zero if none.

    :param northed_direction: device name to its northed direction (deg) on ``index``
    :param power: device name to its power on ``index``
    :param wind_speed: device name to its nacelle wind speed on ``index``; combined with power where
        present and trustworthy, else power alone is used
    :param usable: device name to the rows valid for northing it; defaults to all rows
    """
    devices = sorted(northed_direction)
    masks = usable if usable is not None else {d: np.ones(len(index), dtype=bool) for d in devices}

    resolved: dict[str, _Nadir] = {}
    for upstream in devices:
        pairs = [
            nadir
            for downstream in _downstream_of(layout, upstream=upstream, devices=devices, cutoff=cutoff_diameters)
            if (
                nadir := _pair_nadir(
                    upstream=upstream,
                    downstream=downstream,
                    beta=float(layout.bearing_deg[layout.index_of(downstream), layout.index_of(upstream)]),
                    northed_up=northed_direction[upstream],
                    power_up=power[upstream],
                    power_down=power[downstream],
                    ws_up=None if wind_speed is None else wind_speed.get(upstream),
                    ws_down=None if wind_speed is None else wind_speed.get(downstream),
                    keep=masks[upstream] & masks[downstream],
                    half_width=sector_half_width_deg,
                )
            )
            is not None
        ]
        if pairs:
            resolved[upstream] = _aggregate(pairs)

    return _fill(layout, devices=devices, resolved=resolved)


def _downstream_of(layout: Layout, *, upstream: str, devices: list[str], cutoff: float) -> list[str]:
    """Return the devices ``upstream`` can wake: those within ``cutoff`` diameters downstream."""
    i_up = layout.index_of(upstream)
    out = []
    for downstream in devices:
        if downstream == upstream:
            continue
        i_down = layout.index_of(downstream)
        diameter = float(layout.frame[ROTOR_DIAMETER_COL].iloc[i_down])
        if layout.distance_m[i_down, i_up] <= cutoff * diameter:
            out.append(downstream)
    return out


def _pair_nadir(
    *,
    upstream: str,
    downstream: str,
    beta: float,
    northed_up: npt.NDArray[np.float64],
    power_up: npt.NDArray[np.float64],
    power_down: npt.NDArray[np.float64],
    ws_up: npt.NDArray[np.float64] | None,
    ws_down: npt.NDArray[np.float64] | None,
    keep: npt.NDArray[np.bool_],
    half_width: float,
) -> _Nadir | None:
    """Return the residual and supporting row count for one pair, or ``None`` if the dip is not resolvable."""
    offset = np.asarray(circ_diff(northed_up, np.full(len(northed_up), beta)), dtype=float)
    rows = keep & np.isfinite(offset) & np.isfinite(power_up) & np.isfinite(power_down) & (np.abs(offset) <= half_width)
    if int(rows.sum()) < MIN_POPULATED_BINS * MIN_BIN_ROWS:
        return None

    n_bins = 2 * math.ceil(half_width)
    bin_of = np.clip((offset[rows] + half_width).astype(int), 0, n_bins - 1)
    counts = np.bincount(bin_of, minlength=n_bins)
    populated = counts >= MIN_BIN_ROWS
    if int(populated.sum()) < MIN_POPULATED_BINS:
        return None

    curve = _deficit_curve(
        bin_of=bin_of,
        n_bins=n_bins,
        power_up=power_up[rows],
        power_down=power_down[rows],
        ws_up=None if ws_up is None else ws_up[rows],
        ws_down=None if ws_down is None else ws_down[rows],
        populated=populated,
    )
    dip_offset = _locate_dip(curve, counts=counts, populated=populated, half_width=half_width)
    if dip_offset is None:
        return None
    volume = int(rows.sum())
    logger.debug("pair %s->%s: nadir offset %.2f deg (%d rows)", upstream, downstream, dip_offset, volume)
    return _Nadir(delta=-dip_offset, volume=float(volume))


def _deficit_curve(
    *,
    bin_of: npt.NDArray[np.int64],
    n_bins: int,
    power_up: npt.NDArray[np.float64],
    power_down: npt.NDArray[np.float64],
    ws_up: npt.NDArray[np.float64] | None,
    ws_down: npt.NDArray[np.float64] | None,
    populated: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Return the combined normalised deficit per bin: mean of the power and wind-speed ratios.

    Each signal is a ratio of the downstream to the upstream bin mean, so the ambient resource
    cancels; each ratio is then divided by its out-of-wake level so the two combine on one scale.
    Unpopulated bins are NaN.
    """
    curves = [_ratio_curve(bin_of=bin_of, n_bins=n_bins, down=power_down, up=power_up, populated=populated)]
    if ws_up is not None and ws_down is not None and np.isfinite(ws_up).all() and np.isfinite(ws_down).all():
        curves.append(_ratio_curve(bin_of=bin_of, n_bins=n_bins, down=ws_down, up=ws_up, populated=populated))
    stack = np.vstack(curves)
    combined = np.full(stack.shape[1], np.nan)
    filled = np.isfinite(stack).any(axis=0)
    combined[filled] = np.nanmean(stack[:, filled], axis=0)
    return combined


def _ratio_curve(
    *,
    bin_of: npt.NDArray[np.int64],
    n_bins: int,
    down: npt.NDArray[np.float64],
    up: npt.NDArray[np.float64],
    populated: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Downstream-over-upstream ratio of bin means, divided by its out-of-wake level; NaN elsewhere."""
    down_sum = np.bincount(bin_of, weights=down, minlength=n_bins)
    up_sum = np.bincount(bin_of, weights=up, minlength=n_bins)
    ratio = np.full(n_bins, np.nan)
    ok = populated & (up_sum > 0)
    ratio[ok] = down_sum[ok] / up_sum[ok]
    baseline = np.nanmax(ratio) if np.isfinite(ratio).any() else np.nan
    return ratio / baseline if baseline and np.isfinite(baseline) else ratio


def _locate_dip(
    curve: npt.NDArray[np.float64],
    *,
    counts: npt.NDArray[np.int64],
    populated: npt.NDArray[np.bool_],
    half_width: float,
) -> float | None:
    """Locate the dip by a weighted quadratic fit near the minimum; return its view-angle offset (deg).

    Rejects a curve whose minimum sits at the sector edge (an unbracketed dip, or a central peak),
    or whose best fit is not convex or not deep enough.
    """
    filled = np.where(populated, curve, np.nan)
    if not np.isfinite(filled).any():
        return None
    order = np.flatnonzero(populated)
    min_bin = int(order[np.nanargmin(filled[order])])
    if min_bin <= order[0] or min_bin >= order[-1]:
        return None

    lo, hi = max(order[0], min_bin - _FIT_HALF_WINDOW), min(order[-1], min_bin + _FIT_HALF_WINDOW)
    window = np.arange(lo, hi + 1)
    window = window[populated[window]]
    if len(window) < 3:  # noqa: PLR2004 - a parabola needs three points
        return None

    centres = window - half_width + 0.5
    values = curve[window]
    weights = np.sqrt(counts[window].astype(float))
    convex = _convex_vertex(centres, values, weights)
    if convex is None:
        return None
    vertex, a, b, c = convex
    depth = float(np.nanmax(curve[populated]) - (c - b * b / (4 * a)))
    if depth < MIN_DIP_DEPTH:
        return None
    return float(vertex)


def _convex_vertex(
    centres: npt.NDArray[np.float64], values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> tuple[float, float, float, float] | None:
    """Weighted parabola fit; return ``(vertex, a, b, c)`` when convex with the vertex in range, else None."""
    a, b, c = (float(coeff) for coeff in np.polyfit(centres, values, 2, w=weights))
    if a <= 0:
        return None
    vertex = -b / (2 * a)
    if vertex < centres.min() or vertex > centres.max():
        return None
    return vertex, a, b, c


def _aggregate(pairs: list[_Nadir]) -> _Nadir:
    """Combine a turbine's pairs by the circular median of their view-angle corrections.

    Each pair mixes the wake with a terrain-driven deflection that biases it by several degrees, and
    that bias is not a measurement variance the fit can report. The median lets a majority of
    consistent pairs outvote a deflected one, where any weighted mean would be dragged toward it.
    """
    deltas = np.array([p.delta for p in pairs], dtype=float)
    volume = float(sum(p.volume for p in pairs))
    return _Nadir(delta=float(circ_median(deltas, range_360=False)), volume=volume)


def _fill(layout: Layout, *, devices: list[str], resolved: dict[str, _Nadir]) -> dict[str, float]:
    """Return every device's correction: its own where resolved, else inherited, else zero."""
    out: dict[str, float] = {}
    for device in devices:
        if device in resolved:
            out[device] = resolved[device].delta
            continue
        neighbours = _nearest_resolved(layout, device=device, resolved=resolved)
        if neighbours:
            out[device] = float(circ_median(np.array([resolved[n].delta for n in neighbours]), range_360=False))
        else:
            out[device] = 0.0
    return out


def _nearest_resolved(layout: Layout, *, device: str, resolved: dict[str, _Nadir]) -> list[str]:
    """Return up to :data:`MAX_INHERIT_NEIGHBOURS` resolved turbines nearest ``device``."""
    i = layout.index_of(device)
    ranked = sorted((float(layout.distance_m[i, layout.index_of(n)]), n) for n in resolved)
    return [n for _, n in ranked[:MAX_INHERIT_NEIGHBOURS]]
