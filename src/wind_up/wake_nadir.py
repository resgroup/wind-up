"""The wake-nadir shift step of :func:`wind_up.northing.north_farm`, described in ``docs/northing.md``."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wind_up.circular_math import circ_diff, circ_median
from wind_up.layout import ROTOR_DIAMETER_COL

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    import pandas as pd

    from wind_up.layout import Layout

logger = logging.getLogger(__name__)

# Rows a 1-degree bin needs to contribute.
MIN_BIN_ROWS = 3
_MIN_POPULATED_BINS = 8


def wake_nadir_offsets(
    layout: Layout,
    *,
    index: pd.DatetimeIndex,
    northed_direction: Mapping[str, npt.NDArray[np.float64]],
    power: Mapping[str, npt.NDArray[np.float64]],
    wind_speed: Mapping[str, npt.NDArray[np.float64]] | None = None,
    usable: Mapping[str, npt.NDArray[np.bool_]] | None = None,
    cutoff_diameters: float = 10.0,
    sector_half_width_deg: float = 15.0,
) -> dict[str, float]:
    """Return one northing correction per turbine, to add to its offsets, from where its wakes land.

    :param northed_direction: device name to its northed direction (deg) on ``index``
    :param power: device name to its power on ``index``
    :param wind_speed: device name to its nacelle wind speed on ``index``; combined with power where
        present and trustworthy, else power alone is used
    :param usable: device name to the rows valid for northing it; defaults to all rows
    :param cutoff_diameters: how far downstream, in the downstream turbine's rotor diameters, a pair counts
    :param sector_half_width_deg: half-width of the direction sector searched around each pair's bearing
    """
    devices = sorted(northed_direction)
    masks = usable if usable is not None else {d: np.ones(len(index), dtype=bool) for d in devices}

    resolved: dict[str, float] = {}
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


@dataclass(frozen=True)
class WakePairCurves:
    """One pair's binned wake against the upstream turbine's direction relative to the pair's bearing.

    Each ratio is downstream over upstream, divided by its out-of-wake level; NaN in a thin bin.
    ``nadir_deg`` is where the nadir sits relative to the bearing, ``None`` when it is not resolvable.
    """

    upstream: str
    downstream: str
    offset_deg: npt.NDArray[np.float64]
    power_ratio: npt.NDArray[np.float64]
    wind_speed_ratio: npt.NDArray[np.float64] | None
    nadir_deg: float | None


def wake_pairs(layout: Layout, *, devices: list[str], cutoff_diameters: float = 10.0) -> list[tuple[str, str]]:
    """Return every ``(upstream, downstream)`` pair the wake-nadir shift reads, nearest first per upstream."""
    out: list[tuple[str, str]] = []
    for upstream in sorted(devices):
        i_up = layout.index_of(upstream)
        downstream = _downstream_of(layout, upstream=upstream, devices=devices, cutoff=cutoff_diameters)
        out.extend((upstream, d) for d in sorted(downstream, key=lambda d: layout.distance_m[layout.index_of(d), i_up]))
    return out


def wake_pair_curves(
    layout: Layout,
    *,
    upstream: str,
    downstream: str,
    northed_direction: npt.NDArray[np.float64],
    power: Mapping[str, npt.NDArray[np.float64]],
    wind_speed: Mapping[str, npt.NDArray[np.float64]] | None = None,
    usable: Mapping[str, npt.NDArray[np.bool_]] | None = None,
    sector_half_width_deg: float = 15.0,
) -> WakePairCurves | None:
    """Return one pair's binned wake as :func:`wake_nadir_offsets` sees it, or ``None`` if too thinly sampled.

    :param northed_direction: the upstream turbine's northed direction (deg)
    :param power: device name to its power, on the same rows
    :param wind_speed: device name to its nacelle wind speed; the wind-speed ratio is ``None`` without it
    :param usable: device name to the rows valid for northing it; defaults to all rows
    """
    keep: npt.NDArray[np.bool_] = np.ones(len(northed_direction), dtype=bool)
    if usable is not None:
        keep = usable[upstream] & usable[downstream]
    ws_up = None if wind_speed is None else wind_speed.get(upstream)
    ws_down = None if wind_speed is None else wind_speed.get(downstream)
    binned = _pair_bins(
        beta=float(layout.bearing_deg[layout.index_of(downstream), layout.index_of(upstream)]),
        northed_up=northed_direction,
        power_up=power[upstream],
        power_down=power[downstream],
        keep=keep,
        half_width=sector_half_width_deg,
    )
    if binned is None:
        return None
    rows, bin_of, counts, populated = binned
    n_bins = len(counts)
    power_ratio = _ratio_curve(
        bin_of=bin_of, n_bins=n_bins, down=power[downstream][rows], up=power[upstream][rows], populated=populated
    )
    ws_ratio = None
    if ws_up is not None and ws_down is not None:
        up, down = ws_up[rows], ws_down[rows]
        finite = np.isfinite(up) & np.isfinite(down)
        ws_populated = populated & (np.bincount(bin_of[finite], minlength=n_bins) >= MIN_BIN_ROWS)
        ws_ratio = _ratio_curve(
            bin_of=bin_of[finite], n_bins=n_bins, down=down[finite], up=up[finite], populated=ws_populated
        )
    curve = _deficit_curve(
        bin_of=bin_of,
        n_bins=n_bins,
        power_up=power[upstream][rows],
        power_down=power[downstream][rows],
        ws_up=None if ws_up is None else ws_up[rows],
        ws_down=None if ws_down is None else ws_down[rows],
        populated=populated,
    )
    return WakePairCurves(
        upstream=upstream,
        downstream=downstream,
        offset_deg=np.arange(n_bins, dtype=float) - sector_half_width_deg + 0.5,
        power_ratio=power_ratio,
        wind_speed_ratio=ws_ratio,
        nadir_deg=_locate_nadir(curve, counts=counts, populated=populated, half_width=sector_half_width_deg),
    )


def _pair_bins(
    *,
    beta: float,
    northed_up: npt.NDArray[np.float64],
    power_up: npt.NDArray[np.float64],
    power_down: npt.NDArray[np.float64],
    keep: npt.NDArray[np.bool_],
    half_width: float,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.bool_]] | None:
    """Return the rows in the sector, their 1-deg bin, the bin counts and which bins are populated; ``None`` if thin."""
    offset = np.asarray(circ_diff(northed_up, np.full(len(northed_up), beta)), dtype=float)
    rows = keep & np.isfinite(offset) & np.isfinite(power_up) & np.isfinite(power_down) & (np.abs(offset) <= half_width)
    if int(rows.sum()) < _MIN_POPULATED_BINS * MIN_BIN_ROWS:
        return None
    n_bins = 2 * math.ceil(half_width)
    bin_of = np.clip((offset[rows] + half_width).astype(int), 0, n_bins - 1)
    counts = np.bincount(bin_of, minlength=n_bins)
    populated = counts >= MIN_BIN_ROWS
    if int(populated.sum()) < _MIN_POPULATED_BINS:
        return None
    return rows, bin_of, counts, populated


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
) -> float | None:
    """Return the correction (deg, to add to the offset) for one pair, or ``None`` if the nadir is not resolvable."""
    binned = _pair_bins(
        beta=beta, northed_up=northed_up, power_up=power_up, power_down=power_down, keep=keep, half_width=half_width
    )
    if binned is None:
        return None
    rows, bin_of, counts, populated = binned
    curve = _deficit_curve(
        bin_of=bin_of,
        n_bins=len(counts),
        power_up=power_up[rows],
        power_down=power_down[rows],
        ws_up=None if ws_up is None else ws_up[rows],
        ws_down=None if ws_down is None else ws_down[rows],
        populated=populated,
    )
    nadir_offset = _locate_nadir(curve, counts=counts, populated=populated, half_width=half_width)
    if nadir_offset is None:
        return None
    logger.debug("pair %s->%s: nadir offset %.2f deg (%d rows)", upstream, downstream, nadir_offset, int(rows.sum()))
    return -nadir_offset


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
    """Return the normalised deficit per bin, the mean of the power and wind-speed ratio curves; NaN if unpopulated."""
    curves = [_ratio_curve(bin_of=bin_of, n_bins=n_bins, down=power_down, up=power_up, populated=populated)]
    if ws_up is not None and ws_down is not None:
        finite = np.isfinite(ws_up) & np.isfinite(ws_down)
        ws_populated = populated & (np.bincount(bin_of[finite], minlength=n_bins) >= MIN_BIN_ROWS)
        if ws_populated.any():
            curves.append(
                _ratio_curve(
                    bin_of=bin_of[finite], n_bins=n_bins, down=ws_down[finite], up=ws_up[finite], populated=ws_populated
                )
            )
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


def _locate_nadir(
    curve: npt.NDArray[np.float64],
    *,
    counts: npt.NDArray[np.int64],
    populated: npt.NDArray[np.bool_],
    half_width: float,
) -> float | None:
    """Return the nadir's offset (deg) from a weighted quadratic fit near the minimum, or ``None`` if unresolvable."""
    fit_half_window = 5
    min_nadir_depth = 0.02
    filled = np.where(populated, curve, np.nan)
    if not np.isfinite(filled).any():
        return None
    order = np.flatnonzero(populated)
    min_bin = int(order[np.nanargmin(filled[order])])
    if min_bin <= order[0] or min_bin >= order[-1]:
        return None

    lo, hi = max(order[0], min_bin - fit_half_window), min(order[-1], min_bin + fit_half_window)
    window = np.arange(lo, hi + 1)
    window = window[populated[window]]
    if len(window) < 3:  # noqa: PLR2004
        return None

    centres = window - half_width + 0.5
    values = curve[window]
    weights = np.sqrt(counts[window].astype(float))
    convex = _convex_vertex(centres, values, weights)
    if convex is None:
        return None
    vertex, a, b, c = convex
    depth = float(np.nanmax(curve[populated]) - (c - b * b / (4 * a)))
    if depth < min_nadir_depth:
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


def _aggregate(pairs: list[float]) -> float:
    """Combine a turbine's pair corrections by their circular median."""
    return float(circ_median(np.array(pairs, dtype=float), range_360=False))


def _fill(layout: Layout, *, devices: list[str], resolved: dict[str, float]) -> dict[str, float]:
    """Return every device's correction: its own where resolved, else inherited, else zero."""
    out: dict[str, float] = {}
    for device in devices:
        if device in resolved:
            out[device] = resolved[device]
            continue
        neighbours = _nearest_resolved(layout, device=device, resolved=resolved)
        if neighbours:
            out[device] = float(circ_median(np.array([resolved[n] for n in neighbours]), range_360=False))
        else:
            out[device] = 0.0
    return out


def _nearest_resolved(layout: Layout, *, device: str, resolved: dict[str, float]) -> list[str]:
    """Return up to four resolved turbines nearest ``device``."""
    i = layout.index_of(device)
    ranked = sorted((float(layout.distance_m[i, layout.index_of(n)]), n) for n in resolved)
    return [n for _, n in ranked[:4]]
