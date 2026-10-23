"""Possible yaw-alignment changes: the small north-table steps, with how strongly apparent Cp moved at each.

A north table's changepoints are either north-calibration changes or candidate yaw-alignment
changes. :func:`yaw_change_candidates` keeps the candidates: every step between two adjacent
segments in frame. A run of segments at least ``excursion_deg`` from the latest in-frame offset is
out of frame, an excursion, unless it lasts ``permanent`` or longer, when its first segment is a
re-calibration that becomes the new frame.

:func:`possible_yaw_changes` reports each candidate with the shift in the turbine's apparent Cp
(power over its own nacelle wind speed cubed, binned by wind speed) between the windows either side,
and the largest shift at any other date of its record for comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from wind_up.circular_math import circ_diff
from wind_up.northing import NORTH_OFFSET_COL, TIMESTAMP_COL

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

EXCURSION_DEG = 30.0
PERMANENT_RECALIBRATION = pd.Timedelta(days=90)
CP_WINDOW = pd.Timedelta(days=60)
CP_SCAN_STEP = pd.Timedelta(days=3)
CP_BIN_EDGES = np.arange(4.0, 12.5, 1.0)
MIN_BIN_HOURS = 6.0

CANDIDATE_COLUMNS = ("turbine", "timestamp", "north_step_deg", "cp_shift", "cp_null_max", "cp_ratio")


def yaw_change_candidates(
    table: pd.DataFrame,
    *,
    record_end: pd.Timestamp,
    excursion_deg: float = EXCURSION_DEG,
    permanent: pd.Timedelta = PERMANENT_RECALIBRATION,
) -> pd.DataFrame:
    """Return the steps of ``table`` between two adjacent in-frame segments.

    :param table: one turbine's north table
    :param record_end: where the table's last segment ends
    :return: ``timestamp`` and ``north_step_deg`` (the signed change in north offset) per candidate
    """
    table = table.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    offsets = table[NORTH_OFFSET_COL].to_numpy(dtype=float)
    times = list(pd.DatetimeIndex(table[TIMESTAMP_COL]))
    ends = [*times[1:], pd.Timestamp(record_end)]
    in_frame = np.zeros(len(offsets), dtype=bool)
    in_frame[:1] = True
    home = offsets[0] if len(offsets) else 0.0
    k = 1
    while k < len(offsets):
        if abs(float(circ_diff(offsets[k], home))) < excursion_deg:
            in_frame[k], home = True, offsets[k]
            k += 1
            continue
        run_end = k
        while run_end < len(offsets) and abs(float(circ_diff(offsets[run_end], home))) >= excursion_deg:
            run_end += 1
        if ends[run_end - 1] - times[k] >= permanent:
            in_frame[k], home = True, offsets[k]
            k += 1
            continue
        k = run_end
    rows = [
        (times[k], float(circ_diff(offsets[k], offsets[k - 1])))
        for k in range(1, len(offsets))
        if in_frame[k] and in_frame[k - 1]
    ]
    rows = [(t, step) for t, step in rows if abs(step) < excursion_deg]
    return pd.DataFrame(
        {
            TIMESTAMP_COL: pd.DatetimeIndex([t for t, _ in rows], tz=table[TIMESTAMP_COL].dt.tz),
            "north_step_deg": [step for _, step in rows],
        }
    )


class _DailyCp:
    """Daily per-bin sums and counts of apparent Cp, cumulated so a window's bin means are O(1)."""

    def __init__(
        self,
        index: pd.DatetimeIndex,
        *,
        power: npt.NDArray[np.float64],
        wind_speed: npt.NDArray[np.float64],
        usable: npt.NDArray[np.bool_],
        bin_edges: npt.NDArray[np.float64],
    ) -> None:
        days = index.floor("D")
        self.day0 = days.min()
        n_days = int((days.max() - self.day0) / pd.Timedelta(days=1)) + 1
        n_bins = len(bin_edges) - 1
        ws = np.asarray(wind_speed, dtype=float)
        p = np.asarray(power, dtype=float)
        bins = np.digitize(ws, bin_edges) - 1
        keep = np.asarray(usable, dtype=bool) & np.isfinite(ws) & np.isfinite(p) & (bins >= 0) & (bins < n_bins)
        day = ((days - self.day0) / pd.Timedelta(days=1)).to_numpy().astype(int)
        cp = p[keep] / ws[keep] ** 3
        sums = np.zeros((n_days, n_bins))
        counts = np.zeros((n_days, n_bins))
        np.add.at(sums, (day[keep], bins[keep]), cp)
        np.add.at(counts, (day[keep], bins[keep]), 1.0)
        self.n_days = n_days
        self.sums = np.vstack([np.zeros(n_bins), np.cumsum(sums, axis=0)])
        self.counts = np.vstack([np.zeros(n_bins), np.cumsum(counts, axis=0)])

    def position(self, t: pd.Timestamp) -> int:
        """Return the day position of ``t``."""
        return int((t.floor("D") - self.day0) / pd.Timedelta(days=1))

    def shift(self, at: int, *, window_days: int, min_count: float) -> float:
        """Median fractional change in binned apparent Cp from the window before day ``at`` to the one from it."""
        lo, mid, hi = max(at - window_days, 0), min(max(at, 0), self.n_days), min(at + window_days, self.n_days)
        pre_n = self.counts[mid] - self.counts[lo]
        post_n = self.counts[hi] - self.counts[mid]
        shared = (pre_n >= min_count) & (post_n >= min_count)
        if not shared.any():
            return float("nan")
        pre = (self.sums[mid] - self.sums[lo])[shared] / pre_n[shared]
        post = (self.sums[hi] - self.sums[mid])[shared] / post_n[shared]
        return float(np.median((post - pre) / pre))


def apparent_cp_scan(
    index: pd.DatetimeIndex,
    *,
    power: npt.NDArray[np.float64],
    wind_speed: npt.NDArray[np.float64],
    usable: npt.NDArray[np.bool_],
    timebase: pd.Timedelta,
    at: pd.DatetimeIndex | None = None,
    window: pd.Timedelta = CP_WINDOW,
    step: pd.Timedelta = CP_SCAN_STEP,
    bin_edges: npt.NDArray[np.float64] = CP_BIN_EDGES,
    min_bin_hours: float = MIN_BIN_HOURS,
) -> pd.Series:
    """Return the apparent-Cp shift across each date: at ``at``, or every ``step`` where both windows fit the record.

    A shift is the median over wind-speed bins of the fractional change in mean apparent Cp from the
    ``window`` before the date to the ``window`` from it. A bin counts when each side has at least
    ``min_bin_hours`` of records; NaN when none does.
    """
    daily = _DailyCp(index, power=power, wind_speed=wind_speed, usable=usable, bin_edges=bin_edges)
    window_days = int(window / pd.Timedelta(days=1))
    if at is None:
        first = daily.day0 + window
        last = daily.day0 + pd.Timedelta(days=daily.n_days) - window
        at = pd.date_range(first, last, freq=step) if first <= last else pd.DatetimeIndex([], tz=index.tz)
    min_count = min_bin_hours / (timebase / pd.Timedelta(hours=1))
    values = [daily.shift(daily.position(t), window_days=window_days, min_count=min_count) for t in at]
    return pd.Series(values, index=at, dtype=float)


def possible_yaw_changes(
    tables: Mapping[str, pd.DataFrame],
    *,
    index: pd.DatetimeIndex,
    power: Mapping[str, npt.NDArray[np.float64]],
    wind_speed: Mapping[str, npt.NDArray[np.float64]],
    usable: Mapping[str, npt.NDArray[np.bool_]],
    timebase: pd.Timedelta,
    window: pd.Timedelta = CP_WINDOW,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Return every candidate yaw-alignment change with its apparent-Cp shift, and each such turbine's scan.

    ``cp_null_max`` is the largest absolute shift in the turbine's scan at dates more than ``window``
    from every candidate; ``cp_ratio`` is the candidate's absolute shift over it. Every turbine's
    arrays are positional on ``index``.

    :return: one row per candidate (:data:`CANDIDATE_COLUMNS`), and the scan of each turbine with a candidate
    """
    rows: list[pd.DataFrame] = []
    scans: dict[str, pd.Series] = {}
    record_end = index.max()
    for turbine in sorted(tables):
        candidates = yaw_change_candidates(tables[turbine], record_end=record_end)
        if candidates.empty:
            continue
        when = pd.DatetimeIndex(candidates[TIMESTAMP_COL])
        scan, shifts = (
            apparent_cp_scan(
                index,
                power=power[turbine],
                wind_speed=wind_speed[turbine],
                usable=usable[turbine],
                timebase=timebase,
                window=window,
                at=at,
            )
            for at in (None, when)
        )
        away = np.array([min(abs(t - c) for c in when) > window for t in scan.index], dtype=bool)
        null = np.abs(scan.to_numpy()[away]) if len(scan) else np.array([])
        null_max = float(np.nanmax(null)) if np.isfinite(null).any() else float("nan")
        rows.append(
            pd.DataFrame(
                {
                    "turbine": turbine,
                    "timestamp": when,
                    "north_step_deg": candidates["north_step_deg"].to_numpy(),
                    "cp_shift": shifts.to_numpy(),
                    "cp_null_max": null_max,
                    "cp_ratio": np.abs(shifts.to_numpy()) / null_max if null_max > 0 else np.nan,
                }
            )
        )
        scans[turbine] = scan
    if not rows:
        return pd.DataFrame(columns=list(CANDIDATE_COLUMNS)), scans
    return pd.concat(rows, ignore_index=True), scans
