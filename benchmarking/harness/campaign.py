"""Campaign-length windows and the two record selections each window implies.

The short-campaign sweep varies the treatment-activity duration (post window for prepost,
toggling duration for toggle) over a grid, e.g. ``{3, 6, 9, 12}`` months, while holding a
fixed baseline before the treatment start. Every length shares the one
``(treatment_start, baseline_start)`` and differs only in ``activity_end``, so shorter
windows are strict leading prefixes of longer ones — the campaign-length curve isolates the
effect of post-duration alone.

From a window two distinct selections are derived, deliberately kept apart:

- :func:`window_row_mask` — the method-facing rows (all turbines, baseline + activity).
- :func:`treated_activity_mask` — the test turbine's treated rows within the activity window,
  for the ground-truth comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.synthetic import treated_mask

if TYPE_CHECKING:
    import numpy.typing as npt

    from benchmarking.synthetic import ToggleSchedule


@dataclass(frozen=True)
class CampaignWindow:
    """One campaign length: a fixed baseline then ``months`` of treatment activity."""

    months: int
    baseline_start: pd.Timestamp
    treatment_start: pd.Timestamp
    activity_end: pd.Timestamp


def campaign_windows(
    treatment_start: pd.Timestamp,
    *,
    min_pre_months: int,
    campaign_months: list[int],
    data_start: pd.Timestamp | None = None,
    data_end: pd.Timestamp | None = None,
) -> list[CampaignWindow]:
    """Build one :class:`CampaignWindow` per campaign length.

    ``baseline_start = treatment_start - min_pre_months`` is fixed across lengths;
    ``activity_end = treatment_start + months`` grows with the length. When ``data_start`` /
    ``data_end`` are given, lengths whose window would fall outside the available data are
    dropped (infeasible ``(replicate, length)`` combinations).
    """
    baseline_start = treatment_start - pd.DateOffset(months=min_pre_months)
    windows = []
    for months in campaign_months:
        activity_end = treatment_start + pd.DateOffset(months=months)
        if data_start is not None and baseline_start < data_start:
            continue
        if data_end is not None and activity_end > data_end:
            continue
        windows.append(
            CampaignWindow(
                months=months,
                baseline_start=baseline_start,
                treatment_start=treatment_start,
                activity_end=activity_end,
            )
        )
    return windows


def window_row_mask(index: pd.DatetimeIndex, window: CampaignWindow) -> npt.NDArray[np.bool_]:
    """Boolean mask of the rows a method sees: ``[baseline_start, activity_end)``."""
    return np.asarray((index >= window.baseline_start) & (index < window.activity_end))


def treated_activity_mask(
    index: pd.DatetimeIndex,
    upgrade_timing: pd.Timestamp | ToggleSchedule,
    *,
    window: CampaignWindow,
) -> npt.NDArray[np.bool_]:
    """Boolean mask of the test turbine's treated rows within ``[treatment_start, activity_end)``.

    For prepost these are the post records; for toggle the on-records. The baseline is never
    treated. ``index`` must be the test turbine's rows in time order (matches ``true_uplift``).
    """
    treated = treated_mask(index, upgrade_timing)
    in_activity = np.asarray((index >= window.treatment_start) & (index < window.activity_end))
    return treated & in_activity
