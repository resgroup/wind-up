"""Campaign-length windows and the two record selections each window implies.

The short-campaign sweep varies the treatment-activity duration (post window for prepost,
toggling duration for toggle) over a grid — e.g. ``{3, 6, 9, 12}`` months, or ``{1, 2, 4, 8}``
weeks where a month is too coarse a step — while holding a fixed baseline before the treatment
start. Every length shares the one ``(treatment_start, baseline_start)`` and differs only in
``activity_end``, so shorter windows are strict leading prefixes of longer ones — the
campaign-length curve isolates the effect of post-duration alone.

A grid is in **either** months or weeks, never both; the unit travels with the window
(:attr:`CampaignWindow.unit`) so results are reported under the right column name.

From a window two distinct selections are derived, deliberately kept apart:

- :func:`window_row_mask` — the method-facing rows (all turbines, baseline + activity).
- :func:`treated_activity_mask` — the test turbine's treated rows within the activity window,
  for the ground-truth comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from benchmarking.synthetic import treated_mask

if TYPE_CHECKING:
    import numpy.typing as npt

    from benchmarking.synthetic import ToggleSchedule

CampaignUnit = Literal["months", "weeks"]


@dataclass(frozen=True)
class CampaignWindow:
    """One campaign length: a fixed baseline then ``length`` ``unit`` of treatment activity.

    The activity length is expressed in either months or weeks. Months is the original grid (the
    overnight studies and their frozen baselines); weeks exists for short campaigns, where a month
    is too coarse a step — a real toggle campaign runs for a handful of weeks.

    :param length: the campaign-length grid value
    :param unit: the grid's unit, ``"months"`` or ``"weeks"``
    """

    length: int
    unit: CampaignUnit
    baseline_start: pd.Timestamp
    treatment_start: pd.Timestamp
    activity_end: pd.Timestamp

    @property
    def months(self) -> int:
        """The activity length in months. Raises for a weeks window.

        Months-only consumers read this. It raises rather than returning ``length`` so a weeks window
        reaching a months-only caller fails loudly instead of silently reporting a week count as a
        month count.
        """
        if self.unit != "months":
            msg = f"CampaignWindow.months is only defined for a months grid, but this window has unit={self.unit!r}"
            raise ValueError(msg)
        return self.length

    @property
    def length_col(self) -> str:
        """The result column this window's length is reported under (``campaign_months``/``_weeks``)."""
        return f"campaign_{self.unit}"


def resolve_campaign_grid(
    *, campaign_months: list[int] | None, campaign_weeks: list[int] | None
) -> tuple[list[int], CampaignUnit]:
    """Return the ``(lengths, unit)`` of whichever grid is set. Raises unless exactly one is.

    Shared by :func:`campaign_windows` and ``StudyConfig`` so the two agree on the rule.

    Keyword-only deliberately: the two parameters have the same type, so a positional call would give
    the reader nothing to check against, and transposing them would silently mislabel weeks as months
    rather than fail.
    """
    if (campaign_months is None) == (campaign_weeks is None):
        msg = (
            "exactly one of campaign_months / campaign_weeks must be set, got "
            f"campaign_months={campaign_months!r}, campaign_weeks={campaign_weeks!r}"
        )
        raise ValueError(msg)
    if campaign_months is not None:
        return campaign_months, "months"
    assert campaign_weeks is not None  # noqa: S101 - narrowed by the check above
    return campaign_weeks, "weeks"


def campaign_windows(
    treatment_start: pd.Timestamp,
    *,
    min_pre_months: int,
    campaign_months: list[int] | None = None,
    campaign_weeks: list[int] | None = None,
    data_start: pd.Timestamp | None = None,
    data_end: pd.Timestamp | None = None,
) -> list[CampaignWindow]:
    """Build one :class:`CampaignWindow` per campaign length.

    Exactly one of ``campaign_months`` / ``campaign_weeks`` gives the grid.
    ``baseline_start = treatment_start - min_pre_months`` is fixed across lengths (and is always in
    months, independent of the grid's unit); ``activity_end = treatment_start + length`` grows with
    the length. When ``data_start`` / ``data_end`` are given, lengths whose window would fall outside
    the available data are dropped (infeasible ``(replicate, length)`` combinations).
    """
    lengths, unit = resolve_campaign_grid(campaign_months=campaign_months, campaign_weeks=campaign_weeks)
    baseline_start = treatment_start - pd.DateOffset(months=min_pre_months)
    windows = []
    for length in lengths:
        activity_end = treatment_start + pd.DateOffset(**{unit: length})
        if data_start is not None and baseline_start < data_start:
            continue
        if data_end is not None and activity_end > data_end:
            continue
        windows.append(
            CampaignWindow(
                length=length,
                unit=unit,
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
