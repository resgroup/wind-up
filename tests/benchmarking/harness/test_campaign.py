"""Tests for campaign windows and their two derived selections."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from benchmarking.harness.campaign import (
    CampaignWindow,
    campaign_windows,
    treated_activity_mask,
    window_row_mask,
)
from benchmarking.synthetic import ToggleSchedule

T0 = pd.Timestamp("2018-01-01", tz="UTC")


def test_window_bounds_use_fixed_baseline_and_activity_length() -> None:
    [window] = campaign_windows(T0, min_pre_months=12, campaign_months=[6])
    assert isinstance(window, CampaignWindow)
    assert window.treatment_start == T0
    assert window.baseline_start == T0 - pd.DateOffset(months=12)
    assert window.activity_end == T0 + pd.DateOffset(months=6)
    assert window.months == 6


def test_one_window_per_campaign_length() -> None:
    windows = campaign_windows(T0, min_pre_months=12, campaign_months=[3, 6, 9, 12])
    assert [w.months for w in windows] == [3, 6, 9, 12]


def test_shorter_windows_share_baseline_and_treatment_start() -> None:
    windows = campaign_windows(T0, min_pre_months=12, campaign_months=[3, 6, 9, 12])
    assert {w.baseline_start for w in windows} == {T0 - pd.DateOffset(months=12)}
    assert {w.treatment_start for w in windows} == {T0}
    activity_ends = [w.activity_end for w in windows]
    assert activity_ends == sorted(activity_ends)  # strictly growing post window


def test_each_windows_row_mask_is_a_prefix_subset_of_the_next() -> None:
    index = pd.date_range("2017-01-01", "2019-01-01", freq="6h", tz="UTC")
    windows = campaign_windows(T0, min_pre_months=12, campaign_months=[3, 6, 9, 12])
    masks = [window_row_mask(index, w) for w in windows]
    for shorter, longer in pairwise(masks):
        # 3 ⊂ 6 ⊂ 9 ⊂ 12: every row in the shorter window is in the longer one
        assert np.all(longer[shorter])
        assert longer.sum() > shorter.sum()


def test_window_row_mask_selects_baseline_through_activity_end() -> None:
    index = pd.date_range("2017-01-01", "2019-01-01", freq="1D", tz="UTC")
    [window] = campaign_windows(T0, min_pre_months=12, campaign_months=[6])
    mask = window_row_mask(index, window)
    selected = index[mask]
    assert selected.min() >= window.baseline_start
    assert selected.max() < window.activity_end
    unselected = index[~mask]
    assert ((unselected < window.baseline_start) | (unselected >= window.activity_end)).all()


def test_prepost_truth_mask_is_the_post_rows_within_activity() -> None:
    index = pd.date_range("2017-01-01", "2019-06-01", freq="1D", tz="UTC")
    [window] = campaign_windows(T0, min_pre_months=12, campaign_months=[6])
    mask = treated_activity_mask(index, T0, window=window)  # prepost: a bare timestamp
    selected = index[mask]
    assert (selected >= T0).all()
    assert (selected < window.activity_end).all()
    assert not mask[index < T0].any()  # baseline never counted as treated


def test_toggle_truth_mask_excludes_baseline_and_keeps_only_on_rows() -> None:
    index = pd.date_range("2017-01-01", "2019-06-01", freq="6h", tz="UTC")
    schedule = ToggleSchedule(period=pd.Timedelta(days=14), start=T0)
    [window] = campaign_windows(T0, min_pre_months=12, campaign_months=[6])
    mask = treated_activity_mask(index, schedule, window=window)
    assert not mask[index < T0].any()  # baseline untreated
    assert mask.any()  # some on-rows inside the toggling window
    assert (index[mask] < window.activity_end).all()
    assert mask.sum() < ((index >= T0) & (index < window.activity_end)).sum()  # only on-blocks


def test_infeasible_lengths_are_dropped_when_data_bounds_given() -> None:
    # data ends only 4 months after t0, so 6/9/12-month campaigns do not fit
    windows = campaign_windows(
        T0,
        min_pre_months=12,
        campaign_months=[3, 6, 9, 12],
        data_start=pd.Timestamp("2016-01-01", tz="UTC"),
        data_end=T0 + pd.DateOffset(months=4),
    )
    assert [w.months for w in windows] == [3]


class TestCampaignWeeks:
    """A weeks grid: the same window machinery on a ``DateOffset(weeks=...)`` activity length."""

    def test_window_bounds_use_week_activity_length(self) -> None:
        [window] = campaign_windows(T0, min_pre_months=12, campaign_weeks=[2])
        assert window.treatment_start == T0
        assert window.baseline_start == T0 - pd.DateOffset(months=12)  # baseline stays months-based
        assert window.activity_end == T0 + pd.DateOffset(weeks=2)
        assert window.length == 2
        assert window.unit == "weeks"
        assert window.length_col == "campaign_weeks"

    def test_one_window_per_campaign_length(self) -> None:
        windows = campaign_windows(T0, min_pre_months=12, campaign_weeks=[1, 2, 4, 8])
        assert [w.length for w in windows] == [1, 2, 4, 8]

    def test_shorter_windows_are_prefixes_of_longer_ones(self) -> None:
        index = pd.date_range("2016-06-01", "2018-06-01", freq="6h", tz="UTC")
        windows = campaign_windows(T0, min_pre_months=12, campaign_weeks=[1, 2, 4, 8])
        masks = [window_row_mask(index, w) for w in windows]
        for shorter, longer in pairwise(masks):
            assert np.all(longer[shorter])
            assert longer.sum() > shorter.sum()

    def test_infeasible_lengths_are_dropped(self) -> None:
        windows = campaign_windows(
            T0,
            min_pre_months=12,
            campaign_weeks=[1, 2, 4, 8],
            data_start=pd.Timestamp("2016-01-01", tz="UTC"),
            data_end=T0 + pd.DateOffset(weeks=3),
        )
        assert [w.length for w in windows] == [1, 2]

    def test_months_property_raises_on_a_weeks_window(self) -> None:
        # the months-only inspect_* scripts read ``window.months``; a weeks window must fail loudly
        # there rather than silently reporting a week count as a month count.
        [window] = campaign_windows(T0, min_pre_months=12, campaign_weeks=[2])
        with pytest.raises(ValueError, match="unit='weeks'"):
            _ = window.months


class TestCampaignLengthValidation:
    """Exactly one of the two grids must be given."""

    def test_neither_grid_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            campaign_windows(T0, min_pre_months=12)

    def test_both_grids_raise(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            campaign_windows(T0, min_pre_months=12, campaign_months=[3], campaign_weeks=[2])


def test_months_window_still_reports_months_unit() -> None:
    # back-compat: the months path is unchanged, so ``months`` keeps working and the length column
    # keeps its existing name.
    [window] = campaign_windows(T0, min_pre_months=12, campaign_months=[6])
    assert window.unit == "months"
    assert window.length == 6
    assert window.months == 6
    assert window.length_col == "campaign_months"
