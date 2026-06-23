"""Tests for campaign windows and their two derived selections."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

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
