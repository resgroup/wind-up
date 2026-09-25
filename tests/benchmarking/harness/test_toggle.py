"""Tests for the shared toggle interpreter consumed by all benchmarking methods.

The interpreter turns a toggle campaign (a periodic ``ToggleSchedule`` or an explicit, possibly
irregular ``toggle_df``) into the three row-sets methods need: ``upgraded`` (the "on" segment),
``campaign_baseline`` (the strict "off" segment, for the on/off comparison and matching) and
``training_baseline`` (the lenient pre-first-on U off segment, for counterfactual model fitting).
Rows that are neither on nor a training-baseline row are excluded from every segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.harness.toggle import (
    ToggleRowSets,
    build_toggle_df,
    is_toggle,
    resolve_toggle,
    toggle_upgrade_start,
)
from benchmarking.synthetic import ToggleSchedule, treated_mask


def _toggle_df(index: pd.DatetimeIndex, on: list[bool], off: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"toggle_on": on, "toggle_off": off}, index=index)


def test_is_toggle_recognises_schedules_and_toggle_frames_but_not_timestamps() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="10min")
    assert is_toggle(ToggleSchedule(period=pd.Timedelta(hours=1))) is True
    assert is_toggle(_toggle_df(index, [True, False, False], [False, True, False])) is True
    assert is_toggle(pd.Timestamp("2025-01-01")) is False


def test_resolve_toggle_explicit_frame_excludes_neither_rows_from_every_segment() -> None:
    """A 'neither' row (both flags False) is in neither upgraded, campaign nor training baseline."""
    index = pd.date_range("2025-01-01", periods=4, freq="10min")
    #                     on      off     neither  on
    toggle_df = _toggle_df(index, on=[True, False, False, True], off=[False, True, False, False])

    rows = resolve_toggle(toggle_df, index)

    assert list(rows.upgraded) == [True, False, False, True]
    assert list(rows.campaign_baseline) == [False, True, False, False]
    # training_baseline = (index < first_upgraded) U off; first upgraded is index[0], so no pre rows.
    assert list(rows.training_baseline) == [False, True, False, False]
    # the neither row (position 2) is excluded everywhere.
    assert not rows.upgraded[2]
    assert not rows.campaign_baseline[2]
    assert not rows.training_baseline[2]


def test_resolve_toggle_training_baseline_includes_pre_first_upgraded_rows() -> None:
    """Rows before the first 'on' row join the lenient training baseline but not the campaign one."""
    index = pd.date_range("2025-01-01", periods=4, freq="10min")
    #  pre-campaign 'neither', then off, then on, then off
    toggle_df = _toggle_df(index, on=[False, False, True, False], off=[False, True, False, True])

    rows = resolve_toggle(toggle_df, index)

    assert list(rows.upgraded) == [False, False, True, False]
    assert list(rows.campaign_baseline) == [False, True, False, True]
    # position 0 is before the first upgraded row (index[2]) -> in training baseline only.
    assert list(rows.training_baseline) == [True, True, False, True]


def test_resolve_toggle_rejects_simultaneous_on_and_off() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="10min")
    bad = _toggle_df(index, on=[True, False], off=[True, False])
    with pytest.raises(ValueError, match="toggle_on and toggle_off"):
        resolve_toggle(bad, index)


def test_resolve_toggle_rejects_toggle_df_missing_a_flag_column() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="10min")
    missing_off = pd.DataFrame({"toggle_on": [True, False]}, index=index)
    with pytest.raises(ValueError, match=r"missing.*toggle_off"):
        resolve_toggle(missing_off, index)


def test_resolve_toggle_rejects_toggle_df_with_duplicate_timestamps() -> None:
    """A downstream frame with duplicate timestamps fails with a clear error, not pandas' opaque one."""
    ts = pd.Timestamp("2025-01-01")
    dup = _toggle_df(pd.DatetimeIndex([ts, ts]), on=[True, False], off=[False, True])
    with pytest.raises(ValueError, match="index must be unique"):
        resolve_toggle(dup, pd.date_range("2025-01-01", periods=2, freq="10min"))


def test_resolve_toggle_reindexes_onto_a_repeated_long_frame_index() -> None:
    """A long frame repeats each timestamp per turbine; the labels broadcast to every row."""
    ts = pd.date_range("2025-01-01", periods=3, freq="10min")
    toggle_df = _toggle_df(ts, on=[True, False, True], off=[False, True, False])
    long_index = ts.repeat(2)  # two turbines per timestamp

    rows = resolve_toggle(toggle_df, long_index)

    assert list(rows.upgraded) == [True, True, False, False, True, True]
    assert list(rows.campaign_baseline) == [False, False, True, True, False, False]


def test_resolve_toggle_prepost_timestamp_splits_on_the_changeover() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="10min")
    changeover = index[2]

    rows = resolve_toggle(changeover, index)

    assert list(rows.upgraded) == [False, False, True, True]
    # prepost: both baselines are the pre-changeover rows.
    assert list(rows.campaign_baseline) == [True, True, False, False]
    assert list(rows.training_baseline) == [True, True, False, False]


def test_resolve_toggle_schedule_matches_legacy_treated_mask_definitions() -> None:
    """On a periodic ToggleSchedule the row-sets equal the legacy binary masks (behaviour-preserving).

    upgraded == treated_mask; campaign_baseline == ~treated & (index>=start);
    training_baseline == ~treated (pre-campaign U off).
    """
    index = pd.date_range("2025-01-01", periods=48, freq="30min")
    start = index[12]
    schedule = ToggleSchedule(period=pd.Timedelta(hours=2), start=start)

    rows = resolve_toggle(schedule, index)
    legacy_treated = np.asarray(treated_mask(index, schedule))
    at_or_after_start = np.asarray(index >= start)

    assert np.array_equal(rows.upgraded, legacy_treated)
    assert np.array_equal(rows.campaign_baseline, ~legacy_treated & at_or_after_start)
    assert np.array_equal(rows.training_baseline, ~legacy_treated)


def test_build_toggle_df_from_schedule_is_neither_before_start_then_exactly_one_flag() -> None:
    index = pd.date_range("2025-01-01", periods=24, freq="1h")
    start = index[6]
    schedule = ToggleSchedule(period=pd.Timedelta(hours=4), start=start)

    toggle_df = build_toggle_df(index, schedule)

    assert list(toggle_df.columns) == ["toggle_on", "toggle_off"]
    pre = index < start
    # before start: neither flag set.
    assert not toggle_df.loc[pre, "toggle_on"].any()
    assert not toggle_df.loc[pre, "toggle_off"].any()
    # from start: exactly one flag true per row.
    post = toggle_df.loc[~pre]
    assert (post["toggle_on"] ^ post["toggle_off"]).all()


def test_resolve_toggle_returns_row_sets_dataclass() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="10min")
    rows = resolve_toggle(_toggle_df(index, [True, False], [False, True]), index)
    assert isinstance(rows, ToggleRowSets)


def test_toggle_upgrade_start_prepost_is_the_changeover() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="10min")
    assert toggle_upgrade_start(index[2], index) == index[2]


def test_toggle_upgrade_start_schedule_is_its_start() -> None:
    index = pd.date_range("2025-01-01", periods=48, freq="30min")
    schedule = ToggleSchedule(period=pd.Timedelta(hours=2), start=index[12])
    assert toggle_upgrade_start(schedule, index) == index[12]


def test_toggle_upgrade_start_frame_is_first_upgraded_row_within_index() -> None:
    """The start aligns to the analysis index: on-rows the analysis never sees do not count."""
    index = pd.date_range("2025-01-01", periods=4, freq="10min")
    # An earlier on-row before the analysis window would win a raw ``.min()``, but is not in ``index``.
    frame_index = pd.DatetimeIndex([pd.Timestamp("2024-12-31"), *index])
    toggle_df = _toggle_df(frame_index, on=[True, False, True, False, False], off=[False, True, False, True, True])
    assert toggle_upgrade_start(toggle_df, index) == index[1]


def test_toggle_upgrade_start_frame_never_on_falls_back_to_index_min() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="10min")
    toggle_df = _toggle_df(index, on=[False, False, False], off=[True, True, True])
    assert toggle_upgrade_start(toggle_df, index) == index.min()
