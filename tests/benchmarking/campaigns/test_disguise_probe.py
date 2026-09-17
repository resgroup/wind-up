"""Tests for the disguise: a toggle campaign's blocks relabelled as a prepost campaign's periods.

The probe's runs are a driver (they need the Hill of Towie download and the power model); what is
unit-tested here is the remap itself -- that it keeps every record, moves each one only in whole
blocks, keeps SCADA and reanalysis aligned, and deals the same two halves the matching toggle
schedule toggles -- and that each leg declares the same farm and window with only its timing
differing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns.disguise_probe import (
    DISGUISE_BLOCK,
    DISGUISE_WINDOW,
    block_count,
    disguise_frame,
    disguise_timestamps,
    disguised_changeover,
    leg_campaign,
    leg_summary,
    matching_toggle,
    source_northed_scada,
    undisguise_timestamps,
)
from benchmarking.campaigns.placebo import PLACEBO_TURBINES
from benchmarking.harness.northing import ERA5_WD_COL, north_scada
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule, treated_mask

TEST_TURBINES = ("T07", "T11", "T12")
SHORT_WINDOW = (pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2018-01-09", tz="UTC"))


def _index(*, window: tuple[pd.Timestamp, pd.Timestamp] = SHORT_WINDOW, freq: str = "10min") -> pd.DatetimeIndex:
    """Return a regular index over ``window``, end exclusive."""
    return pd.date_range(window[0], window[1], freq=freq, tz="UTC", inclusive="left", name="timestamp")


def test_the_default_window_is_an_even_number_of_whole_blocks() -> None:
    assert block_count() == 730
    assert disguised_changeover() == pd.Timestamp("2018-09-01", tz="UTC")


def test_a_window_that_does_not_divide_into_whole_blocks_is_refused() -> None:
    with pytest.raises(ValueError, match="whole number"):
        block_count(window=(SHORT_WINDOW[0], SHORT_WINDOW[1] + pd.Timedelta(hours=6)))


def test_a_window_of_an_odd_number_of_blocks_is_refused() -> None:
    with pytest.raises(ValueError, match="odd number"):
        block_count(window=(SHORT_WINDOW[0], SHORT_WINDOW[1] + DISGUISE_BLOCK))


def test_a_block_that_is_not_whole_hours_is_refused() -> None:
    with pytest.raises(ValueError, match="whole number of hours"):
        block_count(window=SHORT_WINDOW, block=pd.Timedelta(minutes=90))


def test_a_backwards_window_is_refused() -> None:
    with pytest.raises(ValueError, match="run forwards"):
        block_count(window=(SHORT_WINDOW[1], SHORT_WINDOW[0]))


def test_every_record_is_kept_inside_the_window_and_none_collides() -> None:
    disguised = disguise_timestamps(_index(), window=SHORT_WINDOW)
    assert len(disguised) == len(_index())
    assert disguised.is_unique
    assert disguised.min() >= SHORT_WINDOW[0]
    assert disguised.max() < SHORT_WINDOW[1]


def test_time_of_day_survives_the_remap() -> None:
    index = _index()
    disguised = disguise_timestamps(index, window=SHORT_WINDOW)
    assert (disguised.time == index.time).all()


def test_even_blocks_become_the_pre_period_and_odd_blocks_the_post_period() -> None:
    index = _index()
    disguised = disguise_timestamps(index, window=SHORT_WINDOW)
    source_block = np.asarray((index - SHORT_WINDOW[0]) // DISGUISE_BLOCK).astype(int)
    is_post = disguised >= disguised_changeover(window=SHORT_WINDOW)
    assert (is_post == (source_block % 2 == 1)).all()
    assert is_post.sum() == (~is_post).sum()


def test_blocks_keep_their_order_within_each_half() -> None:
    index = _index()
    disguised = disguise_timestamps(index, window=SHORT_WINDOW)
    changeover = disguised_changeover(window=SHORT_WINDOW)
    for half in (disguised < changeover, disguised >= changeover):
        moved = pd.Series(disguised[half], index=index[half]).sort_index()
        assert moved.is_monotonic_increasing


def test_the_matching_toggle_treats_exactly_the_rows_the_disguise_makes_post() -> None:
    index = _index()
    is_post = disguise_timestamps(index, window=SHORT_WINDOW) >= disguised_changeover(window=SHORT_WINDOW)
    toggled = treated_mask(index, matching_toggle(window=SHORT_WINDOW))
    assert (toggled == is_post).all()


def test_the_default_window_deals_each_half_the_same_months() -> None:
    index = _index(window=DISGUISE_WINDOW, freq="1h")
    disguised = disguise_timestamps(index)
    pre = index[disguised < disguised_changeover()].month.value_counts().sort_index()
    post = index[disguised >= disguised_changeover()].month.value_counts().sort_index()
    assert (pre == post).all()


def test_the_disguise_can_be_undone() -> None:
    index = _index()
    there = disguise_timestamps(index, window=SHORT_WINDOW)
    back = undisguise_timestamps(there, window=SHORT_WINDOW)
    assert (back == index).all()


def test_undoing_the_disguise_is_itself_inverted_by_the_disguise() -> None:
    index = _index()
    back = undisguise_timestamps(index, window=SHORT_WINDOW)
    assert (disguise_timestamps(back, window=SHORT_WINDOW) == index).all()


def test_a_frame_is_trimmed_to_the_window_and_keeps_its_rows_with_their_timestamps() -> None:
    index = _index()
    frame = pd.DataFrame({"source": index}, index=index)
    outside = pd.DataFrame(
        {"source": [pd.NaT, pd.NaT]},
        index=pd.DatetimeIndex([SHORT_WINDOW[0] - pd.Timedelta(hours=1), SHORT_WINDOW[1]], name=index.name),
    )
    disguised = disguise_frame(pd.concat([outside, frame]).sort_index(), window=SHORT_WINDOW)
    assert len(disguised) == len(frame)
    assert disguised.index.is_monotonic_increasing
    assert disguised.index.name == index.name
    assert (disguise_timestamps(pd.DatetimeIndex(disguised["source"]), window=SHORT_WINDOW) == disguised.index).all()


def test_a_long_frame_keeps_every_turbines_record_at_the_same_new_timestamp() -> None:
    index = _index()
    long = pd.concat([pd.DataFrame({"turbine": wtg, "source": index}, index=index) for wtg in TEST_TURBINES])
    disguised = disguise_frame(long, window=SHORT_WINDOW)
    assert len(disguised) == len(long)
    counts = disguised.groupby("turbine").size()
    assert (counts == len(index)).all()
    per_turbine = {wtg: group["source"].to_numpy() for wtg, group in disguised.groupby("turbine")}
    for wtg in TEST_TURBINES[1:]:
        assert (per_turbine[wtg] == per_turbine[TEST_TURBINES[0]]).all()


def test_scada_and_reanalysis_are_moved_together() -> None:
    hourly = pd.DataFrame({"source": _index(freq="1h")}, index=_index(freq="1h"))
    ten_minute = pd.DataFrame({"source": _index()}, index=_index())
    disguised_hourly = disguise_frame(hourly, window=SHORT_WINDOW)
    disguised_scada = disguise_frame(ten_minute, window=SHORT_WINDOW)
    on_the_hour = disguised_scada[disguised_scada.index.minute == 0]
    assert (on_the_hour["source"].to_numpy() == disguised_hourly["source"].to_numpy()).all()


def test_the_prepost_leg_changes_over_in_the_middle_of_the_window() -> None:
    campaign = leg_campaign("prepost", turbines=TEST_TURBINES, window=SHORT_WINDOW)
    assert campaign.upgrade_timing == disguised_changeover(window=SHORT_WINDOW)
    assert campaign.analysis_period == SHORT_WINDOW


def test_the_toggle_leg_toggles_every_block_from_the_start_of_the_window() -> None:
    campaign = leg_campaign("toggle", turbines=TEST_TURBINES, window=SHORT_WINDOW)
    assert campaign.upgrade_timing == ToggleSchedule(period=2 * DISGUISE_BLOCK, start=SHORT_WINDOW[0], start_on=False)
    assert campaign.analysis_period == SHORT_WINDOW


def test_both_legs_declare_the_same_farm_and_inject_nothing() -> None:
    campaigns = [
        leg_campaign(leg, test_wtg="T12", turbines=TEST_TURBINES, window=SHORT_WINDOW) for leg in ("prepost", "toggle")
    ]
    assert len({tuple(c.turbines) for c in campaigns}) == 1
    assert len({tuple(c.candidate_references) for c in campaigns}) == 1
    assert len({c.analysis_period for c in campaigns}) == 1
    assert all(c.upgrades == [] and c.faults == [] for c in campaigns)


def test_the_test_turbine_is_left_out_of_the_reference_pool() -> None:
    campaign = leg_campaign("prepost", test_wtg="T12", turbines=TEST_TURBINES, window=SHORT_WINDOW)
    assert campaign.upgraded_turbines == ["T12"]
    assert "T12" not in campaign.candidate_references
    assert len(campaign.candidate_references) == len(TEST_TURBINES) - 1


def test_the_default_leg_declares_the_placebo_farm() -> None:
    campaign = leg_campaign("prepost", window=SHORT_WINDOW)
    assert set(campaign.turbines) == set(PLACEBO_TURBINES)


def test_an_unknown_leg_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown leg"):
        leg_campaign("interleaved", turbines=TEST_TURBINES, window=SHORT_WINDOW)  # type: ignore[arg-type]


def test_leg_summary_is_one_row_per_leg_and_method() -> None:
    readings = pd.DataFrame(
        {
            "leg": ["prepost", "prepost", "toggle", "toggle"],
            "method": ["power_model"] * 4,
            "turbine": ["T01", "T02", "T01", "T02"],
            "reading_pp": [0.0, 1.0, -2.0, 4.0],
        }
    )
    summary = leg_summary(readings).set_index("leg")
    assert len(summary) == 2
    assert summary.loc["prepost", "mean_pp"] == pytest.approx(0.5)
    assert summary.loc["toggle", "median_pp"] == pytest.approx(1.0)
    assert summary.loc["toggle", "max_abs_pp"] == pytest.approx(4.0)
    assert (summary["n"] == 2).all()


def test_neither_leg_rediscovers_northing() -> None:
    for leg in ("prepost", "toggle"):
        campaign = leg_campaign(leg, turbines=TEST_TURBINES, window=SHORT_WINDOW)
        assert campaign.north_offsets == []


def _northing_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Return two turbines' SCADA whose nacelle position steps part way through, and the anchor."""
    index = _index(window=SHORT_WINDOW)
    era5_wd = pd.Series(np.linspace(0.0, 359.0, len(index)) % 360.0, index=index)
    step = SHORT_WINDOW[0] + pd.Timedelta(days=4)
    frames = []
    for turbine, offset in (("T07", 12.0), ("T11", -9.0), ("T12", 0.0)):
        applied = np.where(index >= step, offset, 0.0)
        frames.append(
            pd.DataFrame(
                {
                    HOT_COLUMNS.turbine: turbine,
                    HOT_COLUMNS.nacelle_position: (era5_wd.to_numpy() + applied) % 360.0,
                    HOT_COLUMNS.active_power: 1000.0,
                    HOT_COLUMNS.availability: 600.0,
                },
                index=index,
            )
        )
    return pd.concat(frames).sort_index(kind="stable"), era5_wd


def test_the_source_northing_is_written_into_the_nacelle_position_itself() -> None:
    scada_df, era5_wd = _northing_frame()
    era5_df = pd.DataFrame({ERA5_WD_COL: era5_wd})
    northed = source_northed_scada(scada_df, era5_df=era5_df, rated_power_kw=2000.0)
    expected = north_scada(
        scada_df,
        columns=HOT_COLUMNS,
        north_offsets=None,
        rated_power_kw=2000.0,
        era5_wd=era5_wd,
    )
    assert HOT_COLUMNS.northed("nacelle_position") not in northed.columns
    assert northed[HOT_COLUMNS.nacelle_position].to_numpy() == pytest.approx(
        expected[HOT_COLUMNS.northed("nacelle_position")].to_numpy()
    )


def test_the_source_northing_survives_the_disguise_row_for_row() -> None:
    scada_df, era5_wd = _northing_frame()
    northed = source_northed_scada(scada_df, era5_df=pd.DataFrame({ERA5_WD_COL: era5_wd}), rated_power_kw=2000.0)
    disguised = disguise_frame(northed, window=SHORT_WINDOW)
    back = disguised.set_index(undisguise_timestamps(pd.DatetimeIndex(disguised.index), window=SHORT_WINDOW))
    back = back.sort_index(kind="stable")
    assert back[HOT_COLUMNS.nacelle_position].to_numpy() == pytest.approx(
        northed[HOT_COLUMNS.nacelle_position].to_numpy()
    )
