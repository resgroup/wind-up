"""Tests for the synthetic trial-then-rollout campaign generator."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns.declaration import SyntheticCampaign, layout_from_coords
from benchmarking.campaigns.loader import read_works
from benchmarking.campaigns.rollout import (
    AEROUP_DELTAS,
    AEROUP_WS_POINTS,
    RolloutSite,
    aeroup_upgrade,
    bank_holidays,
    draw_rollout,
    hot_site,
    is_working_day,
    rollout_campaign,
    works_schedule,
    works_table,
)
from benchmarking.synthetic import HOT_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

SEEDS = range(8)
DAY = pd.Timedelta(days=1)


def _day(window_edge: pd.Timestamp) -> dt.date:
    return window_edge.date()


def _working_days_in(first: dt.date, last: dt.date) -> int:
    return sum(is_working_day(d.date()) for d in pd.date_range(first, last, freq="D"))


def small_site() -> RolloutSite:
    """Six turbines on a line 400 m apart, with five years of data."""
    return RolloutSite(
        name="line",
        coords={f"T{i:02d}": (57.5 + i * 0.0036, -3.25) for i in range(1, 7)},
        rotor_diameter_m=82.0,
        rated_power_kw=2300.0,
        columns=HOT_COLUMNS,
        data_start=pd.Timestamp("2016-01-01", tz="UTC"),
        data_end=pd.Timestamp("2021-01-01", tz="UTC"),
    )


# --- the calendar ------------------------------------------------------------------------------


def test_bank_holidays_follow_scottish_rules() -> None:
    assert bank_holidays(2019) == {
        dt.date(2019, 1, 1),
        dt.date(2019, 1, 2),
        dt.date(2019, 4, 19),  # Good Friday
        dt.date(2019, 5, 6),  # early May: first Monday
        dt.date(2019, 5, 27),  # spring: last Monday of May
        dt.date(2019, 8, 5),  # summer: first Monday of August
        dt.date(2019, 12, 2),  # St Andrew's Day, 30 November, a Saturday
        dt.date(2019, 12, 25),
        dt.date(2019, 12, 26),
    }


def test_bank_holidays_substitute_weekend_days() -> None:
    # 2022: 1 and 2 January on a weekend, Christmas Day on a Sunday
    assert {dt.date(2022, 1, 3), dt.date(2022, 1, 4)} <= bank_holidays(2022)
    assert {dt.date(2022, 12, 26), dt.date(2022, 12, 27)} <= bank_holidays(2022)
    # 2021: 2 January on a Saturday
    assert dt.date(2021, 1, 4) in bank_holidays(2021)
    # 2020: Boxing Day on a Saturday; St Andrew's Day on a Monday
    assert {dt.date(2020, 12, 28), dt.date(2020, 11, 30)} <= bank_holidays(2020)


def test_bank_holidays_carry_one_off_moves() -> None:
    assert dt.date(2020, 5, 8) in bank_holidays(2020)
    assert dt.date(2020, 5, 4) not in bank_holidays(2020)


def test_working_days_skip_weekends_bank_holidays_and_the_christmas_shutdown() -> None:
    assert is_working_day(dt.date(2019, 5, 7))
    assert not is_working_day(dt.date(2019, 5, 4))  # Saturday
    assert not is_working_day(dt.date(2019, 5, 6))  # early May bank holiday
    assert is_working_day(dt.date(2019, 4, 22))  # Easter Monday is not a Scottish bank holiday
    assert is_working_day(dt.date(2019, 8, 26))  # nor is the last Monday of August
    assert not is_working_day(dt.date(2019, 12, 24))  # shutdown starts
    assert not is_working_day(dt.date(2019, 12, 30))  # a weekday inside the shutdown
    assert not is_working_day(dt.date(2020, 1, 2))  # shutdown ends
    assert is_working_day(dt.date(2020, 1, 3))
    assert is_working_day(dt.date(2019, 12, 23))


# --- the works schedule ------------------------------------------------------------------------


def test_one_team_works_turbines_back_to_back_on_working_days() -> None:
    order = ["T01", "T02", "T03"]
    works = works_schedule(
        order, start=dt.date(2019, 12, 18), n_teams=1, working_days=(5, 5), rng=np.random.default_rng(0)
    )
    assert list(works) == order
    # 18-20 and 23 Dec, then 3 Jan: the weekend and the shutdown are skipped
    assert works["T01"] == (pd.Timestamp("2019-12-18", tz="UTC"), pd.Timestamp("2020-01-04", tz="UTC"))
    # the team moves on the next working day
    assert works["T02"][0] == pd.Timestamp("2020-01-06", tz="UTC")


@pytest.mark.parametrize("n_teams", [1, 2, 3])
def test_schedule_respects_teams_and_working_days(n_teams: int) -> None:
    order = [f"T{i:02d}" for i in range(1, 13)]
    works = works_schedule(
        order, start=dt.date(2019, 3, 30), n_teams=n_teams, working_days=(5, 12), rng=np.random.default_rng(3)
    )
    assert set(works) == set(order)
    for start, end in works.values():
        first, last = _day(start), _day(end - DAY)
        assert is_working_day(first)
        assert is_working_day(last)
        assert 5 <= _working_days_in(first, last) <= 12
    days = pd.date_range(pd.Timestamp("2019-03-30", tz="UTC"), max(e for _, e in works.values()), freq="D")
    busy = [sum(s <= d < e for s, e in works.values()) for d in days]
    assert max(busy) == n_teams
    # the first turbines start together on the first working day
    assert sorted(s for s, _ in works.values())[n_teams - 1] == pd.Timestamp("2019-04-01", tz="UTC")


def test_works_table_round_trips_through_the_loader(tmp_path: Path) -> None:
    works = works_schedule(
        ["T03", "T01"], start=dt.date(2019, 6, 3), n_teams=1, working_days=(5, 12), rng=np.random.default_rng(1)
    )
    path = tmp_path / "works.csv"
    works_table(works).to_csv(path, index=False)
    assert read_works(path) == {t: [w] for t, w in works.items()}


# --- the trial-then-rollout draw ---------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_trial_is_a_compliant_design_below_its_maximum(seed: int) -> None:
    draw = draw_rollout(hot_site(), seed=seed, full_rollout=False)
    assert 1 <= len(draw.trial) <= 10 - 1  # Hill of Towie allows 10 test turbines
    assert set(draw.works) == set(draw.trial)
    assert draw.rollout == ()


@pytest.mark.parametrize("seed", SEEDS)
def test_trial_start_leaves_the_pre_and_post_data(seed: int) -> None:
    site = hot_site()
    draw = draw_rollout(site, seed=seed, full_rollout=False)
    assert site.data_start + pd.DateOffset(months=12) <= draw.trial_start
    assert draw.trial_start <= site.data_start + pd.DateOffset(months=24)
    assert draw.last_trial_end + pd.DateOffset(months=12) <= site.data_end
    assert min(s for s, _ in draw.works.values()) == draw.trial_start


@pytest.mark.parametrize("seed", SEEDS)
def test_full_rollout_works_every_other_turbine_after_the_lag(seed: int) -> None:
    site = hot_site()
    draw = draw_rollout(site, seed=seed, full_rollout=True)
    assert set(draw.rollout) == set(site.coords) - set(draw.trial)
    assert set(draw.works) == set(site.coords)
    rollout_start = min(draw.works[t][0] for t in draw.rollout)
    assert draw.last_trial_end + pd.DateOffset(months=6) <= rollout_start
    assert rollout_start <= draw.last_trial_end + pd.DateOffset(months=9)


def test_the_same_seed_gives_the_same_draw() -> None:
    assert draw_rollout(hot_site(), seed=5, full_rollout=True) == draw_rollout(hot_site(), seed=5, full_rollout=True)
    assert draw_rollout(hot_site(), seed=5, full_rollout=True) != draw_rollout(hot_site(), seed=6, full_rollout=True)


def test_a_small_site_still_draws_one_trial_turbine() -> None:
    for seed in SEEDS:
        assert len(draw_rollout(small_site(), seed=seed, full_rollout=False).trial) >= 1


# --- the campaign ------------------------------------------------------------------------------


def test_aeroup_upgrade_scales_the_shape() -> None:
    assert aeroup_upgrade(1.0).deltas == AEROUP_DELTAS
    assert aeroup_upgrade(-1.0).deltas == tuple(-d for d in AEROUP_DELTAS)
    assert aeroup_upgrade(1.0).ws_points == AEROUP_WS_POINTS


def test_rollout_campaign_declares_works_and_leaves_the_span_to_the_selector() -> None:
    site = hot_site()
    draw = draw_rollout(site, seed=2, full_rollout=True)
    campaign = rollout_campaign(draw, multiplier=1.0, post_months=12)
    assert sorted(campaign.upgraded_turbines) == sorted(draw.trial)
    assert campaign.upgrade_timing is None
    assert campaign.analysis_period is None
    assert set(campaign.candidate_references) == set(site.coords) - set(draw.trial)
    assert campaign.north_offsets is None
    assert campaign.upgrades == [aeroup_upgrade(1.0)]
    # rollout works after the data end are not declared, nor injected
    end = draw.data_end(12)
    assert all(start < end for windows in campaign.works.values() for start, _ in windows)
    assert set(campaign.other_upgraded) == {t for t in draw.rollout if draw.works[t][0] < end}


def test_a_zero_multiplier_injects_nothing() -> None:
    draw = draw_rollout(hot_site(), seed=2, full_rollout=True)
    assert rollout_campaign(draw, multiplier=0.0, post_months=6).upgrades == []


@pytest.mark.parametrize("post_months", [1, 2, 3, 6, 9, 12])
def test_data_end_is_the_last_trial_end_plus_the_post_length(post_months: int) -> None:
    draw = draw_rollout(hot_site(), seed=4, full_rollout=False)
    assert draw.data_end(post_months) == draw.last_trial_end + pd.DateOffset(months=post_months)
    assert draw.data_window(post_months) == (hot_site().data_start, draw.data_end(post_months))


def test_other_upgraded_turbines_are_injected_from_their_own_works_end() -> None:
    coords = {"T01": (57.50, -3.25), "T02": (57.504, -3.25), "T03": (57.508, -3.25)}
    index = pd.date_range("2020-01-01", periods=10, freq="1D", tz="UTC")
    scada = pd.concat(
        [
            pd.DataFrame(
                {
                    HOT_COLUMNS.turbine: wtg,
                    HOT_COLUMNS.active_power: 900.0,
                    HOT_COLUMNS.active_power_min: 850.0,
                    HOT_COLUMNS.wind_speed: 8.0,
                    HOT_COLUMNS.wind_speed_sd: 1.0,
                    HOT_COLUMNS.gen_rpm: 1500.0,
                },
                index=index,
            )
            for wtg in coords
        ]
    )
    campaign = SyntheticCampaign(
        upgraded_turbines=["T01"],
        upgrade_timing=None,
        candidate_references=["T02", "T03"],
        upgrades=[aeroup_upgrade(1.0)],
        layout=layout_from_coords(coords, rotor_diameter_m=82.0),
        north_offsets=None,
        rated_power_kw=2300.0,
        analysis_period=None,
        works={
            "T01": [(index[1], index[2])],
            "T02": [(index[4], index[6])],
        },
        other_upgraded=["T02"],
    )
    data = campaign.generate(scada)
    changed = data.synthetic_df[HOT_COLUMNS.active_power] != data.original_df[HOT_COLUMNS.active_power]
    by_turbine = changed.groupby(data.synthetic_df[HOT_COLUMNS.turbine])
    first_changed = by_turbine.apply(lambda s: s[s].index.min())
    assert first_changed["T01"] == index[2]
    assert first_changed["T02"] == index[6]
    assert not changed[(data.synthetic_df[HOT_COLUMNS.turbine] == "T03").to_numpy()].any()
