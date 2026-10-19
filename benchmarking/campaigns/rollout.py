"""Synthetic trial-then-rollout campaigns: realistic works schedules on real SCADA.

A campaign draws a trial of a few turbines from :func:`~wind_up.campaign_design.design_campaign`,
schedules their works on UK working days with one or two teams, and optionally rolls the upgrade
out to every other turbine some months later. The upgrade injected is the AeroUp shape, scaled by
a multiplier. The works become a works table, so the period selector chooses each analysed
turbine's span as it would for a real campaign. A site's campaigns tile its data: each tile's
trial starts at its own place in the data, and tests turbines the earlier tiles did not.

Draw a campaign and build it for one multiplier, on 12 months of data before it and 6 after::

    draw = draw_rollout(hot_site(), seed=7, full_rollout=True, tile=2)
    campaign = rollout_campaign(draw, multiplier=1.0, post_months=6)
    start, end = draw.data_window(pre_months=12, post_months=6)
    scada, _ = load_hot_scada(start_dt=start, end_dt_excl=end)
    dataset = campaign.generate(scada)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.campaigns.declaration import SyntheticCampaign, Window, layout_from_coords
from benchmarking.campaigns.loader import centroid, era5_window
from benchmarking.harness.wind_rose import era5_rose
from benchmarking.synthetic import HOT_COLUMNS, WindSpeedCpChange
from benchmarking.synthetic.sources.greenbyte import (
    GREENBYTE_COLUMNS,
    KELMARSH,
    PENMANSHIEL,
    GreenbyteFarm,
    load_greenbyte_metadata,
)
from benchmarking.synthetic.sources.hill_of_towie import HOT_COORDINATES, HOT_RATED_POWER_KW, HOT_ROTOR_DIAMETER_M
from wind_up.campaign_design import design_campaign
from wind_up.wake_reach import reach_from_uplift, uniform_rose
from wind_up_v0.era5 import get_era5_hourly_df

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from pathlib import Path

    from benchmarking.synthetic import ColumnSchema
    from wind_up.campaign_design import CampaignDesign
    from wind_up.wake_reach import Rose

# The AeroUp shape: a region-2 Cp gain tailing to nothing by rated.
AEROUP_WS_POINTS = (4.0, 8.0, 12.0, 16.0)
AEROUP_DELTAS = (0.07, 0.07, 0.045, 0.0)
# What a vendor would claim for the AeroUp, declared by every rollout campaign whatever its multiplier.
AEROUP_EXPECTED_UPLIFT = 0.05
# The multiplier of the AeroUp shape that gives about 1 % AEP, and what a campaign of it declares.
SMALL_MULTIPLIER = 0.22
SMALL_EXPECTED_UPLIFT = 0.01

WORKING_DAYS_PER_TURBINE = (5, 12)
TRIAL_TEAMS = (1, 2)
ROLLOUT_TEAMS = (1, 2)
# A trial starts this far into the site's data, so a campaign can see up to this much before it.
TRIAL_PRE_MONTHS = 24
TRIAL_MIN_POST_MONTHS = 12
TRIAL_MAX_TEST_TURBINES = 3
# Room left after a trial's start for its works, beyond their scheduled length.
TRIAL_START_MARGIN_DAYS = 31
# Tile i's trial starts at the fractional part of i times this along the span trials may start in.
TILE_STEP = (np.sqrt(5.0) - 1.0) / 2.0
ROLLOUT_LAG_MONTHS = (6, 9)
# The Christmas shutdown runs from 24 December to 2 January inclusive.
SHUTDOWN_FROM = (12, 24)
SHUTDOWN_TO = (1, 2)

WORKS_COLUMNS = ("Turbine", "First date of works", "Last date of works")

# One-off changes to UK bank holidays: (moved away, added).
_ONE_OFF_HOLIDAYS: dict[int, tuple[tuple[dt.date, ...], tuple[dt.date, ...]]] = {
    2011: ((), (dt.date(2011, 4, 29),)),
    2012: ((dt.date(2012, 5, 28),), (dt.date(2012, 6, 4), dt.date(2012, 6, 5))),
    2020: ((dt.date(2020, 5, 4),), (dt.date(2020, 5, 8),)),
    2022: ((dt.date(2022, 5, 30),), (dt.date(2022, 6, 2), dt.date(2022, 6, 3), dt.date(2022, 9, 19))),
    2023: ((), (dt.date(2023, 5, 8),)),
}

_SATURDAY = 5
_SUNDAY = 6


@dataclass(frozen=True)
class RolloutSite:
    """A wind farm whose real SCADA a rollout campaign is built on.

    :param name: the site's short name
    :param coords: turbine name to ``(latitude, longitude)``
    :param rotor_diameter_m: every turbine's rotor diameter
    :param rated_power_kw: the turbines' rated power
    :param columns: the source-native column schema of its SCADA
    :param data_start: the first SCADA timestamp
    :param data_end: the end of its SCADA, exclusive
    :param excluded: turbines neither tested nor used as references; they still block wakes and are
        worked in a full rollout
    """

    name: str
    coords: Mapping[str, tuple[float, float]]
    rotor_diameter_m: float
    rated_power_kw: float
    columns: ColumnSchema
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    excluded: frozenset[str] = frozenset()

    def layout_frame(self) -> pd.DataFrame:
        """Return the campaign-design layout of the site."""
        return pd.DataFrame(
            {
                "name": list(self.coords),
                "latitude": [lat for lat, _ in self.coords.values()],
                "longitude": [lon for _, lon in self.coords.values()],
                "rotor_diameter_m": self.rotor_diameter_m,
                "wind_farm": self.name,
            }
        )


def hot_site() -> RolloutSite:
    """Return Hill of Towie over 2016-2020, before its real AeroUp works, without its eastern turbines.

    T16 to T21 are neither tested nor used as references: T16's noise-mode sectors move with its
    2017 yaw re-calibration, T17 sits in its wake, and the rest of the eastern group is left out
    with them.
    """
    return RolloutSite(
        name="Hill of Towie",
        coords=dict(HOT_COORDINATES),
        rotor_diameter_m=HOT_ROTOR_DIAMETER_M,
        rated_power_kw=HOT_RATED_POWER_KW,
        columns=HOT_COLUMNS,
        data_start=pd.Timestamp("2016-01-01", tz="UTC"),
        data_end=pd.Timestamp("2021-01-01", tz="UTC"),
        excluded=frozenset({"T16", "T17", "T18", "T19", "T20", "T21"}),
    )


def greenbyte_site(farm: GreenbyteFarm, *, data_dir: Path | None = None) -> RolloutSite:
    """Return a Greenbyte farm from its data start to the end of its last published year.

    The turbine metadata is read from ``data_dir``.
    """
    metadata = load_greenbyte_metadata(farm, data_dir=data_dir)
    return RolloutSite(
        name=farm.name,
        coords={
            str(n): (float(lat), float(lon))
            for n, lat, lon in zip(metadata["Name"], metadata["Latitude"], metadata["Longitude"], strict=True)
        },
        rotor_diameter_m=farm.rotor_diameter_m,
        rated_power_kw=farm.rated_power_kw,
        columns=GREENBYTE_COLUMNS,
        data_start=farm.data_start,
        data_end=pd.Timestamp(year=max(farm.years) + 1, month=1, day=1, tz="UTC"),
    )


def site_rose(site: RolloutSite) -> Rose:
    """Return the site's wind rose from the reanalysis at its centroid over its whole data span.

    Fetches through the reanalysis cache, which needs network on a miss.
    """
    lat, lon = centroid(dict(site.coords))
    start, end = era5_window(site.data_start, site.data_end)
    era5 = get_era5_hourly_df(lat=lat, lon=lon, start_date=start, end_date=end)
    return era5_rose(era5, start=site.data_start, end=site.data_end)


def penmanshiel_site(*, data_dir: Path | None = None) -> RolloutSite:
    """Return Penmanshiel from its commercial operation to the end of its last published year."""
    return greenbyte_site(PENMANSHIEL, data_dir=data_dir)


def kelmarsh_site(*, data_dir: Path | None = None) -> RolloutSite:
    """Return Kelmarsh from its commercial operation to the end of its last published year."""
    return greenbyte_site(KELMARSH, data_dir=data_dir)


# --- the calendar ------------------------------------------------------------------------------


@cache
def bank_holidays(year: int) -> frozenset[dt.date]:
    """Return the bank holidays of ``year`` in Scotland or in England and Wales, weekend substitutes included."""
    easter = (pd.Timestamp(year=year, month=1, day=1) + pd.offsets.Easter()).date()
    days = {
        *_new_year(year),
        easter - dt.timedelta(days=2),
        easter + dt.timedelta(days=1),
        _first_monday(year, 5),
        _last_monday(year, 5),
        _first_monday(year, 8),
        _last_monday(year, 8),
        _next_weekday(dt.date(year, 11, 30)),
        *_christmas(year),
    }
    moved, added = _ONE_OFF_HOLIDAYS.get(year, ((), ()))
    return frozenset((days - set(moved)) | set(added))


def is_working_day(day: dt.date) -> bool:
    """Whether works happen on ``day``: a weekday, not a bank holiday, outside the Christmas shutdown."""
    in_shutdown = (day.month, day.day) >= SHUTDOWN_FROM or (day.month, day.day) <= SHUTDOWN_TO
    return day.weekday() < _SATURDAY and not in_shutdown and day not in bank_holidays(day.year)


def working_day_on_or_after(day: dt.date) -> dt.date:
    """Return ``day`` if it is a working day, else the next one."""
    while not is_working_day(day):
        day += dt.timedelta(days=1)
    return day


def working_days_between(first: dt.date, last: dt.date) -> list[dt.date]:
    """Return the working days from ``first`` to ``last``, both inclusive."""
    return [d.date() for d in pd.date_range(first, last, freq="D") if is_working_day(d.date())]


def _next_weekday(day: dt.date) -> dt.date:
    return day + dt.timedelta(days={_SATURDAY: 2, _SUNDAY: 1}.get(day.weekday(), 0))


def _new_year(year: int) -> tuple[dt.date, dt.date]:
    first, second = dt.date(year, 1, 1), dt.date(year, 1, 2)
    if first.weekday() == _SATURDAY:
        return dt.date(year, 1, 3), dt.date(year, 1, 4)
    if first.weekday() == _SUNDAY:
        return second, dt.date(year, 1, 3)
    if second.weekday() == _SATURDAY:
        return first, dt.date(year, 1, 4)
    return first, second


def _christmas(year: int) -> tuple[dt.date, dt.date]:
    christmas, boxing = dt.date(year, 12, 25), dt.date(year, 12, 26)
    if christmas.weekday() == _SATURDAY:
        return dt.date(year, 12, 27), dt.date(year, 12, 28)
    if christmas.weekday() == _SUNDAY:
        return dt.date(year, 12, 27), boxing
    if boxing.weekday() == _SATURDAY:
        return christmas, dt.date(year, 12, 28)
    return christmas, boxing


def _first_monday(year: int, month: int) -> dt.date:
    day = dt.date(year, month, 1)
    return day + dt.timedelta(days=-day.weekday() % 7)


def _last_monday(year: int, month: int) -> dt.date:
    last = (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).date()
    return last - dt.timedelta(days=last.weekday())


# --- the works schedule ------------------------------------------------------------------------


def works_schedule(
    order: Sequence[str],
    *,
    start: dt.date,
    n_teams: int,
    working_days: tuple[int, int] = WORKING_DAYS_PER_TURBINE,
    rng: np.random.Generator,
) -> dict[str, Window]:
    """Schedule works on ``order``'s turbines, each team taking the next turbine when it finishes.

    Every team starts on the first working day on or after ``start``. Each turbine takes a number of
    working days drawn uniformly from ``working_days`` (inclusive), so its works start and end on
    working days. Windows are whole days, ``[first 00:00, last + 1 day 00:00)`` UTC.

    :param order: the turbines, in the order they are worked
    :param start: the earliest day works may start
    :param n_teams: how many turbines may be in works at once
    :param working_days: the least and most working days a turbine takes
    :param rng: draws each turbine's working days
    """
    free = [working_day_on_or_after(start)] * n_teams
    works: dict[str, Window] = {}
    for turbine in order:
        team = min(range(n_teams), key=free.__getitem__)
        days = int(rng.integers(working_days[0], working_days[1] + 1))
        first = last = free[team]
        for _ in range(days - 1):
            last = working_day_on_or_after(last + dt.timedelta(days=1))
        works[turbine] = (_utc(first), _utc(last + dt.timedelta(days=1)))
        free[team] = working_day_on_or_after(last + dt.timedelta(days=1))
    return works


def works_table(works: Mapping[str, Window]) -> pd.DataFrame:
    """Return ``works`` as a works table in the Zenodo shape: turbine, first and last date of works."""
    rows = [
        (turbine, start.date().isoformat(), (end - pd.Timedelta(days=1)).date().isoformat())
        for turbine, (start, end) in works.items()
    ]
    return pd.DataFrame(rows, columns=list(WORKS_COLUMNS))


def _utc(day: dt.date) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


# --- the trial-then-rollout draw ---------------------------------------------------------------


@dataclass(frozen=True)
class RolloutDraw:
    """A drawn trial-then-rollout campaign: which turbines, and when each is worked.

    :param site: the site it is drawn on
    :param seed: the seed it was drawn from
    :param trial: the trial turbines, the analysed ones
    :param rollout: the turbines worked in the full rollout; empty without one
    :param works: every worked turbine's works window
    """

    site: RolloutSite
    seed: int
    trial: tuple[str, ...]
    rollout: tuple[str, ...]
    works: Mapping[str, Window]

    @property
    def trial_start(self) -> pd.Timestamp:
        """When the first trial works start."""
        return min(self.works[t][0] for t in self.trial)

    @property
    def last_trial_end(self) -> pd.Timestamp:
        """When the last trial works end."""
        return max(self.works[t][1] for t in self.trial)

    def data_start(self, pre_months: int) -> pd.Timestamp:
        """Return the start of the data a campaign with ``pre_months`` before the first trial works sees."""
        return max(self.site.data_start, self.trial_start - pd.DateOffset(months=pre_months))

    def data_end(self, post_months: int) -> pd.Timestamp:
        """Return the end of the data a campaign with ``post_months`` after the last trial works sees."""
        return min(self.site.data_end, self.last_trial_end + pd.DateOffset(months=post_months))

    def data_window(self, *, pre_months: int, post_months: int) -> Window:
        """Return the ``(start, end)`` of the SCADA a campaign with these pre and post lengths is built on."""
        return self.data_start(pre_months), self.data_end(post_months)


def draw_rollout(
    site: RolloutSite,
    *,
    seed: int,
    full_rollout: bool,
    tile: int = 0,
    tested: Collection[str] = (),
    rose: Rose | None = None,
    expected_uplift: float = AEROUP_EXPECTED_UPLIFT,
) -> RolloutDraw:
    """Draw a trial-then-rollout campaign on ``site``, as its tile ``tile``.

    The trial is a compliant campaign design of one to three test turbines, never more than one
    fewer than the site allows, none of ``site.excluded``: the design with the highest total kept
    share for an upgrade expected to give ``expected_uplift`` under ``rose``, among the turbines not
    in ``tested``, or among all of them when those cannot make the trial.
    Its works start at the fractional part of ``tile * TILE_STEP`` along the days from 24 months
    into the data to the last day that leaves room for the works and 12 months after them, with
    one or two teams. How much data before the trial a campaign sees is chosen when the campaign
    is built (see :meth:`RolloutDraw.data_window`). A full rollout works every other turbine,
    starting 6 to 9 months after the last trial works end.

    :param site: the site to draw on
    :param seed: the draw's seed; the same seed, tile and ``tested`` give the same draw
    :param full_rollout: whether every other turbine is worked after the trial
    :param tile: which of the site's tiles; sets where in the data the trial starts
    :param tested: turbines earlier tiles tested
    :param rose: the site's wind rose (see :func:`site_rose`); uniform when not given
    :param expected_uplift: what the campaign declares, which sets the wake reach the trial is designed for
    :raises ValueError: if the site's data is too short for the trial
    """
    rng = np.random.default_rng(seed)
    layout = site.layout_frame()
    excluded = sorted(site.excluded)
    rose = rose if rose is not None else uniform_rose()
    most = design_campaign(layout, excluded=excluded, expected_uplift=expected_uplift, wind_rose=rose).max_test_turbines
    order = [str(t) for t in rng.permutation(list(site.coords)) if t not in site.excluded]
    n_trial = int(rng.integers(1, max(1, min(most - 1, TRIAL_MAX_TEST_TURBINES)) + 1))
    design = _trial_design(
        layout,
        excluded=excluded,
        tested=sorted(set(tested) - site.excluded),
        n_trial=n_trial,
        rose=rose,
        expected_uplift=expected_uplift,
        seed=seed,
    )
    trial = tuple(t for t in order if t in design.test_turbines)

    n_teams = int(rng.choice(TRIAL_TEAMS))
    days_seed = int(rng.integers(2**32))
    earliest = (site.data_start + pd.DateOffset(months=TRIAL_PRE_MONTHS)).date()
    nominal = works_schedule(trial, start=earliest, n_teams=n_teams, rng=np.random.default_rng(days_seed))
    length = max(end for _, end in nominal.values()) - min(start for start, _ in nominal.values())
    latest = (
        site.data_end
        - pd.DateOffset(months=TRIAL_MIN_POST_MONTHS)
        - length
        - pd.Timedelta(days=TRIAL_START_MARGIN_DAYS)
    ).date()
    if latest < earliest:
        msg = f"{site.name}: no room for a trial {TRIAL_PRE_MONTHS} months in and {TRIAL_MIN_POST_MONTHS} from the end"
        raise ValueError(msg)
    offset = int(np.floor(((tile * TILE_STEP) % 1.0) * ((latest - earliest).days + 1)))
    start = earliest + dt.timedelta(days=offset)
    works = works_schedule(trial, start=start, n_teams=n_teams, rng=np.random.default_rng(days_seed))
    last_trial_end = max(end for _, end in works.values())
    if last_trial_end + pd.DateOffset(months=TRIAL_MIN_POST_MONTHS) > site.data_end:
        msg = (
            f"{site.name}: the trial works end {last_trial_end.date()}, leaving less than "
            f"{TRIAL_MIN_POST_MONTHS} months of data after them"
        )
        raise ValueError(msg)
    rollout: tuple[str, ...] = ()
    if full_rollout:
        rollout = tuple(str(t) for t in rng.permutation([t for t in site.coords if t not in trial]))
        lag_start = _working_day_between(
            (last_trial_end + pd.DateOffset(months=ROLLOUT_LAG_MONTHS[0])).date(),
            (last_trial_end + pd.DateOffset(months=ROLLOUT_LAG_MONTHS[1])).date(),
            rng=rng,
        )
        works |= works_schedule(rollout, start=lag_start, n_teams=int(rng.choice(ROLLOUT_TEAMS)), rng=rng)
    return RolloutDraw(site=site, seed=seed, trial=trial, rollout=rollout, works=works)


def _trial_design(
    layout: pd.DataFrame,
    *,
    excluded: Sequence[str],
    tested: Sequence[str],
    n_trial: int,
    rose: Rose,
    expected_uplift: float,
    seed: int,
) -> CampaignDesign:
    """Return the best trial of ``n_trial`` among the untested turbines, else among all of them."""

    def design(reference_only: Sequence[str]) -> CampaignDesign:
        return design_campaign(
            layout,
            excluded=excluded,
            reference_only=reference_only,
            n_test=n_trial,
            expected_uplift=expected_uplift,
            wind_rose=rose,
            seed=seed,
        )

    try:
        return design(tested)
    except ValueError:
        return design(())


def draw_tiles(
    site: RolloutSite,
    *,
    seeds: Sequence[int],
    full_rollouts: Sequence[bool],
    rose: Rose | None = None,
    expected_uplift: float = AEROUP_EXPECTED_UPLIFT,
) -> list[RolloutDraw]:
    """Draw ``site``'s tiles in order, tile ``i`` from ``seeds[i]``, each among turbines no earlier tile tested.

    Once every turbine that may be tested has been, the preference starts afresh. Tile ``i`` depends
    only on the first ``i + 1`` seeds and rollouts.

    :param site: the site to draw on
    :param seeds: each tile's seed
    :param full_rollouts: whether each tile rolls the upgrade out to every other turbine
    :param rose: the site's wind rose; uniform when not given
    :param expected_uplift: what the campaigns declare (see :func:`draw_rollout`)
    """
    testable = set(site.coords) - site.excluded
    tested: set[str] = set()
    draws = []
    for tile, (seed, full_rollout) in enumerate(zip(seeds, full_rollouts, strict=True)):
        if testable <= tested:
            tested = set()
        draw = draw_rollout(
            site,
            seed=seed,
            full_rollout=full_rollout,
            tile=tile,
            tested=frozenset(tested),
            rose=rose,
            expected_uplift=expected_uplift,
        )
        tested |= set(draw.trial)
        draws.append(draw)
    return draws


def _working_day_between(first: dt.date, last: dt.date, *, rng: np.random.Generator) -> dt.date:
    """Draw a working day from ``first`` to ``last`` inclusive, uniformly."""
    days = working_days_between(first, last)
    if not days:
        msg = f"no working day from {first} to {last}"
        raise ValueError(msg)
    return days[int(rng.integers(len(days)))]


# --- the campaign ------------------------------------------------------------------------------


def aeroup_upgrade(multiplier: float) -> WindSpeedCpChange:
    """Return the AeroUp Cp change scaled by ``multiplier``."""
    return WindSpeedCpChange(ws_points=AEROUP_WS_POINTS, deltas=tuple(multiplier * d for d in AEROUP_DELTAS))


def rollout_campaign(
    draw: RolloutDraw, *, multiplier: float, post_months: int, expected_uplift: float = AEROUP_EXPECTED_UPLIFT
) -> SyntheticCampaign:
    """Build the campaign of ``draw`` with the AeroUp change scaled by ``multiplier``.

    The trial turbines are analysed; every other turbine not in ``site.excluded`` is a candidate
    reference. The data ends ``post_months`` after the last trial works (see
    :meth:`RolloutDraw.data_window`), and only works starting before then are declared; how much
    data the campaign sees before the trial is the SCADA window's choice. The change is injected
    into the trial and any rollout turbine from its own works end. The analysis span is left to the
    period selector. The campaign declares ``expected_uplift`` whatever ``multiplier`` is.

    :param draw: the drawn campaign
    :param multiplier: scales the AeroUp change; 0 injects nothing
    :param post_months: months of data after the last trial works end
    :param expected_uplift: the uplift the campaign declares, which sets its wake reach
    """
    site = draw.site
    end = draw.data_end(post_months)
    works = {t: [w] for t, w in draw.works.items() if w[0] < end}
    return SyntheticCampaign(
        upgraded_turbines=sorted(draw.trial),
        upgrade_timing=None,
        candidate_references=[t for t in site.coords if t not in draw.trial and t not in site.excluded],
        upgrades=[] if multiplier == 0 else [aeroup_upgrade(multiplier)],
        layout=layout_from_coords(site.coords, rotor_diameter_m=site.rotor_diameter_m),
        north_offsets=None,
        rated_power_kw=site.rated_power_kw,
        analysis_period=None,
        columns=site.columns,
        seed=draw.seed,
        works=works,
        other_upgraded=[t for t in draw.rollout if t in works],
        wake_reach_d=reach_from_uplift(expected_uplift),
    )
