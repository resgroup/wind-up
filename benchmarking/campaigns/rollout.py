"""Synthetic trial-then-rollout campaigns: realistic works schedules on real SCADA.

A campaign draws a trial of a few turbines from :func:`~wind_up.campaign_design.design_campaign`,
schedules their works on UK working days with one or two teams, and optionally rolls the upgrade
out to every other turbine some months later. The upgrade injected is the AeroUp shape, scaled by
a multiplier. The works become a works table, so the period selector chooses each analysed
turbine's span as it would for a real campaign.

Draw a campaign and build it for one multiplier and post length::

    draw = draw_rollout(hot_site(), seed=7, full_rollout=True)
    campaign = rollout_campaign(draw, multiplier=1.0, post_months=6)
    scada, _ = load_hot_scada(start_dt=draw.data_window(6)[0], end_dt_excl=draw.data_window(6)[1])
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

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from benchmarking.synthetic import ColumnSchema

# The AeroUp shape: a region-2 Cp gain tailing to nothing by rated.
AEROUP_WS_POINTS = (4.0, 8.0, 12.0, 16.0)
AEROUP_DELTAS = (0.07, 0.07, 0.045, 0.0)

WORKING_DAYS_PER_TURBINE = (5, 12)
TRIAL_TEAMS = (1, 2)
ROLLOUT_TEAMS = (1, 2)
TRIAL_PRE_MONTHS = (12, 24)
TRIAL_MIN_POST_MONTHS = 12
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
    """

    name: str
    coords: Mapping[str, tuple[float, float]]
    rotor_diameter_m: float
    rated_power_kw: float
    columns: ColumnSchema
    data_start: pd.Timestamp
    data_end: pd.Timestamp

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
    """Return Hill of Towie over 2016-2020, before its real AeroUp works."""
    return RolloutSite(
        name="Hill of Towie",
        coords=dict(HOT_COORDINATES),
        rotor_diameter_m=HOT_ROTOR_DIAMETER_M,
        rated_power_kw=HOT_RATED_POWER_KW,
        columns=HOT_COLUMNS,
        data_start=pd.Timestamp("2016-01-01", tz="UTC"),
        data_end=pd.Timestamp("2021-01-01", tz="UTC"),
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

    def data_end(self, post_months: int) -> pd.Timestamp:
        """Return the end of the data a campaign with ``post_months`` after the last trial works sees."""
        return min(self.site.data_end, self.last_trial_end + pd.DateOffset(months=post_months))

    def data_window(self, post_months: int) -> Window:
        """Return the ``(start, end)`` of the SCADA a campaign with ``post_months`` is built on."""
        return self.site.data_start, self.data_end(post_months)


def draw_rollout(site: RolloutSite, *, seed: int, full_rollout: bool) -> RolloutDraw:
    """Draw a trial-then-rollout campaign on ``site``.

    The trial is a compliant campaign design with between one and one fewer than the most test
    turbines the site allows. Its works start 12 to 24 months into the data, with one or two
    teams, and leave at least 12 months of data after them. A full rollout works every other
    turbine, starting 6 to 9 months after the last trial works end.

    :param site: the site to draw on
    :param seed: the draw's seed; the same seed gives the same draw
    :param full_rollout: whether every other turbine is worked after the trial
    :raises ValueError: if the site's data is too short for the trial
    """
    rng = np.random.default_rng(seed)
    layout = site.layout_frame()
    most = design_campaign(layout).max_test_turbines
    priority = [str(t) for t in rng.permutation(list(site.coords))]
    n_trial = int(rng.integers(1, max(1, most - 1) + 1))
    design = design_campaign(layout, test_priority=priority, n_test=n_trial)
    trial = tuple(t for t in priority if t in design.test_turbines)

    earliest = (site.data_start + pd.DateOffset(months=TRIAL_PRE_MONTHS[0])).date()
    latest = min(
        site.data_start + pd.DateOffset(months=TRIAL_PRE_MONTHS[1]),
        site.data_end - pd.DateOffset(months=TRIAL_MIN_POST_MONTHS),
    ).date()
    start = _working_day_between(earliest, latest, rng=rng)
    works = works_schedule(trial, start=start, n_teams=int(rng.choice(TRIAL_TEAMS)), rng=rng)
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


def rollout_campaign(draw: RolloutDraw, *, multiplier: float, post_months: int) -> SyntheticCampaign:
    """Build the campaign of ``draw`` with the AeroUp change scaled by ``multiplier``.

    The trial turbines are analysed; every other turbine is a candidate reference. The data ends
    ``post_months`` after the last trial works (see :meth:`RolloutDraw.data_window`), and only works
    starting before then are declared. The change is injected into the trial and any rollout
    turbine from its own works end. The analysis span is left to the period selector.

    :param draw: the drawn campaign
    :param multiplier: scales the AeroUp change; 0 injects nothing
    :param post_months: months of data after the last trial works end
    """
    site = draw.site
    end = draw.data_end(post_months)
    works = {t: [w] for t, w in draw.works.items() if w[0] < end}
    return SyntheticCampaign(
        upgraded_turbines=sorted(draw.trial),
        upgrade_timing=None,
        candidate_references=[t for t in site.coords if t not in draw.trial],
        upgrades=[] if multiplier == 0 else [aeroup_upgrade(multiplier)],
        layout=layout_from_coords(site.coords, rotor_diameter_m=site.rotor_diameter_m),
        north_offsets=None,
        rated_power_kw=site.rated_power_kw,
        analysis_period=None,
        columns=site.columns,
        seed=draw.seed,
        works=works,
        other_upgraded=[t for t in draw.rollout if t in works],
    )
