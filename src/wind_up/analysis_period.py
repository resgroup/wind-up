"""Choose each test turbine's analysis span and power references from the campaign timeline.

Time is measured in records valid for uplift: a side's length is the test turbine's valid time on that
side. A power reference must be valid for uplift over at least ``reference_share`` of the span's records
and have no works window inside it. A chosen span has at least ``min_references`` of them. Among the
candidate spans, the pool rule comes first, then the shorter side, then post, then pre, then how many power
references there are and how far the furthest is.

Candidates are ranked by distance, except that a candidate the test turbine is upwind of for more
than ``demotion_share`` of the wind rose, within the wake effect reach, ranks behind every candidate
that is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from wind_up.layout import NAME_COL, ROTOR_DIAMETER_COL, Layout
from wind_up.wake_reach import DEFAULT_DEMOTION_SHARE, MAX_REACH_D, REFERENCES_PER_TEST_TURBINE, candidate_cost

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from wind_up.wake_reach import Rose

Window = tuple[pd.Timestamp, pd.Timestamp]

MONTH = pd.Timedelta(days=365.25 / 12)

TABLE_COLUMNS = ("turbine", "role", "distance_d", "cost_share", "reason")


@dataclass(frozen=True)
class PlanSettings:
    """How spans and power references are chosen.

    :param k: power references per test turbine
    :param pre_cap: the longest pre side sought
    :param side_cap: the longest short side, and the longest post side, sought
    :param min_side: the shortest side a chosen span may have
    :param reference_share: the least share of a span's records a power reference must be valid for uplift over
    :param pool_size: how many nearest candidates the pool rule looks at
    :param pool_min_eligible: how many of those must be eligible and within the distance limit
    :param max_reference_distance_d: the furthest a power reference may be, in rotor diameters
    :param min_references: the fewest power references a chosen span may have
    :param wake_reach_d: how far the test turbine's wake change reaches, in its rotor diameters
    :param rose: the site's wind rose; ``None`` ranks by distance alone
    :param demotion_share: the share of the rose above which a candidate is demoted
    :param rose_source: what the rose was built from, for the reports
    """

    k: int = 4
    pre_cap: pd.Timedelta = 24 * MONTH
    side_cap: pd.Timedelta = 12 * MONTH
    min_side: pd.Timedelta = 3 * MONTH
    reference_share: float = 0.5
    pool_size: int = 4
    pool_min_eligible: int = REFERENCES_PER_TEST_TURBINE
    max_reference_distance_d: float = 20.0
    min_references: int = REFERENCES_PER_TEST_TURBINE
    wake_reach_d: float = MAX_REACH_D
    rose: Rose | None = field(default=None, compare=False, repr=False)
    demotion_share: float = DEFAULT_DEMOTION_SHARE
    rose_source: str = "none"


DEFAULT_PLAN_SETTINGS = PlanSettings()


@dataclass(frozen=True)
class AnalysisPlan:
    """One test turbine's span, its power references, and why every other turbine is not one.

    :param turbine: the test turbine
    :param start: the span's start
    :param end: the span's end, exclusive
    :param works: the test turbine's works window; no length for a shared changeover
    :param pre: the test turbine's time valid for uplift before the works window
    :param post: the test turbine's time valid for uplift after the works window
    :param power_references: best-ranked first, then any forced in
    :param reserves: further eligible turbines within the distance limit, best-ranked first
    :param waking_only: every other turbine that is not a power reference, with the reason
    :param distances_d: every other turbine's distance, in the test turbine's rotor diameters
    :param forced: declared references used although their works overlap the span
    :param declared: the span was declared rather than chosen
    :param pool_rule_met: whether the pool rule holds over the span
    :param pool_rule_reason: why it does not; empty when it does
    :param k: power references sought
    :param costs: every other turbine's rose share with the test turbine upwind of it within the reach
    :param settings: how the plan was chosen
    """

    turbine: str
    start: pd.Timestamp
    end: pd.Timestamp
    works: Window
    pre: pd.Timedelta
    post: pd.Timedelta
    power_references: tuple[str, ...]
    reserves: tuple[str, ...]
    waking_only: dict[str, str]
    distances_d: dict[str, float]
    forced: tuple[str, ...]
    declared: bool
    pool_rule_met: bool
    pool_rule_reason: str
    k: int
    costs: dict[str, float] = field(default_factory=dict)
    settings: PlanSettings = field(default_factory=PlanSettings, repr=False)

    def demoted(self, other: str) -> bool:
        """Return whether ``other`` ranks behind the candidates the test turbine does not wake as much."""
        return self.costs.get(other, 0.0) > self.settings.demotion_share

    @property
    def before(self) -> Window:
        """The before period: the span's start to the start of the works window."""
        return (self.start, self.works[0])

    @property
    def after(self) -> Window:
        """The after period: the end of the works window to the span's end."""
        return (self.works[1], self.end)

    def table(self) -> pd.DataFrame:
        """One row per other turbine, nearest first, with :data:`TABLE_COLUMNS`."""
        rows = []
        for other, distance in sorted(self.distances_d.items(), key=lambda item: (item[1], item[0])):
            if other in self.forced:
                role, reason = "forced", "declared reference; its works overlap the span"
            elif other in self.power_references:
                role, reason = "power_reference", ""
            elif other in self.reserves:
                role, reason = "reserve", self.waking_only[other]
            else:
                role, reason = "waking_only", self.waking_only[other]
            cost = self.costs.get(other, 0.0)
            if self.demoted(other):
                demotion = (
                    f"demoted: the test turbine is upwind of it {cost * 100:.0f} % of the time within "
                    f"{self.settings.wake_reach_d:g} D"
                )
                reason = demotion if role in ("power_reference", "reserve") else f"{reason}; {demotion}"
            rows.append({"turbine": other, "role": role, "distance_d": distance, "cost_share": cost, "reason": reason})
        return pd.DataFrame(rows, columns=list(TABLE_COLUMNS))

    def summary(self) -> dict[str, object]:
        """Return the plan's headline facts as plain data."""
        return {
            "start": str(self.start),
            "end": str(self.end),
            "works": [str(self.works[0]), str(self.works[1])],
            "pre_days": round(self.pre / pd.Timedelta(days=1), 2),
            "post_days": round(self.post / pd.Timedelta(days=1), 2),
            "declared": self.declared,
            "k": self.k,
            "power_references": list(self.power_references),
            "forced": list(self.forced),
            "pool_rule_met": self.pool_rule_met,
            "pool_rule_reason": self.pool_rule_reason,
            "wake_reach_d": self.settings.wake_reach_d,
            "demotion_share": self.settings.demotion_share,
            "rose": self.settings.rose_source,
            "demoted": sorted(o for o in self.costs if self.demoted(o)),
        }


def distances_in_diameters(layout: Layout, *, turbine: str) -> dict[str, float]:
    """Every other named turbine's distance from ``turbine``, in ``turbine``'s rotor diameters."""
    frame = layout.frame
    row = layout.index_of(turbine)
    diameter = float(frame[ROTOR_DIAMETER_COL].iloc[row])
    return {
        str(name): float(layout.distance_m[row, j]) / diameter
        for j, name in enumerate(frame[NAME_COL])
        if name is not None and str(name) != turbine
    }


def valid_extents(valid: pd.DataFrame) -> dict[str, Window]:
    """Each turbine's first record valid for uplift and the end of its last, one timebase after it."""
    timebase = _timebase(pd.DatetimeIndex(valid.index))
    extents: dict[str, Window] = {}
    for turbine in valid.columns:
        rows = pd.DatetimeIndex(valid.index[valid[turbine].to_numpy(dtype=bool)])
        if len(rows):
            extents[str(turbine)] = (rows[0], rows[-1] + timebase)
    return extents


def _flips(positions: np.ndarray, qualifies: np.ndarray) -> list[int]:
    """Return, at each change of ``qualifies``, the position on its qualifying side."""
    changes = np.flatnonzero(qualifies[:-1] != qualifies[1:])
    return [int(p) for p in np.where(qualifies[changes], positions[changes], positions[changes + 1])]


def _timebase(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Return the median step of ``index``; ten minutes for fewer than two records."""
    return pd.Timedelta(np.median(np.diff(index.to_numpy()))) if len(index) > 1 else pd.Timedelta(minutes=10)


@dataclass(frozen=True)
class _Option:
    """One candidate span, evaluated."""

    start: pd.Timestamp
    end: pd.Timestamp
    pre: pd.Timedelta
    post: pd.Timedelta
    references: tuple[str, ...]
    pool_met: bool
    worst_d: float


class _Timeline:
    """The facts one test turbine's plan is chosen from."""

    def __init__(
        self,
        layout: Layout,
        *,
        turbine: str,
        works: Mapping[str, Sequence[Window]],
        valid: pd.DataFrame,
        candidates: Sequence[str],
        settings: PlanSettings,
    ) -> None:
        if len(works.get(turbine, [])) != 1:
            msg = f"{turbine} needs exactly one works window to plan its span"
            raise ValueError(msg)
        self.layout = layout
        self.turbine = turbine
        self.works = works
        self.settings = settings
        self.a, self.b = works[turbine][0]
        self.index = pd.DatetimeIndex(valid.index)
        self.timebase = _timebase(self.index)
        self.extents = valid_extents(valid)
        if turbine not in self.extents:
            msg = f"{turbine}: no record is valid for uplift"
            raise ValueError(msg)
        # Per turbine, the count of valid records before each position of the index.
        self.counts = {str(t): np.concatenate([[0], np.cumsum(valid[t].to_numpy(dtype=bool))]) for t in valid.columns}
        self.distances_d = self.distances_from(turbine)
        offered = set(candidates) - {turbine}
        self.costs = self.wake_costs()
        self.ranked = sorted(
            offered & set(self.distances_d),
            key=lambda r: (self.costs[r] > settings.demotion_share, self.distances_d[r], r),
        )

    def wake_costs(self) -> dict[str, float]:
        """Return each other turbine's rose share with the test turbine upwind of it, 0 without a rose."""
        rose = self.settings.rose
        if rose is None:
            return dict.fromkeys(self.distances_d, 0.0)
        return {
            other: candidate_cost(
                self.layout, test=self.turbine, candidate=other, reach_d=self.settings.wake_reach_d, rose=rose
            )
            for other in self.distances_d
        }

    def distances_from(self, turbine: str) -> dict[str, float]:
        return distances_in_diameters(self.layout, turbine=turbine)

    def within(self, other: str) -> bool:
        return self.distances_d[other] <= self.settings.max_reference_distance_d

    def overlapping_works(self, other: str, *, start: pd.Timestamp, end: pd.Timestamp) -> Window | None:
        return next(((w0, w1) for w0, w1 in self.works.get(other, []) if w0 < end and w1 > start), None)

    def position(self, stamp: pd.Timestamp) -> int:
        return int(self.index.searchsorted(stamp, side="left"))

    def valid_records(self, turbine: str, *, start: pd.Timestamp, end: pd.Timestamp) -> int:
        counts = self.counts.get(turbine)
        if counts is None or end <= start:
            return 0
        return int(counts[self.position(end)] - counts[self.position(start)])

    def valid_time(self, turbine: str, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timedelta:
        return self.valid_records(turbine, start=start, end=end) * self.timebase

    def valid_share(self, other: str, *, start: pd.Timestamp, end: pd.Timestamp) -> float:
        records = self.position(end) - self.position(start)
        return self.valid_records(other, start=start, end=end) / records if records > 0 else 0.0

    def has_data(self, other: str, *, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        return self.valid_share(other, start=start, end=end) >= self.settings.reference_share

    def eligible(self, other: str, *, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        return (
            self.has_data(other, start=start, end=end) and self.overlapping_works(other, start=start, end=end) is None
        )

    def why_not(self, other: str, *, start: pd.Timestamp, end: pd.Timestamp) -> str:
        """Why ``other`` is not eligible over ``[start, end)``, or its distance when that is the reason."""
        if other not in self.ranked:
            return "not offered as a reference"
        clash = self.overlapping_works(other, start=start, end=end)
        if clash is not None:
            return f"works {_day(clash[0])}..{_day(clash[1] - pd.Timedelta(days=1))} overlap the span"
        if not self.has_data(other, start=start, end=end):
            share = self.valid_share(other, start=start, end=end)
            return (
                f"valid for uplift over {share * 100:.0f} % of the span, under "
                f"{self.settings.reference_share * 100:.0f} %"
            )
        return f"beyond {self.settings.max_reference_distance_d:g} D ({self.distances_d[other]:.1f} D)"

    def usable_eligible(self, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
        """Eligible candidates within the distance limit, best-ranked first."""
        return [r for r in self.ranked if self.within(r) and self.eligible(r, start=start, end=end)]

    def pool_met(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        pool = self.ranked[: self.settings.pool_size]
        good = [r for r in pool if self.within(r) and self.eligible(r, start=start, end=end)]
        return bool(pool) and pool[0] in good and len(good) >= self.settings.pool_min_eligible

    def pool_reason(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        """Why the pool rule fails over ``[start, end)``; empty when it holds."""
        if self.pool_met(start, end):
            return ""
        pool = self.ranked[: self.settings.pool_size]
        if not pool:
            return "no candidate references"
        parts = []
        best = pool[0]
        if not (self.within(best) and self.eligible(best, start=start, end=end)):
            parts.append(f"best-ranked {best} is not eligible ({self.why_not(best, start=start, end=end)})")
        good = [r for r in pool if self.within(r) and self.eligible(r, start=start, end=end)]
        if len(good) < self.settings.pool_min_eligible:
            names = f": {', '.join(good)}" if good else ""
            parts.append(
                f"only {len(good)} of the {len(pool)} best-ranked are eligible within "
                f"{self.settings.max_reference_distance_d:g} D{names}"
            )
        return "; ".join(parts)

    def evaluate(self, start: pd.Timestamp, end: pd.Timestamp) -> _Option:
        pre = self.valid_time(self.turbine, start=start, end=self.a)
        post = self.valid_time(self.turbine, start=self.b, end=end)
        references = tuple(self.usable_eligible(start, end)[: self.settings.k])
        worst = max((self.distances_d[r] for r in references), default=math.inf)
        return _Option(start, end, pre, post, references, self.pool_met(start, end), worst)

    def edges(self) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
        """Return the candidate starts and ends the search runs over.

        Besides the caps, the works windows and the valid extents, these are the starts and ends at which a
        candidate's valid share crosses ``reference_share`` against each of the other edges.
        """
        starts, ends = self.timeline_edges()
        crossing_starts, crossing_ends = self.share_crossings(starts, ends)
        return sorted({*starts, *crossing_starts}), sorted({*ends, *crossing_ends})

    def share_crossings(
        self, starts: Sequence[pd.Timestamp], ends: Sequence[pd.Timestamp]
    ) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
        """Return the starts and ends where a candidate's valid share of the span crosses ``reference_share``.

        Starts are found against each of ``ends`` and ends against each of ``starts``, for every candidate
        within the distance limit, among the starts and ends leaving the test turbine ``min_side`` of valid time.
        """
        first, last = self.extents[self.turbine]
        start_range = (self.position(first), self.position(self.reach(self.settings.min_side, before=True)))
        end_range = (self.position(self.reach(self.settings.min_side, before=False)), self.position(last))
        crossing_starts: set[pd.Timestamp] = set()
        crossing_ends: set[pd.Timestamp] = set()
        for other in self.ranked:
            counts = self.counts.get(other)
            if counts is None or not self.within(other):
                continue
            for end in ends:
                at = self.position(end)
                positions = np.arange(start_range[0], min(start_range[1], at - 1) + 1)
                shares = (counts[at] - counts[positions]) / (at - positions)
                crossing_starts.update(
                    self.stamp(p) for p in _flips(positions, shares >= self.settings.reference_share)
                )
            for start in starts:
                at = self.position(start)
                positions = np.arange(max(end_range[0], at + 1), end_range[1] + 1)
                shares = (counts[positions] - counts[at]) / (positions - at)
                crossing_ends.update(self.stamp(p) for p in _flips(positions, shares >= self.settings.reference_share))
        return sorted(crossing_starts), sorted(crossing_ends)

    def stamp(self, position: int) -> pd.Timestamp:
        """Return the timestamp at ``position`` of the index, one timebase after the last past its end."""
        return self.index[position] if position < len(self.index) else self.index[-1] + self.timebase

    def timeline_edges(self) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
        """Return the starts and ends set by the caps, the works windows and the valid extents."""
        first, last = self.extents[self.turbine]
        starts = {first, self.a - self.settings.pre_cap, self.reach(self.settings.pre_cap, before=True)}
        ends = {last, self.b + self.settings.side_cap, self.reach(self.settings.side_cap, before=False)}
        for other in self.ranked:
            for w0, w1 in self.works.get(other, []):
                if w1 <= self.a:
                    starts.add(w1)
                if w0 >= self.b:
                    ends.add(w0)
            # A reference whose data starts late or ends early is worth a shorter side when the
            # pool rule or min_references needs it; the ranking decides whether it is.
            extent = self.extents.get(other)
            if extent is not None and self.within(other):
                starts.add(extent[0])
                ends.add(extent[1])
        return (
            sorted({max(s, first) for s in starts if s < self.a}),
            sorted({min(e, last) for e in ends if e > self.b}),
        )

    def reach(self, length: pd.Timedelta, *, before: bool) -> pd.Timestamp:
        """Return the nearest edge giving the test turbine ``length`` of valid time before the works window, or after.

        The record's first or last edge when no edge gives that much.
        """
        counts = self.counts[self.turbine]
        needed = math.ceil(length / self.timebase)
        if before:
            target = counts[self.position(self.a)] - needed
            if target < 0:
                return self.index[0]
            return self.index[int(np.searchsorted(counts, target, side="right")) - 1]
        target = counts[self.position(self.b)] + needed
        at = int(np.searchsorted(counts, target, side="left"))
        return self.index[at] if at < len(self.index) else self.index[-1] + self.timebase

    def key(self, option: _Option) -> tuple:
        cap = self.settings.side_cap
        long_enough = option.pre >= self.settings.min_side and option.post >= self.settings.min_side
        return (
            not (option.pool_met and long_enough),
            -min(option.pre, option.post, cap),
            -min(option.post, cap),
            -min(option.pre, self.settings.pre_cap),
            -len(option.references),
            option.worst_d,
            -option.start.value,
            option.end.value,
        )

    def choose(self) -> _Option:
        """Return the best candidate span; raise when none gives both sides ``min_side`` and ``min_references``."""
        starts, ends = self.edges()
        options = [self.evaluate(s, e) for s in starts for e in ends]
        long_enough = [o for o in options if o.pre >= self.settings.min_side and o.post >= self.settings.min_side]
        needed = max(self.settings.min_references, 1)
        viable = [o for o in long_enough if len(o.references) >= needed]
        if not viable:
            floor = _length(self.settings.min_side)
            if long_enough:
                most = max(len(o.references) for o in long_enough)
                wanted = "a power reference" if needed == 1 else f"{needed} power references"
                msg = (
                    f"{self.turbine}: no span with both sides at least {floor} has {wanted}; the most any has is {most}"
                )
                raise ValueError(msg)
            best = max((min(o.pre, o.post) for o in options), default=pd.Timedelta(0))
            msg = f"{self.turbine}: no span gives both sides at least {floor}; the best is {_length(best, digits=1)}"
            raise ValueError(msg)
        return min(viable, key=self.key)


def plan_analysis(
    layout: Layout,
    *,
    turbine: str,
    works: Mapping[str, Sequence[Window]],
    valid: pd.DataFrame,
    candidates: Sequence[str],
    span: Window | None = None,
    allow_works_overlap: Sequence[str] = (),
    settings: PlanSettings = DEFAULT_PLAN_SETTINGS,
) -> AnalysisPlan:
    """Choose ``turbine``'s span and power references, or use the declared ``span``.

    :param layout: every turbine, rotor diameters included
    :param turbine: the test turbine
    :param works: works windows per turbine; the test turbine has exactly one, ``(c, c)`` for a
        shared changeover ``c``
    :param valid: timestamps x turbines, True where a record is valid for uplift, on every timestamp of the
        record; a turbine with no column has no valid record
    :param candidates: the turbines that may be power references
    :param span: a declared span, used as given; it must bracket the works window
    :param allow_works_overlap: declared references to use as power references over a declared ``span``
        even when their works overlap it; they are the plan's ``forced``
    :param settings: how spans and references are chosen
    :raises ValueError: when no span gives both sides ``settings.min_side`` with at least one power
        reference, a declared span does not bracket the works window, or it has no power reference
    """
    timeline = _Timeline(
        layout,
        turbine=turbine,
        works=works,
        valid=valid,
        candidates=candidates,
        settings=settings,
    )
    if span is not None and not (span[0] < timeline.a and span[1] > timeline.b):
        msg = (
            f"{turbine}: the declared span {span[0]}..{span[1]} does not bracket the works window "
            f"{timeline.a}..{timeline.b}; it must start before the works and end after them"
        )
        raise ValueError(msg)
    option = timeline.choose() if span is None else timeline.evaluate(*span)
    start, end = option.start, option.end
    forced_in: tuple[str, ...] = ()
    if span is not None:
        forced_in = tuple(
            r
            for r in allow_works_overlap
            if r in timeline.distances_d
            and r not in option.references
            and timeline.has_data(r, start=start, end=end)
            and timeline.overlapping_works(r, start=start, end=end) is not None
        )
    references = (*option.references, *forced_in)
    if not references:
        msg = f"{turbine}: the declared span {start}..{end} has no power reference"
        raise ValueError(msg)

    eligible = timeline.usable_eligible(start, end)
    reserves = tuple(r for r in eligible if r not in references)
    waking_only = {
        other: (
            f"reserve: beyond the best-ranked {settings.k}"
            if other in reserves
            else timeline.why_not(other, start=start, end=end)
        )
        for other in timeline.distances_d
        if other not in references
    }
    return AnalysisPlan(
        turbine=turbine,
        start=start,
        end=end,
        works=(timeline.a, timeline.b),
        pre=option.pre,
        post=option.post,
        power_references=references,
        reserves=reserves,
        waking_only=waking_only,
        distances_d=timeline.distances_d,
        forced=forced_in,
        declared=span is not None,
        pool_rule_met=option.pool_met,
        pool_rule_reason=timeline.pool_reason(start, end),
        k=settings.k,
        costs=timeline.costs,
        settings=settings,
    )


def _length(length: pd.Timedelta, *, digits: int | None = None) -> str:
    """Return ``length`` in months, or in days when it is under a month."""
    if length < MONTH:
        days = length / pd.Timedelta(days=1)
        return f"{days:g} days" if digits is None else f"{days:.{digits}f} days"
    months = length / MONTH
    return f"{months:g} months" if digits is None else f"{months:.{digits}f} months"


def _day(stamp: pd.Timestamp) -> str:
    return stamp.strftime("%Y-%m-%d")
