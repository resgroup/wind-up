"""Choose each test turbine's analysis span and power references from the campaign timeline.

A power reference must have data over the whole span and no works window inside it. Among the
candidate spans, the pool rule comes first, then the shorter side, then post, then pre, then how many
power references there are and how far the furthest is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from wind_up.campaign_design import REFERENCES_PER_TEST_TURBINE
from wind_up.layout import NAME_COL, ROTOR_DIAMETER_COL, Layout

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Window = tuple[pd.Timestamp, pd.Timestamp]
# (turbine, start, end), end exclusive; turbine None is farm-wide.
Exclusion = tuple[str | None, pd.Timestamp, pd.Timestamp]

MONTH = pd.Timedelta(days=365.25 / 12)

TABLE_COLUMNS = ("turbine", "role", "distance_d", "reason")

# Masked time can extend an edge into more masked time; a few steps settle any realistic timeline.
_REACH_ITERATIONS = 20


@dataclass(frozen=True)
class PlanSettings:
    """How spans and power references are chosen.

    :param k: power references per test turbine
    :param pre_cap: the longest pre side sought
    :param side_cap: the longest short side, and the longest post side, sought
    :param min_side: the shortest side a chosen span may have
    :param pool_size: how many nearest candidates the pool rule looks at
    :param pool_min_eligible: how many of those must be eligible and within the distance limit
    :param max_reference_distance_d: the furthest a power reference may be, in rotor diameters
    """

    k: int = 4
    pre_cap: pd.Timedelta = 24 * MONTH
    side_cap: pd.Timedelta = 12 * MONTH
    min_side: pd.Timedelta = 3 * MONTH
    pool_size: int = 4
    pool_min_eligible: int = REFERENCES_PER_TEST_TURBINE
    max_reference_distance_d: float = 20.0


DEFAULT_PLAN_SETTINGS = PlanSettings()


@dataclass(frozen=True)
class AnalysisPlan:
    """One test turbine's span, its power references, and why every other turbine is not one.

    :param turbine: the test turbine
    :param start: the span's start
    :param end: the span's end, exclusive
    :param works: the test turbine's works window; no length for a shared changeover
    :param pre: usable time before the works window
    :param post: usable time after the works window
    :param power_references: nearest first, then any forced in
    :param reserves: further eligible turbines within the distance limit, nearest first
    :param reading_pools: for each power reference and reserve, the other eligible turbines within
        the distance limit of it, nearest first, the test turbine left out
    :param waking_only: every other turbine that is not a power reference, with the reason
    :param distances_d: every other turbine's distance, in the test turbine's rotor diameters
    :param forced: declared references used although their works overlap the span
    :param declared: the span was declared rather than chosen
    :param pool_rule_met: whether the pool rule holds over the span
    :param pool_rule_reason: why it does not; empty when it does
    :param k: power references sought
    """

    turbine: str
    start: pd.Timestamp
    end: pd.Timestamp
    works: Window
    pre: pd.Timedelta
    post: pd.Timedelta
    power_references: tuple[str, ...]
    reserves: tuple[str, ...]
    reading_pools: dict[str, tuple[str, ...]]
    waking_only: dict[str, str]
    distances_d: dict[str, float]
    forced: tuple[str, ...]
    declared: bool
    pool_rule_met: bool
    pool_rule_reason: str
    k: int

    def table(self) -> pd.DataFrame:
        """One row per other turbine, nearest first: ``turbine``, ``role``, ``distance_d``, ``reason``."""
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
            rows.append({"turbine": other, "role": role, "distance_d": distance, "reason": reason})
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
        }


def intervals_duration(windows: Sequence[Window], *, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timedelta:
    """Total length of the union of ``windows`` clipped to ``[start, end)``."""
    clipped = sorted((max(s, start), min(e, end)) for s, e in windows if s < end and e > start)
    total = pd.Timedelta(0)
    cursor = start
    for s, e in clipped:
        begin = max(s, cursor)
        if e > begin:
            total += e - begin
            cursor = e
    return total


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
        exclusions: Sequence[Exclusion],
        extents: Mapping[str, Window],
        candidates: Sequence[str],
        settings: PlanSettings,
    ) -> None:
        if len(works.get(turbine, [])) != 1:
            msg = f"{turbine} needs exactly one works window to plan its span"
            raise ValueError(msg)
        self.layout = layout
        self.turbine = turbine
        self.works = works
        self.extents = extents
        self.settings = settings
        self.a, self.b = works[turbine][0]
        self.masked = [(s, e) for who, s, e in exclusions if who in (None, turbine)]
        self.distances_d = self.distances_from(turbine)
        offered = set(candidates) - {turbine}
        self.ranked = sorted(offered & set(self.distances_d), key=lambda r: (self.distances_d[r], r))

    def distances_from(self, turbine: str) -> dict[str, float]:
        """Every other named turbine's distance from ``turbine``, in ``turbine``'s rotor diameters."""
        frame = self.layout.frame
        row = self.layout.index_of(turbine)
        diameter = float(frame[ROTOR_DIAMETER_COL].iloc[row])
        return {
            str(name): float(self.layout.distance_m[row, j]) / diameter
            for j, name in enumerate(frame[NAME_COL])
            if name is not None and str(name) != turbine
        }

    def within(self, other: str) -> bool:
        return self.distances_d[other] <= self.settings.max_reference_distance_d

    def overlapping_works(self, other: str, start: pd.Timestamp, end: pd.Timestamp) -> Window | None:
        return next(((w0, w1) for w0, w1 in self.works.get(other, []) if w0 < end and w1 > start), None)

    def has_data(self, other: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        extent = self.extents.get(other)
        return extent is not None and extent[0] <= start and extent[1] >= end

    def eligible(self, other: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        return self.has_data(other, start, end) and self.overlapping_works(other, start, end) is None

    def why_not(self, other: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
        """Why ``other`` is not eligible over ``[start, end)``, or its distance when that is the reason."""
        if other not in self.ranked:
            return "not offered as a reference"
        clash = self.overlapping_works(other, start, end)
        if clash is not None:
            return f"works {_day(clash[0])}..{_day(clash[1] - pd.Timedelta(days=1))} overlap the span"
        if not self.has_data(other, start, end):
            return "no data over the whole span"
        return f"beyond {self.settings.max_reference_distance_d:g} D ({self.distances_d[other]:.1f} D)"

    def usable_eligible(self, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
        """Eligible candidates within the distance limit, nearest first."""
        return [r for r in self.ranked if self.within(r) and self.eligible(r, start, end)]

    def pool_met(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        pool = self.ranked[: self.settings.pool_size]
        good = [r for r in pool if self.within(r) and self.eligible(r, start, end)]
        return bool(pool) and pool[0] in good and len(good) >= self.settings.pool_min_eligible

    def pool_reason(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        """Why the pool rule fails over ``[start, end)``; empty when it holds."""
        if self.pool_met(start, end):
            return ""
        pool = self.ranked[: self.settings.pool_size]
        if not pool:
            return "no candidate references"
        parts = []
        nearest = pool[0]
        if not (self.within(nearest) and self.eligible(nearest, start, end)):
            parts.append(f"nearest {nearest} is not eligible ({self.why_not(nearest, start, end)})")
        good = [r for r in pool if self.within(r) and self.eligible(r, start, end)]
        if len(good) < self.settings.pool_min_eligible:
            names = f": {', '.join(good)}" if good else ""
            parts.append(
                f"only {len(good)} of the {len(pool)} nearest are eligible within "
                f"{self.settings.max_reference_distance_d:g} D{names}"
            )
        return "; ".join(parts)

    def evaluate(self, start: pd.Timestamp, end: pd.Timestamp) -> _Option:
        pre = (self.a - start) - intervals_duration(self.masked, start=start, end=self.a)
        post = (end - self.b) - intervals_duration(self.masked, start=self.b, end=end)
        references = tuple(self.usable_eligible(start, end)[: self.settings.k])
        worst = max((self.distances_d[r] for r in references), default=math.inf)
        return _Option(start, end, pre, post, references, self.pool_met(start, end), worst)

    def edges(self) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
        """Return the candidate starts and ends the search is exact over."""
        first, last = self.extents[self.turbine]
        starts = {first, self.a - self.settings.pre_cap, self.reach(self.settings.pre_cap, before=True)}
        ends = {last, self.b + self.settings.side_cap, self.reach(self.settings.side_cap, before=False)}
        for other in self.ranked:
            for w0, w1 in self.works.get(other, []):
                if w1 <= self.a:
                    starts.add(w1)
                if w0 >= self.b:
                    ends.add(w0)
        return (
            sorted({max(s, first) for s in starts if s < self.a}),
            sorted({min(e, last) for e in ends if e > self.b}),
        )

    def reach(self, length: pd.Timedelta, *, before: bool) -> pd.Timestamp:
        """Return the edge that gives ``length`` of usable time before the works window, or after it."""
        edge = self.a - length if before else self.b + length
        for _ in range(_REACH_ITERATIONS):
            if before:
                moved = self.a - length - intervals_duration(self.masked, start=edge, end=self.a)
            else:
                moved = self.b + length + intervals_duration(self.masked, start=self.b, end=edge)
            if moved == edge:
                break
            edge = moved
        return edge

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
        """Return the best candidate span; raise when none gives both sides ``min_side`` and a reference."""
        starts, ends = self.edges()
        options = [self.evaluate(s, e) for s in starts for e in ends]
        viable = [
            o for o in options if o.pre >= self.settings.min_side and o.post >= self.settings.min_side and o.references
        ]
        if not viable:
            months = self.settings.min_side / MONTH
            if any(o.pre >= self.settings.min_side and o.post >= self.settings.min_side for o in options):
                msg = f"{self.turbine}: no span with both sides at least {months:g} months has a power reference"
                raise ValueError(msg)
            best = max((min(o.pre, o.post) for o in options), default=pd.Timedelta(0))
            msg = (
                f"{self.turbine}: no span gives both sides at least {months:g} months; the best is "
                f"{best / MONTH:.1f} months"
            )
            raise ValueError(msg)
        return min(viable, key=self.key)


def plan_analysis(
    layout: Layout,
    *,
    turbine: str,
    works: Mapping[str, Sequence[Window]],
    exclusions: Sequence[Exclusion],
    extents: Mapping[str, Window],
    candidates: Sequence[str],
    span: Window | None = None,
    forced: Sequence[str] = (),
    settings: PlanSettings = DEFAULT_PLAN_SETTINGS,
) -> AnalysisPlan:
    """Choose ``turbine``'s span and power references, or use the declared ``span``.

    :param layout: every turbine, rotor diameters included
    :param turbine: the test turbine
    :param works: works windows per turbine; the test turbine has exactly one, ``(c, c)`` for a
        shared changeover ``c``
    :param exclusions: ``(turbine, start, end)``, turbine None for farm-wide, end exclusive
    :param extents: each turbine's data extent, ``(first, end)`` with end exclusive
    :param candidates: the turbines that may be power references
    :param span: a declared span, used as given
    :param forced: declared references to use as power references over a declared ``span`` even
        when their works overlap it
    :param settings: how spans and references are chosen
    :raises ValueError: when no span gives both sides ``settings.min_side`` with at least one power
        reference, or a declared span has no power reference
    """
    timeline = _Timeline(
        layout,
        turbine=turbine,
        works=works,
        exclusions=exclusions,
        extents=extents,
        candidates=candidates,
        settings=settings,
    )
    option = timeline.choose() if span is None else timeline.evaluate(*span)
    start, end = option.start, option.end
    forced_in: tuple[str, ...] = ()
    if span is not None:
        forced_in = tuple(
            r
            for r in forced
            if r in timeline.distances_d
            and r not in option.references
            and timeline.has_data(r, start, end)
            and timeline.overlapping_works(r, start, end) is not None
        )
    references = (*option.references, *forced_in)
    if not references:
        msg = f"{turbine}: the declared span {start}..{end} has no power reference"
        raise ValueError(msg)

    eligible = timeline.usable_eligible(start, end)
    reserves = tuple(r for r in eligible if r not in references)
    waking_only = {
        other: (
            f"reserve: beyond the nearest {settings.k}" if other in reserves else timeline.why_not(other, start, end)
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
        reading_pools={r: _reading_pool(timeline, r, start=start, end=end) for r in (*references, *reserves)},
        waking_only=waking_only,
        distances_d=timeline.distances_d,
        forced=forced_in,
        declared=span is not None,
        pool_rule_met=option.pool_met,
        pool_rule_reason=timeline.pool_reason(start, end),
        k=settings.k,
    )


def _reading_pool(timeline: _Timeline, reference: str, *, start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, ...]:
    """Return the eligible turbines within the distance limit of ``reference``, nearest first, bar the test turbine."""
    distances = timeline.distances_from(reference)
    limit = timeline.settings.max_reference_distance_d
    pool = [r for r in timeline.ranked if r != reference and distances[r] <= limit and timeline.eligible(r, start, end)]
    return tuple(sorted(pool, key=lambda r: (distances[r], r)))


def _day(stamp: pd.Timestamp) -> str:
    return stamp.strftime("%Y-%m-%d")
