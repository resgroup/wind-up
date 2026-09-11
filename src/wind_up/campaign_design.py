"""Choose test turbines for an uplift validation campaign, and check a choice against the rules.

The rules and how a design is chosen are described in ``docs/designing-a-campaign.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp

from wind_up.campaign_design_plots import save_design_maps
from wind_up.layout import LATITUDE_COL, LONGITUDE_COL, NAME_COL, ROTOR_DIAMETER_COL, WIND_FARM_COL, Layout, front_row

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt

REFERENCES_PER_TEST_TURBINE = 3


@dataclass(frozen=True, eq=False)
class ComplianceReport:
    """The outcome of :func:`check_design`; ``table`` has one row per test turbine."""

    compliant: bool
    problems: tuple[str, ...]
    table: pd.DataFrame
    summary: dict[str, object]

    def __eq__(self, other: object) -> bool:
        """Compare field by field, the table by value."""
        if not isinstance(other, ComplianceReport):
            return NotImplemented
        return (
            self.compliant == other.compliant
            and self.problems == other.problems
            and self.summary == other.summary
            and self.table.equals(other.table)
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CandidateOutcome:
    """What happened to one candidate in the priority walk.

    ``rank`` is its 1-based place in the resolved priority order; ``from_test_priority`` says
    whether it was listed in ``test_priority`` or appended in the seeded order.
    """

    name: str
    rank: int
    from_test_priority: bool
    outcome: Literal["test", "skipped", "not needed"]
    reason: str | None


@dataclass(frozen=True)
class CampaignDesign:
    """The outcome of :func:`design_campaign`.

    ``reference_limit_d`` is the reference-distance limit, in rotor diameters, the candidates not in
    ``test_priority`` were chosen under: the least that still allows the design's size alongside the
    listed turbines kept.
    """

    wind_farm: str | None
    test_turbines: tuple[str, ...]
    references: dict[str, tuple[str, ...]]
    front_row: frozenset[str]
    max_test_turbines: int
    candidates: tuple[CandidateOutcome, ...]
    reference_only: tuple[str, ...]
    excluded: tuple[str, ...]
    filled_rotor_diameters: tuple[str, ...]
    reference_pool_size: int
    max_reference_distance_d: float
    reference_limit_d: float
    front_row_min_clear_deg: float
    compliance: ComplianceReport
    layout: Layout = field(repr=False)

    def roles(self) -> dict[str, list[str]]:
        """Return the campaign declaration roles: every available non-test farm turbine is a reference."""
        tests = set(self.test_turbines)
        excluded = set(self.excluded)
        frame = self.layout.frame
        farm = [str(n) for n, f in zip(frame[NAME_COL], frame[WIND_FARM_COL], strict=True) if f == self.wind_farm]
        return {
            "upgraded": list(self.test_turbines),
            "references": [n for n in farm if n not in tests and n not in excluded],
            "excluded": [n for n in farm if n in excluded],
        }


def check_design(
    layout: pd.DataFrame,
    *,
    test_turbines: Sequence[str],
    wind_farm: str | None = None,
    reference_only: Sequence[str] = (),
    excluded: Sequence[str] = (),
    reference_pool_size: int = 4,
    max_reference_distance_d: float = 20.0,
    front_row_min_clear_deg: float = 90.0,
) -> ComplianceReport:
    """Check ``test_turbines`` against the rules in the module docstring.

    Raises only on malformed input: an unknown or repeated name, a name in more than one role, an
    ambiguous ``wind_farm``. Non-compliance is reported in the returned :class:`ComplianceReport`.
    """
    site = _Site.build(
        layout,
        wind_farm=wind_farm,
        reference_only=reference_only,
        excluded=excluded,
        reference_pool_size=reference_pool_size,
        max_reference_distance_d=max_reference_distance_d,
        front_row_min_clear_deg=front_row_min_clear_deg,
    )
    if len(set(test_turbines)) != len(test_turbines):
        msg = f"test_turbines names a turbine more than once: {list(test_turbines)}"
        raise ValueError(msg)
    return site.check([site.row_of(name) for name in test_turbines])


def design_campaign(
    layout: pd.DataFrame,
    *,
    wind_farm: str | None = None,
    test_priority: Sequence[str] = (),
    reference_only: Sequence[str] = (),
    excluded: Sequence[str] = (),
    n_test: int | None = None,
    seed: int = 0,
    reference_pool_size: int = 4,
    max_reference_distance_d: float = 20.0,
    front_row_min_clear_deg: float = 90.0,
) -> CampaignDesign:
    """Choose test turbines for a campaign on the farm ``wind_farm`` of ``layout``.

    :param layout: one row per turbine: ``latitude``, ``longitude``, and optionally ``name``,
        ``rotor_diameter_m`` and ``wind_farm``; see :class:`wind_up.layout.Layout`. Turbines outside
        the farm under design only block wakes.
    :param wind_farm: the farm under design; defaults to the only ``wind_farm`` in the layout, or
        to every turbine when the layout names no wind farm
    :param test_priority: farm turbines to test first, highest priority first; every other
        candidate follows in an order drawn from ``seed``, restricted to the nearest references
        a design of this size allows
    :param reference_only: farm turbines that are never tested but may be references
    :param excluded: farm turbines that are neither tested nor references; they still block wakes
    :param n_test: how many test turbines; ``None`` takes the most a compliant design allows
    :param seed: seeds the order of the candidates not in ``test_priority``
    :param reference_pool_size: how many nearest reference-eligible turbines form a test
        turbine's pool
    :param max_reference_distance_d: the furthest a reference may be, in the test turbine's rotor
        diameters
    :param front_row_min_clear_deg: the contiguous clear arc that makes a turbine front row
    :raises ValueError: on malformed input, when no compliant design exists, or when ``n_test`` is
        more than the most a compliant design allows
    """
    site = _Site.build(
        layout,
        wind_farm=wind_farm,
        reference_only=reference_only,
        excluded=excluded,
        reference_pool_size=reference_pool_size,
        max_reference_distance_d=max_reference_distance_d,
        front_row_min_clear_deg=front_row_min_clear_deg,
    )
    order, listed = site.priority_order(test_priority=test_priority, seed=seed)
    most = site.max_count()
    if most == 0:
        msg = f"no compliant design exists for {site.farm_label}: {site.why_no_design()}"
        raise ValueError(msg)
    n = most if n_test is None else n_test
    if n < 1 or n > most:
        msg = f"n_test={n_test} is not possible: {site.farm_label} supports at most {most} test turbines"
        raise ValueError(msg)

    committed: list[int] = []
    outcomes: list[CandidateOutcome] = []
    walking = site
    limit_d = None
    for rank, row in enumerate(order, start=1):
        if limit_d is None and row not in listed:
            limit_d = site.tightest_limit_d(n=n, fixed=committed)
            walking = site.at_limit(limit_d)
        outcome: Literal["test", "skipped", "not needed"]
        reason = None
        if len(committed) == n:
            outcome = "not needed"
        elif len(walking.within[row]) < REFERENCES_PER_TEST_TURBINE:
            outcome, reason = "skipped", walking.too_few_within(row)
        elif walking.solve(n=n, fixed=[*committed, row]) is not None:
            outcome = "test"
            committed.append(row)
        else:
            outcome, reason = "skipped", walking.skip_reason(row, committed=committed, n=n)
        outcomes.append(
            CandidateOutcome(
                name=site.name(row), rank=rank, from_test_priority=row in listed, outcome=outcome, reason=reason
            )
        )
    if limit_d is None:
        limit_d = site.tightest_limit_d(n=n, fixed=committed)

    report = site.check(committed)
    table = report.table.set_index("test_turbine")
    references = {
        name: tuple(str(table.loc[name, f"reference_{k}"]) for k in range(1, REFERENCES_PER_TEST_TURBINE + 1))
        for name in table.index
    }
    return CampaignDesign(
        wind_farm=site.farm,
        test_turbines=tuple(site.name(r) for r in committed),
        references=references,
        front_row=frozenset(site.name(r) for r in site.farm_rows if site.front[r]),
        max_test_turbines=most,
        candidates=tuple(outcomes),
        reference_only=tuple(site.name(r) for r in site.farm_rows if r in site.reference_only),
        excluded=tuple(site.name(r) for r in site.farm_rows if r in site.excluded),
        filled_rotor_diameters=site.layout.filled_rotor_diameters,
        reference_pool_size=reference_pool_size,
        max_reference_distance_d=max_reference_distance_d,
        reference_limit_d=limit_d,
        front_row_min_clear_deg=front_row_min_clear_deg,
        compliance=report,
        layout=site.layout,
    )


def write_design(design: CampaignDesign, *, out_dir: str | Path) -> None:
    """Write ``design`` to ``out_dir``.

    Files: ``compliance.csv`` (one row per test turbine), ``summary.yaml`` (counts, the front-row
    share, the most test turbines possible, any problems), ``turbines.csv`` (every farm turbine's
    role, references served, priority and outcome), ``roles.yaml`` (a campaign declaration
    ``turbines:`` block), and the maps ``design_map.png``, ``design_map_latlon.png`` and
    ``front_row_map.png``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_report(
        design.compliance,
        out_dir=out,
        extra={"max_test_turbines": design.max_test_turbines, "reference_limit_d": design.reference_limit_d},
    )
    _turbines_table(design).to_csv(out / "turbines.csv", index=False)
    (out / "roles.yaml").write_text(yaml.safe_dump({"turbines": design.roles()}, sort_keys=False))
    save_design_maps(design, out_dir=out)


def write_compliance(report: ComplianceReport, *, out_dir: str | Path) -> None:
    """Write ``compliance.csv`` and ``summary.yaml`` for ``report`` to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_report(report, out_dir=out, extra={})


def _write_report(report: ComplianceReport, *, out_dir: Path, extra: dict[str, object]) -> None:
    report.table.to_csv(out_dir / "compliance.csv", index=False)
    summary = {"compliant": report.compliant, "problems": list(report.problems), **extra, **report.summary}
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))


def _turbines_table(design: CampaignDesign) -> pd.DataFrame:
    """One row per farm turbine: role, front row, the test turbines it serves, priority and outcome."""
    frame = design.layout.frame
    outcomes = {o.name: o for o in design.candidates}
    serves: dict[str, list[str]] = {}
    for test, refs in design.references.items():
        for ref in refs:
            serves.setdefault(ref, []).append(test)
    records = []
    for _, row in frame.iterrows():
        if design.wind_farm is not None and row[WIND_FARM_COL] != design.wind_farm:
            continue
        name = str(row[NAME_COL])
        outcome = outcomes.get(name)
        if name in design.test_turbines:
            role = "test"
        elif name in design.excluded:
            role = "excluded"
        elif name in design.reference_only:
            role = "reference-only"
        else:
            role = "reference"
        records.append(
            {
                "name": name,
                "role": role,
                "front_row": name in design.front_row,
                "reference_for": ", ".join(serves.get(name, [])),
                "priority_rank": outcome.rank if outcome else None,
                "from_test_priority": outcome.from_test_priority if outcome else None,
                "outcome": outcome.outcome if outcome else None,
                "reason": outcome.reason if outcome else None,
                "latitude": row[LATITUDE_COL],
                "longitude": row[LONGITUDE_COL],
                "rotor_diameter_m": row[ROTOR_DIAMETER_COL],
            }
        )
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class _Site:
    """The farm under design, its roles, and each turbine's ranked neighbours."""

    layout: Layout
    farm: str | None
    farm_rows: tuple[int, ...]
    excluded: frozenset[int]
    reference_only: frozenset[int]
    front: npt.NDArray[np.bool_]
    reference_pool_size: int
    max_reference_distance_d: float
    distance_d: npt.NDArray[np.float64]
    ranked: dict[int, list[int]]
    within: dict[int, list[int]]

    @classmethod
    def build(
        cls,
        layout_frame: pd.DataFrame,
        *,
        wind_farm: str | None,
        reference_only: Sequence[str],
        excluded: Sequence[str],
        reference_pool_size: int,
        max_reference_distance_d: float,
        front_row_min_clear_deg: float,
    ) -> _Site:
        layout = Layout.from_frame(layout_frame)
        frame = layout.frame
        farm = _resolve_farm(frame, wind_farm=wind_farm)
        farm_rows = tuple(int(i) for i in np.flatnonzero((frame[WIND_FARM_COL] == farm).to_numpy()))
        if farm is None:
            farm_rows = tuple(range(len(frame)))
        unnamed = [i for i in farm_rows if frame[NAME_COL].iloc[i] is None]
        if unnamed:
            msg = f"every turbine of the farm under design needs a name; rows {unnamed} have none"
            raise ValueError(msg)

        names = {str(frame[NAME_COL].iloc[i]): i for i in farm_rows}
        _check_roles(names, roles={"reference_only": reference_only, "excluded": excluded})
        excluded_rows = frozenset(names[n] for n in excluded)
        reference_only_rows = frozenset(names[n] for n in reference_only)

        available = [i for i in farm_rows if i not in excluded_rows]
        distance_d = layout.distance_m / frame[ROTOR_DIAMETER_COL].to_numpy()[:, None]
        ranked: dict[int, list[int]] = {}
        within: dict[int, list[int]] = {}
        for i in farm_rows:
            others = sorted(
                (j for j in available if j != i), key=lambda j: (layout.distance_m[i, j], frame[NAME_COL].iloc[j])
            )
            ranked[i] = others
            within[i] = [j for j in others if distance_d[i, j] <= max_reference_distance_d]
        return cls(
            layout=layout,
            farm=farm,
            farm_rows=farm_rows,
            excluded=excluded_rows,
            reference_only=reference_only_rows,
            front=front_row(layout, min_clear_deg=front_row_min_clear_deg),
            reference_pool_size=reference_pool_size,
            max_reference_distance_d=max_reference_distance_d,
            distance_d=distance_d,
            ranked=ranked,
            within=within,
        )

    @property
    def farm_label(self) -> str:
        return self.farm if self.farm is not None else "the farm"

    @property
    def available(self) -> list[int]:
        return [i for i in self.farm_rows if i not in self.excluded]

    @property
    def candidates(self) -> list[int]:
        return [i for i in self.available if i not in self.reference_only]

    def name(self, row: int) -> str:
        return str(self.layout.frame[NAME_COL].iloc[row])

    def row_of(self, name: str) -> int:
        return self.layout.index_of(name)

    def pool(self, row: int) -> list[int]:
        return self.within[row][: self.reference_pool_size]

    def nearest(self, row: int) -> int | None:
        return self.ranked[row][0] if self.ranked[row] else None

    def share_counts(self, n: int) -> list[int]:
        """Return the front-row test counts within 0.5 of the fair share for ``n`` test turbines."""
        a = len(self.available)
        f = int(self.front[self.available].sum())
        return [c for c in range(n + 1) if 2 * n * f - a <= 2 * a * c <= 2 * n * f + a]

    def too_few_within(self, row: int) -> str:
        return (
            f"only {len(self.within[row])} reference-eligible turbines within "
            f"{self.max_reference_distance_d:g} rotor diameters"
        )

    def priority_order(self, *, test_priority: Sequence[str], seed: int) -> tuple[list[int], set[int]]:
        """Return the candidates in resolved priority order, and the set that came from ``test_priority``."""
        names = {self.name(i): i for i in self.farm_rows}
        _check_roles(
            names,
            roles={
                "test_priority": test_priority,
                "reference_only": [self.name(i) for i in self.reference_only],
                "excluded": [self.name(i) for i in self.excluded],
            },
        )
        listed = [names[n] for n in test_priority]
        tail = [i for i in self.candidates if i not in set(listed)]
        rng = np.random.default_rng(seed)
        return listed + [tail[k] for k in rng.permutation(len(tail))], set(listed)

    def check(self, tests: list[int]) -> ComplianceReport:
        """Check the test turbines at rows ``tests``."""
        test_set = set(tests)
        problems: list[str] = []
        for row in tests:
            if row not in self.farm_rows:
                problems.append(
                    f"{self.name(row)} is not a turbine of {self.farm_label}, so it cannot be a test turbine"
                )
            elif row in self.excluded:
                problems.append(f"{self.name(row)} is excluded, so it cannot be a test turbine")
            elif row in self.reference_only:
                problems.append(f"{self.name(row)} is reference-only, so it cannot be a test turbine")

        records = []
        for row in tests:
            ok = row in self.farm_rows
            refs: list[int] = []
            if ok:
                pool = self.pool(row)
                refs = [j for j in pool if j not in test_set][:REFERENCES_PER_TEST_TURBINE]
                if len(self.within[row]) < REFERENCES_PER_TEST_TURBINE:
                    problems.append(f"{self.name(row)} has {self.too_few_within(row)}")
                    ok = False
                elif len(refs) < REFERENCES_PER_TEST_TURBINE:
                    in_pool = [self.name(j) for j in pool if j in test_set]
                    problems.append(
                        f"{self.name(row)} has only {len(refs)} non-test turbines among its {len(pool)} nearest "
                        f"({_and_list(in_pool)} {'is a test turbine' if len(in_pool) == 1 else 'are test turbines'})"
                    )
                    ok = False
                nearest = self.nearest(row)
                if nearest is not None and nearest in test_set:
                    problems.append(f"{self.name(row)}'s nearest neighbour {self.name(nearest)} is a test turbine")
                    ok = False
            records.append(self._table_row(row, refs=refs, compliant=ok and row in self.candidates))

        n = len(tests)
        front_tests = sum(1 for r in tests if r in self.farm_rows and self.front[r])
        counts = self.share_counts(n)
        a, f = len(self.available), int(self.front[self.available].sum())
        fair = n * f / a if a else 0.0
        if n and a and front_tests not in counts:
            allowed = (
                f"a compliant count is {counts[0]}" if len(counts) == 1 else f"compliant counts are {_or_list(counts)}"
            )
            problems.append(
                f"{front_tests} front-row test turbines against a fair share of {fair:.2f} "
                f"({f} of {a} available turbines are front row); {allowed}"
            )

        table = pd.DataFrame.from_records(records, columns=_table_columns())
        summary: dict[str, object] = {
            "wind_farm": self.farm,
            "farm_turbines": len(self.farm_rows),
            "available_turbines": a,
            "reference_eligible_turbines": a,
            "test_turbines": n,
            "available_front_row": f,
            "front_row_test_turbines": front_tests,
            "fair_front_row_share": fair,
            "compliant_front_row_counts": counts,
            "reference_pool_size": self.reference_pool_size,
            "max_reference_distance_d": self.max_reference_distance_d,
            "filled_rotor_diameters": list(self.layout.filled_rotor_diameters),
        }
        return ComplianceReport(compliant=not problems, problems=tuple(problems), table=table, summary=summary)

    def _table_row(self, row: int, *, refs: list[int], compliant: bool) -> dict[str, object]:
        frame = self.layout.frame
        record: dict[str, object] = {"test_turbine": self.name(row), "front_row": bool(self.front[row])}
        diameter = float(frame[ROTOR_DIAMETER_COL].iloc[row])
        for k in range(1, REFERENCES_PER_TEST_TURBINE + 1):
            if k <= len(refs):
                ref = refs[k - 1]
                distance = float(self.layout.distance_m[row, ref])
                record |= {
                    f"reference_{k}": self.name(ref),
                    f"reference_{k}_distance_m": distance,
                    f"reference_{k}_distance_d": distance / diameter,
                    f"reference_{k}_front_row": bool(self.front[ref]),
                    f"reference_{k}_rank": self.ranked[row].index(ref) + 1,
                }
            else:
                record |= {
                    f"reference_{k}": None,
                    f"reference_{k}_distance_m": math.nan,
                    f"reference_{k}_distance_d": math.nan,
                    f"reference_{k}_front_row": None,
                    f"reference_{k}_rank": None,
                }
        record["compliant"] = compliant
        return record

    def solve(self, *, n: int | None, fixed: Iterable[int] = ()) -> npt.NDArray[np.float64] | None:
        """Return a compliant assignment of ``n`` test turbines including ``fixed``, or ``None`` if none exists.

        With ``n=None`` the assignment maximises the count under the pool and nearest-neighbour rules
        alone, ignoring the front-row share.
        """
        cands = self.candidates
        if not cands:
            return None
        index = {row: k for k, row in enumerate(cands)}
        m = len(cands)
        lower = np.zeros(m)
        upper = np.ones(m)
        for row in fixed:
            lower[index[row]] = 1
        rows: list[npt.NDArray[np.float64]] = []
        lows: list[float] = []
        highs: list[float] = []
        for row in cands:
            k = index[row]
            if len(self.within[row]) < REFERENCES_PER_TEST_TURBINE:
                upper[k] = 0
                continue
            pool = self.pool(row)
            coeffs = np.zeros(m)
            coeffs[k] = REFERENCES_PER_TEST_TURBINE
            for j in pool:
                if j in index:
                    coeffs[index[j]] += 1
            rows.append(coeffs)
            lows.append(-np.inf)
            highs.append(len(pool))
            nearest = self.nearest(row)
            if nearest is not None and nearest in index:
                pair = np.zeros(m)
                pair[k] = pair[index[nearest]] = 1
                rows.append(pair)
                lows.append(-np.inf)
                highs.append(1)
        if (lower > upper).any():
            return None
        if n is not None:
            rows.append(np.ones(m))
            lows.append(n)
            highs.append(n)
            a, f = len(self.available), int(self.front[self.available].sum())
            rows.append(np.array([2.0 * a if self.front[row] else 0.0 for row in cands]))
            lows.append(2 * n * f - a)
            highs.append(2 * n * f + a)
        objective = np.zeros(m) if n is not None else -np.ones(m)
        constraints = [LinearConstraint(np.array(rows), lows, highs)] if rows else []
        result = milp(objective, integrality=np.ones(m), bounds=Bounds(lower, upper), constraints=constraints)
        return np.round(result.x) if result.status == 0 else None

    def max_count(self) -> int:
        """Return the most test turbines a compliant design allows (0 if none)."""
        relaxed = self.solve(n=None)
        if relaxed is None:
            return 0
        for n in range(int(relaxed.sum()), 0, -1):
            if self.solve(n=n) is not None:
                return n
        return 0

    def at_limit(self, limit_d: float) -> _Site:
        """Return this site with references limited to ``limit_d`` rotor diameters."""
        within = {i: [j for j in self.ranked[i] if self.distance_d[i, j] <= limit_d] for i in self.farm_rows}
        return replace(self, within=within, max_reference_distance_d=limit_d)

    def tightest_limit_d(self, *, n: int, fixed: list[int]) -> float:
        """Return the least reference-distance limit at which a design of ``n`` including ``fixed`` exists.

        The site's own limit is the fallback, since the walk only calls this when a design exists there.
        """
        limits = sorted(
            {float(self.distance_d[i, j]) for i in self.candidates for j in self.within[i]}
            | {self.max_reference_distance_d}
        )
        low, high = 0, len(limits) - 1
        while low < high:
            middle = (low + high) // 2
            if self.at_limit(limits[middle]).solve(n=n, fixed=fixed) is not None:
                high = middle
            else:
                low = middle + 1
        return limits[low]

    def why_no_design(self) -> str:
        """Say why no set of test turbines complies."""
        needed = REFERENCES_PER_TEST_TURBINE + 1
        if len(self.available) < needed:
            return (
                f"it needs at least {needed} available turbines (one test turbine and its "
                f"{REFERENCES_PER_TEST_TURBINE} references) and has {len(self.available)}"
            )
        if not any(len(self.within[row]) >= REFERENCES_PER_TEST_TURBINE for row in self.candidates):
            return (
                f"no candidate has {REFERENCES_PER_TEST_TURBINE} reference-eligible turbines within "
                f"{self.max_reference_distance_d:g} rotor diameters"
            )
        return "no set of test turbines meets the pool, nearest-neighbour and front-row share rules together"

    def skip_reason(self, row: int, *, committed: list[int], n: int) -> str:
        """Say why ``row`` cannot join the ``committed`` test turbines in a design of ``n``."""
        return (
            self._spacing_conflict(row, committed=committed)
            or self._share_conflict(row, committed=committed, n=n)
            or f"no compliant design of {n} includes it alongside the higher-priority picks"
        )

    def _spacing_conflict(self, row: int, *, committed: list[int]) -> str | None:
        """Return how ``row`` breaks the pool or nearest-neighbour rule beside ``committed``, if it does."""
        tests = {*committed, row}
        conflict = next((tk for tk in committed if self.nearest(tk) == row), None)
        if conflict is not None:
            return f"reference of {self.name(conflict)} (its nearest neighbour)"
        nearest = self.nearest(row)
        if nearest is not None and nearest in tests:
            return f"{self.name(nearest)} is its nearest neighbour"
        starved = next(
            (tk for tk in committed if sum(j not in tests for j in self.pool(tk)) < REFERENCES_PER_TEST_TURBINE), None
        )
        if starved is not None:
            return f"would leave {self.name(starved)} with fewer than 3 references"
        crowding = [self.name(j) for j in self.pool(row) if j in tests]
        if len(self.pool(row)) - len(crowding) < REFERENCES_PER_TEST_TURBINE:
            return f"too few references among its {len(self.pool(row))} nearest ({_and_list(crowding)} tested)"
        return None

    def _share_conflict(self, row: int, *, committed: list[int], n: int) -> str | None:
        """Return which front-row share ``row`` would overfill, if any."""
        counts = self.share_counts(n)
        front_committed = sum(1 for r in committed if self.front[r])
        if self.front[row] and front_committed >= max(counts):
            return "the front-row share is full"
        if not self.front[row] and len(committed) - front_committed >= n - min(counts):
            return "the rest-of-farm share is full"
        return None


def _table_columns() -> list[str]:
    columns = ["test_turbine", "front_row"]
    for k in range(1, REFERENCES_PER_TEST_TURBINE + 1):
        columns += [
            f"reference_{k}",
            f"reference_{k}_distance_m",
            f"reference_{k}_distance_d",
            f"reference_{k}_front_row",
            f"reference_{k}_rank",
        ]
    return [*columns, "compliant"]


def _resolve_farm(frame: pd.DataFrame, *, wind_farm: str | None) -> str | None:
    """Return the farm under design, defaulting to the only one the layout names."""
    farms = sorted({f for f in frame[WIND_FARM_COL] if f is not None})
    if wind_farm is not None:
        if wind_farm not in farms:
            msg = f"the layout has no wind farm called {wind_farm!r}; it names {farms}"
            raise ValueError(msg)
        return wind_farm
    if len(farms) > 1:
        msg = f"the layout names several wind farms {farms}; pass wind_farm to choose one"
        raise ValueError(msg)
    return farms[0] if farms else None


def _check_roles(names: dict[str, int], *, roles: dict[str, Sequence[str]]) -> None:
    """Raise if a role names a turbine outside the farm, or a turbine holds more than one role."""
    seen: dict[str, str] = {}
    for role, members in roles.items():
        for name in members:
            if name not in names:
                msg = f"{role} names {name!r}, which is not a turbine of the farm under design"
                raise ValueError(msg)
            if name in seen:
                msg = f"{name} is in both {seen[name]} and {role}; a turbine may hold one role"
                raise ValueError(msg)
            seen[name] = role


def _and_list(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _or_list(items: Sequence[int]) -> str:
    return " or ".join(str(i) for i in items)
