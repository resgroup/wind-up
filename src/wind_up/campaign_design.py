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
from wind_up.wake_reach import (
    DEFAULT_DEMOTION_SHARE,
    REFERENCES_PER_TEST_TURBINE,
    candidate_cost,
    kept_wind_share,
    reach_from_uplift,
    union_share,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt

    from wind_up.wake_reach import LossSector, Rose

DEFAULT_MIN_KEPT_WIND_SHARE = 0.5
# How far below the best total kept wind share a design may fall and still count as optimal.
OBJECTIVE_TOLERANCE = 1e-6
# A pairwise loss under this share of the rose is left out of the solver.
_MIN_PAIR_LOSS = 1e-9
WAKE_LOSS_COLUMNS = ("test_turbine", "waked", "waker", "distance_d", "centre_deg", "half_width_deg", "rose_share")
# The columns of ComplianceReport.table, in order: one row per test turbine.
COMPLIANCE_COLUMNS: tuple[str, ...] = (
    "test_turbine",
    "front_row",
    *(
        f"reference_{k}{suffix}"
        for k in range(1, REFERENCES_PER_TEST_TURBINE + 1)
        for suffix in ("", "_distance_m", "_distance_d", "_front_row", "_rank")
    ),
    "kept_wind_share",
    "lost_own_references",
    "lost_other_tests",
    "compliant",
)


@dataclass(frozen=True, eq=False)
class ComplianceReport:
    """The outcome of :func:`check_design`.

    ``table`` has one row per test turbine; ``wake_losses`` one row per sector a test turbine loses
    to a changed turbine upwind of it or of one of its references, with :data:`WAKE_LOSS_COLUMNS`.
    """

    compliant: bool
    problems: tuple[str, ...]
    table: pd.DataFrame
    summary: dict[str, object]
    wake_losses: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=list(WAKE_LOSS_COLUMNS)))

    def __eq__(self, other: object) -> bool:
        """Compare field by field, the tables by value."""
        if not isinstance(other, ComplianceReport):
            return NotImplemented
        return (
            self.compliant == other.compliant
            and self.problems == other.problems
            and self.summary == other.summary
            and self.table.equals(other.table)
            and self.wake_losses.equals(other.wake_losses)
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

    ``reference_limit_d`` is the furthest reference any test turbine uses, in its rotor diameters.
    ``lone_kept_wind_share`` is each available farm turbine's kept wind share as a lone test turbine with its
    own best-ranked references; NaN where it has too few.
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
    wake_reach_d: float = 0.0
    expected_uplift: float | None = None
    min_kept_wind_share: float = DEFAULT_MIN_KEPT_WIND_SHARE
    demotion_share: float = DEFAULT_DEMOTION_SHARE
    wind_rose: Rose | None = field(default=None, repr=False, compare=False)
    rose_label: str = "none"
    lone_kept_wind_share: dict[str, float] = field(default_factory=dict)

    def roles(self) -> dict[str, list[str]]:
        """Return the campaign declaration roles: every available non-test farm turbine is a reference."""
        tests = set(self.test_turbines)
        excluded = set(self.excluded)
        frame = self.layout.frame
        farm = [str(n) for n, f in zip(frame[NAME_COL], frame[WIND_FARM_COL], strict=True) if f == self.wind_farm]
        return {
            "upgraded": sorted(self.test_turbines),
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
    expected_uplift: float | None = None,
    wake_reach_d: float | None = None,
    wind_rose: Rose | None = None,
    offshore: bool = False,
    min_kept_wind_share: float = DEFAULT_MIN_KEPT_WIND_SHARE,
    demotion_share: float = DEFAULT_DEMOTION_SHARE,
    rose_label: str = "given",
) -> ComplianceReport:
    """Check ``test_turbines`` against the rules in ``docs/designing-a-campaign.md``.

    Raises only on malformed input: an unknown or repeated name, a name in more than one role, an
    ambiguous ``wind_farm``, a reach above 0 without a rose. Non-compliance is reported in the
    returned :class:`ComplianceReport`. The wake arguments are those of :func:`design_campaign`.
    """
    site = _Site.build(
        layout,
        wind_farm=wind_farm,
        reference_only=reference_only,
        excluded=excluded,
        reference_pool_size=reference_pool_size,
        max_reference_distance_d=max_reference_distance_d,
        front_row_min_clear_deg=front_row_min_clear_deg,
        wake=_Wake.resolve(
            expected_uplift=expected_uplift,
            wake_reach_d=wake_reach_d,
            wind_rose=wind_rose,
            offshore=offshore,
            min_kept_wind_share=min_kept_wind_share,
            demotion_share=demotion_share,
            rose_label=rose_label,
        ),
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
    expected_uplift: float | None = None,
    wake_reach_d: float | None = None,
    wind_rose: Rose | None = None,
    offshore: bool = False,
    min_kept_wind_share: float = DEFAULT_MIN_KEPT_WIND_SHARE,
    demotion_share: float = DEFAULT_DEMOTION_SHARE,
    rose_label: str = "given",
) -> CampaignDesign:
    """Choose test turbines for a campaign on the farm ``wind_farm`` of ``layout``.

    :param layout: one row per turbine, as described in ``docs/designing-a-campaign.md``. Turbines
        outside the farm under design only block wakes.
    :param wind_farm: the farm under design; defaults to the only ``wind_farm`` in the layout, or
        to every turbine when the layout names no wind farm
    :param test_priority: farm turbines to test first, highest priority first; the rest of the
        design is the one with the highest mean kept wind share, then the nearest references, then
        the first in an order drawn from ``seed``
    :param reference_only: farm turbines that are never tested but may be references
    :param excluded: farm turbines that are neither tested nor references; they still block wakes
    :param n_test: how many test turbines; ``None`` takes the most a compliant design allows
    :param seed: seeds the order of the candidates not in ``test_priority``
    :param reference_pool_size: how many best-ranked reference-eligible turbines form a test
        turbine's pool
    :param max_reference_distance_d: the furthest a reference may be, in the test turbine's rotor
        diameters
    :param front_row_min_clear_deg: the contiguous clear arc that makes a turbine front row
    :param expected_uplift: the AEP uplift expected of the change, a fraction; sets the wake reach
    :param wake_reach_d: the wake reach in rotor diameters, overriding ``expected_uplift``; with
        neither, the reach is 0 and no wind is lost to a test turbine's wake change
    :param wind_rose: the site's wind rose, needed for a reach above 0;
        :func:`~wind_up.wake_reach.uniform_rose` for a site without one
    :param offshore: doubles the reach taken from ``expected_uplift``
    :param min_kept_wind_share: the least share of the rose a test turbine may keep clear of changed turbines
    :param demotion_share: a candidate reference the test turbine is upwind of for more than this share
        of the rose ranks behind the others
    :param rose_label: what the rose is, for the reports
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
        wake=_Wake.resolve(
            expected_uplift=expected_uplift,
            wake_reach_d=wake_reach_d,
            wind_rose=wind_rose,
            offshore=offshore,
            min_kept_wind_share=min_kept_wind_share,
            demotion_share=demotion_share,
            rose_label=rose_label,
        ),
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
            # The unlisted rest: the best mean kept wind share, then the nearest references, then the seeded order.
            floored = replace(site, floor=site.best_objective(n=n, fixed=committed))
            limit_d = floored.tightest_limit_d(n=n, fixed=committed)
            walking = floored.at_limit(limit_d)
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
    report = site.check(committed)
    table = report.table.set_index("test_turbine")
    distances = table[[f"reference_{k}_distance_d" for k in range(1, REFERENCES_PER_TEST_TURBINE + 1)]]
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
        reference_limit_d=float(np.nanmax(distances.to_numpy(dtype=float))),
        front_row_min_clear_deg=front_row_min_clear_deg,
        compliance=report,
        layout=site.layout,
        wake_reach_d=site.wake.reach_d,
        expected_uplift=site.wake.expected_uplift,
        min_kept_wind_share=site.wake.min_kept_wind_share,
        demotion_share=site.wake.demotion_share,
        wind_rose=site.wake.rose,
        rose_label=site.wake.rose_label,
        lone_kept_wind_share={site.name(r): site.lone.get(r, math.nan) for r in site.available},
    )


def write_design(design: CampaignDesign, *, out_dir: str | Path) -> None:
    """Write ``design`` to ``out_dir``.

    Files: ``compliance.csv`` (one row per test turbine), ``wake_losses.csv`` (one row per sector a
    test turbine loses), ``summary.yaml`` (counts, the front-row share, the wake reach and kept
    shares, the most test turbines possible, any problems), ``turbines.csv`` (every farm turbine's
    role, references served, priority, outcome and kept wind share as a lone test turbine), ``roles.yaml``
    (a campaign declaration ``turbines:`` block), and the maps ``design_map.png``,
    ``design_map_latlon.png``, ``front_row_map.png`` and ``coverage_map.png``.
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
    """Write ``compliance.csv``, ``wake_losses.csv`` and ``summary.yaml`` for ``report`` to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_report(report, out_dir=out, extra={})


def _write_report(report: ComplianceReport, *, out_dir: Path, extra: dict[str, object]) -> None:
    report.table.to_csv(out_dir / "compliance.csv", index=False)
    report.wake_losses.to_csv(out_dir / "wake_losses.csv", index=False)
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
                "lone_kept_wind_share": design.lone_kept_wind_share.get(name, math.nan),
                "latitude": row[LATITUDE_COL],
                "longitude": row[LONGITUDE_COL],
                "rotor_diameter_m": row[ROTOR_DIAMETER_COL],
            }
        )
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class _Wake:
    """How far a test turbine's wake change reaches, the rose it is weighed by, and the rules on it."""

    reach_d: float
    rose: Rose | None = field(repr=False, compare=False)
    min_kept_wind_share: float
    demotion_share: float
    expected_uplift: float | None
    rose_label: str

    @classmethod
    def resolve(
        cls,
        *,
        expected_uplift: float | None,
        wake_reach_d: float | None,
        wind_rose: Rose | None,
        offshore: bool,
        min_kept_wind_share: float,
        demotion_share: float,
        rose_label: str,
    ) -> _Wake:
        """Return the wake settings: the given reach, else the one of ``expected_uplift``, else 0."""
        if wake_reach_d is not None:
            reach = float(wake_reach_d)
        elif expected_uplift is not None:
            reach = reach_from_uplift(expected_uplift, offshore=offshore)
        else:
            reach = 0.0
        if not reach >= 0:
            msg = f"wake_reach_d must be at least 0 rotor diameters, not {wake_reach_d}"
            raise ValueError(msg)
        if reach > 0 and wind_rose is None:
            msg = (
                f"a wake reach of {reach:g} D needs a wind rose; pass wind_up.wake_reach.uniform_rose() for a "
                "site without one"
            )
            raise ValueError(msg)
        return cls(
            reach_d=reach,
            rose=wind_rose,
            min_kept_wind_share=min_kept_wind_share,
            demotion_share=demotion_share,
            expected_uplift=expected_uplift,
            rose_label=rose_label if wind_rose is not None else "none",
        )

    @property
    def active(self) -> bool:
        """Whether a test turbine's wake change costs any wind."""
        return self.reach_d > 0 and self.rose is not None


@dataclass(frozen=True)
class _Site:
    """The farm under design, its roles, and each turbine's ranked neighbours.

    ``lone`` is each available turbine's kept wind share as a lone test turbine with its first
    references in ``within``, and ``pair_loss[(j, i)]`` the further share candidate ``i`` loses when
    candidate ``j`` is a test turbine too. ``floor`` is the least total kept wind share a design may have.
    ``cuts`` are the configurations found to break the kept-wind-share rule exactly, ``(tested, untested)``:
    no design tests every turbine of the first and none of the second. They hold for the site's
    pools, so copies under a floor share them and copies at another limit start afresh.
    """

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
    wake: _Wake
    lone: dict[int, float]
    pair_loss: dict[tuple[int, int], float]
    floor: float | None = None
    cuts: list[tuple[frozenset[int], frozenset[int]]] = field(default_factory=list)

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
        wake: _Wake,
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

            def demoted(j: int, *, test: int = i) -> bool:
                if not wake.active:
                    return False
                cost = candidate_cost(
                    layout,
                    test=str(frame[NAME_COL].iloc[test]),
                    candidate=str(frame[NAME_COL].iloc[j]),
                    reach_d=wake.reach_d,
                    rose=wake.rose,  # type: ignore[arg-type]
                )
                return cost > wake.demotion_share

            others = sorted(
                (j for j in available if j != i),
                key=lambda j: (demoted(j), layout.distance_m[i, j], frame[NAME_COL].iloc[j]),
            )
            ranked[i] = others
            within[i] = [j for j in others if distance_d[i, j] <= max_reference_distance_d]
        candidates = [i for i in available if i not in reference_only_rows]
        pools = {i: within[i][:reference_pool_size] for i in available}
        lone, pair_loss = _wake_terms(layout, pools=pools, available=available, candidates=candidates, wake=wake)
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
            wake=wake,
            lone=lone,
            pair_loss=pair_loss,
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
        losses: list[dict[str, object]] = []
        kept_total = 0.0
        for row in tests:
            ok = row in self.farm_rows
            refs: list[int] = []
            kept: tuple[float, float] = (1.0, 0.0)
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
                    word = self.neighbour_word(row)
                    problems.append(f"{self.name(row)}'s {word} neighbour {self.name(nearest)} is a test turbine")
                    ok = False
                share, lost_own, sectors = self.kept(row, refs=refs, tests=tests)
                kept = (share, lost_own)
                losses += [{"test_turbine": self.name(row), **_sector_record(sector)} for sector in sectors]
                if share < self.wake.min_kept_wind_share:
                    problems.append(
                        f"{self.name(row)} keeps {share:.0%} of the wind clear of changed turbines upwind of it or "
                        f"its references, under the minimum {self.wake.min_kept_wind_share:.0%}"
                    )
                    ok = False
            kept_total += kept[0]
            records.append(self._table_row(row, refs=refs, kept=kept, compliant=ok and row in self.candidates))

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

        table = pd.DataFrame.from_records(records, columns=list(COMPLIANCE_COLUMNS))
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
            "wake_reach_d": self.wake.reach_d,
            "expected_uplift": self.wake.expected_uplift,
            "min_kept_wind_share": self.wake.min_kept_wind_share,
            "demotion_share": self.wake.demotion_share,
            "mean_kept_wind_share": kept_total / n if n else math.nan,
            "rose": self.wake.rose_label,
        }
        return ComplianceReport(
            compliant=not problems,
            problems=tuple(problems),
            table=table,
            summary=summary,
            wake_losses=pd.DataFrame.from_records(losses, columns=list(WAKE_LOSS_COLUMNS)),
        )

    def kept(
        self, row: int, *, refs: Sequence[int], tests: Sequence[int]
    ) -> tuple[float, float, tuple[LossSector, ...]]:
        """Return ``row``'s kept wind share, the share it loses to its own wake change, and its loss sectors."""
        if not self.wake.active:
            return 1.0, 0.0, ()
        rose = self.wake.rose
        share, sectors = kept_wind_share(
            self.layout,
            test=self.name(row),
            references=[self.name(r) for r in refs],
            other_tests=[self.name(t) for t in tests if t != row],
            reach_d=self.wake.reach_d,
            rose=rose,  # type: ignore[arg-type]
        )
        own = [sector for sector in sectors if sector.waker == self.name(row)]
        return share, union_share(own, rose=rose), sectors  # type: ignore[arg-type]

    def neighbour_word(self, row: int) -> str:
        """Return how the rules name ``row``'s first-ranked neighbour: nearest, unless demotion moved it."""
        by_distance = min(self.ranked[row], key=lambda j: self.layout.distance_m[row, j], default=None)
        return "nearest" if by_distance == self.nearest(row) else "best-ranked"

    def _table_row(self, row: int, *, refs: list[int], kept: tuple[float, float], compliant: bool) -> dict[str, object]:
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
        share, lost_own = kept
        record |= {
            "kept_wind_share": share,
            "lost_own_references": lost_own,
            "lost_other_tests": 1.0 - share - lost_own,
        }
        record["compliant"] = compliant
        if tuple(record) != COMPLIANCE_COLUMNS:
            msg = f"compliance row columns {list(record)} are not COMPLIANCE_COLUMNS"
            raise RuntimeError(msg)
        return record

    def solve(
        self, *, n: int | None, fixed: Iterable[int] = (), optimise: bool = False
    ) -> npt.NDArray[np.float64] | None:
        """Return a compliant assignment of ``n`` test turbines including ``fixed``, or ``None`` if none exists.

        With ``n=None`` the assignment maximises the count under the pool, nearest-neighbour and kept-wind-share
        rules alone, ignoring the front-row share. With ``optimise`` it maximises the total kept wind share,
        each pair of test turbines within reach adding its loss. Rule 5 and the total are linear in the
        pair losses, which overcount where sectors overlap. Under a :attr:`floor`, only assignments whose
        total reaches it are compliant.
        """
        cands = self.candidates
        if not cands:
            return None
        index = {row: k for k, row in enumerate(cands)}
        m = len(cands)
        pairs = sorted(
            {(min(index[i], index[j]), max(index[i], index[j])) for j, i in self.pair_loss if i in index and j in index}
        )
        program = _Program(width=m + len(pairs))
        for row in fixed:
            program.lower[index[row]] = 1
        self._spacing_rows(program, index=index)
        weights = self._wake_rows(program, index=index, pairs=pairs)
        if self.floor is not None:
            program.add(weights, low=self.floor - OBJECTIVE_TOLERANCE, high=np.inf)
        if (program.lower > program.upper).any():
            return None
        count = np.zeros(program.width)
        count[:m] = 1
        if n is not None:
            program.add(count, low=n, high=n)
            a, f = len(self.available), int(self.front[self.available].sum())
            share = np.zeros(program.width)
            share[:m] = [2.0 * a if self.front[row] else 0.0 for row in cands]
            program.add(share, low=2 * n * f - a, high=2 * n * f + a)
        objective: npt.NDArray[np.float64] = np.zeros(program.width)
        if optimise:
            objective = -weights
        elif n is None:
            objective = -count
        solution = self._solve_with_cuts(program, objective=objective, index=index)
        return None if solution is None else solution[:m]

    def _solve_with_cuts(
        self, program: _Program, *, objective: npt.NDArray[np.float64], index: dict[int, int]
    ) -> npt.NDArray[np.float64] | None:
        """Solve ``program`` under every known cut, adding a cut and solving again while a test turbine fails rule 5."""
        for cut in self.cuts:
            _add_cut(program, cut=cut, index=index)
        while True:
            solution = program.solve(objective)
            if solution is None:
                return None
            found = self.exact_cuts([row for row, k in index.items() if solution[k]])
            if not found:
                return solution
            for cut in found:
                self.cuts.append(cut)
                _add_cut(program, cut=cut, index=index)

    def exact_cuts(self, tests: Sequence[int]) -> list[tuple[frozenset[int], frozenset[int]]]:
        """Return a cut for each of ``tests`` whose exact kept wind share is under the minimum.

        The cut is the test turbine with the tested turbines of its pool and those whose sectors it
        loses, against its pool's untested turbines: with exactly those its references are the same,
        and more test turbines only lose it more.
        """
        if not self.wake.active or self.wake.min_kept_wind_share <= 0:
            return []
        cuts = []
        for test in tests:
            pool = self.pool(test)
            refs = [j for j in pool if j not in tests][:REFERENCES_PER_TEST_TURBINE]
            share, _, sectors = self.kept(test, refs=refs, tests=tests)
            if share >= self.wake.min_kept_wind_share:
                continue
            tested = {test, *(j for j in pool if j in tests), *(self.row_of(x.waker) for x in sectors)}
            cuts.append((frozenset(tested), frozenset(j for j in pool if j not in tests)))
        return cuts

    def _spacing_rows(self, program: _Program, *, index: dict[int, int]) -> None:
        """Add the pool and nearest-neighbour rules; a candidate with too few references is never tested."""
        for row, k in index.items():
            if len(self.within[row]) < REFERENCES_PER_TEST_TURBINE:
                program.upper[k] = 0
                continue
            pool = self.pool(row)
            coeffs = np.zeros(program.width)
            coeffs[k] = REFERENCES_PER_TEST_TURBINE
            for j in pool:
                if j in index:
                    coeffs[index[j]] += 1
            program.add(coeffs, low=-np.inf, high=len(pool))
            nearest = self.nearest(row)
            if nearest is not None and nearest in index:
                pair = np.zeros(program.width)
                pair[k] = pair[index[nearest]] = 1
                program.add(pair, low=-np.inf, high=1)

    def _wake_rows(
        self, program: _Program, *, index: dict[int, int], pairs: Sequence[tuple[int, int]]
    ) -> npt.NDArray[np.float64]:
        """Add a product variable per pair and the kept-wind-share rule; return the objective's coefficients."""
        cands = list(index)
        m = len(cands)
        weights = np.zeros(program.width)
        kept_rows = {row: np.zeros(program.width) for row in cands}
        for row, k in index.items():
            weights[k] = self.lone.get(row, 0.0)
            kept_rows[row][k] = self.lone.get(row, 0.0) - self.wake.min_kept_wind_share
        for p, (a, b) in enumerate(pairs, start=m):
            i, j = cands[a], cands[b]
            for row, other in ((i, j), (j, i)):
                loss = self.pair_loss.get((other, row), 0.0)
                weights[p] -= loss
                kept_rows[row][p] -= loss
            for k in (a, b):
                link = np.zeros(program.width)
                link[p], link[k] = 1, -1
                program.add(link, low=-np.inf, high=0)
            both = np.zeros(program.width)
            both[a], both[b], both[p] = 1, 1, -1
            program.add(both, low=-np.inf, high=1)
        if self.wake.active and self.wake.min_kept_wind_share > 0:
            for row, k in index.items():
                if program.upper[k]:
                    program.add(kept_rows[row], low=0, high=np.inf)
        return weights

    def best_objective(self, *, n: int, fixed: Iterable[int] = ()) -> float | None:
        """Return the highest total kept wind share of a compliant design of ``n`` including ``fixed``, or None."""
        if not self.wake.active:
            return None
        cands = self.candidates
        solution = self.solve(n=n, fixed=fixed, optimise=True)
        if solution is None:
            return None
        chosen = [row for row, x in zip(cands, solution, strict=True) if x]
        return self.linear_total(chosen)

    def linear_total(self, tests: Sequence[int]) -> float:
        """Return the solver's total kept wind share of ``tests``: lone shares less every pair loss."""
        chosen = set(tests)
        return sum(self.lone.get(t, 0.0) for t in tests) - sum(
            loss for (j, i), loss in self.pair_loss.items() if i in chosen and j in chosen
        )

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
        """Return this site with references limited to ``limit_d`` rotor diameters, its wake terms on the new pools."""
        within = {i: [j for j in self.ranked[i] if self.distance_d[i, j] <= limit_d] for i in self.farm_rows}
        site = replace(self, within=within, max_reference_distance_d=limit_d, cuts=[])
        if all(site.pool(i) == self.pool(i) for i in self.available):
            return site
        lone, pair_loss = _wake_terms(
            self.layout,
            pools={i: site.pool(i) for i in self.available},
            available=self.available,
            candidates=self.candidates,
            wake=self.wake,
        )
        return replace(site, lone=lone, pair_loss=pair_loss)

    def tightest_limit_d(self, *, n: int, fixed: list[int]) -> float:
        """Return the least reference-distance limit at which a design of ``n`` including ``fixed`` exists.

        A binary search, then a scan up from where it ends, since a test turbine's references, and so
        its kept wind share, can change with the limit. The site's own limit is the fallback, since the walk
        only calls this when a design exists there.
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
        return next(
            (limit for limit in limits[low:-1] if self.at_limit(limit).solve(n=n, fixed=fixed) is not None),
            limits[-1],
        )

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
            or self._wake_conflict(row, committed=committed)
            or f"no compliant design of {n} includes it alongside the higher-priority picks"
        )

    def _spacing_conflict(self, row: int, *, committed: list[int]) -> str | None:
        """Return how ``row`` breaks the pool or nearest-neighbour rule beside ``committed``, if it does."""
        tests = {*committed, row}
        conflict = next((tk for tk in committed if self.nearest(tk) == row), None)
        if conflict is not None:
            return f"reference of {self.name(conflict)} (its {self.neighbour_word(conflict)} neighbour)"
        nearest = self.nearest(row)
        if nearest is not None and nearest in tests:
            return f"{self.name(nearest)} is its {self.neighbour_word(row)} neighbour"
        starved = next(
            (tk for tk in committed if sum(j not in tests for j in self.pool(tk)) < REFERENCES_PER_TEST_TURBINE), None
        )
        if starved is not None:
            return f"would leave {self.name(starved)} with fewer than 3 references"
        crowding = [self.name(j) for j in self.pool(row) if j in tests]
        if len(self.pool(row)) - len(crowding) < REFERENCES_PER_TEST_TURBINE:
            return f"too few references among its {len(self.pool(row))} nearest ({_and_list(crowding)} tested)"
        return None

    def _wake_conflict(self, row: int, *, committed: list[int]) -> str | None:
        """Return which test turbine would keep too little wind beside ``committed``, if any."""
        if not self.wake.active:
            return None
        tests = [*committed, row]
        for test in tests:
            share = self.lone.get(test, 0.0) - sum(self.pair_loss.get((j, test), 0.0) for j in tests)
            if share < self.wake.min_kept_wind_share:
                whose = "it" if test == row else self.name(test)
                return (
                    f"{whose} would keep about {share:.0%} of the wind clear of changed turbines, under the "
                    f"minimum {self.wake.min_kept_wind_share:.0%}"
                )
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


@dataclass
class _Program:
    """A binary linear program's bounds and constraint rows, built up one row at a time."""

    width: int
    rows: list[npt.NDArray[np.float64]] = field(default_factory=list)
    lows: list[float] = field(default_factory=list)
    highs: list[float] = field(default_factory=list)
    lower: npt.NDArray[np.float64] = field(init=False)
    upper: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        """Start every variable free between 0 and 1."""
        self.lower = np.zeros(self.width)
        self.upper = np.ones(self.width)

    def add(self, coeffs: npt.NDArray[np.float64], *, low: float, high: float) -> None:
        """Add the constraint ``low <= coeffs @ x <= high``."""
        self.rows.append(coeffs)
        self.lows.append(low)
        self.highs.append(high)

    def solve(self, objective: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
        """Return the binary ``x`` minimising ``objective @ x``, or None when there is none."""
        constraints = [LinearConstraint(np.array(self.rows), self.lows, self.highs)] if self.rows else []
        result = milp(
            objective,
            integrality=np.ones(self.width),
            bounds=Bounds(self.lower, self.upper),
            constraints=constraints,
        )
        return np.round(result.x) if result.status == 0 else None


def _add_cut(program: _Program, *, cut: tuple[frozenset[int], frozenset[int]], index: dict[int, int]) -> None:
    """Add a cut: not every turbine of its first set tested while none of its second is."""
    tested, untested = cut
    if not tested <= set(index):
        return
    coeffs = np.zeros(program.width)
    for row in tested:
        coeffs[index[row]] = 1
    for row in untested:
        if row in index:
            coeffs[index[row]] = -1
    program.add(coeffs, low=-np.inf, high=len(tested) - 1)


def _wake_terms(
    layout: Layout,
    *,
    pools: dict[int, list[int]],
    available: Sequence[int],
    candidates: Sequence[int],
    wake: _Wake,
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Return each available turbine's lone kept wind share and each ordered pair of candidates' further loss.

    A turbine's references are the first :data:`REFERENCES_PER_TEST_TURBINE` of its pool that are not
    tested; one with too few has no kept wind share. ``(j, i)`` is the share ``i`` loses when ``j`` is
    tested too, beyond what it keeps alone, with ``j`` taken out of its references; negative when
    the reference that replaces ``j`` costs less.
    """
    names = [str(n) for n in layout.frame[NAME_COL]]

    def share(i: int, tested: Sequence[int]) -> float | None:
        refs = [j for j in pools[i] if j not in tested][:REFERENCES_PER_TEST_TURBINE]
        if len(refs) < REFERENCES_PER_TEST_TURBINE:
            return None
        if not wake.active:
            return 1.0
        kept, _ = kept_wind_share(
            layout,
            test=names[i],
            references=[names[j] for j in refs],
            other_tests=[names[j] for j in tested],
            reach_d=wake.reach_d,
            rose=wake.rose,  # type: ignore[arg-type]
        )
        return kept

    lone = {i: kept for i in available if (kept := share(i, ())) is not None}
    pair_loss = {}
    if wake.active:
        for i in candidates:
            if i not in lone:
                continue
            for j in candidates:
                kept = share(i, (j,)) if j != i else None
                if kept is not None and abs(lone[i] - kept) > _MIN_PAIR_LOSS:
                    pair_loss[j, i] = lone[i] - kept
    return lone, pair_loss


def _sector_record(sector: LossSector) -> dict[str, object]:
    """Return a loss sector as a ``wake_losses.csv`` row, without its test turbine."""
    return {
        "waked": sector.waked,
        "waker": sector.waker,
        "distance_d": sector.distance_d,
        "centre_deg": sector.centre_deg,
        "half_width_deg": sector.half_width_deg,
        "rose_share": sector.rose_share,
    }


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
