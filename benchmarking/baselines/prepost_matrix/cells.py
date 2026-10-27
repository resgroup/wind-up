"""The prepost campaign matrix's cells: one full ``wind-up`` run each, named and seeded stably.

A cell is one (campaign, multiplier, K, post length). The cell list is a pure function of the
:class:`MatrixSettings`. A synthetic site's campaigns are its tiles, seed index ``i`` being tile ``i``
(see :func:`benchmarking.campaigns.rollout.draw_tiles`); tile ``i`` depends only on ``master_seed``,
the site and the seed indices up to ``i``, so a cell's id and its campaign do not depend on how many
seeds the matrix runs. A real campaign runs
the post lengths its data holds, and Lmax, all of it, where :data:`REAL_SITES` says so.
"""

from __future__ import annotations

import itertools
import zlib
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from benchmarking.campaigns.declaration import Exclusion

Arm = Literal["main", "excl", "small", "real"]


@dataclass(frozen=True)
class RealSite:
    """A real campaign of the matrix.

    :param campaign: its name in :data:`benchmarking.campaigns.real.REAL_CAMPAIGNS`
    :param longest_pre_months: the most whole months its usable data holds before the changeover
    :param longest_post_months: the most whole months its data holds after the changeover
    :param lmax: also run one cell on all the data after the changeover
    """

    campaign: str
    longest_pre_months: int
    longest_post_months: int
    lmax: bool = False


REAL_SITES: dict[str, RealSite] = {
    "hot_t13": RealSite("hot_aeroup_t13", longest_pre_months=24, longest_post_months=15),
    "hot_tuneup": RealSite("hot_tuneup", longest_pre_months=13, longest_post_months=5, lmax=True),
}
# The arms whose references are given synthetic exclusions.
EXCLUSION_ARMS: tuple[str, ...] = ("excl",)
# The arms whose campaigns are a small change: about 1 % AEP, declared as 1 %.
SMALL_ARMS: tuple[str, ...] = ("small",)
_ARM_SUFFIX = {"main": "", "excl": "_excl", "small": "_1pc", "real": ""}

# Synthetic reference exclusions for the exclusion arms: per turbine, this many periods, each this many days.
EXCLUSIONS_PER_TURBINE = (0, 2)
EXCLUSION_DAYS = (3, 14)


@dataclass(frozen=True)
class MatrixSettings:
    """The axes of the matrix and the seed every campaign is drawn from.

    :param seeds: campaigns (tiles) per synthetic site, keyed ``hot``, ``pen``, ``kel``
    :param multipliers: the AeroUp shape's scalings, paired within a campaign
    :param ks: power references per test turbine
    :param pre_months: months of data before the first trial works start (before the changeover for a real campaign)
    :param post_months: months of data after the last trial works end (after the changeover for a real campaign)
    :param exclusion_k: the one K the exclusion arms run at; the small-change arms run at every K of ``ks``
    :param exclusion_arms: which of the exclusion arms run
    :param small_arms: which of the small-change arms run
    :param real: the real campaigns included, keys of :data:`REAL_SITES`
    :param min_side_days: the shortest side the period selector may choose, so that every post length is analysed
    :param min_total_months: the fewest months before and after together a cell may have; shorter shapes are left out
    :param master_seed: every campaign draw is keyed on it
    :param season_shifts: months each synthetic campaign's works are moved by (see
        :meth:`~benchmarking.campaigns.rollout.RolloutDraw.shifted`), each a cell of its own
    """

    seeds: dict[str, int] = field(default_factory=lambda: {"hot": 16, "pen": 12, "kel": 4})
    multipliers: tuple[int, ...] = (1, 0, -1)
    ks: tuple[int, ...] = (3, 4, 6)
    pre_months: tuple[int, ...] = (3, 6, 12, 24)
    post_months: tuple[int, ...] = (1, 2, 3, 6, 9, 12)
    exclusion_k: int = 4
    exclusion_arms: tuple[str, ...] = EXCLUSION_ARMS
    small_arms: tuple[str, ...] = SMALL_ARMS
    real: tuple[str, ...] = tuple(REAL_SITES)
    min_side_days: int = 14
    min_total_months: int = 9
    master_seed: int = 0
    season_shifts: tuple[int, ...] = (0,)

    def to_json(self) -> dict[str, object]:
        """Return the settings as plain data, lists in place of tuples."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> MatrixSettings:
        """Return the settings :meth:`to_json` wrote; an older ``real: true`` reads as the T13 campaign alone."""
        data = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
        if isinstance(data.get("real"), bool):
            data["real"] = ("hot_t13",) if data["real"] else ()
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Cell:
    """One run of the matrix.

    :param arm: ``main``; ``excl``, whose references have synthetic exclusions; ``small``, a change of
        about 1 % declared as 1 %; or ``real``
    :param site: ``hot``, ``pen``, ``kel`` or a key of :data:`REAL_SITES`
    :param seed_index: which of the site's campaigns; ``None`` for a real campaign
    :param multiplier: the AeroUp scaling; ``None`` for a real campaign
    :param k: power references per test turbine
    :param pre_months: months of data before the (first trial) works start; ``None`` for all of a real
        campaign's data, Pmax
    :param post_months: months of data after the (last trial) works end; ``None`` for all of a real
        campaign's data, Lmax
    :param season_shift: months the synthetic campaign's works are moved by; 0 for a real campaign
    """

    arm: Arm
    site: str
    seed_index: int | None
    multiplier: int | None
    k: int
    pre_months: int | None
    post_months: int | None
    season_shift: int = 0

    @property
    def cell_id(self) -> str:
        """A readable, stable id: ``pen_s07_m+1_K4_P12_L6``, ``hot_t13_K4_P24_L6`` or ``hot_tuneup_K4_Pmax_Lmax``.

        A season shift adds ``_S<months>`` after the periods: ``pen_s07_m+1_K4_P12_L6_S3``.
        """
        pre = "max" if self.pre_months is None else self.pre_months
        post = "max" if self.post_months is None else self.post_months
        periods = f"P{pre}_L{post}" + (f"_S{self.season_shift}" if self.season_shift else "")
        if self.arm == "real":
            return f"{self.site}_K{self.k}_{periods}"
        return f"{self.site}_s{self.seed_index:02d}_m{self.multiplier:+d}_K{self.k}_{periods}{_ARM_SUFFIX[self.arm]}"

    @property
    def has_exclusions(self) -> bool:
        """Whether this cell's references are given synthetic exclusions."""
        return self.arm in EXCLUSION_ARMS

    @property
    def is_small(self) -> bool:
        """Whether this cell's campaign is a small change, about 1 % AEP, declared as 1 %."""
        return self.arm in SMALL_ARMS

    @property
    def full_rollout(self) -> bool:
        """Whether this campaign rolls the upgrade out to every other turbine: every even seed index."""
        return self.seed_index is not None and is_full_rollout(self.seed_index)


def is_full_rollout(seed_index: int) -> bool:
    """Whether a synthetic site's campaign ``seed_index`` rolls the upgrade out to every other turbine."""
    return seed_index % 2 == 0


@dataclass(frozen=True)
class StudySize:
    """One size of study: its matrix and how many cells run at once by default."""

    settings: MatrixSettings
    workers: int


SIZES: dict[str, StudySize] = {
    # A laptop smoke test of every arm and the real campaign; a synthetic HOT cell alone costs ~20 min.
    "small": StudySize(
        MatrixSettings(seeds={"hot": 0, "pen": 1, "kel": 1}, ks=(4,), pre_months=(24,), post_months=(3, 12)),
        workers=3,
    ),
    # About 142 CPU-hours of cells, 5 % and 1 % campaigns in equal numbers at K3 and K4; check with ``plan``
    # before a run.
    "big": StudySize(
        MatrixSettings(seeds={"hot": 6, "pen": 7, "kel": 6}, ks=(3, 4), exclusion_arms=()),
        workers=10,
    ),
    # The big matrix's campaigns in the validation domain, each with its works in all four seasons, so
    # the season of the upgrade can be judged within a campaign; no real campaign.
    "season": StudySize(
        MatrixSettings(
            seeds={"hot": 6, "pen": 7, "kel": 6},
            multipliers=(1, 0),
            ks=(4,),
            pre_months=(12, 24),
            post_months=(6, 12),
            exclusion_arms=(),
            real=(),
            season_shifts=(0, 3, 6, 9),
        ),
        workers=10,
    ),
}

# Median wall seconds of one cell per (site, K), measured on the HPC at 3742175 over 12 to 24 months of
# data before the trial. Post length matters less than the pre-period, which dominates the data; the
# exclusion arms run at K4 and cost about what main K4 does.
CELL_COST_S: dict[tuple[str, int], float] = {
    ("hot", 3): 180.0,
    ("hot", 4): 187.0,
    ("hot", 6): 394.0,
    ("pen", 3): 108.0,
    ("pen", 4): 102.0,
    ("pen", 6): 136.0,
    ("kel", 3): 46.0,
    ("kel", 4): 59.0,
    ("kel", 6): 64.0,
    ("hot_t13", 3): 219.0,
    ("hot_t13", 4): 242.0,
    ("hot_t13", 6): 266.0,
    ("hot_tuneup", 3): 361.0,
    ("hot_tuneup", 4): 414.0,
    ("hot_tuneup", 6): 525.0,
}
# The largest peak resident memory of one cell per site, in MB, measured alongside CELL_COST_S.
PEAK_RSS_MB: dict[str, float] = {
    "hot": 7800.0,
    "pen": 4800.0,
    "kel": 3600.0,
    "hot_t13": 8600.0,
    "hot_tuneup": 5600.0,
}


def cell_cost_s(cell: Cell, costs: dict[tuple[str, int], float] = CELL_COST_S) -> float:
    """Return the expected wall seconds of ``cell``."""
    return costs[(cell.site, cell.k)]


def matrix_cells(settings: MatrixSettings) -> list[Cell]:
    """Return every cell, the most expensive first, so no long cell is left to run alone at the end."""
    cells: list[Cell] = []
    for site in settings.real:
        real = REAL_SITES[site]
        cells.extend(
            Cell("real", site, None, None, k, p, n)
            for k in settings.ks
            for p in real_pre_months(site, settings)
            for n in real_post_months(site, settings)
            if _long_enough(
                real.longest_pre_months if p is None else p, real.longest_post_months if n is None else n, settings
            )
        )
    for site, n_seeds in settings.seeds.items():
        for index in range(n_seeds):
            for multiplier, p, n, shift in itertools.product(
                settings.multipliers, settings.pre_months, settings.post_months, settings.season_shifts
            ):
                if not _long_enough(p, n, settings):
                    continue
                for k in settings.ks:
                    cells.append(Cell("main", site, index, multiplier, k, p, n, shift))
                    cells.extend(Cell(arm, site, index, multiplier, k, p, n, shift) for arm in settings.small_arms)  # type: ignore[arg-type]
                cells.extend(
                    Cell(arm, site, index, multiplier, settings.exclusion_k, p, n, shift)  # type: ignore[arg-type]
                    for arm in settings.exclusion_arms
                )
    return sorted(cells, key=lambda c: (-cell_cost_s(c), c.cell_id))


def _long_enough(pre_months: int, post_months: int, settings: MatrixSettings) -> bool:
    """Whether a cell's before and after lengths together reach ``min_total_months``."""
    return pre_months + post_months >= settings.min_total_months


def real_pre_months(site: str, settings: MatrixSettings) -> list[int | None]:
    """Return the pre lengths a real campaign runs: the matrix's that its data holds, then ``None`` for Pmax.

    Pmax is left out when the longest length its data holds is one of the matrix's.
    """
    real = REAL_SITES[site]
    fitting: list[int | None] = [p for p in settings.pre_months if p <= real.longest_pre_months]
    return fitting if real.longest_pre_months in settings.pre_months else [*fitting, None]


def real_post_months(site: str, settings: MatrixSettings) -> list[int | None]:
    """Return the post lengths a real campaign runs: the matrix's that its data holds, then ``None`` for Lmax."""
    real = REAL_SITES[site]
    fitting: list[int | None] = [n for n in settings.post_months if n <= real.longest_post_months]
    return [*fitting, None] if real.lmax else fitting


def campaign_seed(master_seed: int, *, site: str, seed_index: int) -> int:
    """Return the seed a site's campaign is drawn from, a function of these three alone."""
    sequence = np.random.SeedSequence([master_seed, zlib.crc32(site.encode()), seed_index])
    return int(sequence.generate_state(1)[0])


def draw_exclusions(
    turbines: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    seed: int,
) -> list[Exclusion]:
    """Draw 0 to 2 exclusions of 3 to 14 whole days per turbine, anywhere in ``[start, end)``.

    :param turbines: the turbines given exclusions
    :param start: the earliest an exclusion may start
    :param end: the latest an exclusion may end
    :param seed: the campaign's seed; the exclusions draw from their own stream of it
    """
    rng = np.random.default_rng([seed, 1])
    days = int((end - start) / pd.Timedelta(days=1))
    exclusions: list[Exclusion] = []
    for turbine in turbines:
        for _ in range(int(rng.integers(EXCLUSIONS_PER_TURBINE[0], EXCLUSIONS_PER_TURBINE[1] + 1))):
            length = int(rng.integers(EXCLUSION_DAYS[0], EXCLUSION_DAYS[1] + 1))
            first = start + pd.Timedelta(days=int(rng.integers(0, days - length + 1)))
            exclusions.append((turbine, first, first + pd.Timedelta(days=length)))
    return exclusions
