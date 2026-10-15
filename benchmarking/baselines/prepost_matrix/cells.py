"""The prepost campaign matrix's cells: one full ``wind-up`` run each, named and seeded stably.

A cell is one (campaign, multiplier, K, post length). The cell list is a pure function of the
:class:`MatrixSettings`. A synthetic site's campaigns are its tiles, seed index ``i`` being tile ``i``
(see :func:`benchmarking.campaigns.rollout.draw_tiles`); tile ``i`` depends only on ``master_seed``,
the site and the seed indices up to ``i``, so a cell's id and its campaign do not depend on how many
seeds the matrix runs. A real campaign runs
the post lengths its data holds, and Lmax, all of it, where :data:`REAL_SITES` says so.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from benchmarking.campaigns.declaration import Exclusion

Arm = Literal["main", "excl", "real"]


@dataclass(frozen=True)
class RealSite:
    """A real campaign of the matrix.

    :param campaign: its name in :data:`benchmarking.campaigns.real.REAL_CAMPAIGNS`
    :param longest_post_months: the most whole months its data holds after the changeover
    :param lmax: also run one cell on all the data after the changeover
    """

    campaign: str
    longest_post_months: int
    lmax: bool = False


REAL_SITES: dict[str, RealSite] = {
    "hot_t13": RealSite("hot_aeroup_t13", longest_post_months=15),
    "hot_tuneup": RealSite("hot_tuneup", longest_post_months=5, lmax=True),
}
# The arms whose references are given synthetic exclusions.
EXCLUSION_ARMS: tuple[str, ...] = ("excl",)
_ARM_SUFFIX = {"main": "", "excl": "_excl", "real": ""}

# Synthetic reference exclusions for the exclusion arms: per turbine, this many periods, each this many days.
EXCLUSIONS_PER_TURBINE = (0, 2)
EXCLUSION_DAYS = (3, 14)


@dataclass(frozen=True)
class MatrixSettings:
    """The axes of the matrix and the seed every campaign is drawn from.

    :param seeds: campaigns (tiles) per synthetic site, keyed ``hot``, ``pen``, ``kel``
    :param multipliers: the AeroUp shape's scalings, paired within a campaign
    :param ks: power references per test turbine
    :param post_months: months of data after the last trial works end (after the changeover for a real campaign)
    :param exclusion_k: the one K the exclusion arms run at
    :param exclusion_arms: which of the exclusion arms run
    :param real: the real campaigns included, keys of :data:`REAL_SITES`
    :param min_side_days: the shortest side the period selector may choose, so that every post length is analysed
    :param master_seed: every campaign draw is keyed on it
    """

    seeds: dict[str, int] = field(default_factory=lambda: {"hot": 16, "pen": 12, "kel": 4})
    multipliers: tuple[int, ...] = (1, 0, -1)
    ks: tuple[int, ...] = (3, 4, 6)
    post_months: tuple[int, ...] = (1, 2, 3, 6, 9, 12)
    exclusion_k: int = 4
    exclusion_arms: tuple[str, ...] = EXCLUSION_ARMS
    real: tuple[str, ...] = tuple(REAL_SITES)
    min_side_days: int = 14
    master_seed: int = 0

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

    :param arm: ``main``; ``excl``, whose references have synthetic exclusions; or ``real``
    :param site: ``hot``, ``pen``, ``kel`` or a key of :data:`REAL_SITES`
    :param seed_index: which of the site's campaigns; ``None`` for a real campaign
    :param multiplier: the AeroUp scaling; ``None`` for a real campaign
    :param k: power references per test turbine
    :param post_months: months of data after the (last trial) works end; ``None`` for all of a real
        campaign's data, Lmax
    """

    arm: Arm
    site: str
    seed_index: int | None
    multiplier: int | None
    k: int
    post_months: int | None

    @property
    def cell_id(self) -> str:
        """A readable, stable id, such as ``pen_s07_m+1_K4_L6``, ``hot_t13_K4_L6`` or ``hot_tuneup_K4_Lmax``."""
        if self.arm == "real":
            return f"{self.site}_K{self.k}_L{'max' if self.post_months is None else self.post_months}"
        return (
            f"{self.site}_s{self.seed_index:02d}_m{self.multiplier:+d}_K{self.k}_L{self.post_months}"
            f"{_ARM_SUFFIX[self.arm]}"
        )

    @property
    def has_exclusions(self) -> bool:
        """Whether this cell's references are given synthetic exclusions."""
        return self.arm in EXCLUSION_ARMS

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
        MatrixSettings(seeds={"hot": 0, "pen": 1, "kel": 1}, ks=(4,), post_months=(3, 12)),
        workers=3,
    ),
    # About 9.4 h of cells on 16 workers; check with ``plan`` before a run.
    "big": StudySize(MatrixSettings(seeds={"hot": 5, "pen": 6, "kel": 4}), workers=16),
}

# Mean wall seconds of one cell per (site, K), measured on the HPC at 57e4aeb; hot_tuneup K4 on a laptop,
# K3 and K6 scaled from it as hot_t13's. Post length barely matters, as the pre-period dominates; the
# exclusion arms run at K4 and cost about what main K4 does.
CELL_COST_S: dict[tuple[str, int], float] = {
    ("hot", 3): 911.0,
    ("hot", 4): 1085.0,
    ("hot", 6): 1616.0,
    ("pen", 3): 154.0,
    ("pen", 4): 169.0,
    ("pen", 6): 287.0,
    ("kel", 3): 83.0,
    ("kel", 4): 91.0,
    ("kel", 6): 92.0,
    ("hot_t13", 3): 298.0,
    ("hot_t13", 4): 346.0,
    ("hot_t13", 6): 458.0,
    ("hot_tuneup", 3): 990.0,
    ("hot_tuneup", 4): 1150.0,
    ("hot_tuneup", 6): 1520.0,
}
# The largest peak resident memory of one cell per site, in MB, measured alongside CELL_COST_S.
PEAK_RSS_MB: dict[str, float] = {
    "hot": 7165.0,
    "pen": 4082.0,
    "kel": 4178.0,
    "hot_t13": 5465.0,
    "hot_tuneup": 7127.0,
}


def cell_cost_s(cell: Cell, costs: dict[tuple[str, int], float] = CELL_COST_S) -> float:
    """Return the expected wall seconds of ``cell``."""
    return costs[(cell.site, cell.k)]


def matrix_cells(settings: MatrixSettings) -> list[Cell]:
    """Return every cell, the most expensive first, so no long cell is left to run alone at the end."""
    cells: list[Cell] = []
    for site in settings.real:
        cells.extend(
            Cell("real", site, None, None, k, n) for k in settings.ks for n in real_post_months(site, settings)
        )
    for site, n_seeds in settings.seeds.items():
        for index in range(n_seeds):
            for multiplier in settings.multipliers:
                for n in settings.post_months:
                    cells.extend(Cell("main", site, index, multiplier, k, n) for k in settings.ks)
                    cells.extend(
                        Cell(arm, site, index, multiplier, settings.exclusion_k, n)  # type: ignore[arg-type]
                        for arm in settings.exclusion_arms
                    )
    return sorted(cells, key=lambda c: (-cell_cost_s(c), c.cell_id))


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
