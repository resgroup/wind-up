"""The prepost campaign matrix's cells: one full ``wind-up`` run each, named and seeded stably.

A cell is one (campaign, multiplier, K, post length). The cell list is a pure function of the
:class:`MatrixSettings`; each campaign draw depends only on ``(master_seed, site, seed index)``, so
a cell's id and its campaign do not depend on which other cells are in the matrix.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from benchmarking.campaigns.declaration import Exclusion

Arm = Literal["main", "excl_booleans", "excl_nan", "real"]

REAL_SITE = "hot_t13"
# The exclusion-channel A/B arms and the power_model setting each runs.
EXCLUSION_ARMS: dict[str, Literal["booleans", "nan"]] = {"excl_booleans": "booleans", "excl_nan": "nan"}
_ARM_SUFFIX = {"main": "", "excl_booleans": "_xbool", "excl_nan": "_xnan", "real": ""}

# Synthetic reference exclusions for the A/B arms: per turbine, this many periods, each this many days.
EXCLUSIONS_PER_TURBINE = (0, 2)
EXCLUSION_DAYS = (3, 14)


@dataclass(frozen=True)
class MatrixSettings:
    """The axes of the matrix and the seed every campaign is drawn from.

    :param seeds: campaigns per synthetic site, keyed ``hot``, ``pen``, ``kel``
    :param multipliers: the AeroUp shape's scalings, paired within a campaign
    :param ks: power references per test turbine
    :param post_months: months of data after the last trial works end (after T13's works for the real campaign)
    :param exclusion_k: the one K the exclusion-channel A/B arms run at
    :param exclusion_arms: which of the exclusion-channel A/B arms run
    :param real: include the real Hill of Towie AeroUp T13 campaign
    :param min_side_days: the shortest side the period selector may choose, so that every post length is analysed
    :param master_seed: every campaign draw is keyed on it
    """

    seeds: dict[str, int] = field(default_factory=lambda: {"hot": 16, "pen": 12, "kel": 4})
    multipliers: tuple[int, ...] = (1, 0, -1)
    ks: tuple[int, ...] = (3, 4, 6)
    post_months: tuple[int, ...] = (1, 2, 3, 6, 9, 12)
    exclusion_k: int = 4
    exclusion_arms: tuple[str, ...] = tuple(EXCLUSION_ARMS)
    real: bool = True
    min_side_days: int = 14
    master_seed: int = 0

    def to_json(self) -> dict[str, object]:
        """Return the settings as plain data, lists in place of tuples."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> MatrixSettings:
        """Return the settings :meth:`to_json` wrote."""
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})  # type: ignore[arg-type]


@dataclass(frozen=True)
class Cell:
    """One run of the matrix.

    :param arm: ``main``; an exclusion-channel A/B arm; or ``real``
    :param site: ``hot``, ``pen``, ``kel`` or :data:`REAL_SITE`
    :param seed_index: which of the site's campaigns; ``None`` for the real campaign
    :param multiplier: the AeroUp scaling; ``None`` for the real campaign
    :param k: power references per test turbine
    :param post_months: months of data after the (last trial) works end
    """

    arm: Arm
    site: str
    seed_index: int | None
    multiplier: int | None
    k: int
    post_months: int

    @property
    def cell_id(self) -> str:
        """A readable, stable id, such as ``pen_s07_m+1_K4_L6`` or ``hot_t13_K4_L6``."""
        if self.arm == "real":
            return f"{self.site}_K{self.k}_L{self.post_months}"
        return (
            f"{self.site}_s{self.seed_index:02d}_m{self.multiplier:+d}_K{self.k}_L{self.post_months}"
            f"{_ARM_SUFFIX[self.arm]}"
        )

    @property
    def exclusion_channels(self) -> Literal["booleans", "nan"] | None:
        """The power_model exclusion channels this cell's arm sets; ``None`` keeps the default."""
        return EXCLUSION_ARMS.get(self.arm)

    @property
    def full_rollout(self) -> bool:
        """Whether this campaign rolls the upgrade out to every other turbine: every even seed index."""
        return self.seed_index is not None and self.seed_index % 2 == 0


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
    # Sized to about 11.5 h of cells on 16 workers; check with ``plan`` before a run.
    "big": StudySize(MatrixSettings(seeds={"hot": 5, "pen": 6, "kel": 4}), workers=16),
}

# Mean wall seconds of one cell per (site, K), measured on the HPC at 57e4aeb. Post length barely
# matters, as the pre-period dominates; the exclusion arms run at K4 and cost about what main K4 does.
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
    (REAL_SITE, 3): 298.0,
    (REAL_SITE, 4): 346.0,
    (REAL_SITE, 6): 458.0,
}
# The largest peak resident memory of one cell per site, in MB, measured alongside CELL_COST_S.
PEAK_RSS_MB: dict[str, float] = {"hot": 7165.0, "pen": 4082.0, "kel": 4178.0, REAL_SITE: 5465.0}


def cell_cost_s(cell: Cell, costs: dict[tuple[str, int], float] = CELL_COST_S) -> float:
    """Return the expected wall seconds of ``cell``."""
    return costs[(cell.site, cell.k)]


def matrix_cells(settings: MatrixSettings) -> list[Cell]:
    """Return every cell, the most expensive first, so no long cell is left to run alone at the end."""
    cells: list[Cell] = []
    if settings.real:
        cells.extend(Cell("real", REAL_SITE, None, None, k, n) for k in settings.ks for n in settings.post_months)
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
