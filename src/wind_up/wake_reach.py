"""How far a changed turbine's wake change reaches, and the share of the wind it costs a measurement.

:func:`reach_from_uplift` converts an expected AEP uplift into a wake effect reach in rotor diameters. A
rose is the weight of each 1 degree bin ``[i, i + 1)`` of wind direction (where the wind comes from,
degrees from true north), summing to 1. A :class:`LossSector` is the IEC disturbed sector, as step 5
sizes it, in which a changed turbine is upwind of a turbine whose records are then lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from wind_up.circular_math import circ_diff
from wind_up.layout import ROTOR_DIAMETER_COL, Layout, iec_disturbed_sector_deg

if TYPE_CHECKING:
    from collections.abc import Sequence

Rose = npt.NDArray[np.float64]

REACH_TABLE_D: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.01, 4.0), (0.03, 8.0), (0.05, 12.0), (0.10, 20.0))
MAX_REACH_D = 20.0
OFFSHORE_FACTOR = 2.0
# How many references each test turbine needs.
REFERENCES_PER_TEST_TURBINE = 3
# A candidate reference the test turbine is upwind of for more than this share of the rose is demoted.
DEFAULT_DEMOTION_SHARE = 0.20

_BIN_CENTRES = np.arange(360) + 0.5


def reach_from_uplift(expected_uplift: float, *, offshore: bool = False) -> float:
    """Return the wake effect reach, in rotor diameters, for an expected AEP uplift (a fraction, either sign).

    Interpolates :data:`REACH_TABLE_D` on the uplift's magnitude, :data:`MAX_REACH_D` beyond its end. Offshore
    the reach is doubled, up to :data:`MAX_REACH_D`.
    """
    uplifts, reaches = zip(*REACH_TABLE_D, strict=True)
    reach = float(np.interp(abs(expected_uplift), uplifts, reaches))
    if offshore:
        reach = min(reach * OFFSHORE_FACTOR, MAX_REACH_D)
    return reach


def wind_rose(directions_deg: npt.ArrayLike) -> Rose:
    """Return the rose of ``directions_deg``, ignoring values that are not finite."""
    directions = np.asarray(directions_deg, dtype=float).ravel()
    directions = directions[np.isfinite(directions)]
    if len(directions) == 0:
        msg = "a wind rose needs at least one valid direction; there is no valid direction"
        raise ValueError(msg)
    bins = np.floor(np.mod(directions, 360.0)).astype(int) % 360
    counts = np.bincount(bins, minlength=360).astype(float)
    return counts / counts.sum()


def uniform_rose() -> Rose:
    """Return the rose with every direction equally likely."""
    return np.full(360, 1 / 360)


@dataclass(frozen=True)
class LossSector:
    """The wind directions in which ``waker``, a changed turbine, is upwind of ``waked``.

    :param waked: the turbine whose records are lost (a subject or a reference)
    :param waker: the changed turbine upwind of it
    :param distance_d: waker to waked, in the waker's rotor diameters
    :param centre_deg: the bearing from waked to waker, the wind direction that puts the waker upwind
    :param half_width_deg: half the IEC disturbed sector at ``distance_d``
    :param rose_share: the share of the rose inside the sector
    """

    waked: str
    waker: str
    distance_d: float
    centre_deg: float
    half_width_deg: float
    rose_share: float

    def bins(self) -> npt.NDArray[np.bool_]:
        """Return which 1 degree bins of the rose fall inside the sector, by their centres."""
        return np.asarray(np.abs(circ_diff(_BIN_CENTRES, self.centre_deg)) < self.half_width_deg)


def loss_sector(layout: Layout, *, waked: str, waker: str, reach_d: float, rose: Rose) -> LossSector | None:
    """Return the sector in which ``waker`` is upwind of ``waked``, or ``None`` beyond ``reach_d``."""
    i, j = layout.index_of(waked), layout.index_of(waker)
    distance_d = float(layout.distance_m[i, j] / layout.frame[ROTOR_DIAMETER_COL].iloc[j])
    half_width = float(iec_disturbed_sector_deg(distance_d)) / 2
    if distance_d > reach_d or half_width == 0:
        return None
    centre = float(layout.bearing_deg[i, j]) % 360.0
    in_sector = np.abs(circ_diff(_BIN_CENTRES, centre)) < half_width
    return LossSector(
        waked=waked,
        waker=waker,
        distance_d=distance_d,
        centre_deg=centre,
        half_width_deg=half_width,
        rose_share=float(rose[in_sector].sum()),
    )


def candidate_cost(layout: Layout, *, test: str, candidate: str, reach_d: float, rose: Rose) -> float:
    """Return the rose share in which ``test`` is upwind of ``candidate`` within ``reach_d``, else 0."""
    sector = loss_sector(layout, waked=candidate, waker=test, reach_d=reach_d, rose=rose)
    return 0.0 if sector is None else sector.rose_share


def loss_sectors(
    layout: Layout,
    *,
    test: str,
    references: Sequence[str],
    other_tests: Sequence[str],
    reach_d: float,
    rose: Rose,
) -> tuple[LossSector, ...]:
    """Return every sector that costs ``test`` records: a changed turbine upwind of it or of a reference.

    The changed turbines are ``test`` itself and ``other_tests``.
    """
    pairs = [(ref, test) for ref in references]
    for other in other_tests:
        pairs += [(test, other)] + [(ref, other) for ref in references if ref != other]
    found = (loss_sector(layout, waked=a, waker=b, reach_d=reach_d, rose=rose) for a, b in pairs)
    return tuple(s for s in found if s is not None)


def union_share(sectors: Sequence[LossSector], *, rose: Rose) -> float:
    """Return the rose share inside any of ``sectors``."""
    lost = np.zeros(360, dtype=bool)
    for sector in sectors:
        lost |= sector.bins()
    return float(rose[lost].sum())


def kept_wind_share(
    layout: Layout,
    *,
    test: str,
    references: Sequence[str],
    other_tests: Sequence[str],
    reach_d: float,
    rose: Rose,
) -> tuple[float, tuple[LossSector, ...]]:
    """Return the rose share ``test`` keeps outside every loss sector of :func:`loss_sectors`, and the sectors."""
    sectors = loss_sectors(
        layout, test=test, references=references, other_tests=other_tests, reach_d=reach_d, rose=rose
    )
    return 1.0 - union_share(sectors, rose=rose), sectors
