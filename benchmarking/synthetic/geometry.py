"""Farm geometry for synthetic wake steering.

Turns turbine coordinates into the directed upstream -> downstream steering pairs a
:class:`~benchmarking.synthetic.upgrades.WakeSteering` upgrade acts on. Pure and
source-agnostic: coordinates are ``(lat, lon)`` in degrees.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    import numpy.typing as npt

_EARTH_RADIUS_M = 6_371_000.0


class WakePair(NamedTuple):
    """A directed steering pair: ``upstream`` wakes ``downstream`` at ``nadir_bearing``.

    :param upstream: steering turbine name
    :param downstream: benefitting turbine name
    :param nadir_bearing: wind-from direction (deg, true north) at which the upstream turbine
        directly wakes the downstream turbine (the wake nadir / line of sight)
    """

    upstream: str
    downstream: str
    nadir_bearing: float


def wrap180(angle_deg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Wrap an angle array in degrees to the half-open range [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def bearing_deg(from_latlon: tuple[float, float], to_latlon: tuple[float, float]) -> float:
    """Return the initial great-circle compass bearing from one ``(lat, lon)`` to another, in [0, 360)."""
    lat1, lon1 = math.radians(from_latlon[0]), math.radians(from_latlon[1])
    lat2, lon2 = math.radians(to_latlon[0]), math.radians(to_latlon[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


def distance_m(a_latlon: tuple[float, float], b_latlon: tuple[float, float]) -> float:
    """Great-circle (haversine) separation between two ``(lat, lon)`` points, in metres."""
    lat1, lon1 = math.radians(a_latlon[0]), math.radians(a_latlon[1])
    lat2, lon2 = math.radians(b_latlon[0]), math.radians(b_latlon[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_M * math.asin(math.sqrt(h))


def derive_wake_steering_pairs(
    coords: Mapping[str, tuple[float, float]],
    *,
    test_wtgs: Sequence[str],
    rotor_diameter_m: float = 82.0,
    max_separation_d: float = 7.0,
) -> list[WakePair]:
    """Directed steering pairs among ``test_wtgs`` within ``max_separation_d`` rotor diameters.

    For each ordered pair ``(A, B)`` of distinct participants closer than the proximity limit,
    yields a :class:`WakePair` with ``A`` upstream, ``B`` downstream and
    ``nadir_bearing = bearing_deg(B, A)`` (the wind-from direction at which A's wake lands on B).
    Turbines absent from ``test_wtgs`` (references) never appear, so they cannot steer or benefit.

    :param coords: ``(lat, lon)`` in degrees for at least every turbine in ``test_wtgs``
    :param test_wtgs: participants that may steer or benefit
    :param rotor_diameter_m: rotor diameter for the proximity limit
    :param max_separation_d: maximum upstream-downstream separation, in rotor diameters
    """
    limit_m = max_separation_d * rotor_diameter_m
    pairs: list[WakePair] = []
    for upstream in test_wtgs:
        for downstream in test_wtgs:
            if upstream == downstream:
                continue
            if distance_m(coords[upstream], coords[downstream]) <= limit_m:
                nadir = bearing_deg(coords[downstream], coords[upstream])
                pairs.append(WakePair(upstream=upstream, downstream=downstream, nadir_bearing=nadir))
    return pairs
