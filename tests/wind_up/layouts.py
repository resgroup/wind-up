"""Small synthetic layouts for the layout and campaign-design tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from geographiclib.geodesic import Geodesic

if TYPE_CHECKING:
    from collections.abc import Sequence

ORIGIN = (57.5, -3.0)


def offset(*, east_m: float, north_m: float, origin: tuple[float, float] = ORIGIN) -> tuple[float, float]:
    """Return the point ``north_m`` north then ``east_m`` east of ``origin``, as ``(lat, lon)``."""
    north = Geodesic.WGS84.Direct(origin[0], origin[1], 0, north_m)
    point = Geodesic.WGS84.Direct(north["lat2"], north["lon2"], 90, east_m)
    return point["lat2"], point["lon2"]


def grid_layout(*, rows: int, cols: int, spacing_m: float, rotor_diameter_m: float = 100.0) -> pd.DataFrame:
    """A rows x cols grid of turbines ``R{row}C{col}``, ``spacing_m`` apart."""
    records = []
    for r in range(rows):
        for c in range(cols):
            lat, lon = offset(east_m=c * spacing_m, north_m=r * spacing_m)
            records.append(
                {"name": f"R{r}C{c}", "latitude": lat, "longitude": lon, "rotor_diameter_m": rotor_diameter_m}
            )
    return pd.DataFrame(records)


def line_layout(positions_m: Sequence[float], *, rotor_diameter_m: float = 100.0) -> pd.DataFrame:
    """Turbines ``T0``, ``T1``, ... on an east-west line at ``positions_m`` metres east of the origin."""
    records = []
    for i, east in enumerate(positions_m):
        lat, lon = offset(east_m=east, north_m=0)
        records.append({"name": f"T{i}", "latitude": lat, "longitude": lon, "rotor_diameter_m": rotor_diameter_m})
    return pd.DataFrame(records)


def scatter_layout(points_m: Sequence[tuple[float, float]], *, rotor_diameter_m: float = 100.0) -> pd.DataFrame:
    """Turbines ``T0``, ``T1``, ... at ``(east_m, north_m)`` offsets from the origin."""
    records = []
    for i, (east, north) in enumerate(points_m):
        lat, lon = offset(east_m=east, north_m=north)
        records.append({"name": f"T{i}", "latitude": lat, "longitude": lon, "rotor_diameter_m": rotor_diameter_m})
    return pd.DataFrame(records)
