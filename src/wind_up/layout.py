"""A wind farm layout and its IEC 61400-12-1 wake geometry.

:class:`Layout` validates a table of turbines and holds the geodesic distance and bearing between
every pair. :func:`front_row` classifies turbines by how wide an arc of wind directions reaches
them with no upwind turbine in the way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from wind_up.circular_math import circ_diff
from wind_up.geodesy import geodesic_matrices

NAME_COL = "name"
LATITUDE_COL = "latitude"
LONGITUDE_COL = "longitude"
ROTOR_DIAMETER_COL = "rotor_diameter_m"
WIND_FARM_COL = "wind_farm"

# Wind directions swept for front-row classification, one per degree.
SWEEP_DIRECTIONS_DEG = np.arange(360.0)


def iec_disturbed_sector_deg(distance_diameters: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """IEC 61400-12-1 disturbed-sector full width (deg) vs upwind separation in rotor diameters.

    Below 2 diameters the whole 180 deg upwind is disturbed; from 2 to 20 diameters the sector is
    ``1.3 * atan(2.5 / Dn + 0.15) + 10``; beyond 20 diameters the wake has dissipated (sector 0).
    """
    dn = np.asarray(distance_diameters, dtype=float)
    sector = np.full(dn.shape, 180.0)
    ge_2 = dn >= 2  # noqa: PLR2004
    sector[ge_2] = 1.3 * np.rad2deg(np.arctan(2.5 / dn[ge_2] + 0.15)) + 10
    sector[dn > 20] = 0.0  # noqa: PLR2004
    return sector


@dataclass(frozen=True)
class Layout:
    """A validated turbine table plus the geodesic distance and bearing between every pair.

    ``frame`` has one row per turbine, positionally indexed, with columns ``name`` (``None`` when
    unknown), ``latitude``, ``longitude``, ``rotor_diameter_m`` and ``wind_farm`` (``None`` when
    unknown). ``distance_m[i, j]`` and ``bearing_deg[i, j]`` are from row ``i`` to row ``j``.
    """

    frame: pd.DataFrame
    filled_rotor_diameters: tuple[str, ...]
    distance_m: npt.NDArray[np.float64]
    bearing_deg: npt.NDArray[np.float64]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> Layout:
        """Validate ``frame`` and compute its geometry.

        Column names are matched case-insensitively. ``latitude`` and ``longitude`` are required;
        ``name`` and ``wind_farm`` are optional. ``rotor_diameter_m`` needs at least one value;
        missing ones take the largest known, and the filled turbines are listed in
        ``filled_rotor_diameters``.
        """
        lookup = {str(c).strip().lower(): c for c in frame.columns}
        missing = [c for c in (LATITUDE_COL, LONGITUDE_COL) if c not in lookup]
        if missing:
            msg = f"the layout has no {missing} column(s); it carries {list(frame.columns)}"
            raise ValueError(msg)

        def column(name: str) -> pd.Series:
            if name in lookup:
                return frame[lookup[name]].reset_index(drop=True)
            return pd.Series([None] * len(frame), dtype=object)

        names = [_text_or_none(v) for v in column(NAME_COL)]
        named = [n for n in names if n is not None]
        duplicates = sorted({n for n in named if named.count(n) > 1})
        if duplicates:
            msg = f"the layout names {duplicates} more than once"
            raise ValueError(msg)

        diameters = pd.to_numeric(column(ROTOR_DIAMETER_COL), errors="coerce").astype(float)
        if diameters.isna().all():
            msg = "the layout has no rotor diameter for any turbine; at least one is needed"
            raise ValueError(msg)
        unknown = diameters.isna()
        filled = tuple(str(n) if n is not None else f"row {i}" for i, n in enumerate(names) if unknown.iloc[i])
        diameters = diameters.fillna(diameters.max())

        tidy = pd.DataFrame(
            {
                NAME_COL: pd.Series(names, dtype=object),
                LATITUDE_COL: column(LATITUDE_COL).astype(float),
                LONGITUDE_COL: column(LONGITUDE_COL).astype(float),
                ROTOR_DIAMETER_COL: diameters,
                WIND_FARM_COL: pd.Series([_text_or_none(v) for v in column(WIND_FARM_COL)], dtype=object),
            }
        )
        distance_m, bearing_deg = geodesic_matrices(latitudes=tidy[LATITUDE_COL], longitudes=tidy[LONGITUDE_COL])
        return cls(frame=tidy, filled_rotor_diameters=filled, distance_m=distance_m, bearing_deg=bearing_deg)

    def index_of(self, name: str) -> int:
        """Return the row of the turbine called ``name``."""
        matches = np.flatnonzero(self.frame[NAME_COL].to_numpy() == name)
        if len(matches) == 0:
            msg = f"the layout has no turbine called {name!r}"
            raise ValueError(msg)
        return int(matches[0])


def upwind_mask(layout: Layout, *, target: int, wind_direction_deg: float) -> npt.NDArray[np.bool_]:
    """Return which turbines are upwind of row ``target`` at ``wind_direction_deg``.

    A turbine is upwind when the bearing to it is within half its IEC disturbed sector of the wind
    direction, the sector sized by its separation in its own rotor diameters. The target itself is
    never upwind of itself.
    """
    return _upwind(layout, target=target, directions_deg=np.array([wind_direction_deg]))[0]


def longest_clear_run_deg(clear: npt.NDArray[np.bool_]) -> int:
    """Return the longest run of ``True`` in a one-per-degree compass array, wrapping through north."""
    if clear.all():
        return len(clear)
    if not clear.any():
        return 0
    start = int(np.flatnonzero(~clear)[0])
    rotated = np.roll(clear, -start)
    longest = current = 0
    for is_clear in rotated:
        current = current + 1 if is_clear else 0
        longest = max(longest, current)
    return longest


def front_row(layout: Layout, *, min_clear_deg: float = 90.0) -> npt.NDArray[np.bool_]:
    """Return which turbines are front row.

    A turbine is front row when at least ``min_clear_deg`` contiguous degrees of wind direction reach
    it with no other layout turbine upwind. Every turbine in the layout counts as a blocker.
    """
    result = np.zeros(len(layout.frame), dtype=bool)
    for target in range(len(layout.frame)):
        clear = np.asarray(~_upwind(layout, target=target, directions_deg=SWEEP_DIRECTIONS_DEG).any(axis=1))
        result[target] = longest_clear_run_deg(clear) >= min_clear_deg
    return result


def _upwind(layout: Layout, *, target: int, directions_deg: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Return an upwind mask of shape ``(directions, turbines)`` for row ``target``."""
    diameters = layout.frame[ROTOR_DIAMETER_COL].to_numpy()
    sector = iec_disturbed_sector_deg(layout.distance_m[target] / diameters)
    bearing = layout.bearing_deg[target]
    relative = np.abs(circ_diff(bearing[None, :], directions_deg[:, None]))
    upwind = relative < sector[None, :] / 2
    upwind[:, target] = False
    return upwind


def _text_or_none(value: object) -> str | None:
    """Return ``value`` as text, or ``None`` when it is missing or blank."""
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
