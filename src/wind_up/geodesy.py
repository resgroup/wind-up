"""Distances, bearings and local east/north coordinates on the WGS84 ellipsoid.

Everything here uses the ellipsoidal geodesic (Karney's algorithm, via ``geographiclib``).
Coordinates are ``(latitude, longitude)`` in degrees; bearings are degrees clockwise from true
north in ``[0, 360)``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from geographiclib.geodesic import Geodesic


def distance_and_bearing(origin: tuple[float, float], destination: tuple[float, float]) -> tuple[float, float]:
    """Return the geodesic distance in metres and the initial bearing from ``origin`` to ``destination``."""
    result = Geodesic.WGS84.Inverse(origin[0], origin[1], destination[0], destination[1])
    return result["s12"], result["azi1"] % 360


def geodesic_matrices(
    *, latitudes: npt.ArrayLike, longitudes: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the distance (m) and bearing (deg) between every ordered pair of points.

    Entry ``[i, j]`` is from point ``i`` to point ``j``. The diagonal has distance 0 and bearing NaN.
    """
    lats = np.asarray(latitudes, dtype=float)
    lons = np.asarray(longitudes, dtype=float)
    n = len(lats)
    distance_m = np.zeros((n, n))
    bearing_deg = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_m[i, j], bearing_deg[i, j] = distance_and_bearing((lats[i], lons[i]), (lats[j], lons[j]))
    return distance_m, bearing_deg


def local_east_north(
    *, latitudes: npt.ArrayLike, longitudes: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return east and north coordinates in metres, with the minimum easting and northing at 0.

    Points are placed by their geodesic distance and bearing from the centroid of all points (an
    azimuthal equidistant projection), so north is true north and farm-scale distances are kept to
    within a few parts per million. The maximum easting and northing give the physical extent.
    """
    lats = np.asarray(latitudes, dtype=float)
    lons = np.asarray(longitudes, dtype=float)
    centroid = (float(lats.mean()), float(lons.mean()))
    east = np.empty(len(lats))
    north = np.empty(len(lats))
    for i, point in enumerate(zip(lats, lons, strict=True)):
        distance_m, bearing_deg = distance_and_bearing(centroid, point)
        east[i] = distance_m * np.sin(np.radians(bearing_deg))
        north[i] = distance_m * np.cos(np.radians(bearing_deg))
    return east - east.min(), north - north.min()
