import math

import numpy as np
import pytest

from wind_up.geodesy import distance_and_bearing, geodesic_matrices, local_east_north

# Two Homer turbines, as placed by the v0 ``test_homer_config`` fixture.
HMR_T01 = (-58.60364145072843, 103.6841410289202052)
HMR_T02 = (-58.60261454305449, 103.688364968451364)
HMR_T01_TO_T02_DISTANCE_M = 270.894287973147
HMR_T01_TO_T02_BEARING_DEG = 245.02500888680734 - 180

# Circumference of the earth at the equator is about 40075 km.
HUNDREDTH_DEGREE_AT_EQUATOR_M = 40075 * 1000 / 360 / 100


def _mirror_lat(point: tuple[float, float]) -> tuple[float, float]:
    return (-point[0], point[1])


def _mirror_lon(point: tuple[float, float]) -> tuple[float, float]:
    return (point[0], -point[1])


def test_bearing_cardinal_directions() -> None:
    assert distance_and_bearing((0, 0), (0, 1))[1] == 90
    assert distance_and_bearing((0, 0), (1, 0))[1] == 0
    assert distance_and_bearing((0, 0), (0, -1))[1] == 270


def test_bearing_between_homer_turbines() -> None:
    assert distance_and_bearing(HMR_T01, HMR_T02)[1] == pytest.approx(HMR_T01_TO_T02_BEARING_DEG)
    assert distance_and_bearing(HMR_T02, HMR_T01)[1] == pytest.approx(
        (HMR_T01_TO_T02_BEARING_DEG + 180) % 360, rel=1e-4
    )
    mirrored = distance_and_bearing(_mirror_lat(HMR_T01), _mirror_lat(HMR_T02))[1]
    assert mirrored == pytest.approx((-HMR_T01_TO_T02_BEARING_DEG + 180) % 360)
    mirrored = distance_and_bearing(_mirror_lon(HMR_T01), _mirror_lon(HMR_T02))[1]
    assert mirrored == pytest.approx((-HMR_T01_TO_T02_BEARING_DEG) % 360)


def test_distance_along_the_equator() -> None:
    assert distance_and_bearing((0, 0), (0, 1 / 100))[0] == pytest.approx(HUNDREDTH_DEGREE_AT_EQUATOR_M)
    assert distance_and_bearing((0, 0), (0, -1 / 100))[0] == pytest.approx(HUNDREDTH_DEGREE_AT_EQUATOR_M)
    assert distance_and_bearing((0, 90), (0, 90 + 1 / 100))[0] == pytest.approx(HUNDREDTH_DEGREE_AT_EQUATOR_M)


def test_distance_between_homer_turbines() -> None:
    for a, b in [
        (HMR_T01, HMR_T02),
        (HMR_T02, HMR_T01),
        (_mirror_lat(HMR_T01), _mirror_lat(HMR_T02)),
        (_mirror_lon(HMR_T01), _mirror_lon(HMR_T02)),
    ]:
        assert distance_and_bearing(a, b)[0] == pytest.approx(HMR_T01_TO_T02_DISTANCE_M)


def test_bearing_is_in_zero_to_360() -> None:
    _, bearing = distance_and_bearing((0, 0), (-1, -1e-9))
    assert 0 <= bearing < 360


def test_geodesic_matrices_match_the_pairwise_function() -> None:
    points = [HMR_T01, HMR_T02, (-58.601587635380, 103.692588907983)]
    distance_m, bearing_deg = geodesic_matrices(latitudes=[p[0] for p in points], longitudes=[p[1] for p in points])
    assert distance_m.shape == bearing_deg.shape == (3, 3)
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if i == j:
                assert distance_m[i, j] == 0
                assert math.isnan(bearing_deg[i, j])
            else:
                assert (distance_m[i, j], bearing_deg[i, j]) == distance_and_bearing(a, b)


def test_local_east_north_origin_is_the_minimum_easting_and_northing() -> None:
    east, north = local_east_north(latitudes=[57.50, 57.52, 57.51], longitudes=[-3.00, -2.99, -3.02])
    assert east.min() == 0
    assert north.min() == 0
    assert east.max() > 0
    assert north.max() > 0


def test_local_east_north_places_a_point_due_north() -> None:
    lat2, lon2 = _point_at(origin=(57.5, -3.0), bearing_deg=0, distance_m=1000)
    east, north = local_east_north(latitudes=[57.5, lat2], longitudes=[-3.0, lon2])
    assert east == pytest.approx([0, 0], abs=1e-3)
    assert north == pytest.approx([0, 1000], abs=1e-3)


def test_local_east_north_places_points_on_one_parallel_side_by_side() -> None:
    east, north = local_east_north(latitudes=[57.5, 57.5], longitudes=[-3.0, -2.98])
    separation_m, _ = distance_and_bearing((57.5, -3.0), (57.5, -2.98))
    assert east == pytest.approx([0, separation_m], abs=1e-3)
    assert north == pytest.approx([0, 0], abs=1e-3)


def test_local_east_north_preserves_farm_scale_distances() -> None:
    rng = np.random.default_rng(0)
    lats = 57.5 + rng.uniform(-0.05, 0.05, size=15)
    lons = -3.0 + rng.uniform(-0.08, 0.08, size=15)
    east, north = local_east_north(latitudes=lats, longitudes=lons)
    distance_m, _ = geodesic_matrices(latitudes=lats, longitudes=lons)
    planar = np.hypot(east[:, None] - east[None, :], north[:, None] - north[None, :])
    off_diagonal = ~np.eye(15, dtype=bool)
    assert planar[off_diagonal] == pytest.approx(distance_m[off_diagonal], rel=1e-5)


def _point_at(*, origin: tuple[float, float], bearing_deg: float, distance_m: float) -> tuple[float, float]:
    from geographiclib.geodesic import Geodesic  # noqa: PLC0415

    result = Geodesic.WGS84.Direct(origin[0], origin[1], bearing_deg, distance_m)
    return result["lat2"], result["lon2"]
