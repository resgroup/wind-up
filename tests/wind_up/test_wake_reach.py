import numpy as np
import pytest

from tests.wind_up.layouts import scatter_layout
from wind_up.layout import Layout, iec_disturbed_sector_deg
from wind_up.wake_reach import (
    MAX_REACH_D,
    candidate_cost,
    kept_wind_share,
    loss_sector,
    reach_from_uplift,
    uniform_rose,
    wind_rose,
)


@pytest.mark.parametrize(
    ("uplift", "reach"),
    [(0.0, 0.0), (0.01, 4.0), (-0.01, 4.0), (0.02, 6.0), (0.05, 12.0), (0.10, 20.0), (0.2, 20.0), (-1.0, 20.0)],
)
def test_reach_from_uplift_interpolates_the_table(uplift: float, reach: float) -> None:
    assert reach_from_uplift(uplift) == pytest.approx(reach)


@pytest.mark.parametrize(("uplift", "reach"), [(0.0, 0.0), (0.01, 8.0), (0.03, 16.0), (0.05, MAX_REACH_D)])
def test_offshore_doubles_the_reach_up_to_the_cap(uplift: float, reach: float) -> None:
    assert reach_from_uplift(uplift, offshore=True) == pytest.approx(reach)


def test_wind_rose_bins_valid_directions_by_degree() -> None:
    rose = wind_rose([0.2, 359.9, 359.5, -0.5, np.nan, 720.4])
    assert rose.shape == (360,)
    assert rose.sum() == pytest.approx(1.0)
    assert rose[0] == pytest.approx(0.4)
    assert rose[359] == pytest.approx(0.6)


def test_wind_rose_refuses_no_valid_direction() -> None:
    with pytest.raises(ValueError, match="no valid direction"):
        wind_rose([np.nan])


def test_uniform_rose() -> None:
    rose = uniform_rose()
    assert rose.shape == (360,)
    assert rose.sum() == pytest.approx(1.0)
    assert np.ptp(rose) == 0


def north_pair(distance_d: float) -> Layout:
    """T0 at the origin and T1 ``distance_d`` rotor diameters due north of it."""
    return Layout.from_frame(scatter_layout([(0.0, 0.0), (0.0, distance_d * 100.0)]))


def test_loss_sector_of_a_waker_due_north_at_five_diameters() -> None:
    sector = loss_sector(north_pair(5.0), waked="T0", waker="T1", reach_d=12.0, rose=uniform_rose())
    assert sector is not None
    assert (sector.waked, sector.waker) == ("T0", "T1")
    assert sector.distance_d == pytest.approx(5.0, abs=1e-6)
    assert sector.centre_deg == pytest.approx(0.0, abs=1e-6)
    assert sector.half_width_deg == pytest.approx(iec_disturbed_sector_deg(5.0) / 2)
    assert sector.half_width_deg == pytest.approx(26.5, abs=0.1)
    assert sector.rose_share == pytest.approx(52 / 360)  # bin centres 0.5 to 25.5 either side of north


def test_no_loss_sector_beyond_the_reach() -> None:
    assert loss_sector(north_pair(5.0), waked="T0", waker="T1", reach_d=4.0, rose=uniform_rose()) is None
    assert loss_sector(north_pair(5.0), waked="T0", waker="T1", reach_d=0.0, rose=uniform_rose()) is None


def test_the_distance_is_in_the_wakers_rotor_diameters() -> None:
    frame = scatter_layout([(0.0, 0.0), (0.0, 500.0)])
    frame.loc[1, "rotor_diameter_m"] = 50.0
    sector = loss_sector(Layout.from_frame(frame), waked="T0", waker="T1", reach_d=12.0, rose=uniform_rose())
    assert sector is not None
    assert sector.distance_d == pytest.approx(10.0, abs=1e-6)


def rose_at(direction_deg: int) -> np.ndarray:
    rose = np.zeros(360)
    rose[direction_deg] = 1.0
    return rose


def test_candidate_cost_is_the_rose_share_with_the_test_turbine_upwind() -> None:
    layout = north_pair(5.0)
    assert candidate_cost(layout, test="T1", candidate="T0", reach_d=12.0, rose=rose_at(0)) == pytest.approx(1.0)
    assert candidate_cost(layout, test="T1", candidate="T0", reach_d=12.0, rose=rose_at(90)) == 0.0
    assert candidate_cost(layout, test="T1", candidate="T0", reach_d=4.0, rose=rose_at(0)) == 0.0


def test_kept_share_is_one_minus_the_union_of_the_sectors() -> None:
    # T1 and T2 are both due north of T0 (the reference): their sectors overlap almost entirely.
    layout = Layout.from_frame(scatter_layout([(0.0, 0.0), (0.0, 500.0), (10.0, 600.0)]))
    share, sectors = kept_wind_share(
        layout, test="T1", references=["T0"], other_tests=["T2"], reach_d=12.0, rose=uniform_rose()
    )
    assert {(s.waked, s.waker) for s in sectors} == {("T0", "T1"), ("T0", "T2"), ("T1", "T2")}
    union = np.zeros(360, dtype=bool)
    for s in sectors:
        union |= np.abs((np.arange(360) + 0.5 - s.centre_deg + 180) % 360 - 180) < s.half_width_deg
    assert share == pytest.approx(1 - union.mean())
    assert share > 1 - sum(s.rose_share for s in sectors)


def test_kept_share_is_one_with_nothing_in_reach() -> None:
    share, sectors = kept_wind_share(
        north_pair(5.0), test="T1", references=["T0"], other_tests=[], reach_d=0.0, rose=uniform_rose()
    )
    assert share == 1.0
    assert sectors == ()
