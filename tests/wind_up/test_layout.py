import math

import numpy as np
import pandas as pd
import pytest

from tests.wind_up.layouts import grid_layout
from wind_up.layout import Layout, front_row, iec_disturbed_sector_deg, longest_clear_run_deg, upwind_mask

# The three Homer turbines of the v0 ``test_homer_with_t00_config`` fixture, 62 m rotors.
HOMER = pd.DataFrame(
    {
        "name": ["HMR_T01", "HMR_T02", "HMR_T00"],
        "latitude": [-58.60364145072843, -58.60261454305449, -58.601587635380],
        "longitude": [103.6841410289202052, 103.688364968451364, 103.692588907983],
        "rotor_diameter_m": [62.0, 62.0, 62.0],
    }
)


def test_disturbed_sector_is_half_the_compass_below_two_diameters() -> None:
    assert iec_disturbed_sector_deg(1.9) == 180


def test_disturbed_sector_follows_the_iec_formula_between_two_and_twenty_diameters() -> None:
    for dn in (2.0, 5.0, 20.0):
        assert iec_disturbed_sector_deg(dn) == pytest.approx(1.3 * math.degrees(math.atan(2.5 / dn + 0.15)) + 10)


def test_disturbed_sector_is_zero_beyond_twenty_diameters() -> None:
    assert iec_disturbed_sector_deg(20.1) == 0


@pytest.mark.parametrize(
    ("target", "wind_direction", "expected"),
    [
        ("HMR_T01", 250 - 180, ["HMR_T00", "HMR_T02"]),
        ("HMR_T02", 250 - 180, ["HMR_T00"]),
        ("HMR_T01", 250, []),
        ("HMR_T02", 250, ["HMR_T01"]),
        ("HMR_T01", 250 - 180 - 90, []),
        ("HMR_T02", 250 - 180 - 90, []),
    ],
)
def test_upwind_turbines_match_v0(target: str, wind_direction: float, expected: list[str]) -> None:
    layout = Layout.from_frame(HOMER)
    mask = upwind_mask(layout, target=layout.index_of(target), wind_direction_deg=wind_direction)
    assert sorted(layout.frame["name"][mask]) == expected


def test_longest_clear_run_all_clear() -> None:
    assert longest_clear_run_deg(np.ones(360, dtype=bool)) == 360


def test_longest_clear_run_all_blocked() -> None:
    assert longest_clear_run_deg(np.zeros(360, dtype=bool)) == 0


def test_longest_clear_run_wraps_through_north() -> None:
    clear = np.zeros(360, dtype=bool)
    clear[350:] = True
    clear[:30] = True
    clear[100:140] = True
    assert longest_clear_run_deg(clear) == 40


def test_front_row_threshold_is_inclusive() -> None:
    clear = np.zeros(360, dtype=bool)
    clear[:90] = True
    assert longest_clear_run_deg(clear) >= 90
    clear[89] = False
    assert longest_clear_run_deg(clear) == 89


def test_front_row_in_a_grid_is_everything_but_the_middle() -> None:
    layout = Layout.from_frame(grid_layout(rows=3, cols=3, spacing_m=300))
    assert dict(zip(layout.frame["name"], front_row(layout), strict=True)) == {
        f"R{r}C{c}": (r, c) != (1, 1) for r in range(3) for c in range(3)
    }


def test_front_row_is_everyone_when_turbines_are_far_apart() -> None:
    layout = Layout.from_frame(grid_layout(rows=2, cols=2, spacing_m=2500))
    assert front_row(layout).all()


def test_layout_columns_are_matched_case_insensitively() -> None:
    frame = HOMER.rename(columns={"name": "Name", "latitude": "Latitude", "longitude": "LONGITUDE"})
    assert list(Layout.from_frame(frame).frame["name"]) == ["HMR_T01", "HMR_T02", "HMR_T00"]


def test_missing_rotor_diameters_take_the_largest_known() -> None:
    frame = HOMER.assign(rotor_diameter_m=[62.0, np.nan, 90.0])
    layout = Layout.from_frame(frame)
    assert list(layout.frame["rotor_diameter_m"]) == [62.0, 90.0, 90.0]
    assert layout.filled_rotor_diameters == ("HMR_T02",)


def test_layout_needs_at_least_one_rotor_diameter() -> None:
    with pytest.raises(ValueError, match="rotor diameter"):
        Layout.from_frame(HOMER.drop(columns="rotor_diameter_m"))


def test_layout_needs_latitude_and_longitude() -> None:
    with pytest.raises(ValueError, match="latitude"):
        Layout.from_frame(HOMER.drop(columns="latitude"))


def test_layout_names_must_be_unique() -> None:
    with pytest.raises(ValueError, match="HMR_T01"):
        Layout.from_frame(HOMER.assign(name=["HMR_T01", "HMR_T01", "HMR_T00"]))


def test_unnamed_turbines_are_allowed() -> None:
    layout = Layout.from_frame(HOMER.assign(name=["HMR_T01", None, np.nan]))
    assert list(layout.frame["name"]) == ["HMR_T01", None, None]
    assert layout.filled_rotor_diameters == ()


def test_nullable_missing_names_are_unnamed() -> None:
    names = pd.Series(["HMR_T01", pd.NA, pd.NA], dtype="string")
    layout = Layout.from_frame(HOMER.assign(name=names))
    assert list(layout.frame["name"]) == ["HMR_T01", None, None]


def test_wind_farm_is_optional() -> None:
    layout = Layout.from_frame(HOMER)
    assert list(layout.frame["wind_farm"]) == [None, None, None]
    layout = Layout.from_frame(HOMER.assign(wind_farm=["Homer", "Homer", np.nan]))
    assert list(layout.frame["wind_farm"]) == ["Homer", "Homer", None]


def test_layout_carries_the_geodesic_matrices() -> None:
    layout = Layout.from_frame(HOMER)
    assert layout.distance_m.shape == (3, 3)
    assert layout.distance_m[0, 1] == pytest.approx(270.894287973147)
    assert layout.bearing_deg[0, 1] == pytest.approx(245.02500888680734 - 180)
