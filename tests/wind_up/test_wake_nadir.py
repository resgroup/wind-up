"""Tests for the wake-nadir shift."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from wind_up.circular_math import circ_diff
from wind_up.layout import Layout
from wind_up.northing import add_wake_nadir_shift, apply_north_table, north_farm
from wind_up.wake_nadir import _aggregate, _own_corrections, wake_nadir_offsets, wake_pair_curves, wake_pairs


def _index(rows: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=rows, freq="600s", tz="UTC")


def _pair_layout() -> Layout:
    """Two turbines a few diameters apart: A upstream (west) wakes B downstream (east)."""
    # ~410 m apart along a parallel, so A lies roughly due west of B and B->A bearing is near 270.
    frame = pd.DataFrame(
        {
            "name": ["A", "B"],
            "latitude": [55.0, 55.0],
            "longitude": [-0.0064, 0.0],
            "rotor_diameter_m": [82.0, 82.0],
        }
    )
    return Layout.from_frame(frame)


def _waked_pair(
    layout: Layout, *, residual_deg: float, rows: int = 8000, seed: int = 0
) -> tuple[
    pd.DatetimeIndex, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], float
]:
    """Fabricate a clean wake on B whose apparent nadir sits ``residual_deg`` off geometry.

    A's *true* direction sweeps the sector around the geometric nadir; B's power and wind speed reach
    their nadir when the true wind is there. A's *northed* direction carries a constant ``residual_deg``
    error, so the nadir appears in A's northed direction at the bearing + residual.
    """
    b_idx, a_idx = layout.index_of("B"), layout.index_of("A")
    beta = float(layout.bearing_deg[b_idx, a_idx])
    rng = np.random.default_rng(seed)
    true_a = beta + rng.uniform(-25.0, 25.0, size=rows)
    deficit = np.exp(-0.5 * (circ_diff(true_a, beta) / 3.0) ** 2)
    index = _index(rows)
    northed = {
        "A": (true_a + residual_deg) % 360.0,
        "B": (true_a + residual_deg) % 360.0,
    }
    power = {
        "A": np.full(rows, 1000.0),
        "B": 1000.0 * (1.0 - 0.4 * deficit),
    }
    wind_speed = {
        "A": np.full(rows, 9.0),
        "B": 9.0 * (1.0 - 0.15 * deficit),
    }
    usable = {"A": np.ones(rows, dtype=bool), "B": np.ones(rows, dtype=bool)}
    return index, northed, power, wind_speed, usable, beta


def test_recovers_a_known_residual_on_the_waking_turbine() -> None:
    layout = _pair_layout()
    residual = 6.0
    index, northed, power, wind_speed, usable, _ = _waked_pair(layout, residual_deg=residual)

    offsets = wake_nadir_offsets(
        layout, index=index, northed_direction=northed, power=power, wind_speed=wind_speed, usable=usable
    )

    # Adding the correction to A's north offset must cancel the injected residual.
    assert circ_diff(offsets["A"], -residual) == pytest.approx(0.0, abs=1.5)


def test_power_only_fallback_recovers_the_residual() -> None:
    """With no nacelle wind speed the power ratio alone still locates the nadir."""
    layout = _pair_layout()
    residual = -5.0
    index, northed, power, _, usable, _ = _waked_pair(layout, residual_deg=residual)

    offsets = wake_nadir_offsets(layout, index=index, northed_direction=northed, power=power, usable=usable)

    assert circ_diff(offsets["A"], -residual) == pytest.approx(0.0, abs=1.5)


def test_a_missing_wind_speed_row_does_not_discard_the_wind_speed_signal() -> None:
    """Real nacelle wind speed has gaps; one NaN row must drop that row, not the whole signal.

    Power is flat here, so only the wind-speed curve carries the nadir.
    """
    layout = _pair_layout()
    residual = 6.0
    index, northed, _, wind_speed, usable, _ = _waked_pair(layout, residual_deg=residual)
    flat_power = {"A": np.full(len(index), 1000.0), "B": np.full(len(index), 1000.0)}
    gappy = {name: ws.copy() for name, ws in wind_speed.items()}
    gappy["B"][0] = np.nan

    offsets = wake_nadir_offsets(
        layout, index=index, northed_direction=northed, power=flat_power, wind_speed=gappy, usable=usable
    )

    assert circ_diff(offsets["A"], -residual) == pytest.approx(0.0, abs=1.5)


def test_a_flat_deficit_leaves_the_turbine_uncorrected() -> None:
    """No nadir anywhere means no turbine resolves, so every correction is zero (graceful)."""
    layout = _pair_layout()
    index, northed, _, _, usable, _ = _waked_pair(layout, residual_deg=6.0)
    flat_power = {"A": np.full(len(index), 1000.0), "B": np.full(len(index), 1000.0)}

    offsets = wake_nadir_offsets(layout, index=index, northed_direction=northed, power=flat_power, usable=usable)

    assert offsets == {"A": 0.0, "B": 0.0}


def test_turbines_beyond_the_cutoff_get_no_correction() -> None:
    """A wake pair further apart than the cutoff yields no geometry, so nothing is corrected."""
    frame = pd.DataFrame(
        {"name": ["A", "B"], "latitude": [55.0, 55.0], "longitude": [-1.0, 0.0], "rotor_diameter_m": [82.0, 82.0]}
    )
    far_layout = Layout.from_frame(frame)
    index, northed, power, wind_speed, usable, _ = _waked_pair(_pair_layout(), residual_deg=6.0)

    offsets = wake_nadir_offsets(
        far_layout, index=index, northed_direction=northed, power=power, wind_speed=wind_speed, usable=usable
    )

    assert offsets == {"A": 0.0, "B": 0.0}


def test_north_farm_applies_the_wake_nadir_shift_when_given_layout_and_power() -> None:
    """north_farm runs wake-nadir-shift when a layout and power are supplied, shifting each device's table."""
    layout = _pair_layout()
    residual = 6.0
    index, northed, power, wind_speed, usable, _ = _waked_pair(layout, residual_deg=residual)
    reanalysis = northed["A"]  # reanalysis matches the reported signal, so the anchor contributes ~zero offset

    without = north_farm(
        index, direction_deg=northed, usable=usable, reanalysis_deg=reanalysis, layout=layout, power=None
    )
    with_p4 = north_farm(
        index,
        direction_deg=northed,
        usable=usable,
        reanalysis_deg=reanalysis,
        layout=layout,
        power=power,
        wind_speed=wind_speed,
    )

    pre_shift = {name: apply_north_table(index, northed[name], north_table=without[name]) for name in northed}
    expected = wake_nadir_offsets(
        layout, index=index, northed_direction=pre_shift, power=power, wind_speed=wind_speed, usable=usable
    )
    assert abs(expected["A"]) > 3.0, "the fixture should produce a real shift to detect"
    for name in northed:
        shift = circ_diff(with_p4[name]["north_offset"].to_numpy(), without[name]["north_offset"].to_numpy())
        assert shift == pytest.approx(expected[name], abs=1e-6), name


def test_add_wake_nadir_shift_returns_the_corrections_it_applied() -> None:
    """The wake-nadir-shift step hands back the per-turbine correction it added, so a caller can report it."""
    layout = _pair_layout()
    index, northed, power, wind_speed, usable, _ = _waked_pair(layout, residual_deg=6.0)
    tables = north_farm(index, direction_deg=northed, usable=usable, reanalysis_deg=northed["A"], layout=layout)

    shifted, corrections = add_wake_nadir_shift(
        tables, layout=layout, index=index, direction_deg=northed, power=power, wind_speed=wind_speed, usable=usable
    )

    pre_shift = {name: apply_north_table(index, northed[name], north_table=tables[name]) for name in northed}
    assert corrections == wake_nadir_offsets(
        layout, index=index, northed_direction=pre_shift, power=power, wind_speed=wind_speed, usable=usable
    )
    for name in northed:
        shift = circ_diff(shifted[name]["north_offset"].to_numpy(), tables[name]["north_offset"].to_numpy())
        assert shift == pytest.approx(corrections[name], abs=1e-9), name


def test_unpopulated_sector_bins_do_not_warn() -> None:
    """A gap in the swept directions leaves some sector bins empty; combining deficits must not warn.

    Real SCADA rarely fills every one-degree bin in the sector, so a bin can be empty in both the
    power and the wind-speed curve. Averaging that all-NaN column must stay silent (warnings are
    errors here) while still resolving the nadir from the populated bins.
    """
    layout = _pair_layout()
    index, northed, power, wind_speed, usable, beta = _waked_pair(layout, residual_deg=6.0)
    # Punch a hole in the sector (measured against geometry, as the code does) away from the nadir,
    # so those one-degree bins have no rows at all.
    offset = circ_diff(northed["A"], np.full(len(index), beta))
    hole = (offset > -12.0) & (offset < -9.0)
    usable = {name: mask & ~hole for name, mask in usable.items()}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        offsets = wake_nadir_offsets(
            layout, index=index, northed_direction=northed, power=power, wind_speed=wind_speed, usable=usable
        )

    assert circ_diff(offsets["A"], -6.0) == pytest.approx(0.0, abs=1.5)


def test_the_aggregate_is_a_median_that_resists_a_biased_pair() -> None:
    """Terrain adds a per-pair bias on top of the wake, so one deflected pair must not swing the answer.

    Real wake nadirs mix wake geometry with terrain deflection, which biases individual pairs by many
    degrees. Aggregating a turbine's pairs by their circular median lets a majority of consistent pairs
    outvote a deflected one, where a mean (however weighted) would be dragged toward it.
    """
    assert _aggregate([6.0, 6.4, -9.0]) == pytest.approx(6.0, abs=1e-9)


def test_the_view_angle_is_wrap_safe_when_the_nadir_sits_at_north() -> None:
    """A pair whose geometric nadir is due north sweeps directions across the 360/0 wrap.

    Measuring the nadir in view angle -- the signed offset from the geometric nadir, in [-180, 180) --
    keeps the swept directions contiguous through north, so the parabola fit is unharmed. A raw
    direction difference would split the sector across the wrap and ruin the fit.
    """
    # A due north of B (~410 m), so the B->A bearing -- the geometric nadir -- is ~0 degrees.
    frame = pd.DataFrame(
        {"name": ["A", "B"], "latitude": [55.0037, 55.0], "longitude": [0.0, 0.0], "rotor_diameter_m": [82.0, 82.0]}
    )
    layout = Layout.from_frame(frame)
    beta = float(layout.bearing_deg[layout.index_of("B"), layout.index_of("A")])
    assert min(beta, 360.0 - beta) < 1.0, f"expected the nadir near north, got {beta}"
    residual = 6.0
    index, northed, power, wind_speed, usable, _ = _waked_pair(layout, residual_deg=residual)

    offsets = wake_nadir_offsets(
        layout, index=index, northed_direction=northed, power=power, wind_speed=wind_speed, usable=usable
    )

    assert circ_diff(offsets["A"], -residual) == pytest.approx(0.0, abs=1.5)


def test_a_downstream_turbine_inherits_its_neighbours_correction() -> None:
    """B never wakes anyone within the cutoff, so it has no nadir of its own and inherits A's."""
    layout = _pair_layout()
    residual = 6.0
    index, northed, power, wind_speed, usable, _ = _waked_pair(layout, residual_deg=residual)

    offsets = wake_nadir_offsets(
        layout, index=index, northed_direction=northed, power=power, wind_speed=wind_speed, usable=usable
    )

    assert circ_diff(offsets["B"], offsets["A"]) == pytest.approx(0.0, abs=1e-6)


def test_wake_pairs_lists_each_pair_within_the_cutoff() -> None:
    layout = _pair_layout()

    assert wake_pairs(layout, devices=["A", "B"]) == [("A", "B"), ("B", "A")]
    assert wake_pairs(layout, devices=["A", "B"], cutoff_diameters=1.0) == []


def test_wake_pair_curves_find_the_nadir_the_offsets_correct() -> None:
    layout = _pair_layout()
    residual = 6.0
    index, northed, power, wind_speed, usable, _ = _waked_pair(layout, residual_deg=residual)

    curves = wake_pair_curves(
        layout, upstream="A", downstream="B", northed_direction=northed["A"], power=power, wind_speed=wind_speed
    )
    offsets = wake_nadir_offsets(
        layout, index=index, northed_direction=northed, power=power, wind_speed=wind_speed, usable=usable
    )

    assert curves is not None
    assert curves.nadir_deg == pytest.approx(-offsets["A"])
    assert curves.wind_speed_ratio is not None
    assert len(curves.offset_deg) == len(curves.power_ratio) == len(curves.wind_speed_ratio)
    assert float(curves.offset_deg[int(np.nanargmin(curves.power_ratio))]) == pytest.approx(residual, abs=1.0)


_EARTH_RADIUS_M = 6_371_000.0


def _place(latitude: float, longitude: float, *, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    north = np.cos(np.deg2rad(bearing_deg)) * distance_m
    east = np.sin(np.deg2rad(bearing_deg)) * distance_m
    return (
        latitude + float(np.rad2deg(north / _EARTH_RADIUS_M)),
        longitude + float(np.rad2deg(east / (_EARTH_RADIUS_M * np.cos(np.deg2rad(latitude))))),
    )


def _skewed_row_layout(skew_deg: float = 4.0) -> Layout:
    """W wakes U and U wakes D, 7 diameters apart, the W-U line ``skew_deg`` off the U-D line.

    D -> U is due west; U -> W is ``skew_deg`` north of west, as in a grid whose rows are not straight
    or whose columns are not square to them.
    """
    spacing = 7 * 82.0
    d = (55.0, 0.0)
    u = _place(*d, bearing_deg=270.0, distance_m=spacing)
    w = _place(*u, bearing_deg=270.0 + skew_deg, distance_m=spacing)
    frame = pd.DataFrame(
        {
            "name": ["W", "U", "D"],
            "latitude": [w[0], u[0], d[0]],
            "longitude": [w[1], u[1], d[1]],
            "rotor_diameter_m": [82.0, 82.0, 82.0],
        }
    )
    return Layout.from_frame(frame)


def _waked_farm(
    layout: Layout, *, rows: int = 20000, seed: int = 0
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Every turbine's power under a Gaussian wake from each neighbour within 10 diameters.

    Directions are true for every turbine (no northing error at all), swept 30 deg either side of west.
    """
    rng = np.random.default_rng(seed)
    true = 270.0 + rng.uniform(-30.0, 30.0, size=rows)
    names = [str(n) for n in layout.frame["name"]]
    power = {}
    for name in names:
        i = layout.index_of(name)
        factor = np.ones(rows)
        for other in names:
            j = layout.index_of(other)
            if other == name or layout.distance_m[i, j] > 10 * 82.0:
                continue
            factor *= 1.0 - 0.4 * np.exp(-0.5 * (circ_diff(true, layout.bearing_deg[i, j]) / 4.0) ** 2)
        power[name] = 1000.0 * factor
    northed = dict.fromkeys(names, true % 360.0)
    usable = {name: np.ones(rows, dtype=bool) for name in names}
    return _index(rows), northed, power, usable


def test_a_waked_upstream_turbine_is_not_shifted_by_its_neighbours_wake() -> None:
    """A pair reads its upstream turbine's error only when that turbine sees free stream on the bearing.

    U wakes D on the D-U bearing, but W wakes U from a few degrees off it, so U's power dips beside the
    bearing and the downstream-over-upstream ratio's nadir is pulled sideways. Every direction here is
    true, so every correction must be about zero; reading the U -> D pair would shift U by degrees.
    """
    layout = _skewed_row_layout()
    index, northed, power, usable = _waked_farm(layout)

    offsets = wake_nadir_offsets(layout, index=index, northed_direction=northed, power=power, usable=usable)

    assert offsets == pytest.approx({"W": 0.0, "U": 0.0, "D": 0.0}, abs=0.2)


def test_wake_pairs_skip_an_upstream_turbine_that_is_itself_waked_on_the_bearing() -> None:
    """U sits between W and D, so it is waked on both of its own pairs' bearings; only the end turbines' pairs count."""
    layout = _skewed_row_layout()

    assert wake_pairs(layout, devices=["W", "U", "D"]) == [("D", "U"), ("W", "U")]


def test_a_turbine_with_one_resolved_pair_takes_its_neighbours_correction_when_others_have_two() -> None:
    """One pair is one curve; a turbine resting on it alone takes its neighbours' answer instead."""
    assert _own_corrections({"A": [1.0, 1.2], "B": [0.8, 1.4, 1.0], "Z": [13.0]}) == pytest.approx({"A": 1.1, "B": 1.0})


def test_single_pairs_still_count_when_no_turbine_has_two() -> None:
    """In a farm too small for any turbine to resolve two pairs, one pair each is all there is."""
    assert _own_corrections({"A": [1.0], "B": [], "Z": [3.0]}) == pytest.approx({"A": 1.0, "Z": 3.0})


def test_wake_pair_curves_are_none_without_enough_rows() -> None:
    layout = _pair_layout()
    _, northed, power, _, _, _ = _waked_pair(layout, residual_deg=0.0, rows=10)

    assert wake_pair_curves(layout, upstream="A", downstream="B", northed_direction=northed["A"], power=power) is None
