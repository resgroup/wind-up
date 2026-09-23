"""Tests for the pass-4 wake-nadir absolute nudge."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from wind_up.circular_math import circ_diff
from wind_up.layout import Layout
from wind_up.northing import apply_north_table, north_farm
from wind_up.wake_nadir import _aggregate, _Nadir, wake_nadir_offsets

TIMEBASE_S = 600


def _index(rows: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=rows, freq=f"{TIMEBASE_S}s", tz="UTC")


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
    """Fabricate a clean wake dip on B whose apparent nadir sits ``residual_deg`` off geometry.

    A's *true* direction sweeps the sector around the geometric nadir; B's power and wind speed dip
    when the true wind is at the nadir. A's *northed* direction carries a constant ``residual_deg``
    error, so the dip appears in A's northed direction at nadir + residual.
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
    """With no nacelle wind speed the power ratio alone still locates the dip."""
    layout = _pair_layout()
    residual = -5.0
    index, northed, power, _, usable, _ = _waked_pair(layout, residual_deg=residual)

    offsets = wake_nadir_offsets(layout, index=index, northed_direction=northed, power=power, usable=usable)

    assert circ_diff(offsets["A"], -residual) == pytest.approx(0.0, abs=1.5)


def test_a_flat_deficit_leaves_the_turbine_uncorrected() -> None:
    """No dip anywhere means no turbine resolves, so every correction is zero (graceful)."""
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


def test_north_farm_applies_the_wake_nudge_when_given_layout_and_power() -> None:
    """north_farm runs pass 4 when a layout and power are supplied, shifting each device's table."""
    layout = _pair_layout()
    residual = 6.0
    index, northed, power, wind_speed, usable, _ = _waked_pair(layout, residual_deg=residual)
    reanalysis = northed["A"]  # anchor pass 1 to the reported signal so it contributes ~zero offset

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

    pre_nudge = {name: apply_north_table(index, northed[name], north_table=without[name]) for name in northed}
    expected = wake_nadir_offsets(
        layout, index=index, northed_direction=pre_nudge, power=power, wind_speed=wind_speed, usable=usable
    )
    assert abs(expected["A"]) > 3.0, "the fixture should produce a real nudge to detect"
    for name in northed:
        shift = circ_diff(with_p4[name]["north_offset"].to_numpy(), without[name]["north_offset"].to_numpy())
        assert shift == pytest.approx(expected[name], abs=1e-6), name


def test_unpopulated_sector_bins_do_not_warn() -> None:
    """A gap in the swept directions leaves some sector bins empty; combining deficits must not warn.

    Real SCADA rarely fills every one-degree bin in the sector, so a bin can be empty in both the
    power and the wind-speed curve. Averaging that all-NaN column must stay silent (warnings are
    errors here) while still resolving the dip from the populated bins.
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
    pairs = [_Nadir(delta=6.0, volume=8000.0), _Nadir(delta=6.4, volume=8000.0), _Nadir(delta=-9.0, volume=8000.0)]
    assert _aggregate(pairs).delta == pytest.approx(6.0, abs=1e-9)


def test_the_view_angle_is_wrap_safe_when_the_nadir_sits_at_north() -> None:
    """A pair whose geometric nadir is due north sweeps directions across the 360/0 wrap.

    Measuring the dip in view angle -- the signed offset from the geometric nadir, in [-180, 180) --
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
