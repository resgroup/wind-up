"""Tests for synthetic wake steering: geometry, solar, the upgrade, ground truth and end-to-end."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless: no display needed for tests

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.naive_ratio import NaiveRatioMethod
from benchmarking.harness.method import MethodInput
from benchmarking.synthetic import (
    HOT_COLUMNS,
    ToggleSchedule,
    WakeSteering,
    apply_upgrades,
    bearing_deg,
    derive_wake_steering_pairs,
    distance_m,
    diurnal_factor,
    generate_dataset,
    plot_wake_steering_by_direction,
    sin_solar_elevation,
    true_net_uplift,
)
from benchmarking.synthetic.cp_core import CpCore
from benchmarking.synthetic.upgrades import UpgradeEffect
from wind_up_v0.constants import TIMESTAMP_COL

if TYPE_CHECKING:
    from pathlib import Path

# A compact three-turbine layout: UP steers for DOWN (a near pair); REF is far away (a clean
# reference, never paired). UP is due south of DOWN, so the wake nadir (wind-from at which UP wakes
# DOWN) is 180 deg. All coordinates are (lat, lon) in degrees.
UP, DOWN, REF = "UP", "DOWN", "REF"
COORDS = {UP: (57.500, -3.250), DOWN: (57.5027, -3.250), REF: (57.400, -3.100)}
NADIR = 180.0
T0 = pd.Timestamp("2020-01-01", tz="UTC")
NORTH_OFFSETS = [(UP, T0, 0.0), (DOWN, T0, 0.0), (REF, T0, 0.0)]


def _one_turbine(turbine: str, *, nacelle: list[float], power: float = 1000.0) -> pd.DataFrame:
    index = pd.date_range("2020-06-01", periods=len(nacelle), freq="10min", tz="UTC")
    frame = pd.DataFrame(
        {
            HOT_COLUMNS.turbine: turbine,
            HOT_COLUMNS.active_power: float(power),
            HOT_COLUMNS.wind_speed: 8.0,
            HOT_COLUMNS.wind_speed_sd: 0.8,
            HOT_COLUMNS.gen_rpm: 1400.0,
            HOT_COLUMNS.nacelle_position: np.array(nacelle, dtype=float),
        },
        index=index,
    )
    frame.index.name = TIMESTAMP_COL
    return frame


def _steering(**overrides: object) -> WakeSteering:
    params: dict = {"coords": COORDS, "test_wtgs": [UP, DOWN], "north_offsets": NORTH_OFFSETS}
    params.update(overrides)
    return WakeSteering(**params)  # type: ignore[arg-type]


# --- geometry -----------------------------------------------------------------------------------


def test_bearing_and_distance_due_north() -> None:
    """A point due north bears 0 deg; the reciprocal bearing is 180 deg."""
    assert bearing_deg(COORDS[UP], COORDS[DOWN]) == pytest.approx(0.0, abs=1e-6)
    assert bearing_deg(COORDS[DOWN], COORDS[UP]) == pytest.approx(180.0, abs=1e-6)
    assert distance_m(COORDS[UP], COORDS[DOWN]) == pytest.approx(300.0, abs=5.0)


def test_derive_pairs_within_7d_and_excludes_references() -> None:
    """Only near test-test pairs appear (both directions); far and reference turbines are dropped."""
    pairs = derive_wake_steering_pairs(COORDS, test_wtgs=[UP, DOWN, REF])
    directed = {(p.upstream, p.downstream): p.nadir_bearing for p in pairs}
    assert set(directed) == {(UP, DOWN), (DOWN, UP)}
    assert directed[(UP, DOWN)] == pytest.approx(NADIR, abs=1e-3)
    assert directed[(DOWN, UP)] == pytest.approx(0.0, abs=1e-3)


# --- solar --------------------------------------------------------------------------------------


def test_solar_elevation_positive_by_day_negative_at_night() -> None:
    """Solar elevation is positive around local noon and negative around local midnight."""
    noon = pd.DatetimeIndex([pd.Timestamp("2020-06-21 12:00", tz="UTC")])
    midnight = pd.DatetimeIndex([pd.Timestamp("2020-06-21 00:00", tz="UTC")])
    assert sin_solar_elevation(noon, lat=57.5, lon=-3.25)[0] > 0.5
    assert sin_solar_elevation(midnight, lat=57.5, lon=-3.25)[0] < 0.0


def test_diurnal_factor_hits_extremes() -> None:
    """The diurnal factor is the night value in darkness and the day value at high sun."""
    noon = pd.DatetimeIndex([pd.Timestamp("2020-06-21 12:00", tz="UTC")])
    midnight = pd.DatetimeIndex([pd.Timestamp("2020-06-21 00:00", tz="UTC")])
    kwargs = {"lat": 57.5, "lon": -3.25, "night_factor": 1.3, "day_factor": 0.6}
    assert diurnal_factor(midnight, **kwargs)[0] == pytest.approx(1.3)
    assert diurnal_factor(noon, **kwargs)[0] == pytest.approx(0.6)


# --- the WakeSteering effect --------------------------------------------------------------------


def test_steer_sign_cw_left_ccw_right_of_nadir() -> None:
    """Applied compass yaw is +ve (CW) left of the nadir and -ve (CCW) right of it; 0 outside."""
    # cal == nacelle (north offset 0): 170 = left of nadir, 190 = right, 100 = outside the sector.
    rows = _one_turbine(UP, nacelle=[170.0, 190.0, 100.0])
    effect = _steering()(rows, HOT_COLUMNS)
    delta = np.asarray(effect.nacelle_position_delta, dtype=float)
    assert delta[0] > 0.0  # left of nadir -> CW
    assert delta[1] < 0.0  # right of nadir -> CCW
    assert delta[2] == pytest.approx(0.0)  # outside sector -> no steer


def test_upstream_cosine_loss_in_sector_only() -> None:
    """The steering turbine loses power in the sector (cp_ratio < 1) and is untouched outside."""
    # 183 is inside the plateau but off the exact nadir (where the sign flip zeroes the offset).
    rows = _one_turbine(UP, nacelle=[183.0, 100.0])
    effect = _steering()(rows, HOT_COLUMNS)
    cp = np.asarray(effect.cp_ratio, dtype=float)
    ws = np.asarray(effect.ws_factor, dtype=float)
    assert cp[0] < 1.0  # near nadir: yaw loss
    assert cp[1] == pytest.approx(1.0)  # outside sector: unchanged
    assert ws[0] == pytest.approx(1.0 + (-0.02))  # plateau flow-distortion ws change
    assert ws[1] == pytest.approx(1.0)


def test_downstream_gain_and_ws_inflation_in_sector() -> None:
    """The benefitting turbine gains power and reads higher wind speed in-sector (past the deadband)."""
    # 184 is in-sector and past the near-nadir deadband; the exact nadir would give no effect.
    rows = _one_turbine(DOWN, nacelle=[184.0, 100.0])
    effect = _steering()(rows, HOT_COLUMNS)
    cp = np.asarray(effect.cp_ratio, dtype=float)
    ws = np.asarray(effect.ws_factor, dtype=float)
    nacelle = np.asarray(effect.nacelle_position_delta, dtype=float)
    assert cp[0] > 1.0  # in-sector: wake-recovery gain
    assert ws[0] > 1.0  # post-treatment nacelle-ws inflation
    assert cp[1] == pytest.approx(1.0)
    assert np.allclose(nacelle, 0.0)  # the benefitting turbine does not steer


def test_steer_magnitude_clipped_by_power() -> None:
    """The upstream steer is clipped to a power-dependent limit: the full offset at/below the
    full-steer power, zero at the zero-steer (rated) power, linear between (and so is the loss)."""
    steering = _steering(max_offset_deg=20.0, steer_full_power_kw=1610.0, steer_zero_power_kw=2300.0)
    powers = [1000.0, 1610.0, 1955.0, 2300.0]  # availability 1, 1, 0.5, 0
    index = pd.date_range("2020-06-01", periods=len(powers), freq="10min", tz="UTC")
    rows = pd.DataFrame(
        {
            HOT_COLUMNS.turbine: UP,
            HOT_COLUMNS.active_power: np.array(powers, dtype=float),
            HOT_COLUMNS.wind_speed: 9.0,
            HOT_COLUMNS.wind_speed_sd: 0.8,
            HOT_COLUMNS.gen_rpm: 1400.0,
            HOT_COLUMNS.nacelle_position: NADIR + 4.0,  # in the plateau, past the sign crossover
        },
        index=index,
    )
    rows.index.name = TIMESTAMP_COL
    effect = steering(rows, HOT_COLUMNS)
    assert np.abs(effect.nacelle_position_delta) == pytest.approx([20.0, 20.0, 10.0, 0.0])
    cp = np.asarray(effect.cp_ratio, dtype=float)
    assert cp[0] == pytest.approx(cp[1])  # no clip below the full-steer power
    assert cp[1] < cp[2] < cp[3]  # loss shrinks as steering fades with power
    assert cp[3] == pytest.approx(1.0)  # no steering, no loss at rated


def test_low_power_clip_zero_at_no_generation() -> None:
    """The steer limit is a trapezoid in power: 0 at 0 kW, rising to full by the low-full power."""
    steering = _steering(max_offset_deg=20.0, steer_cutin_power_kw=0.0, steer_low_full_power_kw=230.0)
    powers = [0.0, 115.0, 230.0, 800.0]  # availability 0, 0.5, 1, 1
    index = pd.date_range("2020-06-01", periods=len(powers), freq="10min", tz="UTC")
    rows = pd.DataFrame(
        {
            HOT_COLUMNS.turbine: UP,
            HOT_COLUMNS.active_power: np.array(powers, dtype=float),
            HOT_COLUMNS.wind_speed: 9.0,
            HOT_COLUMNS.wind_speed_sd: 0.8,
            HOT_COLUMNS.gen_rpm: 1400.0,
            HOT_COLUMNS.nacelle_position: NADIR + 4.0,  # in the plateau, past the sign crossover
        },
        index=index,
    )
    rows.index.name = TIMESTAMP_COL
    effect = steering(rows, HOT_COLUMNS)
    assert np.abs(effect.nacelle_position_delta) == pytest.approx([0.0, 10.0, 20.0, 20.0])


def test_steer_magnitude_clipped_by_wind_speed() -> None:
    """The upstream steer is also clipped by a high-wind-speed gate: full at/below 12 m/s, zero at/above
    14 m/s, linear between (so the loss fades with it), independent of the power gate."""
    steering = _steering(max_offset_deg=20.0)  # ws gate defaults: fade 12 -> 0 by 14 m/s
    winds = [8.0, 12.0, 13.0, 14.0, 20.0]  # ws availability 1, 1, 0.5, 0, 0
    index = pd.date_range("2020-06-01", periods=len(winds), freq="10min", tz="UTC")
    rows = pd.DataFrame(
        {
            HOT_COLUMNS.turbine: UP,
            HOT_COLUMNS.active_power: 1000.0,  # power availability 1 throughout
            HOT_COLUMNS.wind_speed: np.array(winds, dtype=float),
            HOT_COLUMNS.wind_speed_sd: 0.8,
            HOT_COLUMNS.gen_rpm: 1400.0,
            HOT_COLUMNS.nacelle_position: NADIR + 4.0,  # in the plateau, past the sign crossover
        },
        index=index,
    )
    rows.index.name = TIMESTAMP_COL
    effect = steering(rows, HOT_COLUMNS)
    assert np.abs(effect.nacelle_position_delta) == pytest.approx([20.0, 20.0, 10.0, 0.0, 0.0])
    cp = np.asarray(effect.cp_ratio, dtype=float)
    assert cp[0] == pytest.approx(cp[1])  # no clip at/below the fade-start wind speed
    assert cp[1] < cp[2] < cp[3]  # loss shrinks as steering fades with wind speed
    assert cp[3] == pytest.approx(1.0)  # at 14 m/s: no steering, no loss
    assert cp[4] == pytest.approx(1.0)  # above 14 m/s: still none


def test_gate_is_min_of_power_and_wind_speed() -> None:
    """The steer envelope is the minimum of the power and wind-speed gates: the more conservative wins."""
    steering = _steering(max_offset_deg=20.0, steer_full_power_kw=1610.0, steer_zero_power_kw=2300.0)
    # Row 0: power-limited (1955 kW -> power gate 0.5) while ws is low (gate 1) -> min 0.5.
    # Row 1: ws-limited (13.5 m/s -> ws gate 0.25) while power is full (gate 1) -> min 0.25.
    index = pd.date_range("2020-06-01", periods=2, freq="10min", tz="UTC")
    rows = pd.DataFrame(
        {
            HOT_COLUMNS.turbine: UP,
            HOT_COLUMNS.active_power: np.array([1955.0, 1000.0], dtype=float),
            HOT_COLUMNS.wind_speed: np.array([8.0, 13.5], dtype=float),
            HOT_COLUMNS.wind_speed_sd: 0.8,
            HOT_COLUMNS.gen_rpm: 1400.0,
            HOT_COLUMNS.nacelle_position: NADIR + 4.0,
        },
        index=index,
    )
    rows.index.name = TIMESTAMP_COL
    effect = steering(rows, HOT_COLUMNS)
    assert np.abs(effect.nacelle_position_delta) == pytest.approx([10.0, 5.0])  # 20*0.5, 20*0.25


def test_high_wind_speed_suppresses_downstream_gain_via_prepare() -> None:
    """After ``prepare`` a high upstream wind speed removes the downstream gain (gated on the upstream),
    even where direction and power would otherwise allow it -- so no steering effect at high wind."""
    index = pd.date_range("2020-06-01", periods=2, freq="10min", tz="UTC")

    def frame(wtg: str, *, wind: list[float]) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                HOT_COLUMNS.turbine: wtg,
                HOT_COLUMNS.active_power: 1000.0,
                HOT_COLUMNS.wind_speed: np.array(wind, dtype=float),
                HOT_COLUMNS.wind_speed_sd: 0.8,
                HOT_COLUMNS.gen_rpm: 1400.0,
                HOT_COLUMNS.nacelle_position: NADIR + 4.0,
            },
            index=index,
        )
        df.index.name = TIMESTAMP_COL
        return df

    upstream = frame(UP, wind=[8.0, 20.0])  # t0 low wind, t1 high wind (> gate zero)
    downstream = frame(DOWN, wind=[8.0, 8.0])
    steering = _steering()
    steering.prepare(pd.concat([upstream, downstream]), columns=HOT_COLUMNS)
    cp = np.asarray(steering(downstream, HOT_COLUMNS).cp_ratio, dtype=float)
    assert cp[0] > 1.0  # low upstream wind: gain applies
    assert cp[1] == pytest.approx(1.0)  # high upstream wind: no gain


def test_steer_and_effect_fade_through_the_nadir_deadband() -> None:
    """Near the nadir the applied yaw ramps to zero (a deadband), and the loss fades with it: the whole
    steering effect shrinks toward the nadir instead of the yaw flipping sign sharply."""
    steering = _steering(max_offset_deg=20.0, crossover_half_deg=2.5)
    offsets = [-5.0, -2.5, -1.25, 0.0, 1.25, 2.5, 5.0]  # deg from nadir, all inside the plateau
    index = pd.date_range("2020-06-01", periods=len(offsets), freq="10min", tz="UTC")
    rows = pd.DataFrame(
        {
            HOT_COLUMNS.turbine: UP,
            HOT_COLUMNS.active_power: 1000.0,
            HOT_COLUMNS.wind_speed: 9.0,
            HOT_COLUMNS.wind_speed_sd: 0.8,
            HOT_COLUMNS.gen_rpm: 1400.0,
            HOT_COLUMNS.nacelle_position: NADIR + np.array(offsets, dtype=float),
        },
        index=index,
    )
    rows.index.name = TIMESTAMP_COL
    effect = steering(rows, HOT_COLUMNS)
    # Applied compass yaw: +max left of nadir, linear through 0 at the nadir, -max right of it.
    assert effect.nacelle_position_delta == pytest.approx([20.0, 20.0, 10.0, 0.0, -10.0, -20.0, -20.0])
    cp = np.asarray(effect.cp_ratio, dtype=float)
    assert cp[3] == pytest.approx(1.0)  # nadir: no steer -> no loss
    assert cp[1] < cp[2] < cp[3]  # loss fades from full (band edge) to none (nadir)
    assert cp[0] == pytest.approx(cp[1])  # full loss at/past the band edge


def test_downstream_gain_gated_on_upstream_via_prepare() -> None:
    """After ``prepare`` the downstream gain follows the UPSTREAM's direction and power: it applies
    when the upstream steers even if the downstream is out of its own sector, and vanishes when the
    upstream is not generating even though the downstream itself is in sector and generating."""
    index = pd.date_range("2020-06-01", periods=2, freq="10min", tz="UTC")

    def frame(wtg: str, *, power: list[float], nacelle: list[float]) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                HOT_COLUMNS.turbine: wtg,
                HOT_COLUMNS.active_power: np.array(power, dtype=float),
                HOT_COLUMNS.wind_speed: 8.0,
                HOT_COLUMNS.wind_speed_sd: 0.8,
                HOT_COLUMNS.gen_rpm: 1400.0,
                HOT_COLUMNS.nacelle_position: np.array(nacelle, dtype=float),
            },
            index=index,
        )
        df.index.name = TIMESTAMP_COL
        return df

    # t0: upstream in-sector + generating, downstream pointing OUT of its own sector.
    # t1: upstream parked (0 kW), downstream in-sector + generating.
    upstream = frame(UP, power=[1000.0, 0.0], nacelle=[NADIR + 4.0, NADIR + 4.0])
    downstream = frame(DOWN, power=[900.0, 900.0], nacelle=[100.0, NADIR + 4.0])
    steering = _steering()
    steering.prepare(pd.concat([upstream, downstream]), columns=HOT_COLUMNS)
    cp = np.asarray(steering(downstream, HOT_COLUMNS).cp_ratio, dtype=float)
    assert cp[0] > 1.0  # gated on the upstream: gain despite the downstream being out of its own sector
    assert cp[1] == pytest.approx(1.0)  # upstream not generating: no gain


def test_time_stepped_northing_shifts_the_gate() -> None:
    """A later north-offset correction moves which raw nacelle positions fall in the sector."""
    # Raw nacelle 210: with +0 offset (before the step) calibrated 210 is outside the 180 sector;
    # with a -33 offset (after the step) calibrated 177 sits just inside the plateau -> loss.
    index = pd.date_range("2020-06-01", periods=2, freq="10min", tz="UTC")
    rows = _one_turbine(UP, nacelle=[210.0, 210.0]).set_axis(index)
    rows[HOT_COLUMNS.turbine] = UP
    step = index[1]
    north = [(UP, T0, 0.0), (UP, step, -33.0), (DOWN, T0, 0.0)]
    effect = _steering(north_offsets=north)(rows, HOT_COLUMNS)
    cp = np.asarray(effect.cp_ratio, dtype=float)
    assert cp[0] == pytest.approx(1.0)  # before step: 210 outside sector
    assert cp[1] < 1.0  # after step: calibrated 180 is on the nadir -> loss


def test_construction_validates_schema_and_northing() -> None:
    """Missing nacelle role or a participant without a northing correction raise clearly."""
    no_nacelle = replace(HOT_COLUMNS, nacelle_position=None)
    with pytest.raises(ValueError, match="nacelle_position"):
        _steering()(_one_turbine(UP, nacelle=[180.0]), no_nacelle)
    with pytest.raises(ValueError, match="north_offsets is missing"):
        WakeSteering(coords=COORDS, test_wtgs=[UP, DOWN], north_offsets=[(UP, T0, 0.0)])


def test_pair_dropped_when_upstream_waked_across_whole_sector() -> None:
    """A pair whose upstream is inside another turbine's disturbed sector across its whole steering
    sector is dropped (that steer can never happen); moving the blocker away restores the pair.

    BLOCK sits just south of UP, so UP is inside BLOCK's disturbed sector for all of UP's southerly
    (~180) steering sector for DOWN.
    """
    coords = {**COORDS, "BLOCK": (57.4985, -3.250)}  # ~170 m due south of UP (upwind at ~180 deg)
    north = [*NORTH_OFFSETS, ("BLOCK", T0, 0.0)]

    blocked = WakeSteering(coords=coords, test_wtgs=[UP, DOWN], north_offsets=north)
    assert (UP, DOWN) not in [(p.upstream, p.downstream) for p in blocked.pairs]
    effect = blocked(_one_turbine(UP, nacelle=[183.0]), HOT_COLUMNS)
    assert np.asarray(effect.cp_ratio, dtype=float)[0] == pytest.approx(1.0)  # no steering loss

    far = {**coords, "BLOCK": (57.400, -3.100)}  # relocate the blocker well outside 20 D
    unblocked = WakeSteering(coords=far, test_wtgs=[UP, DOWN], north_offsets=north)
    assert (UP, DOWN) in [(p.upstream, p.downstream) for p in unblocked.pairs]


def test_is_waked_per_row_matches_iec_sector() -> None:
    """``_is_waked`` flags exactly the rows whose wind comes from within a neighbour's IEC sector."""
    # BLOCK is due south of UP (~2 D): a wide IEC sector centred on 180 deg. Wind from 180 -> waked;
    # wind from 90 (neighbour off to the side) -> not waked.
    coords = {**COORDS, "BLOCK": (57.4985, -3.250)}
    north = [*NORTH_OFFSETS, ("BLOCK", T0, 0.0)]
    ws = WakeSteering(coords=coords, test_wtgs=[UP, DOWN], north_offsets=north)
    waked = ws._is_waked(UP, np.array([180.0, 90.0]))  # noqa: SLF001
    assert bool(waked[0]) is True
    assert bool(waked[1]) is False


# --- mechanism ----------------------------------------------------------------------------------


def test_apply_upgrades_composes_ws_and_nacelle_and_writes_back() -> None:
    """apply_upgrades multiplies ws factors, sums nacelle deltas (mod 360) and writes nacelle back."""

    class _Fake:
        def __init__(self, ws: float, delta: float) -> None:
            self.ws, self.delta = ws, delta

        def __call__(self, rows: pd.DataFrame, columns: object) -> UpgradeEffect:  # noqa: ARG002
            n = len(rows)
            return UpgradeEffect(ws_factor=np.full(n, self.ws), nacelle_position_delta=np.full(n, self.delta))

    rows = _one_turbine(UP, nacelle=[350.0])
    out = apply_upgrades(rows, [_Fake(1.1, 20.0), _Fake(1.2, 5.0)], cp=CpCore(), columns=HOT_COLUMNS)
    assert out[HOT_COLUMNS.wind_speed].iloc[0] == pytest.approx(8.0 * 1.1 * 1.2)
    assert out[HOT_COLUMNS.nacelle_position].iloc[0] == pytest.approx((350.0 + 25.0) % 360.0)  # 15.0


def test_plain_upgrade_leaves_nacelle_unchanged() -> None:
    """A Cp-only upgrade returns the original nacelle position (no spurious write-back)."""
    from benchmarking.synthetic.upgrades import ConstantCpChange  # noqa: PLC0415

    rows = _one_turbine(UP, nacelle=[123.0, 45.0])
    out = apply_upgrades(rows, [ConstantCpChange(delta=0.05)], cp=CpCore(), columns=HOT_COLUMNS)
    assert np.allclose(out[HOT_COLUMNS.nacelle_position].to_numpy(), [123.0, 45.0])


# --- ground truth -------------------------------------------------------------------------------


def test_true_net_uplift_closed_form() -> None:
    """The pair net equals (sum syn) / (sum orig) - 1 over changed records."""
    synthetic = pd.concat(
        [_one_turbine(UP, nacelle=[0.0], power=900.0), _one_turbine(DOWN, nacelle=[0.0], power=1100.0)]
    )
    original = pd.concat(
        [_one_turbine(UP, nacelle=[0.0], power=1000.0), _one_turbine(DOWN, nacelle=[0.0], power=1000.0)]
    )
    mask = np.array([True])
    net = true_net_uplift(synthetic, original, upstream=UP, downstream=DOWN, mask=mask)
    assert net == pytest.approx((900.0 + 1100.0) / (1000.0 + 1000.0) - 1.0)


def test_true_net_uplift_uses_union_of_changed_records() -> None:
    """Both turbines are summed over the union of changed timestamps, not each over its own.

    Row 0 (nadir-like): only the downstream changes; row 1: only the upstream changes. Both rows must
    count for both turbines, so the upstream's row-0 baseline energy is not dropped (which would
    inflate the net).
    """
    synthetic = pd.concat(
        [_one_turbine(UP, nacelle=[0.0, 0.0], power=1000.0), _one_turbine(DOWN, nacelle=[0.0, 0.0], power=1000.0)]
    )
    synthetic.loc[(synthetic[HOT_COLUMNS.turbine] == DOWN), HOT_COLUMNS.active_power] = [1100.0, 1000.0]
    synthetic.loc[(synthetic[HOT_COLUMNS.turbine] == UP), HOT_COLUMNS.active_power] = [1000.0, 950.0]
    original = pd.concat(
        [_one_turbine(UP, nacelle=[0.0, 0.0], power=1000.0), _one_turbine(DOWN, nacelle=[0.0, 0.0], power=1000.0)]
    )
    net = true_net_uplift(synthetic, original, upstream=UP, downstream=DOWN)
    # Union of both rows: (1000+1100 + 950+1000) / (4 * 1000) - 1 = +1.25%, not the +2.5% each-own-mask gives.
    assert net == pytest.approx((1000.0 + 1100.0 + 950.0 + 1000.0) / 4000.0 - 1.0)


# --- end to end ---------------------------------------------------------------------------------


def _farm_sweeping_sector(periods: int = 288) -> pd.DataFrame:
    """A toggle-length farm whose nacelle positions sweep across the 180 deg steering sector."""
    index = pd.date_range("2020-06-01", periods=periods, freq="10min", tz="UTC")
    # A slow sweep 150..210 deg keeps a good fraction of rows inside the 180 +/- 15 sector.
    sweep = 150.0 + 60.0 * (np.arange(periods) % 60) / 59.0
    frames = []
    for turbine, power in ((UP, 1000.0), (DOWN, 1000.0), (REF, 1000.0)):
        frames.append(
            pd.DataFrame(
                {
                    HOT_COLUMNS.turbine: turbine,
                    HOT_COLUMNS.active_power: power,
                    HOT_COLUMNS.wind_speed: 8.0,
                    HOT_COLUMNS.wind_speed_sd: 0.8,
                    HOT_COLUMNS.gen_rpm: 1400.0,
                    HOT_COLUMNS.nacelle_position: sweep,
                    HOT_COLUMNS.availability: 600.0,
                },
                index=index,
            )
        )
    wf = pd.concat(frames)
    wf.index.name = TIMESTAMP_COL
    return wf


def test_end_to_end_ground_truth_naive_and_plot(tmp_path: Path) -> None:
    """A full wake-steering dataset: signed ground truth, a naive run, and a rendered plot."""
    wf = _farm_sweeping_sector()
    schedule = ToggleSchedule(period=pd.Timedelta(minutes=100), start=wf.index.min())
    # Tuned so the downstream gain clearly outweighs the upstream loss in energy (robust net > 0).
    steering = _steering(peak_gain=0.10, max_offset_deg=12.0)
    dataset = generate_dataset(
        scada_df=wf, test_wtgs=[UP, DOWN], upgrades=[steering], mode="toggle", upgrade_timing=schedule
    )

    assert dataset.true_uplift(test_wtg=UP).overall < 0.0
    assert dataset.true_uplift(test_wtg=DOWN).overall > 0.0
    assert dataset.true_net_uplift(upstream=UP, downstream=DOWN) > 0.0

    # No change outside the pair's turbines: the reference is untouched.
    ref_syn = dataset.synthetic_df[dataset.synthetic_df[HOT_COLUMNS.turbine] == REF][HOT_COLUMNS.active_power]
    ref_orig = dataset.original_df[dataset.original_df[HOT_COLUMNS.turbine] == REF][HOT_COLUMNS.active_power]
    assert np.allclose(ref_syn.to_numpy(), ref_orig.to_numpy())

    result = NaiveRatioMethod(columns=HOT_COLUMNS).estimate(
        MethodInput(scada_df=dataset.synthetic_df, test_wtg=DOWN, upgrade_timing=schedule)
    )
    assert np.isfinite(result.p50_overall)

    save_path = tmp_path / "wake_steering.png"
    plot_wake_steering_by_direction(dataset, upstream=UP, downstream=DOWN, save_path=save_path)
    assert save_path.exists()


def test_wake_steering_example_driver_saves_dataset_and_plots(tmp_path: Path) -> None:
    """The HoT wake-steering driver builds coords from metadata and writes a dataset plus plots."""
    from benchmarking.synthetic.make_example_datasets import (  # noqa: PLC0415
        WAKE_STEERING_CLUSTER,
        generate_wake_steering_example,
    )

    # A tight cluster (all within 7 D) using the real campaign turbine names the driver expects.
    latlon = {
        "T02": (57.5000, -3.2500),
        "T03": (57.5015, -3.2500),
        "T05": (57.5000, -3.2505),
        "T07": (57.5015, -3.2505),
    }
    periods = 288
    index = pd.date_range("2020-06-01", periods=periods, freq="10min", tz="UTC")
    sweep = 360.0 * np.arange(periods) / periods  # sweep the full circle so every pair sees its sector
    frames = [
        pd.DataFrame(
            {
                HOT_COLUMNS.turbine: wtg,
                HOT_COLUMNS.active_power: 1000.0,
                HOT_COLUMNS.wind_speed: 8.0,
                HOT_COLUMNS.wind_speed_sd: 0.8,
                HOT_COLUMNS.gen_rpm: 1400.0,
                HOT_COLUMNS.nacelle_position: sweep,
                HOT_COLUMNS.availability: 600.0,
            },
            index=index,
        )
        for wtg in WAKE_STEERING_CLUSTER
    ]
    scada_df = pd.concat(frames)
    scada_df.index.name = TIMESTAMP_COL
    metadata_df = pd.DataFrame([{"Name": wtg, "Latitude": lat, "Longitude": lon} for wtg, (lat, lon) in latlon.items()])

    dataset = generate_wake_steering_example(
        scada_df=scada_df, metadata_df=metadata_df, start_dt=index.min(), out_root=tmp_path
    )

    assert (tmp_path / "wake_steering" / "synthetic.parquet").exists()
    assert dataset.run_metadata["upgrades"][0]["kind"] == "wake_steering"
    plots = list((tmp_path / "wake_steering").glob("steering_*_to_*.png"))
    assert plots, "expected at least one per-pair steering plot"
