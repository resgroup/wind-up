"""Tests for the synthetic upgrade callables and their resolution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.synthetic.cp_core import CpCore, region2_fraction
from benchmarking.synthetic.upgrades import (
    ConditionCpChange,
    ConstantCpChange,
    RatedPowerChange,
    WindSpeedCpChange,
    apply_upgrades,
)
from wind_up.constants import DataColumns


def _rows(
    powers: list[float],
    *,
    ws: list[float] | None = None,
    sd: list[float] | None = None,
    rpm: list[float] | None = None,
) -> pd.DataFrame:
    n = len(powers)
    return pd.DataFrame(
        {
            DataColumns.active_power_mean: np.array(powers, dtype=float),
            DataColumns.wind_speed_mean: np.array(ws if ws is not None else [8.0] * n, dtype=float),
            DataColumns.wind_speed_sd: np.array(sd if sd is not None else [0.8] * n, dtype=float),
            DataColumns.gen_rpm_mean: np.array(rpm if rpm is not None else [1400.0] * n, dtype=float),
        }
    )


def test_constant_cp_change_scales_region2_power() -> None:
    """At 2000 kW (region-2 fraction 0.25) a +10% Cp gives 2050 kW (not 2200 due to 25% region 2)."""
    rows = _rows([2000.0])
    out = apply_upgrades(rows, [ConstantCpChange(delta=0.10)], CpCore(rated_power_kw=2300.0))
    assert out[DataColumns.active_power_mean].iloc[0] == pytest.approx(2050.0)


def test_empty_upgrade_list_leaves_rows_unchanged() -> None:
    """Applying no upgrades returns power identical to the original."""
    rows = _rows([500.0, 1500.0, 2200.0])
    out = apply_upgrades(rows, [], CpCore())
    pd.testing.assert_series_equal(out[DataColumns.active_power_mean], rows[DataColumns.active_power_mean])


def test_constant_cp_change_only_treats_producing_below_rated_records() -> None:
    """Non-producing (<=0) and virtually-rated rows are not treated; region-2 rows are.

    Locks the user-visible behaviour: a Cp change only scales genuine producing records
    below pure rated, so idling/curtailed and at-rated records keep their original power.
    """
    rows = _rows([-5.0, 0.0, 1000.0, 2295.0])
    out = apply_upgrades(rows, [ConstantCpChange(delta=0.03)], CpCore(rated_power_kw=2300.0))
    power = out[DataColumns.active_power_mean].to_numpy()
    assert power[0] == pytest.approx(-5.0)  # negative: untouched
    assert power[1] == pytest.approx(0.0)  # zero: untouched
    assert power[2] > 1000.0  # producing region 2: still changed
    assert power[3] == pytest.approx(2295.0)  # virtually rated: untouched


def test_constant_cp_change_does_not_mutate_input() -> None:
    """apply_upgrades returns a new frame and leaves the caller's rows untouched."""
    rows = _rows([1500.0])
    original = rows[DataColumns.active_power_mean].copy()
    apply_upgrades(rows, [ConstantCpChange(delta=0.05)], CpCore())
    pd.testing.assert_series_equal(rows[DataColumns.active_power_mean], original)


def test_wind_speed_cp_change_applies_delta_by_original_wind_speed() -> None:
    """Cp delta is interpolated over the original wind speed; zero-delta bins are unchanged."""
    rows = _rows([600.0, 1000.0, 1500.0], ws=[4.0, 8.0, 12.0])
    upgrade = WindSpeedCpChange(ws_points=[4.0, 8.0, 12.0], deltas=[0.0, 0.04, 0.0])
    out = apply_upgrades(rows, [upgrade], CpCore())
    power = out[DataColumns.active_power_mean].to_numpy()
    f8 = region2_fraction(1000.0)
    assert power[0] == pytest.approx(600.0)  # ws 4: zero delta
    assert power[1] == pytest.approx(1000.0 * (1.0 + f8 * 0.04))  # ws 8: +4% Cp in region 2
    assert power[2] == pytest.approx(1500.0)  # ws 12: zero delta


def test_condition_cp_change_varies_by_original_turbulence_intensity() -> None:
    """Cp delta is interpolated over original TI = WindSpeedSD / WindSpeedMean."""
    # ws 8 m/s with sd 0.8 and 1.6 -> TI 0.10 and 0.20
    rows = _rows([1000.0, 1000.0], ws=[8.0, 8.0], sd=[0.8, 1.6])
    upgrade = ConditionCpChange(by="ti", points=[0.10, 0.20], deltas=[0.0, 0.05])
    out = apply_upgrades(rows, [upgrade], CpCore())
    power = out[DataColumns.active_power_mean].to_numpy()
    f = region2_fraction(1000.0)
    assert power[0] == pytest.approx(1000.0)  # TI 0.10: zero delta
    assert power[1] == pytest.approx(1000.0 * (1.0 + f * 0.05))  # TI 0.20: +5% Cp in region 2


def test_rated_power_downrate_clips_at_new_rated() -> None:
    """A downrate caps power at the new rated and leaves region-2 power unchanged."""
    rows = _rows([1000.0, 2200.0])
    out = apply_upgrades(rows, [RatedPowerChange(new_rated_power_kw=2000.0)], CpCore(rated_power_kw=2300.0))
    power = out[DataColumns.active_power_mean].to_numpy()
    assert power[0] == pytest.approx(1000.0)  # well below new rated: unchanged
    assert power[1] == pytest.approx(2000.0)  # above new rated: clipped


def test_rated_power_uprate_lifts_region3_power() -> None:
    """An uprate lifts near-rated power toward the new rated, leaving deep region 2 alone."""
    rows = _rows([1000.0, 2200.0])
    out = apply_upgrades(rows, [RatedPowerChange(new_rated_power_kw=2400.0)], CpCore(rated_power_kw=2300.0))
    power = out[DataColumns.active_power_mean].to_numpy()
    assert power[0] == pytest.approx(1000.0, rel=1e-3)  # deep region 2: ~unchanged
    assert 2200.0 < power[1] < 2400.0  # near rated: lifted but capped at new rated


def test_ws_delta_scales_nacelle_wind_speed() -> None:
    """An upgrade's ws_delta scales the nacelle WindSpeedMean."""
    rows = _rows([1000.0], ws=[8.0])
    out = apply_upgrades(rows, [ConstantCpChange(delta=0.0, ws_delta=0.02)], CpCore())
    assert out[DataColumns.wind_speed_mean].iloc[0] == pytest.approx(8.0 * 1.02)


def test_rpm_increases_when_power_increases() -> None:
    """A positive Cp change drags generator rpm up with the power increase."""
    rows = _rows([1000.0], rpm=[1392.0])
    out = apply_upgrades(rows, [ConstantCpChange(delta=0.10)], CpCore())
    assert out[DataColumns.gen_rpm_mean].iloc[0] > 1392.0


def test_upgrades_compose() -> None:
    """A Cp change and a downrate combine: region-2 power rises, near-rated power clips."""
    rows = _rows([1000.0, 2200.0])
    upgrades = [ConstantCpChange(delta=0.10), RatedPowerChange(new_rated_power_kw=2000.0)]
    out = apply_upgrades(rows, upgrades, CpCore(rated_power_kw=2300.0))
    power = out[DataColumns.active_power_mean].to_numpy()
    f = region2_fraction(1000.0)
    assert power[0] == pytest.approx(1000.0 * (1.0 + f * 0.10))  # region 2: +Cp, below downrate
    assert power[1] == pytest.approx(2000.0)  # near rated: clipped to the downrate
