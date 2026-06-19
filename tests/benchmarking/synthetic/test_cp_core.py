"""Tests for the analytic Hill of Towie Cp surface and Cp-space physics core."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarking.synthetic.cp_core import (
    HOT_CP_MODEL,
    CpCore,
    cp_surface,
    power_from_cp_change,
    region2_fraction,
    rpm_from_power,
    rpm_from_power_change,
)


def test_cp_peaks_at_optimal_tsr_and_pitch() -> None:
    """At optimal TSR and pitch the surface returns Cp_max."""
    cp = cp_surface(tsr=HOT_CP_MODEL.opt_tsr, pitch=HOT_CP_MODEL.opt_pitch, params=HOT_CP_MODEL)
    assert cp == HOT_CP_MODEL.cp_max


def test_cp_off_optimal_pitch_matches_hand_calculation() -> None:
    """At optimal TSR, pitch 0 deg: Cp = Cp_max * (1 - pitch_scale * 1^2) = 0.45 * 179/180."""
    cp = cp_surface(tsr=HOT_CP_MODEL.opt_tsr, pitch=0.0, params=HOT_CP_MODEL)
    assert cp == pytest.approx(0.4475)


def test_cp_surface_is_vectorised() -> None:
    """Array inputs return an array of the matching Cp values."""
    cp = cp_surface(tsr=[7.0, 7.0], pitch=[-1.0, 0.0], params=HOT_CP_MODEL)
    assert cp.shape == (2,)
    np.testing.assert_allclose(cp, [0.45, 0.4475])


def test_cp_clamps_to_zero_far_from_peak() -> None:
    """Far from the optimal TSR the quadratic falloff is clamped to zero, never negative."""
    cp = cp_surface(tsr=20.0, pitch=HOT_CP_MODEL.opt_pitch, params=HOT_CP_MODEL)
    assert cp == 0.0


def test_region2_fraction_is_quarter_at_sigmoid_midpoint() -> None:
    """At the sigmoid midpoint (2000 kW) the squared sigmoid is exactly 0.25."""
    assert region2_fraction(2000.0) == pytest.approx(0.25)


def test_region2_fraction_near_one_deep_in_region2() -> None:
    """Deep in region 2 (low power) almost all of the period is region 2."""
    assert region2_fraction(500.0) > 0.99


def test_region2_fraction_tails_off_near_rated() -> None:
    """Approaching rated power (2300 kW) the region-2 fraction tails toward zero."""
    assert region2_fraction(2300.0) < 0.01


def test_region2_fraction_decreases_monotonically_with_power() -> None:
    """The region-2 fraction is monotonically non-increasing in power."""
    powers = np.arange(0.0, 2400.0, 50.0)
    fractions = region2_fraction(powers)
    assert np.all(np.diff(fractions) <= 0.0)


def test_power_from_cp_change_applies_region2_weighted_ratio() -> None:
    """At 2000 kW (region-2 fraction 0.25) a +10% Cp gives +2.5% power: 2000 -> 2050."""
    new = power_from_cp_change(2000.0, cp_ratio=1.10, rated_power_kw=2300.0)
    assert new == pytest.approx(2050.0)


def test_power_from_cp_change_clips_at_rated() -> None:
    """A large Cp ratio near rated cannot push power above the rated clip."""
    new = power_from_cp_change(2280.0, cp_ratio=5.0, rated_power_kw=2300.0)
    assert new == pytest.approx(2300.0)


def test_power_from_cp_change_reduces_power_for_negative_delta() -> None:
    """A Cp ratio below 1 reduces power roughly proportionally deep in region 2."""
    new = power_from_cp_change(800.0, cp_ratio=0.90, rated_power_kw=2300.0)
    assert new == pytest.approx(800.0 * 0.90, rel=1e-3)


def test_power_from_cp_change_is_vectorised() -> None:
    """Array inputs return an array of modified powers."""
    new = power_from_cp_change(np.array([500.0, 2000.0]), cp_ratio=1.10, rated_power_kw=2300.0)
    assert new.shape == (2,)
    assert new[1] == pytest.approx(2050.0)


def test_rpm_from_power_matches_curve_anchor() -> None:
    """The ported rpm-vs-power curve passes through its known knot (1000 kW -> 1392 rpm)."""
    assert rpm_from_power(1000.0) == pytest.approx(1392.0)


def test_rpm_unchanged_when_power_unchanged() -> None:
    """If power does not change, rpm is unchanged."""
    new_rpm = rpm_from_power_change(baseline_rpm=1392.0, baseline_power_kw=1000.0, new_power_kw=1000.0)
    assert new_rpm == pytest.approx(1392.0)


def test_rpm_increases_with_power() -> None:
    """Higher power scales rpm up by the curve ratio."""
    new_rpm = rpm_from_power_change(baseline_rpm=1392.0, baseline_power_kw=1000.0, new_power_kw=1100.0)
    assert new_rpm == pytest.approx(rpm_from_power(1100.0))
    assert new_rpm > 1392.0


def test_rpm_unchanged_when_new_power_not_positive() -> None:
    """A non-positive new power leaves rpm at its baseline value."""
    new_rpm = rpm_from_power_change(baseline_rpm=1392.0, baseline_power_kw=1000.0, new_power_kw=0.0)
    assert new_rpm == pytest.approx(1392.0)


def test_cpcore_defaults_to_hot_model() -> None:
    """A default CpCore uses the HoT Cp model and 2300 kW rated power."""
    core = CpCore()
    assert core.cp_params == HOT_CP_MODEL
    assert core.rated_power_kw == pytest.approx(2300.0)


def test_cpcore_apply_cp_ratio_uses_its_rated_power() -> None:
    """CpCore.apply_cp_ratio clips at the core's configured rated power."""
    core = CpCore(rated_power_kw=2300.0)
    assert core.apply_cp_ratio(2000.0, cp_ratio=1.10) == pytest.approx(2050.0)
    assert core.apply_cp_ratio(2280.0, cp_ratio=5.0) == pytest.approx(2300.0)


def test_cpcore_rpm_after_tracks_power() -> None:
    """CpCore.rpm_after scales rpm by the operating-curve ratio."""
    core = CpCore()
    new_rpm = core.rpm_after(baseline_rpm=1392.0, baseline_power_kw=1000.0, new_power_kw=1100.0)
    assert new_rpm == pytest.approx(rpm_from_power(1100.0))
