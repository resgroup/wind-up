"""Recovery / correctness tests for ``PowerModelMethod`` (the §8-analog bias guard).

Builds a toy dataset where the test turbine's power is a known function of the references plus a
known multiplicative uplift in the upgraded window, and asserts the counterfactual power model
recovers the uplift — for both prepost and toggle. Also checks the reference-only rule end-to-end
(a leak-bait test-turbine column cannot change the estimate).
"""

from __future__ import annotations

import dataclasses
import logging
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.power_model import PowerModelMethod
from benchmarking.baselines.power_model.features import ERA5_FEATURES, QUALIFIER
from benchmarking.baselines.power_model.method import (
    _DEFAULT_SCREEN_MIN_CAMPAIGN_DAYS,
    _TIME_DECAY_CAMPAIGN_MULTIPLE,
    REVERSAL_COLUMNS,
    _clip_predictions,
    _combine_uplift,
    _implied_shrinkage,
    _reference_input,
    reference_measurement_dirname,
    reference_overall_uplift,
)
from benchmarking.baselines.power_model.screening import ScreenResult
from benchmarking.diagnostics import stages
from benchmarking.diagnostics.context import era5_source_label
from benchmarking.harness.conditions import CONDITION_BINS
from benchmarking.harness.context import CampaignContext
from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.harness.operating_state import VALID_UPLIFT_COL
from benchmarking.harness.record_selection import EXCLUDED, RARE_CONFIGURATION, UNPAIRED, USABLE, ChangedInterval
from benchmarking.harness.scoring import _merge_diagnostics
from benchmarking.harness.toggle import resolve_toggle
from benchmarking.harness.waking import UPWIND_COL, WAKING_SCENARIO_COL
from benchmarking.synthetic import ColumnSchema, ToggleSchedule

if TYPE_CHECKING:
    from pathlib import Path


_TURBINE = "TurbineName"
_POWER = "wtc_ActPower_mean"
_AVAIL = "wtc_ScReToOp_timeon"
_WS = "wtc_AcWindSp_mean"
_WS_SD = "wtc_AcWindSp_stddev"
_POWER_MAX = "wtc_ActPower_max"
_POWER_MIN = "wtc_ActPower_min"
_POWER_SD = "wtc_ActPower_stddev"
_YAW = "wtc_NacelPos_mean"
_NORTHED_YAW = f"northed_{_YAW}"
_COLUMNS = ColumnSchema(
    turbine=_TURBINE,
    active_power=_POWER,
    active_power_min=_POWER_MIN,
    wind_speed=_WS,
    wind_speed_sd=_WS_SD,
    gen_rpm="wtc_GenRpm_mean",
    availability=_AVAIL,
    nacelle_position=_YAW,
)

# Per-turbine north miscalibration the northed column removes.
_YAW_OFFSETS = {"T1": 0.0, "R1": 7.0, "R2": -5.0, "R3": 3.0}

# Small/fast LightGBM so the toy data (a few thousand rows) is fit well. One thread per fit: the
# toy frames are too small to gain from LightGBM's threading, and the test run is parallel.
_FAST_PARAMS = {
    "n_estimators": 60,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_child_samples": 20,
    "n_jobs": 1,
}


def _toy_scada(n: int, *, uplift: float, treated: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Long SCADA: references drive the test power; the upgrade scales test power on ``treated`` rows.

    Weather is i.i.d. across the whole window so baseline and upgraded share a distribution (this
    isolates the estimator mechanics from the prepost confounding that the real study probes).
    """
    idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC", name="timestamp")
    rng = np.random.default_rng(seed)
    r1 = rng.normal(900, 150, n)
    r2 = rng.normal(850, 150, n)
    r3 = rng.normal(800, 150, n)
    base_test = 0.4 * r1 + 0.35 * r2 + 0.25 * r3 + rng.normal(0, 15, n)
    test_power = np.where(treated, base_test * (1.0 + uplift), base_test)
    # One site-wide direction every turbine sees, so the direction features are plausible rather
    # than noise; each turbine reports it through its own north miscalibration.
    wind_direction = 180.0 + 60.0 * np.sin(2.0 * np.pi * np.arange(n) / 1000.0)
    frames = {
        "T1": test_power,
        "R1": r1,
        "R2": r2,
        "R3": r3,
    }
    parts = [
        pd.DataFrame(
            {
                _TURBINE: name,
                _POWER: power,
                _AVAIL: 600.0,
                _WS: power / 100.0,
                _WS_SD: power / 1000.0,
                # active-power companion statistics (Issue 11 reference_stat_cols candidates)
                _POWER_MAX: power * 1.15,
                _POWER_MIN: power * 0.85,
                _POWER_SD: np.abs(power) / 20.0,
                _YAW: (wind_direction + _YAW_OFFSETS[name]) % 360.0,
                _NORTHED_YAW: wind_direction % 360.0,
            },
            index=idx,
        )
        for name, power in frames.items()
    ]
    return pd.concat(parts)


class TestReversalCorrection:
    """The reversal correction (train-baseline-predict-upgraded contrasted with its reverse).

    The reverse fit and the ``_combine_uplift`` formula are already exercised by the conditional
    path; these check the wiring at the headline: the field defaults to the shipped forward ratio,
    the reversal correction still recovers a known uplift and reads near zero on a placebo (the
    common shrinkage cancels, ``u`` survives), and a bad value is rejected.
    """

    def test_defaults_to_forward(self) -> None:
        method = PowerModelMethod(columns=_COLUMNS, baseline_rated_power_kw=2300.0)
        assert method.headline_estimator == "forward"

    def test_recovers_known_uplift_prepost(self) -> None:
        mi, _ = _prepost_case(uplift=0.05)
        out = _fundamentals_method(model_params=_FAST_PARAMS, headline_estimator="reversal").estimate(mi)
        assert out.p50_overall == pytest.approx(0.05, abs=0.02)

    def test_placebo_reads_near_zero(self) -> None:
        mi, _ = _prepost_case(uplift=0.0)
        out = _fundamentals_method(model_params=_FAST_PARAMS, headline_estimator="reversal").estimate(mi)
        assert out.p50_overall == pytest.approx(0.0, abs=0.02)

    def test_invalid_headline_estimator_raises(self) -> None:
        mi, _ = _prepost_case(n=200)
        method = _fundamentals_method(model_params=_FAST_PARAMS, headline_estimator="sideways")
        with pytest.raises(ValueError, match="headline_estimator"):
            method.estimate(mi)


class TestRecovery:
    def test_recovers_known_uplift_prepost(self) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            conditions=(),  # overall-only; the conditional path needs ERA5 (not supplied here)
            model_params=_FAST_PARAMS,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        out = method.estimate(mi)
        assert out.p50_overall == pytest.approx(0.05, abs=0.02)

    def test_recovers_known_uplift_toggle(self) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        schedule = ToggleSchedule(period=pd.Timedelta(hours=4))
        treated = np.asarray((((idx - idx.min()) // (schedule.period / 2)).astype(int) % 2) == 1)
        scada = _toy_scada(n, uplift=0.04, treated=treated)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            conditions=(),  # overall-only; the conditional path needs ERA5 (not supplied here)
            model_params=_FAST_PARAMS,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE)
        out = method.estimate(mi)
        assert out.p50_overall == pytest.approx(0.04, abs=0.02)

    def test_placebo_reads_near_zero(self) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.0, treated=treated)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            conditions=(),  # overall-only; the conditional path needs ERA5 (not supplied here)
            model_params=_FAST_PARAMS,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        out = method.estimate(mi)
        assert out.p50_overall == pytest.approx(0.0, abs=0.02)


def _prepost_case(n: int = 4000, *, uplift: float = 0.05) -> tuple[MethodInput, pd.Timestamp]:
    """A toy prepost MethodInput with a known uplift, for the model-fundamentals config trials."""
    idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
    changeover = idx[n // 2]
    treated = np.asarray(idx >= changeover)
    scada = _toy_scada(n, uplift=uplift, treated=treated)
    mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
    return mi, changeover


def _fundamentals_method(**overrides: object) -> PowerModelMethod:
    kwargs: dict[str, object] = {
        "columns": _COLUMNS,
        "baseline_rated_power_kw": 2300.0,
        "conditions": (),
        **overrides,
    }
    return PowerModelMethod(**kwargs)  # type: ignore[arg-type]


class TestModelFundamentals:
    """The campaign-proximity weighting, off by default, and the toggle campaign mask."""

    def test_time_decay_weights_recover_uplift(self) -> None:
        mi, _ = _prepost_case()
        out = _fundamentals_method(
            model_params=_FAST_PARAMS, adaptive_time_decay=False, time_decay_half_life_days=30.0
        ).estimate(mi)
        assert out.p50_overall == pytest.approx(0.05, abs=0.02)

    def test_time_decay_weight_values(self) -> None:
        # the expert fixed-half-life path (adaptive_time_decay=False)
        method = _fundamentals_method(adaptive_time_decay=False, time_decay_half_life_days=10.0)
        index = pd.date_range("2019-01-01", periods=5, freq="10D", tz="UTC")
        # campaign interval [index[2], index[3]]: inside weighs 1, outside decays both ways
        weights = method._time_decay_weights(index, campaign_start=index[2], campaign_end=index[3])  # noqa: SLF001
        np.testing.assert_allclose(weights, [0.25, 0.5, 1.0, 1.0, 0.5])
        no_decay = _fundamentals_method(adaptive_time_decay=False, time_decay_half_life_days=None)
        assert no_decay._time_decay_weights(index, campaign_start=index[2], campaign_end=index[3]) is None  # noqa: SLF001

    def test_the_weighting_is_off_unless_asked_for(self) -> None:
        method = _fundamentals_method()
        assert method.adaptive_time_decay is False
        assert method.time_decay_half_life_days is None
        index = pd.date_range("2019-01-01", periods=5, freq="10D", tz="UTC")
        assert method._time_decay_weights(index, campaign_start=index[2], campaign_end=index[3]) is None  # noqa: SLF001

    def test_adaptive_time_decay_half_life_scales_with_campaign_duration(self) -> None:
        # opted in: half_life = k * campaign_duration_days, in both modes
        method = _fundamentals_method(adaptive_time_decay=True)
        start = pd.Timestamp("2019-04-01", tz="UTC")
        for duration_days in (30.0, 90.0, 365.0):
            end = start + pd.Timedelta(days=duration_days)
            hl = method._effective_half_life(campaign_start=start, campaign_end=end)  # noqa: SLF001
            assert hl == pytest.approx(_TIME_DECAY_CAMPAIGN_MULTIPLE * duration_days)

    def test_adaptive_time_decay_weight_values(self) -> None:
        method = _fundamentals_method(adaptive_time_decay=True)
        index = pd.date_range("2019-01-01", periods=5, freq="10D", tz="UTC")
        start, end = index[2], index[3]  # 10-day campaign -> half_life = k * 10
        hl = _TIME_DECAY_CAMPAIGN_MULTIPLE * 10.0
        days_outside = np.array([20.0, 10.0, 0.0, 0.0, 10.0])  # distance to [index[2], index[3]]
        expected = 0.5 ** (days_outside / hl)
        weights = method._time_decay_weights(index, campaign_start=start, campaign_end=end)  # noqa: SLF001
        np.testing.assert_allclose(weights, expected)

    def test_effective_half_life_fixed_and_off(self) -> None:
        start = pd.Timestamp("2019-04-01", tz="UTC")
        end = start + pd.Timedelta(days=90)
        fixed = _fundamentals_method(adaptive_time_decay=False, time_decay_half_life_days=42.0)
        assert fixed._effective_half_life(campaign_start=start, campaign_end=end) == 42.0  # noqa: SLF001
        off = _fundamentals_method(adaptive_time_decay=False, time_decay_half_life_days=None)
        assert off._effective_half_life(campaign_start=start, campaign_end=end) is None  # noqa: SLF001

    def test_adaptive_with_explicit_half_life_conflict_raises(self) -> None:
        mi, _ = _prepost_case(n=200)
        with pytest.raises(ValueError, match="adaptive_time_decay"):
            _fundamentals_method(adaptive_time_decay=True, time_decay_half_life_days=90.0).estimate(mi)

    def test_time_decay_half_life_must_be_positive(self) -> None:
        mi, _ = _prepost_case(n=200)
        with pytest.raises(ValueError, match="must be positive"):
            _fundamentals_method(adaptive_time_decay=False, time_decay_half_life_days=0.0).estimate(mi)

    def test_started_toggle_baselines_split_pre_campaign_from_off_blocks(self) -> None:
        # The old ``_campaign_mask`` folded into the shared ``resolve_toggle``: the strict
        # campaign_baseline (the conditional matching's off rows) excludes pre-campaign, while the
        # lenient training_baseline (the headline fit's rows) includes them. period=20D, half=10D.
        index = pd.date_range("2019-01-01", periods=4, freq="10D", tz="UTC")
        rows = resolve_toggle(ToggleSchedule(period=pd.Timedelta(days=20), start=index[2]), index)
        pre = np.asarray(index < index[2])  # index[0], index[1]
        assert not rows.campaign_baseline[pre].any()  # off-only baseline drops pre-campaign
        assert rows.training_baseline[pre].all()  # fitting baseline keeps pre-campaign
        # prepost: both baselines are exactly the pre-changeover rows (no pre-campaign concept).
        prepost = resolve_toggle(pd.Timestamp(index[2]), index)
        np.testing.assert_array_equal(prepost.campaign_baseline, ~prepost.upgraded)
        np.testing.assert_array_equal(prepost.training_baseline, ~prepost.upgraded)

    def test_toggle_all_data_with_conditional_recovers_uplift(self) -> None:
        # A toggle whose headline fit trains on the pre-campaign baseline too (the adaptive default,
        # no campaign-only restriction): the conditional step still matches within the campaign only.
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        start = idx[n // 2]  # first half pre-campaign baseline, second half interleaved toggle
        schedule = ToggleSchedule(period=pd.Timedelta(hours=4), start=start)
        within = (((idx - start) // (schedule.period / 2)).astype(int) % 2) == 1
        treated = np.asarray((idx >= start) & within)
        scada = _toy_scada(n, uplift=0.04, treated=treated)
        method = _fundamentals_method(
            model_params=_FAST_PARAMS,
            conditions=PowerModelMethod.conditions,
            era5_hourly_df=_toy_era5(idx),
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE)
        out = method.estimate(mi)
        assert out.p50_overall == pytest.approx(0.04, abs=0.02)
        assert out.p50_by_condition is not None


class TestReferenceOnly:
    def test_leak_bait_test_column_does_not_change_estimate(self) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            conditions=(),  # overall-only; the conditional path needs ERA5 (not supplied here)
            model_params=_FAST_PARAMS,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        baseline = method.estimate(mi).p50_overall

        # Add a column that perfectly reveals the (post-treatment) test power on the test turbine.
        leaked = scada.copy()
        leaked["wtc_NacWdSp_mean"] = np.where(leaked[_TURBINE] == "T1", leaked[_POWER], np.nan)
        mi_leak = MethodInput(
            scada_df=leaked, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE
        )
        with_leak = method.estimate(mi_leak).p50_overall
        # the reference-only builder ignores test-turbine columns, so the estimate is unchanged
        assert with_leak == pytest.approx(baseline, abs=1e-9)


class TestClipPredictions:
    def test_out_of_range_pulled_to_bounds_in_range_untouched(self) -> None:
        # lower = min(0, 0) = 0; upper = max(2300, 1000) = 2300
        pred = np.array([-50.0, 500.0, 1500.0, 2400.0])
        clipped = _clip_predictions(pred, y_train=np.array([0.0, 500.0, 1000.0]), rated_power_kw=2300.0)
        assert clipped.tolist() == [0.0, 500.0, 1500.0, 2300.0]

    def test_upper_bound_is_max_of_rated_and_train(self) -> None:
        # an observed outcome above rated raises the ceiling above rated_power_kw
        clipped = _clip_predictions(np.array([3000.0]), y_train=np.array([0.0, 2500.0]), rated_power_kw=2300.0)
        assert clipped.tolist() == [2500.0]

    def test_floors_at_zero_for_nonnegative_training_data(self) -> None:
        clipped = _clip_predictions(np.array([-5.0]), y_train=np.array([10.0, 100.0]), rated_power_kw=2300.0)
        assert clipped.tolist() == [0.0]

    def test_lower_bound_allows_negative_training_data(self) -> None:
        # min(0, min(y_train)) never clips a genuinely-negative observation up to 0
        clipped = _clip_predictions(np.array([-100.0]), y_train=np.array([-30.0, 100.0]), rated_power_kw=2300.0)
        assert clipped.tolist() == [-30.0]


class TestConditionsSelection:
    """``conditions`` selects which axes are reported; ``()`` skips the conditional step entirely."""

    def _run(self, **kwargs: object) -> MethodInput:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            era5_hourly_df=_toy_era5(idx),
            model_params=_FAST_PARAMS,
            **kwargs,  # type: ignore[arg-type]
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        return method.estimate(mi)  # type: ignore[return-value]

    def test_power_only_reports_power_alone(self) -> None:
        out = self._run(conditions=("power",))
        assert set(out.p50_by_condition["condition"]) == {"power"}

    def test_era5_ws_only_reports_era5_ws_alone(self) -> None:
        out = self._run(conditions=("era5_ws",))
        assert set(out.p50_by_condition["condition"]) == {"era5_ws"}

    def test_default_reports_power_and_era5_wind_speed(self) -> None:
        out = self._run()
        assert set(out.p50_by_condition["condition"]) == {"power", "era5_ws"}

    def test_the_subject_turbines_own_signals_are_not_condition_axes(self) -> None:
        for axis in ("ws", "ti"):
            with pytest.raises(ValueError, match="does not support"):
                PowerModelMethod(columns=_COLUMNS, baseline_rated_power_kw=2300.0, conditions=(axis,))

    def test_empty_conditions_skips_the_conditional_step(self) -> None:
        out = self._run(conditions=())
        assert out.p50_by_condition is None

    def test_unknown_condition_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown condition"):
            PowerModelMethod(columns=_COLUMNS, baseline_rated_power_kw=2300.0, conditions=("bogus",))


class TestConditionalWithoutItsMatchingColumns:
    """A reanalysis delivery without a matching column stops the run."""

    def _case(self, **overrides: object) -> tuple[PowerModelMethod, MethodInput]:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        scada = _toy_scada(n, uplift=0.05, treated=np.asarray(idx >= changeover))
        era5 = _toy_era5(idx).drop(columns=["wind_gusts_10m"])  # a matching axis never arrived
        kwargs: dict[str, object] = {
            "columns": _COLUMNS,
            "baseline_rated_power_kw": 2300.0,
            "era5_hourly_df": era5,
            "model_params": _FAST_PARAMS,
            **overrides,
        }
        method = PowerModelMethod(**kwargs)  # type: ignore[arg-type]
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        return method, mi

    def test_a_missing_matching_column_raises(self) -> None:
        method, mi = self._case()
        with pytest.raises(ValueError, match="wind_gusts_10m"):
            method.estimate(mi)

    def test_an_explicitly_named_missing_column_raises(self) -> None:
        method, mi = self._case(matching_vars=("wind_speed_100m", "wind_gusts_10m"))
        with pytest.raises(ValueError, match="wind_gusts_10m"):
            method.estimate(mi)


class TestConditionalUplift:
    def test_emits_conditional_uplift_by_power_and_era5_wind_speed(self) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)  # now includes _WS_SD
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            era5_hourly_df=_toy_era5(idx),  # conditional uplift (default on) matches on ERA5 weather
            model_params=_FAST_PARAMS,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        out = method.estimate(mi)
        bc = out.p50_by_condition
        assert list(bc.columns) == ["condition", "condition_bin", "p50_uplift", "sigma_uplift"]
        assert set(bc["condition"]) == {"power", "era5_ws"}
        # power uses the 6 fraction-of-rated bins; era5_ws the shared 2 m/s wind-speed bins
        assert (bc["condition"] == "power").sum() == 6
        assert (bc["condition"] == "era5_ws").sum() == len(CONDITION_BINS["era5_ws"]) - 1
        # Issue 14: imputation fills every uncovered bin, so the reported per-bin estimate is never NaN
        # (a bare NaN would let abstention game the conditional score, which drops non-finite errors).
        assert bc["p50_uplift"].notna().all()

    def test_conditional_csv_carries_covered_flag(self, tmp_path: Path) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            era5_hourly_df=_toy_era5(idx),
            model_params=_FAST_PARAMS,
            out_dir=tmp_path,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        method.estimate(mi)
        files = sorted(tmp_path.rglob("*_conditional_by_bin_*.csv"))
        assert files, "no conditional_by_bin CSV written"
        per_bin = pd.read_csv(files[0])
        assert "covered" in per_bin.columns
        # don't assert the CSV round-trip dtype (read_csv bool inference is version-dependent); the
        # column's meaning is what matters — at least some bins measured in well-populated toy data.
        assert per_bin["covered"].any()
        assert per_bin["p50_uplift"].notna().all()  # measured-or-imputed, never bare NaN

    def test_count_floor_marks_sparse_bins_uncovered(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # force every bin below an impossibly-high floor (string target avoids a function-level import)
        monkeypatch.setattr("benchmarking.baselines.power_model.method._MIN_BIN_MATCHED_HOURS", 1e9)
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            era5_hourly_df=_toy_era5(idx),
            model_params=_FAST_PARAMS,
            out_dir=tmp_path,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        method.estimate(mi)
        per_bin = pd.read_csv(sorted(tmp_path.rglob("*_conditional_by_bin_*.csv"))[0])
        assert (~per_bin["covered"]).all()  # nothing clears an impossibly-high floor
        assert per_bin["p50_uplift"].notna().all()  # all imputed, still never NaN


def _toy_era5(scada_idx: pd.DatetimeIndex, *, seed: int = 0) -> pd.DataFrame:
    """Hourly ERA5 covering the toy window with the three matching columns, i.i.d. over the window.

    Weather is drawn independently per hour, so the baseline and upgraded periods share a distribution
    and CEM finds well-populated two-sided cells. Values sit in modest ranges so the default matching
    bin edges give a handful of populated cells rather than one row each.
    """
    hours = pd.date_range(
        scada_idx.min().floor("h") - pd.Timedelta(hours=2), scada_idx.max().ceil("h") + pd.Timedelta(hours=2), freq="h"
    )
    rng = np.random.default_rng(seed + 7)
    ws = rng.uniform(4.0, 12.0, len(hours))
    return pd.DataFrame(
        {
            "wind_speed_100m": ws,
            "wind_gusts_10m": ws * 1.4 + rng.uniform(0.0, 2.0, len(hours)),
            "wind_direction_100m": rng.uniform(200.0, 260.0, len(hours)),
            # extra raw columns so the Issue 9 derivations have their inputs
            "wind_speed_10m": ws * 0.75,
            "wind_direction_10m": rng.uniform(190.0, 250.0, len(hours)),
            "temperature_2m": rng.uniform(0.0, 15.0, len(hours)),
            "surface_pressure": rng.uniform(980.0, 1030.0, len(hours)),
            "relative_humidity_2m": rng.uniform(50.0, 100.0, len(hours)),
        },
        index=hours,
    )


class TestFeatureConfig:
    """The feature config: columns reach the model and estimates stay sound."""

    def _prepost_mi(self, n: int = 4000, *, uplift: float = 0.05) -> MethodInput:
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        treated = np.asarray(idx >= idx[n // 2])
        scada = _toy_scada(n, uplift=uplift, treated=treated)
        return MethodInput(
            scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(idx[n // 2]), turbine_col=_TURBINE
        )

    def _fitted_feature_names(self, out_dir: Path) -> set[str]:
        files = sorted(out_dir.rglob("*_feature_importance_*.csv"))
        assert files, f"no feature-importance CSV under {out_dir}"
        return set(pd.read_csv(files[-1])["feature"])

    def test_reference_stat_cols_reach_model_and_recovery_holds(self, tmp_path: Path) -> None:
        mi = self._prepost_mi()
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            conditions=(),
            model_params=_FAST_PARAMS,
            reference_stat_cols=(_POWER_MAX, _POWER_MIN, _POWER_SD),
            out_dir=tmp_path,
        )
        out = method.estimate(mi)
        assert out.p50_overall == pytest.approx(0.05, abs=0.02)
        fitted = self._fitted_feature_names(tmp_path)
        assert {f"{_POWER_SD} @ R1", f"{_POWER_MAX} @ R2", f"{_POWER_MIN} @ R3"} <= fitted
        assert not any(name.endswith(" @ T1") for name in fitted)

    def test_the_era5_features_are_the_curated_list(self, tmp_path: Path) -> None:
        mi = self._prepost_mi()
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            era5_hourly_df=_toy_era5(pd.DatetimeIndex(mi.scada_df.index)),
            conditions=(),
            model_params=_FAST_PARAMS,
            out_dir=tmp_path,
        )
        out = method.estimate(mi)
        assert out.p50_overall == pytest.approx(0.05, abs=0.02)
        fitted = self._fitted_feature_names(tmp_path)
        era5 = {name.removesuffix(" @ ERA5") for name in fitted if name.endswith(" @ ERA5")}
        assert era5 == set(ERA5_FEATURES)

    def test_power_and_direction_are_all_a_reference_contributes(self, tmp_path: Path) -> None:
        mi = self._prepost_mi()
        method = PowerModelMethod(
            columns=_COLUMNS, baseline_rated_power_kw=2300.0, conditions=(), model_params=_FAST_PARAMS, out_dir=tmp_path
        )
        out = method.estimate(mi)
        assert out.p50_overall == pytest.approx(0.05, abs=0.02)
        fitted = self._fitted_feature_names(tmp_path)
        assert sorted(name for name in fitted if name.endswith(" @ R1")) == sorted(
            [f"{_POWER} @ R1", f"{_NORTHED_YAW}_sin @ R1", f"{_NORTHED_YAW}_cos @ R1"]
        )


class TestPromotedDefaults:
    def test_effective_lgbm_params_include_tuned_min_child_samples(self) -> None:
        m = PowerModelMethod(columns=_COLUMNS, baseline_rated_power_kw=2300.0)
        assert m._make_model().get_params()["min_child_samples"] == 50  # noqa: SLF001

    def test_explicit_model_params_override_the_tuned_default(self) -> None:
        m = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            model_params={"min_child_samples": 123},
        )
        assert m._make_model().get_params()["min_child_samples"] == 123  # noqa: SLF001


# The re-level is now the pinned-imputed ``relevel_conditional`` in power_model.conditional; its unit
# coverage lives in test_power_model_conditional.py (TestRelevelConditionalPinned). Kept here only:
# the direction-combine helpers, still in method.py.


class TestCombineDirections:
    def test_recovers_uplift_and_shrinkage_from_ratios(self) -> None:
        # construct the two directions from a known uplift u and shrinkage s:
        #   1 + r_fwd = (1 + u) / s ;  1 + r_rev = 1 / (s (1 + u))
        u, s = 0.06, 0.85
        r_fwd = (1 + u) / s - 1
        r_rev = 1 / (s * (1 + u)) - 1
        assert _combine_uplift(np.array([r_fwd]), np.array([r_rev]))[0] == pytest.approx(u)
        assert _implied_shrinkage(np.array([r_fwd]), np.array([r_rev]))[0] == pytest.approx(s)

    def test_nonpositive_ratio_gives_nan(self) -> None:
        # (1 + r) <= 0 on either side is unphysical -> NaN, not a complex/blown-up number
        out = _combine_uplift(np.array([-1.5, 0.1]), np.array([0.1, -2.0]))
        assert np.isnan(out).tolist() == [True, True]


class TestConditional:
    def test_requires_era5(self) -> None:
        n = 300
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        treated = np.asarray(idx >= idx[n // 2])
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            model_params=_FAST_PARAMS,  # conditional on by default, but no era5_hourly_df -> must raise
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(idx[n // 2]), turbine_col=_TURBINE)
        with pytest.raises(ValueError, match="ERA5"):
            method.estimate(mi)

    def test_recovers_known_uplift_through_two_directions(self) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            era5_hourly_df=_toy_era5(idx),
            model_params=_FAST_PARAMS,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        out = method.estimate(mi)
        # matched i.i.d. weather -> shrinkage ~1, forward-only overall recovers the true uplift
        assert out.p50_overall == pytest.approx(0.05, abs=0.02)
        assert set(out.p50_by_condition["condition"]) == {"power", "era5_ws"}
        assert list(out.p50_by_condition.columns) == ["condition", "condition_bin", "p50_uplift", "sigma_uplift"]

    def test_overall_matches_conditional_off_and_bins_aggregate_to_it(self, tmp_path: Path) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.03, treated=treated)
        config = {
            "columns": _COLUMNS,
            "baseline_rated_power_kw": 2300.0,
            "era5_hourly_df": _toy_era5(idx),  # same features both ways, so the headline is comparable
            "model_params": _FAST_PARAMS,
        }
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        overall_only = PowerModelMethod(**config, conditions=()).estimate(mi).p50_overall
        method = PowerModelMethod(**config, out_dir=tmp_path)  # conditional on by default
        out = method.estimate(mi)

        # 1. the headline is the single full-data fit; computing the conditional step leaves it unchanged
        assert out.p50_overall == pytest.approx(overall_only, rel=1e-9)
        # 2. self-consistency: each of the ws and ti decompositions energy-aggregates back to that overall
        run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
        by_bin = pd.read_csv(next((run_dir / "conditional").glob("*_conditional_by_bin_*.csv")))
        for _cond, g in by_bin.groupby("condition"):
            good = g[np.isfinite(g["p50_uplift"])]
            agg = good["sum_actual"].sum() / (good["sum_actual"] / (1.0 + good["p50_uplift"])).sum()
            assert agg == pytest.approx(1.0 + out.p50_overall, rel=1e-6)

    def test_writes_shrinkage_and_cem_balance_diagnostics(self, tmp_path: Path) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            era5_hourly_df=_toy_era5(idx),
            out_dir=tmp_path,
            model_params=_FAST_PARAMS,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        method.estimate(mi)

        run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(run_dirs) == 1
        conditional_dir = run_dirs[0] / "conditional"
        overall = pd.read_csv(next(conditional_dir.glob("*_conditional_overall_*.csv")))
        by_bin = pd.read_csv(next(conditional_dir.glob("*_conditional_by_bin_*.csv")))
        balance = pd.read_csv(next(conditional_dir.glob("*_cem_balance_*.csv")))
        assert next(conditional_dir.glob("*_cem_cells_*.csv"), None) is not None
        # implied shrinkage s is surfaced overall and per-bin; matched weather -> s ~ 1
        assert "implied_shrinkage" in overall.columns
        assert overall["implied_shrinkage"].iloc[0] == pytest.approx(1.0, abs=0.1)
        assert {"condition", "condition_bin", "r_fwd", "r_rev", "implied_shrinkage", "p50_uplift"} <= set(
            by_bin.columns
        )
        # CEM balance carries the coverage numbers
        assert {"n_matched_per_side", "retained_fraction_baseline", "n_cells_one_sided"} <= set(balance.columns)


def _shrinkage_scada(n: int, *, uplift: float, treated: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Attenuation-shrinkage toy: references are *noisy* proxies of a steep power curve.

    Because the references (the model's features) are noisy measurements of the same weather-driven
    power, the counterfactual model learns an attenuated conditional mean — it over-predicts where power
    is low and under-predicts where it is high (multiplicative shrinkage). The test wind speed is the
    *clean* driver, so binning by it exposes that compression as a spurious per-bin uplift tilt even at
    the placebo (the shrinkage mechanism). Weather is i.i.d. across the window, so baseline and upgraded are
    distribution-matched and the shrinkage is common to both cross-predict directions -> it cancels.
    """
    idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC", name="timestamp")
    rng = np.random.default_rng(seed)
    w = rng.uniform(3.0, 12.0, n)  # latent wind speed, i.i.d. -> matched across periods
    curve = 20.0 * w**2  # steep power curve (≈180..2880 kW), so per-ws-bin compression is visible
    test_power = np.where(treated, curve * (1.0 + uplift), curve) + rng.normal(0.0, 20.0, n)
    wind_direction = 180.0 + 60.0 * np.sin(2.0 * np.pi * np.arange(n) / 1000.0)
    parts = [
        pd.DataFrame(
            {
                _TURBINE: "T1",
                _POWER: test_power,
                _POWER_MIN: test_power * 0.85,
                _AVAIL: 600.0,
                _WS: w,
                _WS_SD: 0.05 * w,
                _YAW: (wind_direction + _YAW_OFFSETS["T1"]) % 360.0,
                _NORTHED_YAW: wind_direction % 360.0,
            },
            index=idx,
        )
    ]
    for i in range(1, 4):
        ref_power = curve + rng.normal(0.0, 500.0, n)  # noisy proxy of the curve -> attenuation shrinkage
        parts.append(
            pd.DataFrame(
                {
                    _TURBINE: f"R{i}",
                    _POWER: ref_power,
                    _POWER_MIN: ref_power * 0.85,
                    _AVAIL: 600.0,
                    _WS: w,
                    _WS_SD: 0.05 * w,
                    _YAW: (wind_direction + _YAW_OFFSETS[f"R{i}"]) % 360.0,
                    _NORTHED_YAW: wind_direction % 360.0,
                },
                index=idx,
            )
        )
    return pd.concat(parts)


def _ws_bin_bias(by_condition: pd.DataFrame) -> pd.Series:
    """Per-ERA5-wind-speed-bin uplift indexed by bin (truth is 0 at placebo, so the value *is* the bias)."""
    ws = by_condition[by_condition["condition"] == "era5_ws"]
    return ws.set_index("condition_bin")["p50_uplift"]


class TestConditionalRegression:
    def test_conditional_flat_at_shrinkage_placebo(self) -> None:
        # Bias guard (design note §8-analog): on a placebo whose references are noisy proxies of a steep
        # power curve, a single counterfactual fit shrinks and reads a spurious per-ws-bin uplift tilt
        # (the shrinkage mechanism). The two-direction matched conditional cancels that common shrinkage, so the
        # (default) conditional uplift must read ~flat-zero in every bin against the flat-0 truth.
        n = 5000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _shrinkage_scada(n, uplift=0.0, treated=treated)  # placebo: true uplift 0 in every bin
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=6000.0,
            era5_hourly_df=_toy_era5(idx),
            model_params=_FAST_PARAMS,
        )
        on_ws = _ws_bin_bias(method.estimate(mi).p50_by_condition)
        bins = on_ws.dropna().index
        on_bias = on_ws.loc[bins].abs().mean()
        # Deterministic (fixed seeds); observed on this data: mean|bias| ≈ 0.0095, max|bias| ≈ 0.020.
        # Thresholds sit ~2.5x above so a version/platform bump won't flake, but a regression in the
        # matched cancellation (which would let the shrinkage tilt back in) will trip them.
        assert on_bias < 0.025
        assert on_ws.loc[bins].abs().max() < 0.05


class TestCampaignContext:
    """The model takes reference membership and row validity from the campaign context."""

    @staticmethod
    def _fixture() -> tuple[pd.DataFrame, pd.Timestamp]:
        n = 2000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        return _toy_scada(n, uplift=0.05, treated=np.asarray(idx >= changeover)), pd.Timestamp(changeover)

    @staticmethod
    def _estimate(scada: pd.DataFrame, **kwargs: object) -> float:
        method = PowerModelMethod(
            columns=_COLUMNS, baseline_rated_power_kw=2300.0, conditions=(), model_params=_FAST_PARAMS
        )
        return method.estimate(MethodInput(scada_df=scada, **kwargs)).p50_overall

    def test_only_offered_references_become_features(self) -> None:
        scada, changeover = self._fixture()
        context = CampaignContext.from_frame(scada, test_wtg="T1", timing=changeover, turbine_col=_TURBINE)
        object.__setattr__(context, "candidate_references", ["R1", "R2"])
        offered = self._estimate(scada, test_wtg="T1", campaign_context=context)

        # Identical to R3 simply not being in the data.
        assert offered == pytest.approx(
            self._estimate(
                scada[scada[_TURBINE] != "R3"], test_wtg="T1", upgrade_timing=changeover, turbine_col=_TURBINE
            )
        )

    def test_an_invalid_record_of_a_power_reference_drops_the_row(self) -> None:
        scada, changeover = self._fixture()
        context = CampaignContext.from_frame(scada, test_wtg="T1", timing=changeover, turbine_col=_TURBINE)
        valid = context.valid_for_uplift.copy()
        valid.loc[valid.index[:200], "R1"] = False
        object.__setattr__(context, "valid_for_uplift", valid)
        with_holes = self._estimate(scada, test_wtg="T1", campaign_context=context)

        cut = scada[scada.index >= valid.index[200]]
        assert with_holes == pytest.approx(
            self._estimate(cut, test_wtg="T1", upgrade_timing=changeover, turbine_col=_TURBINE)
        )

    def test_an_invalid_record_of_a_wake_contributor_keeps_the_row(self) -> None:
        scada, changeover = self._fixture()
        context = CampaignContext.from_frame(scada, test_wtg="T1", timing=changeover, turbine_col=_TURBINE)
        context = dataclasses.replace(context, candidate_references=["R1", "R2"], wake_contributors=["R3"])
        whole = self._estimate(scada, test_wtg="T1", campaign_context=context)
        valid = context.valid_for_uplift.copy()
        valid.loc[valid.index[:200], "R3"] = False
        holed = self._estimate(
            scada, test_wtg="T1", campaign_context=dataclasses.replace(context, valid_for_uplift=valid)
        )
        assert holed == pytest.approx(whole)

    def test_a_changed_turbine_upwind_drops_the_row(self) -> None:
        scada, changeover = self._fixture()
        first = scada.index.unique()[:300]
        upwind = np.where((scada[_TURBINE] == "R2") & scada.index.isin(first), "T1", "none")
        scada = scada.assign(**{UPWIND_COL: upwind})
        context = CampaignContext.from_frame(scada, test_wtg="T1", timing=changeover, turbine_col=_TURBINE)
        assert context.changed_turbines == {"T1": (ChangedInterval(None, None),)}
        dropped = self._estimate(scada, test_wtg="T1", campaign_context=context)

        cut = scada[~scada.index.isin(first)]
        assert dropped == pytest.approx(
            self._estimate(cut, test_wtg="T1", upgrade_timing=changeover, turbine_col=_TURBINE)
        )


def _scada_with_a_stepped_reference(n: int, *, changeover: pd.Timestamp, step: float) -> pd.DataFrame:
    """Toy placebo SCADA in which reference R1 alone changes performance at the changeover.

    The test turbine's power is built from the clean references first, so R1's step is a genuine
    change in R1 and not something the test turbine followed -- exactly the R3 failure mode.
    """
    scada = _toy_scada(n, uplift=0.0, treated=np.zeros(n, dtype=bool))
    is_r1 = scada[_TURBINE] == "R1"
    stepped = is_r1 & (scada.index >= changeover)
    for col in (_POWER, _POWER_MIN, _POWER_MAX):
        scada.loc[stepped, col] = scada.loc[stepped, col] * (1.0 + step)
    return scada


def _with_a_fourth_reference(scada: pd.DataFrame, *, seed: int = 7) -> pd.DataFrame:
    """Add reference R4, R3's rows carrying their own noise.

    Three references are the fewest the screen can rule with, so a three-reference pool cannot show
    one candidate being left out of the screen while the rest are still screened.
    """
    r3 = scada[scada[_TURBINE] == "R3"]
    power = r3[_POWER].to_numpy() + np.random.default_rng(seed).normal(0, 60, len(r3))
    r4 = r3.assign(
        **{
            _TURBINE: "R4",
            _POWER: power,
            _POWER_MAX: power * 1.15,
            _POWER_MIN: power * 0.85,
            _POWER_SD: np.abs(power) / 20.0,
            _WS: power / 100.0,
            _WS_SD: power / 1000.0,
            _YAW: (r3[_YAW].to_numpy() - 4.0) % 360.0,
        }
    )
    return pd.concat([scada, r4])


def _screen_case(*, step: float, n: int = 4000) -> tuple[MethodInput, pd.Timestamp]:
    idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
    changeover = pd.Timestamp(idx[n // 2])
    scada = _scada_with_a_stepped_reference(n, changeover=changeover, step=step)
    mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=changeover, turbine_col=_TURBINE)
    return mi, changeover


def _screen_method(**overrides: object) -> PowerModelMethod:
    """A screening method for the toy cases below.

    ``screen_min_campaign_days=0`` because these toy campaigns are a fortnight long and exist to
    exercise the outlier rule, not the minimum-data gate — which has its own tests, at realistic
    campaign lengths, in :class:`TestScreenNeedsEnoughCampaign`.
    """
    kwargs: dict[str, object] = {
        "columns": _COLUMNS,
        "baseline_rated_power_kw": 2300.0,
        "conditions": (),
        "screen_floor": 0.01,
        "screen_min_campaign_days": 0.0,
        "model_params": _FAST_PARAMS,
        **overrides,
    }
    return PowerModelMethod(**kwargs)  # type: ignore[arg-type]


class TestReferenceScreen:
    """The screen finds a reference that changed on its own, and the estimate stops following it."""

    def test_a_clean_pool_screens_nobody(self) -> None:
        mi, _ = _screen_case(step=0.0)
        assert _screen_method().screen_references(mi).screened == ()

    def test_the_stepped_reference_is_found(self) -> None:
        mi, _ = _screen_case(step=0.03)
        assert _screen_method().screen_references(mi).screened == ("R1",)

    def test_a_degrading_reference_is_found_too(self) -> None:
        mi, _ = _screen_case(step=-0.03)
        assert _screen_method().screen_references(mi).screened == ("R1",)

    def test_the_screen_removes_most_of_the_bias(self) -> None:
        """Truth is 0: an unscreened run follows R1's step, a screened one should not."""
        mi, _ = _screen_case(step=0.03)
        unscreened = _screen_method(reference_screen=False).estimate(mi).p50_overall
        screened = _screen_method(reference_screen=True).estimate(mi).p50_overall
        assert abs(unscreened) > 0.005
        assert abs(screened) < abs(unscreened) / 2

    def test_a_clean_pool_estimates_identically_screened_or_not(self) -> None:
        """The screen finding nothing must cost the estimate nothing."""
        mi, _ = _screen_case(step=0.0)
        off = _screen_method(reference_screen=False).estimate(mi).p50_overall
        on = _screen_method(reference_screen=True).estimate(mi).p50_overall
        assert on == pytest.approx(off)

    def test_the_screened_reference_contributes_nothing(self) -> None:
        """Its power changed, and whatever changed it may have moved where it points too."""
        mi, _ = _screen_case(step=0.03)
        features = _screen_method().reference_features(mi, power_free=("R1",))
        assert not [c for c in features.columns if c.endswith(" @ R1")]

    def test_the_screen_is_on_by_default(self) -> None:
        assert PowerModelMethod(columns=_COLUMNS, baseline_rated_power_kw=2300.0).reference_screen


class TestScreeningContrast:
    """A reference never toggles, so screening it is always a prepost question."""

    def _toggle_mi(self, n: int = 2000) -> MethodInput:
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        start = pd.Timestamp(idx[n // 4])
        schedule = ToggleSchedule(period=pd.Timedelta(minutes=100), start=start)
        treated = np.asarray(resolve_toggle(schedule, pd.DatetimeIndex(idx)).upgraded)
        scada = _toy_scada(n, uplift=0.0, treated=treated)
        return MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE)

    def test_a_prepost_campaign_screens_at_its_own_changeover(self) -> None:
        mi, changeover = _screen_case(step=0.0)
        assert _screen_method().screening_timing(mi) == changeover

    def test_a_toggle_campaign_screens_at_a_timestamp_not_a_schedule(self) -> None:
        """Handing a reference the toggle schedule asks it an on-vs-off question it cannot answer."""
        timing = _screen_method().screening_timing(self._toggle_mi())
        assert isinstance(timing, pd.Timestamp)

    def test_a_toggle_campaign_splits_inside_the_test_period(self) -> None:
        """The toggle test is the valuable data, so the split protects it rather than straddling it."""
        mi = self._toggle_mi()
        index = pd.DatetimeIndex(pd.unique(mi.scada_df.index)).sort_values()
        campaign_start = pd.Timestamp(index[len(index) // 4])
        timing = _screen_method().screening_timing(mi)
        assert campaign_start < timing < index.max()

    def test_the_toggle_split_halves_the_campaign_by_data_volume(self) -> None:
        mi = self._toggle_mi()
        index = pd.DatetimeIndex(pd.unique(mi.scada_df.index)).sort_values()
        timing = _screen_method().screening_timing(mi)
        campaign = index[index >= pd.Timestamp(index[len(index) // 4])]
        before = int((campaign < timing).sum())
        after = int((campaign >= timing).sum())
        assert abs(before - after) <= 1


class TestReferenceReporting:
    """power_model reports what the screen did, and what the surviving references read."""

    def test_reference_uplifts_are_reported_for_every_candidate(self) -> None:
        mi, _ = _screen_case(step=0.0)
        out = _screen_method().estimate(mi)
        assert out.reference_uplifts is not None
        assert set(out.reference_uplifts["turbine"]) == {"R1", "R2", "R3"}

    def test_each_reference_carries_its_energy_so_it_can_be_combined_like_a_test_turbine(self) -> None:
        mi, _ = _screen_case(step=0.0)
        refs = _screen_method().estimate(mi).reference_uplifts
        assert refs is not None
        assert set(refs.columns) >= {"turbine", "uplift", "actual_energy", "n_records", "screened"}
        assert (refs["actual_energy"] > 0).all()
        assert (refs["n_records"] > 0).all()

    def test_a_healthy_campaign_reads_near_zero_reference_uplift(self) -> None:
        """The standard sanity check: references should show no uplift of their own."""
        mi, _ = _screen_case(step=0.0)
        refs = _screen_method().estimate(mi).reference_uplifts
        assert refs is not None
        assert abs(reference_overall_uplift(refs, rated_power_kw=2300.0)) < 0.01

    def test_the_screened_reference_is_reported_but_excluded_from_the_headline(self) -> None:
        """Post-screen: a ruled-out reference stays visible, and stops dragging the sanity check."""
        mi, _ = _screen_case(step=0.05)
        out = _screen_method().estimate(mi)
        refs = out.reference_uplifts
        assert refs is not None
        assert bool(refs.loc[refs["turbine"] == "R1", "screened"].iloc[0])
        surviving = refs[~refs["screened"]]
        assert set(surviving["turbine"]) == {"R2", "R3"}

    def test_the_screening_detail_is_reported(self) -> None:
        """An analyst has to be able to see, and disagree with, what the screen dropped."""
        mi, _ = _screen_case(step=0.05)
        passes = _screen_method().estimate(mi).screen_passes
        assert passes is not None
        assert set(passes.columns) >= {"pass", "turbine", "estimate", "deviation", "dropped"}
        assert bool(passes[passes["dropped"]]["turbine"].eq("R1").any())

    def test_nothing_is_reported_when_the_screen_is_off(self) -> None:
        mi, _ = _screen_case(step=0.0)
        out = _screen_method(reference_screen=False).estimate(mi)
        assert out.screen_passes is None


class TestReferenceOverallUplift:
    def test_it_combines_by_energy_sums(self) -> None:
        """A big turbine at +2% and a small one at 0% must not average to +1%."""
        refs = pd.DataFrame(
            [
                {"turbine": "R1", "uplift": 0.02, "actual_energy": 900.0, "n_records": 100, "screened": False},
                {"turbine": "R2", "uplift": 0.00, "actual_energy": 100.0, "n_records": 100, "screened": False},
            ]
        )
        combined = reference_overall_uplift(refs, rated_power_kw=2300.0)
        assert 0.015 < combined < 0.02

    def test_screened_references_are_excluded(self) -> None:
        refs = pd.DataFrame(
            [
                {"turbine": "R1", "uplift": 0.50, "actual_energy": 500.0, "n_records": 100, "screened": True},
                {"turbine": "R2", "uplift": 0.00, "actual_energy": 500.0, "n_records": 100, "screened": False},
            ]
        )
        assert reference_overall_uplift(refs, rated_power_kw=2300.0) == pytest.approx(0.0)

    def test_an_all_screened_pool_has_no_reference_uplift(self) -> None:
        refs = pd.DataFrame(
            [{"turbine": "R1", "uplift": 0.5, "actual_energy": 500.0, "n_records": 100, "screened": True}]
        )
        assert np.isnan(reference_overall_uplift(refs, rated_power_kw=2300.0))


class TestReferenceUpliftReuse:
    """A healthy prepost campaign must not refit the whole pool twice for the same numbers."""

    def test_a_clean_prepost_pool_reuses_the_screens_final_pass(self) -> None:
        mi, _ = _screen_case(step=0.0)
        method = _screen_method()
        screen = method.screen_references(mi)
        assert screen.screened == ()
        reused = method.reference_uplifts(mi, power_free=(), screen=screen)
        final = screen.passes[screen.passes["pass"] == screen.passes["pass"].max()]
        expected = dict(zip(final["turbine"], final["estimate"], strict=True))
        for row in reused.itertuples():
            assert row.uplift == pytest.approx(expected[row.turbine])

    def test_a_screened_pool_refits_because_the_pools_changed(self) -> None:
        """Dropping a reference changes what every survivor reads, so its old estimate is stale."""
        mi, _ = _screen_case(step=0.05)
        method = _screen_method()
        screen = method.screen_references(mi)
        assert screen.screened == ("R1",)
        refits = method.reference_uplifts(mi, power_free=screen.power_free, screen=screen)
        first = screen.passes[screen.passes["pass"] == 1].set_index("turbine")["estimate"]
        survivor = refits[refits["turbine"] == "R2"].iloc[0]
        assert survivor["uplift"] != pytest.approx(first["R2"])


class TestReferencePlots:
    """Optionally, each reference measurement writes its own plots, in a folder naming its test turbine."""

    def test_each_reference_gets_a_folder_named_for_its_test_turbine(self, tmp_path: Path) -> None:
        mi, _ = _screen_case(step=0.0)
        method = _screen_method(reference_plots_dir=tmp_path)
        refs = method.reference_uplifts(mi, power_free=(), screen=method.screen_references(mi))
        folders = sorted(p.name for p in tmp_path.iterdir())
        assert folders == sorted(f"{r}_reference_for_T1" for r in refs["turbine"])
        assert all(any((tmp_path / f).rglob("*.png")) for f in folders)

    def test_the_numbers_are_those_without_plots(self, tmp_path: Path) -> None:
        mi, _ = _screen_case(step=0.0)
        plain = _screen_method().reference_uplifts(mi)
        plotted = _screen_method(reference_plots_dir=tmp_path).reference_uplifts(mi)
        assert list(plotted["uplift"]) == pytest.approx(list(plain["uplift"]))

    def test_the_plots_the_test_turbine_run_already_has_are_skipped(self, tmp_path: Path) -> None:
        mi, _ = _screen_case(step=0.0)
        _screen_method(reference_plots_dir=tmp_path).reference_uplifts(mi)
        written = {p.parent.name for p in tmp_path.rglob("*.png")}
        assert not written & {stages.CHANGES, stages.OPERATING_STATES, stages.NORTHING}
        assert stages.UPLIFT in written

    def test_the_folder_name(self) -> None:
        assert reference_measurement_dirname("T14", test_wtg="T13") == "T14_reference_for_T13"


class TestADeadReference:
    """A power reference without valid records costs every row; its own uplift is reported as unknown."""

    def _mi_with_a_dead_reference(self) -> tuple[MethodInput, pd.Timestamp]:
        mi, changeover = _screen_case(step=0.0)
        scada = mi.scada_df.copy()
        scada.loc[scada[_TURBINE] == "R1", _POWER] = np.nan  # R1 never reports power
        return MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=changeover, turbine_col=_TURBINE), changeover

    def test_as_a_power_reference_it_leaves_nothing_to_fit(self) -> None:
        mi, _ = self._mi_with_a_dead_reference()
        with pytest.raises(ValueError, match="usable hours in the before period"):
            _screen_method(reference_screen=False, report_reference_uplifts=False).estimate(mi)

    def test_its_own_uplift_is_reported_as_unknown(self) -> None:
        mi, _ = self._mi_with_a_dead_reference()
        refs = _screen_method().reference_uplifts(mi, power_free=("R1",))
        assert not np.isfinite(refs.loc[refs["turbine"] == "R1", "uplift"]).any()
        assert np.isfinite(refs.loc[refs["turbine"] != "R1", "uplift"]).all()


class TestAShrunkenPoolIsAnnounced:
    """A candidate reference the campaign offers but the data does not carry is said out loud."""

    def _case(self) -> tuple[PowerModelMethod, MethodInput]:
        mi, changeover = _screen_case(step=0.0)
        full = mi.scada_df
        context = CampaignContext.from_frame(full, test_wtg="T1", timing=changeover, turbine_col=_TURBINE)
        delivered = full[full[_TURBINE] != "R1"]  # R1 is offered, but never turned up
        return _screen_method(), MethodInput(scada_df=delivered, test_wtg="T1", campaign_context=context)

    def test_the_absent_reference_is_named(self, caplog: pytest.LogCaptureFixture) -> None:
        method, mi = self._case()
        with caplog.at_level(logging.WARNING):
            method.estimate(mi)
        assert "R1" in caplog.text

    def test_the_estimate_still_runs_on_what_is_there(self) -> None:
        method, mi = self._case()
        assert np.isfinite(method.estimate(mi).p50_overall)

    def test_a_complete_delivery_says_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        mi, _ = _screen_case(step=0.0)
        with caplog.at_level(logging.WARNING):
            _screen_method().estimate(mi)
        assert "carries no data" not in caplog.text


class TestTheConditionalGuardRunsBeforeTheScreen:
    """An explicitly-named missing matching column stops the run before any model is fitted."""

    def test_it_raises_without_ever_screening(self, monkeypatch: pytest.MonkeyPatch) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        scada = _toy_scada(n, uplift=0.05, treated=np.asarray(idx >= changeover))
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            era5_hourly_df=_toy_era5(idx).drop(columns=["wind_gusts_10m"]),
            matching_vars=("wind_speed_100m", "wind_gusts_10m"),
            model_params=_FAST_PARAMS,
            screen_min_campaign_days=0.0,
        )
        called: list[str] = []
        monkeypatch.setattr(
            PowerModelMethod,
            "screen_references",
            lambda self, mi: called.append(mi.test_wtg),  # noqa: ARG005
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        with pytest.raises(ValueError, match="wind_gusts_10m"):
            method.estimate(mi)
        assert called == []


class TestTheScreenGateJudgesThePoolThatExists:
    """A declared reference with no rows must not silently disable screening for the rest."""

    def test_an_absent_declared_reference_does_not_disable_the_screen(self) -> None:
        mi, changeover = _screen_case(step=0.0)
        full = mi.scada_df
        context = CampaignContext.from_frame(full, test_wtg="T1", timing=changeover, turbine_col=_TURBINE)
        # R4 is offered by the campaign but never turned up; R1/R2/R3 did.
        context = dataclasses.replace(context, candidate_references=[*context.candidate_references, "R4"])
        starved = MethodInput(scada_df=full, test_wtg="T1", campaign_context=context)
        assert _screen_method(screen_min_campaign_days=5.0).screen_references(starved).screenable


class TestScreenFailureNamesItsCause:
    """When no reference can be estimated, the raised error carries why, not just the verdict."""

    def _mi_without_the_northed_direction(self) -> MethodInput:
        """Every reference becomes unestimatable for one reason: a column the features need is gone."""
        mi, changeover = _screen_case(step=0.0)
        scada = mi.scada_df.drop(columns=[_NORTHED_YAW])
        return MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=changeover, turbine_col=_TURBINE)

    def test_the_error_names_the_missing_column(self) -> None:
        """Without this the analyst is told the farm is broken when one column is absent."""
        with pytest.raises(ValueError, match=_NORTHED_YAW):
            _screen_method().estimate(self._mi_without_the_northed_direction())

    def test_the_error_still_carries_the_screen_s_own_verdict(self) -> None:
        with pytest.raises(ValueError, match="majority"):
            _screen_method().estimate(self._mi_without_the_northed_direction())

    def test_the_underlying_cause_is_chained(self) -> None:
        with pytest.raises(ValueError, match="What stopped each of them") as excinfo:
            _screen_method().estimate(self._mi_without_the_northed_direction())
        assert excinfo.value.__cause__ is not None


class TestReferenceUpliftsSchema:
    """The reported frame keeps its documented columns even when it has no rows."""

    def _single_reference_mi(self) -> MethodInput:
        mi, changeover = _screen_case(step=0.0)
        scada = mi.scada_df
        pair = scada[scada[_TURBINE].isin(["T1", "R1"])]
        return MethodInput(scada_df=pair, test_wtg="T1", upgrade_timing=changeover, turbine_col=_TURBINE)

    def test_a_lone_reference_yields_an_empty_but_typed_frame(self) -> None:
        """One candidate reference leaves it no pool to be estimated against."""
        refs = _screen_method().reference_uplifts(self._single_reference_mi())
        assert refs.empty
        assert list(refs.columns) == [
            "turbine",
            "uplift",
            "actual_energy",
            "n_records",
            "screened",
            "unjudged",
            "sigma",
        ]
        # the documented mask still works on an empty frame
        assert refs.loc[refs["screened"], "turbine"].tolist() == []


class TestScreenIsPrepostOnly:
    """Toggle is not vulnerable to this failure mode, and the screen cannot see it there anyway."""

    def _toggle_mi(self, n: int = 2000) -> MethodInput:
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        start = pd.Timestamp(idx[n // 4])
        schedule = ToggleSchedule(period=pd.Timedelta(minutes=100), start=start)
        treated = np.asarray(resolve_toggle(schedule, pd.DatetimeIndex(idx)).upgraded)
        scada = _toy_scada(n, uplift=0.0, treated=treated)
        return MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE)

    def test_a_toggle_campaign_screens_nobody(self) -> None:
        result = _screen_method().screen_references(self._toggle_mi())
        assert result.screened == ()
        assert not result.screenable

    def test_a_toggle_campaign_still_reports_reference_uplifts(self) -> None:
        """The sanity check is not the screen: references should read ~0 in toggle too."""
        out = _screen_method().estimate(self._toggle_mi())
        assert out.reference_uplifts is not None
        assert set(out.reference_uplifts["turbine"]) == {"R1", "R2", "R3"}
        assert not out.reference_uplifts["screened"].any()

    def test_a_prepost_campaign_still_screens(self) -> None:
        mi, _ = _screen_case(step=0.08)
        assert _screen_method().screen_references(mi).screened == ("R1",)


class TestScreenNeedsEnoughCampaign:
    """A short campaign makes screening estimates too noisy to tell a bad reference from a good one."""

    def _prepost_days(self, days: float, *, baseline_days: int = 120) -> MethodInput:
        """A prepost case whose campaign holds exactly ``days`` of 10-minute records."""
        per_day = 144  # 10-minute records
        n = int(per_day * (baseline_days + days))
        # _toy_scada builds its own index from 2019-01-01, so the changeover is taken from that.
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = pd.Timestamp(idx[per_day * baseline_days])
        scada = _with_a_fourth_reference(_scada_with_a_stepped_reference(n, changeover=changeover, step=0.08))
        return MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=changeover, turbine_col=_TURBINE)

    def _gated_method(self, **overrides: object) -> PowerModelMethod:
        """The screening method with the real minimum-campaign default, which is what is under test."""
        return _screen_method(**{"screen_min_campaign_days": _DEFAULT_SCREEN_MIN_CAMPAIGN_DAYS, **overrides})

    def test_a_short_campaign_is_not_screened(self) -> None:
        result = self._gated_method().screen_references(self._prepost_days(30))
        assert result.screened == ()
        assert not result.screenable

    def test_a_long_enough_campaign_is_screened(self) -> None:
        result = self._gated_method().screen_references(self._prepost_days(200, baseline_days=200))
        assert result.screenable

    def test_the_threshold_is_configurable(self) -> None:
        mi = self._prepost_days(30)
        assert self._gated_method(screen_min_campaign_days=10.0).screen_references(mi).screenable

    def test_a_candidate_short_on_available_data_is_left_out(self) -> None:
        """Readings are not fits: a reference available for a fortnight has a fortnight of data."""
        mi = self._prepost_days(200, baseline_days=200)
        control = self._gated_method().screen_references(mi)
        assert set(control.passes["turbine"].astype(str)) == {"R1", "R2", "R3", "R4"}

        scada = mi.scada_df.copy()
        in_campaign = np.asarray(scada.index >= pd.Timestamp(mi.upgrade_timing))
        is_r1 = (scada[_TURBINE] == "R1").to_numpy()
        starved_rows = np.flatnonzero(is_r1 & in_campaign)[144 * 10 :]  # R1 keeps 10 available days
        scada.iloc[starved_rows, scada.columns.get_loc(_AVAIL)] = 0.0
        starved = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=mi.upgrade_timing, turbine_col=_TURBINE)
        result = self._gated_method().screen_references(starved)
        assert result.screenable
        assert set(result.passes["turbine"].astype(str)) == {"R2", "R3", "R4"}

    def test_the_default_excludes_a_three_month_campaign(self) -> None:
        """The benchmark sweep set this: at 90 days a 3-month campaign still false-positived."""
        default = PowerModelMethod(columns=_COLUMNS, baseline_rated_power_kw=2300.0).screen_min_campaign_days
        assert default > 92, "a 3-month campaign must not be screened"
        assert default < 180, "a 6-month campaign must still be screened"


class TestReferenceUpliftReportingIsOptional:
    """The reference pass costs N fits per estimate; a method sweep does not need it."""

    def test_it_is_reported_by_default(self) -> None:
        mi, _ = _screen_case(step=0.0)
        assert _screen_method().estimate(mi).reference_uplifts is not None

    def test_it_can_be_skipped(self) -> None:
        mi, _ = _screen_case(step=0.0)
        assert _screen_method(report_reference_uplifts=False).estimate(mi).reference_uplifts is None

    def test_skipping_it_does_not_change_the_estimate(self) -> None:
        """It is a report, not an input: turning it off must not move the headline."""
        mi, _ = _screen_case(step=0.0)
        on = _screen_method(report_reference_uplifts=True).estimate(mi).p50_overall
        off = _screen_method(report_reference_uplifts=False).estimate(mi).p50_overall
        assert on == pytest.approx(off)

    def test_the_screen_still_runs_and_is_still_reported(self) -> None:
        """Skipping the report must not silently skip the screening that changes the estimate."""
        mi, _ = _screen_case(step=0.08)
        out = _screen_method(report_reference_uplifts=False).estimate(mi)
        assert out.screen_passes is not None
        assert bool(out.screen_passes["dropped"].any())


class TestCloneRecursionGuards:
    """A clone must never relaunch the passes that created it, or the work is combinatorial.

    The screen and the reference pass each estimate every candidate reference with a clone of this
    method. If a clone still has those passes enabled it runs them too, and its clones run them
    again, down to a pool of one. Setting `reference_screen=False` alone stopped only half of it.
    """

    def _method(self) -> PowerModelMethod:
        return PowerModelMethod(columns=_COLUMNS, baseline_rated_power_kw=2300.0, conditions=())

    def test_the_screening_clone_runs_neither_pass(self) -> None:
        clone = self._method()._screening_clone()  # noqa: SLF001 - the guard is the private clone
        assert not clone.reference_screen
        assert not clone.report_reference_uplifts

    def test_the_reference_clone_runs_neither_pass(self) -> None:
        clone = self._method()._reference_clone()  # noqa: SLF001 - the guard is the private clone
        assert not clone.reference_screen
        assert not clone.report_reference_uplifts

    def test_a_clone_name_never_nests(self) -> None:
        """The runaway showed up as a name with seventeen `_reference` suffixes."""
        method = self._method()
        for clone in (method._screening_clone(), method._reference_clone()):  # noqa: SLF001 - as above
            assert clone.name.count("_reference") <= 1
            assert clone.name.count("_screen") <= 1


class TestPassClonesWriteNoDiagnostics:
    """`out_dir=None` does not suppress diagnostics — it makes a temp dir per run, O(N) per estimate.

    A production run of the shipped configuration left 5845 `/tmp/power_model_*` directories
    totalling 151 MB before this was fixed.
    """

    def test_a_pass_clone_writes_nothing(self) -> None:
        method = PowerModelMethod(columns=_COLUMNS, baseline_rated_power_kw=2300.0, conditions=())
        for clone in (method._screening_clone(), method._reference_clone()):  # noqa: SLF001 - the guard is the clone
            assert not clone.write_diagnostics

    def test_diagnostics_are_written_by_default(self) -> None:
        assert PowerModelMethod(columns=_COLUMNS, baseline_rated_power_kw=2300.0).write_diagnostics

    def test_screening_leaves_no_temp_directories(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # its own temp root, so a parallel worker's directories are not counted as this run's
        temp_root = tmp_path / "tmp"
        temp_root.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
        mi, _ = _screen_case(step=0.08)
        _screen_method(out_dir=tmp_path / "out").estimate(mi)
        assert list(temp_root.glob("power_model_*")) == []

    def test_the_screen_config_reaches_the_run_config(self) -> None:
        """Without it the run-config YAML cannot reproduce whether screening was on, or at what floor."""
        params = _screen_method()._config_params()  # noqa: SLF001 - the recorded config is the point
        assert set(params) >= {
            "reference_screen",
            "screen_floor",
            "screen_min_campaign_days",
            "report_reference_uplifts",
        }


def test_no_diagnostics_with_conditions_raises() -> None:
    """The conditional step writes into the run directory, so it needs one."""
    with pytest.raises(ValueError, match="write_diagnostics"):
        PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            conditions=("power",),
            write_diagnostics=False,
        ).estimate(_screen_case(step=0.0)[0])


class TestGateJudgesEachCandidateSeparately:
    """The gate must reflect the data each screening estimate has, candidate by candidate.

    One turbine's outage says nothing about the others, so it leaves itself out of the screen
    rather than switching the screen off for the whole pool.
    """

    def _mi_with_a_sparse_turbine(
        self, *, campaign_days: int, sparse_days: int, sparse: str = "R1", fourth_reference: bool = True
    ) -> MethodInput:
        per_day = 144
        baseline_days = 200
        n = per_day * (baseline_days + campaign_days)
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = pd.Timestamp(idx[per_day * baseline_days])
        scada = _scada_with_a_stepped_reference(n, changeover=changeover, step=0.0)
        if fourth_reference:
            scada = _with_a_fourth_reference(scada)
        # `sparse` has data for only the first `sparse_days` of the campaign.
        cutoff = changeover + pd.Timedelta(days=sparse_days)
        scada = scada[~((scada[_TURBINE] == sparse) & (scada.index >= cutoff))]
        return MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=changeover, turbine_col=_TURBINE)

    @staticmethod
    def _screened_turbines(result: ScreenResult) -> set[str]:
        return set(result.passes["turbine"].astype(str))

    def test_a_sparse_candidate_is_left_out_and_the_rest_are_still_screened(self) -> None:
        """One candidate with 10 campaign days must not cost the other three their screen."""
        mi = self._mi_with_a_sparse_turbine(campaign_days=200, sparse_days=10)
        result = _screen_method(screen_min_campaign_days=150.0).screen_references(mi)
        assert result.screenable
        assert self._screened_turbines(result) == {"R2", "R3", "R4"}

    def test_a_candidate_left_out_is_not_thereby_ruled_out(self) -> None:
        """Thin data is not evidence against a reference, so it keeps its power channels."""
        mi = self._mi_with_a_sparse_turbine(campaign_days=200, sparse_days=10)
        assert "R1" not in _screen_method(screen_min_campaign_days=150.0).screen_references(mi).screened

    def test_a_pool_that_all_has_coverage_is_screened_whole(self) -> None:
        mi = self._mi_with_a_sparse_turbine(campaign_days=200, sparse_days=200)
        result = _screen_method(screen_min_campaign_days=150.0).screen_references(mi)
        assert result.screenable
        assert self._screened_turbines(result) == {"R1", "R2", "R3", "R4"}

    def test_the_test_turbine_s_outage_does_not_disable_the_screen(self) -> None:
        """The screen never estimates the test turbine, so its coverage does not gate the pool."""
        mi = self._mi_with_a_sparse_turbine(campaign_days=200, sparse_days=10, sparse="T1")
        result = _screen_method(screen_min_campaign_days=150.0).screen_references(mi)
        assert result.screenable
        assert self._screened_turbines(result) == {"R1", "R2", "R3", "R4"}

    def test_too_few_candidates_left_with_coverage_is_not_screened(self) -> None:
        """Two survivors cannot form a majority, so the screen still stands down."""
        mi = self._mi_with_a_sparse_turbine(campaign_days=200, sparse_days=10, fourth_reference=False)
        assert not _screen_method(screen_min_campaign_days=150.0).screen_references(mi).screenable

    def test_a_campaign_short_for_everyone_is_not_screened(self) -> None:
        mi = self._mi_with_a_sparse_turbine(campaign_days=200, sparse_days=200)
        assert not _screen_method(screen_min_campaign_days=250.0).screen_references(mi).screenable


class TestAnUnjudgedCandidateLosesItsPowerEverywhere:
    """What the screen could not judge, the headline must not lean on.

    A screening estimate already runs without the held-out candidates -- `_reference_input` puts
    any candidate outside the pool it is handed among the wake contributors. The headline and the
    reference report must read that same pool, or the screen rules on one farm and the campaign
    reports another.
    """

    @staticmethod
    def _starved() -> MethodInput:
        return TestGateJudgesEachCandidateSeparately()._mi_with_a_sparse_turbine(  # noqa: SLF001 - the shared fixture
            campaign_days=200, sparse_days=10
        )

    def _method(self) -> PowerModelMethod:
        return _screen_method(screen_min_campaign_days=150.0)

    def test_the_screen_names_who_it_held_out(self) -> None:
        assert self._method().screen_references(self._starved()).unjudged == ("R1",)

    def test_power_free_covers_the_held_out_and_the_ruled_out(self) -> None:
        screen = ScreenResult(screened=("R2",), passes=pd.DataFrame(), screenable=True, unjudged=("R1",))
        assert set(screen.power_free) == {"R1", "R2"}

    def test_the_headline_carries_no_power_from_a_held_out_candidate(self) -> None:
        """The whole point: R1's power is a feature the screen was not allowed to vouch for."""
        mi = self._starved()
        method = self._method()
        screen = method.screen_references(mi)
        features = method.reference_features(mi, power_free=screen.power_free).columns
        assert not [c for c in features if c.endswith(f"{QUALIFIER}R1")]
        assert f"{_POWER}{QUALIFIER}R2" in features

    def test_the_reference_report_keeps_it_out_of_every_pool_but_still_reports_it(self) -> None:
        mi = self._starved()
        method = self._method()
        screen = method.screen_references(mi)
        refs = method.reference_uplifts(mi, power_free=screen.power_free, screen=screen)
        row = refs[refs["turbine"] == "R1"].iloc[0]
        assert bool(row["unjudged"])
        assert not bool(row["screened"]), "held out for thin data is not the same as ruled out"

    def test_a_campaign_too_short_to_screen_demotes_nobody(self) -> None:
        """Holding a pool out of a screen that never ran would strip every reference at once."""
        mi = self._starved()
        screen = _screen_method(screen_min_campaign_days=1e6).screen_references(mi)
        assert not screen.screenable
        assert screen.power_free == ()

    def test_toggle_demotes_nobody(self) -> None:
        screen = self._method().screen_references(TestScreenIsPrepostOnly()._toggle_mi())  # noqa: SLF001 - shared
        assert not screen.screenable
        assert screen.power_free == ()


class TestTheReferenceReportRefitsAPartlyScreenedPool:
    """Screening estimates are reusable only when they answered the reference report's question."""

    def test_a_partly_screened_pool_is_not_reused(self) -> None:
        """A candidate left out of the screen was not in the pools the screening estimates used."""
        mi = TestGateJudgesEachCandidateSeparately()._mi_with_a_sparse_turbine(  # noqa: SLF001 - the shared fixture
            campaign_days=200, sparse_days=10
        )
        method = _screen_method(screen_min_campaign_days=150.0)
        screen = method.screen_references(mi)
        assert method._reusable_screen_estimates(mi, screen=screen, ruled_out=set()) == {}  # noqa: SLF001 - the guard

    def test_a_wholly_screened_pool_is_reused(self) -> None:
        mi = TestGateJudgesEachCandidateSeparately()._mi_with_a_sparse_turbine(  # noqa: SLF001 - as above
            campaign_days=200, sparse_days=200
        )
        method = _screen_method(screen_min_campaign_days=150.0)
        screen = method.screen_references(mi)
        reusable = method._reusable_screen_estimates(mi, screen=screen, ruled_out=set())  # noqa: SLF001 - as above
        assert set(reusable) == {"R1", "R2", "R3", "R4"}


_NEIGHBOUR_STEP = 0.10


def _with_a_changed_neighbour(*, as_reference: bool = False, n: int = 4000) -> MethodInput:
    """T1 under test at +5% beside W1, another changed turbine whose power steps +10% at the changeover.

    W1 sees T1's own flow, so its power, were it a feature, would pass its own step off as T1's.
    """
    idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
    changeover = pd.Timestamp(idx[n // 2])
    treated = np.asarray(idx >= changeover)
    scada = _toy_scada(n, uplift=0.05, treated=treated)
    w1 = scada[scada[_TURBINE] == "T1"].assign(**{_TURBINE: "W1"})
    for col in (_POWER, _POWER_MIN, _POWER_MAX):
        w1[col] = w1[col] * np.where(treated, (1.0 + _NEIGHBOUR_STEP) / 1.05, 1.0)
    scada = pd.concat([scada, w1])
    references = ["R1", "R2", "R3", *(["W1"] if as_reference else [])]
    context = CampaignContext(
        test_wtg="T1",
        timing=changeover,
        turbine_col=_TURBINE,
        candidate_references=references,
        wake_contributors=[] if as_reference else ["W1"],
        valid_for_uplift=pd.DataFrame(data=True, index=idx, columns=["T1", "R1", "R2", "R3", "W1"]),
    )
    return MethodInput(scada_df=scada, test_wtg="T1", campaign_context=context)


class TestReanalysisIsIdentified:
    """Output names the reanalysis point, so a reader can tell which series a run used."""

    def test_era5_features_carry_their_source_in_the_importance_table(self, tmp_path: Path) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        treated = np.asarray(idx >= idx[n // 2])
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            era5_hourly_df=_toy_era5(idx),
            conditions=(),
            model_params=_FAST_PARAMS,
            out_dir=tmp_path,
            era5_label=era5_source_label(57.4979, -3.2513),
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(idx[n // 2]), turbine_col=_TURBINE)
        method.estimate(mi)
        importance = pd.read_csv(sorted(tmp_path.rglob("*_feature_importance_*.csv"))[-1])
        assert "wind_speed_100m @ ERA5_57.50_-3.25" in set(importance["feature"])
        assert any(f.endswith(" @ R1") for f in importance["feature"]), "turbine features keep their own source"

    def test_unlocated_reanalysis_is_still_named(self) -> None:
        assert PowerModelMethod(columns=_COLUMNS, baseline_rated_power_kw=2300.0).era5_label == "ERA5"


class TestOrderDoesNotReachTheAnswer:
    """The model is not invariant to column order, so nothing that decides it may vary by accident."""

    def _mi(self, references: list[str]) -> MethodInput:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        treated = np.asarray(idx >= idx[n // 2])
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        context = CampaignContext(
            test_wtg="T1",
            timing=pd.Timestamp(idx[n // 2]),
            turbine_col=_TURBINE,
            candidate_references=references,
            wake_contributors=[],
            valid_for_uplift=pd.DataFrame(data=True, index=idx, columns=["T1", "R1", "R2", "R3"]),
        )
        return MethodInput(scada_df=scada, test_wtg="T1", campaign_context=context)

    def test_the_declaration_order_of_the_references_does_not_change_the_estimate(self) -> None:
        # the same campaign, its references typed in two different orders
        method = PowerModelMethod(
            columns=_COLUMNS, baseline_rated_power_kw=2300.0, conditions=(), model_params=_FAST_PARAMS
        )
        as_declared = method.estimate(self._mi(["R1", "R2", "R3"])).p50_overall
        reversed_order = method.estimate(self._mi(["R3", "R2", "R1"])).p50_overall
        assert as_declared == reversed_order

    def test_every_test_turbine_screens_a_reference_identically(self) -> None:
        # a campaign testing T1 and W1: the wake set behind a screening estimate is the same set
        # whichever of them is under test, so only its order could ever have differed
        mi = _with_a_changed_neighbour()
        from_t1 = _reference_input(mi, target="R1", references=["R2", "R3"])
        as_w1 = dataclasses.replace(mi.context, test_wtg="W1", wake_contributors=["T1"])
        from_w1 = _reference_input(
            MethodInput(scada_df=mi.scada_df, test_wtg="W1", campaign_context=as_w1),
            target="R1",
            references=["R2", "R3"],
        )
        assert from_t1.context.wake_contributors == from_w1.context.wake_contributors == ["T1", "W1"]


class TestTheScreenRunsOncePerCampaign:
    def test_a_shared_cache_reuses_the_verdict_for_the_next_test_turbine(self) -> None:
        cache: dict = {}
        mi, _ = _screen_case(step=0.08)
        first = _screen_method(screen_cache=cache).screen_references(mi)
        assert cache, "the screen recorded nothing to reuse"
        again = _screen_method(screen_cache=cache).screen_references(mi)
        # The verdict is the cached one -- `passes` is the expensive artefact, so sharing that object
        # is what "did not screen again" means. The result itself is re-stamped per test turbine,
        # since two of them can share a screening pool but hold different candidates out of it.
        assert again.passes is first.passes
        assert again.screened == first.screened

    def test_the_changed_times_are_part_of_the_key(self) -> None:
        cache: dict = {}
        mi, _ = _screen_case(step=0.08)
        first = _screen_method(screen_cache=cache).screen_references(mi)
        other = MethodInput(
            scada_df=mi.scada_df,
            test_wtg=mi.test_wtg,
            campaign_context=dataclasses.replace(
                mi.context,
                changed_turbines={"T1": (ChangedInterval(None, None),), "R9": (ChangedInterval(None, None),)},
            ),
        )
        again = _screen_method(screen_cache=cache).screen_references(other)
        assert again.passes is not first.passes
        assert len(cache) == 2

    def test_a_change_of_changed_times_alone_is_a_new_key(self) -> None:
        cache: dict = {}
        mi, _ = _screen_case(step=0.08)
        late = mi.scada_df.index[-1]
        keys = [
            {"T1": (ChangedInterval(None, None),), "R9": (ChangedInterval(None, None),)},
            {"T1": (ChangedInterval(None, None),), "R9": (ChangedInterval(late, None),)},
        ]
        for changed in keys:
            other = MethodInput(
                scada_df=mi.scada_df,
                test_wtg=mi.test_wtg,
                campaign_context=dataclasses.replace(mi.context, changed_turbines=changed),
            )
            _screen_method(screen_cache=cache).screen_references(other)
        assert len(cache) == 2

    def test_without_a_cache_it_screens_every_time(self) -> None:
        mi, _ = _screen_case(step=0.08)
        first = _screen_method().screen_references(mi)
        again = _screen_method().screen_references(mi)
        assert again is not first
        assert again.screened == first.screened


class TestWakeContributors:
    """Another changed turbine stays in the estimate for its wake, and never for its power."""

    def test_it_contributes_no_features(self) -> None:
        features = _screen_method().reference_features(_with_a_changed_neighbour())
        assert not [c for c in features.columns if c.endswith(" @ W1")]

    def test_its_change_does_not_reach_the_estimate(self) -> None:
        method = _screen_method(reference_screen=False, report_reference_uplifts=False)
        as_wake = method.estimate(_with_a_changed_neighbour()).p50_overall
        as_reference = method.estimate(_with_a_changed_neighbour(as_reference=True)).p50_overall
        assert abs(as_reference - 0.05) > 0.01, "the control: W1's power would carry its step into T1's estimate"
        assert as_wake == pytest.approx(0.05, abs=0.005)

    def test_it_is_never_screened_or_reported_as_a_reference(self) -> None:
        out = _screen_method().estimate(_with_a_changed_neighbour())
        assert out.reference_uplifts is not None
        assert set(out.reference_uplifts["turbine"]) == {"R1", "R2", "R3"}
        assert out.screen_passes is not None
        assert set(out.screen_passes["turbine"]) <= {"R1", "R2", "R3"}

    def test_every_estimate_keeps_every_turbine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Screening and reporting estimate each reference too, and every other turbine still wakes it."""
        contexts = _recorded_contexts(monkeypatch, _with_a_changed_neighbour())
        assert {c.test_wtg for c in contexts} == {"T1", "R1", "R2", "R3"}
        for c in contexts:
            assert {c.test_wtg, *c.candidate_references, *c.wake_contributors} == {"T1", "R1", "R2", "R3", "W1"}

    def test_a_screened_reference_still_wakes_the_others(self, monkeypatch: pytest.MonkeyPatch) -> None:
        contexts = _recorded_contexts(monkeypatch, _screen_case(step=0.05)[0])
        ruled_out = [c for c in contexts if c.test_wtg != "R1" and "R1" not in c.candidate_references]
        assert ruled_out, "the screen rules R1 out, so some estimate runs without it as a reference"
        assert all("R1" in c.wake_contributors for c in ruled_out)


def _recorded_contexts(monkeypatch: pytest.MonkeyPatch, mi: MethodInput) -> list[CampaignContext]:
    """Run the screening method on ``mi`` and return the context of every estimate it made, its own first."""
    contexts: list[CampaignContext] = []
    estimate = PowerModelMethod.estimate

    def recording(self: PowerModelMethod, mi: MethodInput) -> object:
        contexts.append(mi.context)
        return estimate(self, mi)

    monkeypatch.setattr(PowerModelMethod, "estimate", recording)
    _screen_method().estimate(mi)
    return contexts


def _with_reference(scada: pd.DataFrame, name: str, *, seed: int) -> pd.DataFrame:
    """Add reference ``name``: R3's rows carrying their own noise."""
    r3 = scada[scada[_TURBINE] == "R3"]
    power = r3[_POWER].to_numpy() + np.random.default_rng(seed).normal(0, 60, len(r3))
    extra = r3.assign(
        **{
            _TURBINE: name,
            _POWER: power,
            _POWER_MAX: power * 1.15,
            _POWER_MIN: power * 0.85,
            _POWER_SD: np.abs(power) / 20.0,
            _WS: power / 100.0,
            _WS_SD: power / 1000.0,
        }
    )
    return pd.concat([scada, extra])


def _planned_case(
    *, step: float, stepped: tuple[str, ...] = ("R1",), reserves: tuple[str, ...] = ("R5", "R6"), n: int = 4000
) -> MethodInput:
    """A pool R1..R4 with ``stepped`` references stepping at the changeover, and clean ``reserves`` waiting."""
    idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
    changeover = pd.Timestamp(idx[n // 2])
    scada = _toy_scada(n, uplift=0.0, treated=np.zeros(n, dtype=bool))
    for k, name in enumerate(("R4", *reserves)):
        scada = _with_reference(scada, name, seed=11 + k)
    for name in stepped:
        rows = (scada[_TURBINE] == name) & (scada.index >= changeover)
        for col in (_POWER, _POWER_MIN, _POWER_MAX):
            scada.loc[rows, col] = scada.loc[rows, col] * (1.0 + step)
    base = CampaignContext.from_frame(scada, test_wtg="T1", timing=changeover, turbine_col=_TURBINE)
    context = dataclasses.replace(
        base,
        candidate_references=["R1", "R2", "R3", "R4"],
        wake_contributors=list(reserves),
        reserve_references=list(reserves),
    )
    return MethodInput(scada_df=scada, test_wtg="T1", campaign_context=context)


class TestTheScreenRefillsItsPool:
    def test_a_ruled_out_reference_is_replaced_by_the_nearest_reserve(self) -> None:
        mi = _planned_case(step=0.05)
        method = _screen_method()
        refilled, screen = method.refill_screened(mi, method.screen_references(mi))
        assert refilled.context.candidate_references == ["R2", "R3", "R4", "R5"]
        assert refilled.context.reserve_references == ["R6"]
        assert "R1" in refilled.context.wake_contributors
        assert "R5" not in refilled.context.wake_contributors
        assert screen.screened == ("R1",)

    def test_the_refilled_pool_is_screened_again_and_every_round_is_recorded(self) -> None:
        mi = _planned_case(step=0.05)
        method = _screen_method()
        _, screen = method.refill_screened(mi, method.screen_references(mi))
        assert "R5" in set(screen.passes["turbine"])
        assert screen.passes["pass"].is_monotonic_increasing

    def test_refill_runs_out_and_the_pool_shrinks(self, caplog: pytest.LogCaptureFixture) -> None:
        mi = _planned_case(step=0.05, stepped=("R1",), reserves=())
        method = _screen_method()
        with caplog.at_level(logging.WARNING):
            refilled, _ = method.refill_screened(mi, method.screen_references(mi))
        assert refilled.context.candidate_references == ["R2", "R3", "R4"]
        assert "reserves ran out" in caplog.text

    def test_the_estimate_runs_on_the_refilled_pool(self) -> None:
        out = _screen_method().estimate(_planned_case(step=0.05))
        refs = out.reference_uplifts
        assert refs is not None
        assert set(refs["turbine"]) == {"R1", "R2", "R3", "R4", "R5"}
        assert bool(refs.loc[refs["turbine"] == "R1", "screened"].iloc[0])
        assert np.isfinite(out.p50_overall)

    @staticmethod
    def _short_r1(reserves: tuple[str, ...] = ("R5", "R6")) -> MethodInput:
        """The clean pool with R1 valid for uplift on only its first two days after the changeover."""
        mi = _planned_case(step=0.0, reserves=reserves)
        valid = mi.context.valid_for_uplift.copy()
        changeover = pd.Timestamp(mi.context.timing)  # type: ignore[arg-type]
        valid.loc[valid.index >= changeover + pd.Timedelta(days=2), "R1"] = False
        return MethodInput(
            scada_df=mi.scada_df,
            test_wtg="T1",
            campaign_context=dataclasses.replace(mi.context, valid_for_uplift=valid),
        )

    def test_a_reference_too_short_to_judge_is_replaced_by_the_nearest_reserve(self) -> None:
        mi = self._short_r1()
        method = _screen_method(screen_min_campaign_days=5.0)
        screen = method.screen_references(mi)
        assert screen.unjudged == ("R1",)
        refilled, after = method.refill_screened(mi, screen)
        assert refilled.context.candidate_references == ["R2", "R3", "R4", "R5"]
        assert "R1" in refilled.context.wake_contributors
        assert after.screened == ()
        assert after.unjudged == ("R1",)

    def test_a_replaced_unjudged_reference_is_still_read_and_flagged(self) -> None:
        out = _screen_method(screen_min_campaign_days=5.0).estimate(self._short_r1())
        refs = out.reference_uplifts.set_index("turbine")
        assert set(refs.index) == {"R1", "R2", "R3", "R4", "R5"}
        assert bool(refs.loc["R1", "unjudged"])
        assert not refs.loc[["R2", "R3", "R4", "R5"], "unjudged"].any()
        assert not refs["screened"].any()

    def test_without_reserves_an_unjudged_reference_stays_in_the_pool_without_power(self) -> None:
        out = _screen_method(screen_min_campaign_days=5.0).estimate(self._short_r1(reserves=()))
        refs = out.reference_uplifts.set_index("turbine")
        assert set(refs.index) == {"R1", "R2", "R3", "R4"}
        assert bool(refs.loc["R1", "unjudged"])

    def test_without_reserves_a_clean_pool_is_untouched(self) -> None:
        mi = _planned_case(step=0.0, reserves=())
        method = _screen_method()
        screen = method.screen_references(mi)
        refilled, after = method.refill_screened(mi, screen)
        assert refilled.context == mi.context
        assert after == screen


class TestTheScreenCacheKnowsTheSpan:
    def test_two_spans_with_the_same_pool_are_screened_separately(self) -> None:
        mi = _planned_case(step=0.0, reserves=())
        cache: dict = {}
        method = _screen_method(screen_cache=cache)
        method.screen_references(mi)
        later = mi.scada_df[mi.scada_df.index >= mi.scada_df.index.min() + pd.Timedelta(hours=6)]
        method.screen_references(dataclasses.replace(mi, scada_df=later))
        assert len(cache) == 2


class TestReferenceReadings:
    def test_each_reference_is_read_against_the_other_candidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mi = _planned_case(step=0.0, reserves=("R5",))
        seen: dict[str, list[str]] = {}

        def record(self: PowerModelMethod, clone: PowerModelMethod, sub_input: MethodInput) -> float:  # noqa: ARG001
            seen[sub_input.test_wtg] = list(sub_input.context.candidate_references)
            return {"uplift": 0.0, "sigma": float("nan")}

        monkeypatch.setattr(PowerModelMethod, "_reference_uplift", record)
        _screen_method(reference_screen=False).reference_uplifts(mi)
        assert seen == {
            "R1": ["R2", "R3", "R4"],
            "R2": ["R1", "R3", "R4"],
            "R3": ["R1", "R2", "R4"],
            "R4": ["R1", "R2", "R3"],
        }

    def test_a_reading_uses_no_record_inside_the_test_turbines_works_window(self) -> None:
        mi = _planned_case(step=0.0, reserves=())
        changeover = pd.Timestamp(mi.context.timing)  # type: ignore[arg-type]
        works = (changeover - pd.Timedelta(days=2), changeover)
        mi = MethodInput(
            scada_df=mi.scada_df, test_wtg="T1", campaign_context=dataclasses.replace(mi.context, works=works)
        )
        valid = _reference_input(mi, target="R1", references=["R2", "R3", "R4"]).context.valid_for_uplift
        inside = (valid.index >= works[0]) & (valid.index < works[1])
        assert not valid.loc[inside, "R1"].any()
        assert valid.loc[~inside, "R1"].all()
        assert valid.loc[inside, "R2"].all()
        assert mi.context.valid_for_uplift["R1"].all()

    def test_the_test_turbine_is_a_wake_contributor_of_every_reading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mi = _planned_case(step=0.0, reserves=())
        wakes: list[bool] = []

        def record(self: PowerModelMethod, clone: PowerModelMethod, sub_input: MethodInput) -> float:  # noqa: ARG001
            wakes.append("T1" in sub_input.context.wake_contributors)
            return {"uplift": 0.0, "sigma": float("nan")}

        monkeypatch.setattr(PowerModelMethod, "_reference_uplift", record)
        _screen_method(reference_screen=False).reference_uplifts(mi)
        assert wakes
        assert all(wakes)


def _held_back_case(*, days: int = 3) -> MethodInput:
    """R1 held back (present but invalid) over its first ``days`` days."""
    mi = _planned_case(step=0.0, reserves=())
    valid = mi.context.valid_for_uplift.copy()
    held = valid.index < valid.index.min() + pd.Timedelta(days=days)
    valid.loc[held, "R1"] = False
    context = dataclasses.replace(mi.context, valid_for_uplift=valid)
    return MethodInput(scada_df=mi.scada_df, test_wtg="T1", campaign_context=context)


class TestAHeldBackReference:
    def test_its_power_is_missing_over_its_exclusion(self) -> None:
        mi = _held_back_case()
        features = _screen_method().reference_features(mi)
        held = features.index < features.index.min() + pd.Timedelta(days=3)
        assert features.loc[held, f"{_POWER}{QUALIFIER}R1"].isna().all()
        assert features.loc[~held, f"{_POWER}{QUALIFIER}R1"].notna().all()


def _with_scenarios(mi: MethodInput) -> MethodInput:
    """``mi`` with a waking scenario per record that depends on the time of day only."""
    scada = mi.scada_df
    hours = pd.DatetimeIndex(scada.index).hour
    labels = np.where(hours < 6, "R2 not waking", "normal")
    return dataclasses.replace(mi, scada_df=scada.assign(**{WAKING_SCENARIO_COL: labels}))


class TestWakingScenarioFeatures:
    def test_the_test_turbine_and_each_power_reference_contribute_their_scenario(self) -> None:
        mi = _with_scenarios(_with_a_changed_neighbour())
        features = _screen_method().reference_features(mi, power_free=("R3",))
        scenarios = sorted(c for c in features.columns if c.startswith(WAKING_SCENARIO_COL))
        assert scenarios == [f"{WAKING_SCENARIO_COL}{QUALIFIER}{t}" for t in ("R1", "R2", "test")]
        assert all(features[c].dtype == "category" for c in scenarios)

    def test_the_estimate_recovers_the_uplift_with_them(self, tmp_path: Path) -> None:
        mi = _with_scenarios(_with_a_changed_neighbour())
        method = _screen_method(reference_screen=False, report_reference_uplifts=False, out_dir=tmp_path)
        assert method.estimate(mi).p50_overall == pytest.approx(0.05, abs=0.005)
        importance = pd.read_csv(sorted(tmp_path.rglob("*_feature_importance_*.csv"))[-1])
        assert f"{WAKING_SCENARIO_COL}{QUALIFIER}test" in set(importance["feature"])

    def test_without_the_column_they_are_left_out_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            features = _screen_method().reference_features(_with_a_changed_neighbour())
        assert not [c for c in features.columns if c.startswith(WAKING_SCENARIO_COL)]
        assert "no waking scenario features" in caplog.text


class TestRunFolders:
    """Studies keep a named, timestamped folder per run; a composed campaign writes flat and plain."""

    def test_a_named_run_folder_by_default(self, tmp_path: Path) -> None:
        mi, _ = _prepost_case(n=2000)
        _fundamentals_method(out_dir=tmp_path, save_plots=False).estimate(mi)
        (run_dir,) = list(tmp_path.iterdir())
        assert run_dir.name.startswith("power_model_T1_")
        assert list(run_dir.glob("power_model_T1_*_results_*.csv"))

    def test_without_a_run_subfolder_it_writes_straight_into_out_dir(self, tmp_path: Path) -> None:
        mi, _ = _prepost_case(n=2000)
        _fundamentals_method(out_dir=tmp_path, save_plots=False, run_subdir=False).estimate(mi)
        names = {p.name for p in tmp_path.iterdir()}
        assert {"results.csv", "data_stats.csv", "feature_importance.csv", "feature_catalogue.csv"} <= names
        assert not [n for n in names if n.startswith("power_model_")]

    def test_the_conditional_csvs_are_plain_too(self, tmp_path: Path) -> None:
        mi, _ = _prepost_case(n=2000)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            era5_hourly_df=_toy_era5(pd.DatetimeIndex(mi.scada_df.index)),
            out_dir=tmp_path,
            save_plots=False,
            run_subdir=False,
        )
        method.estimate(mi)
        names = {p.name for p in (tmp_path / "conditional").iterdir()}
        assert {"conditional_overall.csv", "cem_balance.csv", "cem_cells.csv"} <= names


class TestValidUpliftRows:
    """The test turbine's rows are selected by the operating states' uplift validity."""

    @staticmethod
    def _selected(scada: pd.DataFrame) -> np.ndarray:
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=scada.index[3], turbine_col=_TURBINE)
        index = pd.DatetimeIndex(pd.unique(scada.index)).sort_values()
        y = scada[scada[_TURBINE] == "T1"][_POWER]
        reasons = _fundamentals_method()._loss_reasons(  # noqa: SLF001
            mi, index=index, references=["R1", "R2", "R3"], timebase=pd.Timedelta(minutes=10)
        )
        return np.asarray(reasons == USABLE) & np.isfinite(y.to_numpy(dtype=float))

    def test_a_labelled_frame_keeps_only_valid_uplift_rows(self) -> None:
        scada = _toy_scada(6, uplift=0.0, treated=np.zeros(6, dtype=bool))
        scada[VALID_UPLIFT_COL] = True
        test = (scada[_TURBINE] == "T1").to_numpy()
        curtailed = test & (scada.index == scada.index[2])
        scada.loc[curtailed, VALID_UPLIFT_COL] = False
        assert self._selected(scada).tolist() == [True, True, False, True, True, True]

    def test_an_unlabelled_frame_drops_out_of_range_records(self) -> None:
        scada = _toy_scada(6, uplift=0.0, treated=np.zeros(6, dtype=bool))
        test = (scada[_TURBINE] == "T1").to_numpy()
        scada.loc[test & (scada.index == scada.index[4]), _POWER] = 5000.0
        assert self._selected(scada).tolist() == [True, True, True, True, False, True]


def _context_estimate(scada: pd.DataFrame, *, changeover: pd.Timestamp, **overrides: object) -> float:
    """Estimate T1 from a campaign-free context over ``scada``, with the fast model."""
    context = CampaignContext.from_frame(scada, test_wtg="T1", timing=changeover, turbine_col=_TURBINE)
    method = PowerModelMethod(
        columns=_COLUMNS,
        baseline_rated_power_kw=2300.0,
        conditions=(),
        model_params=_FAST_PARAMS,
        **overrides,  # type: ignore[arg-type]
    )
    return method.estimate(MethodInput(scada_df=scada, test_wtg="T1", campaign_context=context)).p50_overall


class TestRecordFloors:
    """A measurement needs 96 hours of usable data in each period (method step 11 i)."""

    @staticmethod
    def _scada(n_before: int, n_after: int) -> tuple[pd.DataFrame, pd.Timestamp]:
        n = n_before + n_after
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = pd.Timestamp(idx[n_before])
        return _toy_scada(n, uplift=0.05, treated=np.asarray(idx >= changeover)), changeover

    def test_a_before_period_under_96_hours_is_refused(self) -> None:
        scada, changeover = self._scada(575, 800)
        with pytest.raises(ValueError, match="96"):
            _context_estimate(scada, changeover=changeover)

    def test_an_after_period_under_96_hours_is_refused(self) -> None:
        scada, changeover = self._scada(800, 575)
        with pytest.raises(ValueError, match="96"):
            _context_estimate(scada, changeover=changeover)

    def test_96_hours_in_each_period_is_enough(self) -> None:
        scada, changeover = self._scada(576, 576)
        assert np.isfinite(_context_estimate(scada, changeover=changeover))


class TestWakingConfiguration:
    """Method step 11 ii: the scenarios of the subject and its references together, 24 hours a period."""

    @staticmethod
    def _fixture() -> tuple[pd.DataFrame, pd.Timestamp]:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        scada = _toy_scada(n, uplift=0.05, treated=np.asarray(idx >= changeover))
        return scada.assign(**{UPWIND_COL: "none", WAKING_SCENARIO_COL: "normal"}), pd.Timestamp(changeover)

    def test_a_configuration_under_24_hours_in_a_period_is_dropped(self) -> None:
        scada, changeover = self._fixture()
        stamps = scada.index.unique()
        rare = stamps[:100]  # 16.7 hours before the changeover, none after
        scada.loc[(scada[_TURBINE] == "R1") & scada.index.isin(rare), WAKING_SCENARIO_COL] = "R3 not waking"
        dropped = _context_estimate(scada, changeover=changeover)
        assert dropped == pytest.approx(_context_estimate(scada[~scada.index.isin(rare)], changeover=changeover))

    def test_a_configuration_with_24_hours_in_both_periods_is_kept(self) -> None:
        scada, changeover = self._fixture()
        stamps = scada.index.unique()
        common = stamps[:200].union(stamps[2000:2200])  # 33 hours each side
        scada.loc[(scada[_TURBINE] == "R1") & scada.index.isin(common), WAKING_SCENARIO_COL] = "R3 not waking"
        with_scenario = _context_estimate(scada, changeover=changeover)
        without = _context_estimate(scada[~scada.index.isin(common)], changeover=changeover)
        assert with_scenario != pytest.approx(without)
        assert with_scenario == pytest.approx(0.05, abs=0.02)

    def test_a_wake_contributors_scenario_is_not_part_of_the_configuration(self) -> None:
        scada, changeover = self._fixture()
        stamps = scada.index.unique()
        scada.loc[(scada[_TURBINE] == "R3") & scada.index.isin(stamps[:100]), WAKING_SCENARIO_COL] = "R1 not waking"
        context = CampaignContext.from_frame(scada, test_wtg="T1", timing=changeover, turbine_col=_TURBINE)
        context = dataclasses.replace(context, candidate_references=["R1", "R2"], wake_contributors=["R3"])
        method = PowerModelMethod(
            columns=_COLUMNS, baseline_rated_power_kw=2300.0, conditions=(), model_params=_FAST_PARAMS
        )
        whole = method.estimate(MethodInput(scada_df=scada, test_wtg="T1", campaign_context=context)).p50_overall
        plain = scada.assign(**{WAKING_SCENARIO_COL: "normal"})
        assert whole == pytest.approx(
            method.estimate(MethodInput(scada_df=plain, test_wtg="T1", campaign_context=context)).p50_overall
        )

    def test_the_dropped_rows_are_named_in_the_row_dump(self, tmp_path: Path) -> None:
        scada, changeover = self._fixture()
        stamps = scada.index.unique()
        scada.loc[(scada[_TURBINE] == "T1") & scada.index.isin(stamps[:100]), WAKING_SCENARIO_COL] = "R3 not waking"
        _context_estimate(scada, changeover=changeover, row_dump_dir=tmp_path)
        rows = pd.read_parquet(tmp_path / "T1" / "rows.parquet")
        assert (rows["loss_reason"].iloc[:100] == RARE_CONFIGURATION).all()
        assert (rows["loss_reason"].iloc[100:] == USABLE).all()


class TestTogglePairing:
    """Method step 11 iv: a toggle record is used only with a record of the other state within a cycle."""

    @staticmethod
    def _fixture() -> tuple[pd.DataFrame, ToggleSchedule, np.ndarray]:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        schedule = ToggleSchedule(period=pd.Timedelta(hours=4))
        treated = np.asarray(resolve_toggle(schedule, idx).upgraded)
        return _toy_scada(n, uplift=0.04, treated=treated), schedule, treated

    @staticmethod
    def _estimate(scada: pd.DataFrame, schedule: ToggleSchedule, **overrides: object) -> float:
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            conditions=(),
            model_params=_FAST_PARAMS,
            **overrides,  # type: ignore[arg-type]
        )
        return method.estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE)
        ).p50_overall

    def test_on_records_a_cycle_from_any_off_record_are_dropped(self, tmp_path: Path) -> None:
        scada, schedule, treated = self._fixture()
        stamps = scada.index.unique()
        # Remove every off record over a day: the on records inside it sit a cycle or more from any off record.
        gap = stamps[1000:1144]
        off_in_gap = gap[~treated[1000:1144]]
        holed = scada[~scada.index.isin(off_in_gap)]
        self._estimate(holed, schedule, row_dump_dir=tmp_path)
        rows = pd.read_parquet(tmp_path / "T1" / "rows.parquet").set_index("timestamp")
        on_in_gap = gap[treated[1000:1144]]
        inside = rows.loc[on_in_gap, "loss_reason"]
        far = inside.index[(inside.index - gap[0] > schedule.period) & (gap[-1] - inside.index > schedule.period)]
        assert (rows.loc[far, "loss_reason"] == UNPAIRED).all()
        assert (rows.loc[rows.index.difference(on_in_gap), "loss_reason"] == USABLE).all()

    def test_a_complete_toggle_drops_nothing(self, tmp_path: Path) -> None:
        scada, schedule, _ = self._fixture()
        self._estimate(scada, schedule, row_dump_dir=tmp_path)
        rows = pd.read_parquet(tmp_path / "T1" / "rows.parquet")
        assert (rows["loss_reason"] == USABLE).all()

    def test_the_gap_can_be_stated(self, tmp_path: Path) -> None:
        scada, schedule, treated = self._fixture()
        stamps = scada.index.unique()
        gap = stamps[1000:1144]
        holed = scada[~scada.index.isin(gap[~treated[1000:1144]])]
        self._estimate(holed, schedule, row_dump_dir=tmp_path, pairing_max_gap=pd.Timedelta(days=2))
        rows = pd.read_parquet(tmp_path / "T1" / "rows.parquet")
        assert (rows["loss_reason"] == USABLE).all()


class TestRecordLosses:
    def test_the_measurement_writes_its_record_losses_in_hours(self, tmp_path: Path) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = pd.Timestamp(idx[n // 2])
        scada = _toy_scada(n, uplift=0.05, treated=np.asarray(idx >= changeover))
        context = CampaignContext.from_frame(scada, test_wtg="T1", timing=changeover, turbine_col=_TURBINE)
        valid = context.valid_for_uplift.copy()
        valid.loc[valid.index[:60], "R1"] = False
        context = dataclasses.replace(context, valid_for_uplift=valid)
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            conditions=(),
            model_params=_FAST_PARAMS,
            out_dir=tmp_path,
            run_subdir=False,
        )
        method.estimate(MethodInput(scada_df=scada, test_wtg="T1", campaign_context=context))
        losses = pd.read_csv(tmp_path / "record_losses.csv").set_index("reason")
        assert losses.loc[EXCLUDED, "hours_before"] == pytest.approx(10.0)
        assert losses.loc[EXCLUDED, "hours_after"] == pytest.approx(0.0)
        assert losses.loc[USABLE, "hours_before"] == pytest.approx((2000 - 60) / 6.0)
        assert losses.loc[USABLE, "hours_after"] == pytest.approx(2000 / 6.0)


def _uncertainty_case(n: int = 4000, *, toggle: bool = False) -> MethodInput:
    """A toy prepost or toggle campaign with a 5 % uplift, for the uncertainty tests."""
    idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
    if toggle:
        schedule = ToggleSchedule(period=pd.Timedelta(hours=4))
        treated = np.asarray(resolve_toggle(schedule, idx).upgraded)
        timing: object = schedule
    else:
        timing = pd.Timestamp(idx[n // 2])
        treated = np.asarray(idx >= timing)
    scada = _toy_scada(n, uplift=0.05, treated=treated)
    return MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=timing, turbine_col=_TURBINE)  # type: ignore[arg-type]


class TestHeadlineUncertainty:
    """The cheap sigma: a circular block bootstrap of the after period's (actual, counterfactual) pairs."""

    @staticmethod
    def _case(n: int = 4000, *, toggle: bool = False) -> MethodInput:
        return _uncertainty_case(n, toggle=toggle)

    @staticmethod
    def _method(**overrides: object) -> PowerModelMethod:
        return _fundamentals_method(model_params=_FAST_PARAMS, **overrides)

    def test_the_headline_carries_a_sigma_by_default(self) -> None:
        out = self._method().estimate(self._case())
        assert out.sigma_overall is not None
        assert np.isfinite(out.sigma_overall)
        assert 0 < out.sigma_overall < 0.02

    def test_the_sweep_over_block_lengths_is_reported_on_the_headline_row(self) -> None:
        out = self._method().estimate(self._case())
        diagnostics = out.uncertainty_diagnostics
        assert diagnostics is not None
        assert list(zip(diagnostics["condition"], diagnostics["condition_bin"], strict=True)) == [
            ("overall", "overall")
        ]
        row = diagnostics.iloc[0]
        assert row["block_hours"] == 24.0
        assert row["sigma_block_24h"] == pytest.approx(out.sigma_overall)
        for hours in (6, 24, 168, 672):
            assert f"sigma_block_{hours}h" in diagnostics.columns
        assert row["n_blocks"] == 14  # 2000 upgraded records of 10 min over 24 h blocks
        assert row["frac_resamples_finite"] == 1.0

    def test_the_default_block_length_is_a_setting(self) -> None:
        out = self._method(block_hours=6.0).estimate(self._case())
        row = out.uncertainty_diagnostics.iloc[0]
        assert out.sigma_overall == pytest.approx(row["sigma_block_6h"])
        assert row["block_hours"] == 6.0

    def test_the_bootstrap_never_moves_the_headline(self) -> None:
        mi = self._case()
        a = self._method().estimate(mi).p50_overall
        b = self._method(n_resamples=50, bootstrap_seed=7).estimate(mi).p50_overall
        assert a == b

    def test_the_headline_row_merges_onto_the_harness_scores(self) -> None:
        out = self._method(record_reversal=True).estimate(self._case())
        rows: list[dict[str, object]] = [{"condition": "overall", "condition_bin": "overall"}]
        _merge_diagnostics(rows, out.uncertainty_diagnostics)
        assert rows[0]["sigma_block_24h"] == pytest.approx(out.sigma_overall)
        assert "uplift_reverse" in rows[0]

    def test_a_toggle_headline_carries_a_sigma_too(self) -> None:
        out = self._method().estimate(self._case(toggle=True))
        assert out.sigma_overall is not None
        assert np.isfinite(out.sigma_overall)

    def test_the_sweep_is_written_to_the_run_folder(self, tmp_path: Path) -> None:
        self._method(out_dir=tmp_path, run_subdir=False).estimate(self._case())
        table = pd.read_csv(tmp_path / "uncertainty.csv")
        assert list(table["block_hours"]) == [6.0, 24.0, 168.0, 672.0]
        assert {"sigma", "sigma_robust", "n_blocks", "frac_resamples_finite"} <= set(table.columns)

    def test_every_reference_reading_carries_its_sigma(self) -> None:
        mi, _ = _screen_case(step=0.0)
        refs = _screen_method().reference_uplifts(mi)
        assert np.isfinite(refs["sigma"]).all()
        assert (refs["sigma"] > 0).all()
        assert "sigma_block_168h" in refs.columns

    def test_a_reference_reused_from_the_screen_keeps_the_sigma_it_was_screened_with(self) -> None:
        mi, _ = _screen_case(step=0.0)
        method = _screen_method()
        screen = method.screen_references(mi)
        final = screen.passes[screen.passes["pass"] == screen.passes["pass"].max()].set_index("turbine")
        reused = method.reference_uplifts(mi, power_free=(), screen=screen).set_index("turbine")
        for turbine in reused.index:
            assert reused.loc[turbine, "sigma"] == pytest.approx(final.loc[turbine, "sigma"])


class TestRecordedReversal:
    """``record_reversal`` records the reverse direction beside the headline without moving it."""

    @staticmethod
    def _method(**overrides: object) -> PowerModelMethod:
        return _fundamentals_method(model_params=_FAST_PARAMS, **overrides)

    def test_is_off_by_default(self) -> None:
        method = PowerModelMethod(columns=_COLUMNS, baseline_rated_power_kw=2300.0)
        assert method.record_reversal is False
        out = self._method().estimate(_uncertainty_case())
        assert not set(REVERSAL_COLUMNS) & set(out.uncertainty_diagnostics.columns)

    def test_records_both_directions_without_moving_the_headline(self) -> None:
        mi = _uncertainty_case()
        forward = self._method().estimate(mi)
        recorded = self._method(record_reversal=True).estimate(mi)
        assert recorded.p50_overall == forward.p50_overall
        assert recorded.sigma_overall == forward.sigma_overall
        row = recorded.uncertainty_diagnostics.iloc[0]
        assert row["uplift_forward"] == forward.p50_overall
        assert row["uplift_reverse"] == pytest.approx(-0.05 / 1.05, abs=0.02)
        assert 0 < row["sigma_reverse"] < 0.02

    def test_the_recorded_directions_combine_to_the_reversal_headline(self) -> None:
        mi = _uncertainty_case()
        row = self._method(record_reversal=True).estimate(mi).uncertainty_diagnostics.iloc[0]
        reversal = self._method(headline_estimator="reversal").estimate(mi).p50_overall
        combined = np.sqrt((1 + row["uplift_forward"]) / (1 + row["uplift_reverse"])) - 1
        assert combined == pytest.approx(reversal)

    def test_a_toggle_estimate_records_nan(self) -> None:
        out = self._method(record_reversal=True).estimate(_uncertainty_case(toggle=True))
        row = out.uncertainty_diagnostics.iloc[0]
        assert all(np.isnan(row[c]) for c in REVERSAL_COLUMNS)

    def test_every_reference_reading_carries_both_directions(self) -> None:
        mi, _ = _screen_case(step=0.0)
        refs = _screen_method(record_reversal=True).reference_uplifts(mi)
        assert set(REVERSAL_COLUMNS) <= set(refs.columns)
        assert np.isfinite(refs["uplift_reverse"]).all()
        assert refs["uplift_forward"].to_numpy() == pytest.approx(refs["uplift"].to_numpy())


class TestPerBinUncertainty:
    """Each measured bin of the uplift distributions carries a sigma from the matched rows' bootstrap."""

    @staticmethod
    def _run(**overrides: object) -> MethodOutput:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        scada = _toy_scada(n, uplift=0.05, treated=np.asarray(idx >= changeover))
        method = PowerModelMethod(
            columns=_COLUMNS,
            baseline_rated_power_kw=2300.0,
            era5_hourly_df=_toy_era5(idx),
            model_params=_FAST_PARAMS,
            **overrides,  # type: ignore[arg-type]
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        return method.estimate(mi)

    def test_measured_bins_carry_a_sigma_and_filled_bins_none(self, tmp_path: Path) -> None:
        out = self._run(out_dir=tmp_path, run_subdir=False)
        bc = out.p50_by_condition
        assert list(bc.columns) == ["condition", "condition_bin", "p50_uplift", "sigma_uplift"]
        per_bin = pd.read_csv(tmp_path / "conditional" / "conditional_by_bin.csv")
        covered = per_bin["covered"].to_numpy(dtype=bool)
        assert covered.any()
        assert not covered.all()
        sigma = per_bin["sigma_uplift"].to_numpy(dtype=float)
        assert np.isfinite(sigma[covered]).all()
        assert (sigma[covered] > 0).all()
        assert np.isnan(sigma[~covered]).all()
        assert (sigma[covered] < 0.05).all()

    def test_the_per_bin_bootstrap_never_moves_the_estimates(self) -> None:
        a = self._run().p50_by_condition
        b = self._run(n_resamples=50, bootstrap_seed=3).p50_by_condition
        pd.testing.assert_series_equal(a["p50_uplift"], b["p50_uplift"])
