"""Recovery / correctness tests for ``PowerModelMethod`` (the §8-analog bias guard).

Builds a toy dataset where the test turbine's power is a known function of the references plus a
known multiplicative uplift in the upgraded window, and asserts the counterfactual power model
recovers the uplift — for both prepost and toggle. Also checks the reference-only rule end-to-end
(a leak-bait test-turbine column cannot change the estimate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.power_model import CURATED_ERA5_EXCLUDE, PowerModelMethod

if TYPE_CHECKING:
    from pathlib import Path
from benchmarking.baselines.power_model.method import (
    _TIME_DECAY_CAMPAIGN_MULTIPLE,
    _clip_predictions,
    _combine_uplift,
    _implied_shrinkage,
)
from benchmarking.harness.method import MethodInput
from benchmarking.synthetic import ToggleSchedule

_TURBINE = "TurbineName"
_POWER = "wtc_ActPower_mean"
_AVAIL = "wtc_ScReToOp_timeon"
_WS = "wtc_AcWindSp_mean"
_WS_SD = "wtc_AcWindSp_stddev"
_POWER_MAX = "wtc_ActPower_max"
_POWER_MIN = "wtc_ActPower_min"
_POWER_SD = "wtc_ActPower_stddev"

# Small/fast LightGBM so the toy data (a few thousand rows) is fit well.
_FAST_PARAMS = {"n_estimators": 120, "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20}


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
            },
            index=idx,
        )
        for name, power in frames.items()
    ]
    return pd.concat(parts)


class TestRecovery:
    def test_recovers_known_uplift_prepost(self) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            conditional_uplift=False,  # overall-only; the conditional path needs ERA5 (not supplied here)
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
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            conditional_uplift=False,  # overall-only; the conditional path needs ERA5 (not supplied here)
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
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            conditional_uplift=False,  # overall-only; the conditional path needs ERA5 (not supplied here)
            model_params=_FAST_PARAMS,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        out = method.estimate(mi)
        assert out.p50_overall == pytest.approx(0.0, abs=0.02)


class TestConfigGuards:
    def test_era5_with_missing_wind_speed_col_raises(self) -> None:
        n = 200
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        treated = np.asarray(idx >= idx[n // 2])
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col="not_a_real_column",
            era5_hourly_df=pd.DataFrame({"wind_speed_100m": [1.0]}),
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(idx[n // 2]), turbine_col=_TURBINE)
        with pytest.raises(ValueError, match="not in scada_df"):
            method.estimate(mi)


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
        "active_power_col": _POWER,
        "availability_col": _AVAIL,
        "baseline_rated_power_kw": 2300.0,
        "wind_speed_col": _WS,
        "conditional_uplift": False,
        **overrides,
    }
    return PowerModelMethod(**kwargs)  # type: ignore[arg-type]


class TestModelFundamentals:
    """The self-configuring time-decay weighting and the toggle campaign mask."""

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

    def test_adaptive_time_decay_half_life_scales_with_campaign_duration(self) -> None:
        # the self-configuring default: half_life = k * campaign_duration_days, in both modes
        method = _fundamentals_method()  # adaptive_time_decay defaults to True
        assert method.adaptive_time_decay is True
        start = pd.Timestamp("2019-04-01", tz="UTC")
        for duration_days in (30.0, 90.0, 365.0):
            end = start + pd.Timedelta(days=duration_days)
            hl = method._effective_half_life(campaign_start=start, campaign_end=end)  # noqa: SLF001
            assert hl == pytest.approx(_TIME_DECAY_CAMPAIGN_MULTIPLE * duration_days)

    def test_adaptive_time_decay_weight_values(self) -> None:
        method = _fundamentals_method()  # adaptive default
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

    def test_campaign_mask_bites_only_for_started_toggle(self) -> None:
        index = pd.date_range("2019-01-01", periods=4, freq="10D", tz="UTC")
        mask = PowerModelMethod._campaign_mask  # noqa: SLF001
        np.testing.assert_array_equal(
            mask(index, upgrade_timing=ToggleSchedule(period=pd.Timedelta(hours=4), start=index[2])),
            [False, False, True, True],
        )
        assert mask(index, upgrade_timing=pd.Timestamp(index[2])).all()  # prepost: all-True
        assert mask(index, upgrade_timing=ToggleSchedule(period=pd.Timedelta(hours=4))).all()  # no start

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
            conditional_uplift=True,
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
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            conditional_uplift=False,  # overall-only; the conditional path needs ERA5 (not supplied here)
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


class TestConditionalUplift:
    def test_emits_conditional_uplift_by_ws_ti_and_power(self) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)  # now includes _WS_SD
        method = PowerModelMethod(
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            wind_speed_sd_col=_WS_SD,
            era5_hourly_df=_toy_era5(idx),  # conditional uplift (default on) matches on ERA5 weather
            model_params=_FAST_PARAMS,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        out = method.estimate(mi)
        bc = out.p50_by_condition
        assert list(bc.columns) == ["condition", "condition_bin", "p50_uplift"]
        assert set(bc["condition"]) == {"ws", "ti", "power"}
        # power uses the 6 fraction-of-rated bins
        assert (bc["condition"] == "power").sum() == 6
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
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            wind_speed_sd_col=_WS_SD,
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
        monkeypatch.setattr("benchmarking.baselines.power_model.method._MIN_BIN_MATCHED_COUNT", 10**9)
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            wind_speed_sd_col=_WS_SD,
            era5_hourly_df=_toy_era5(idx),
            model_params=_FAST_PARAMS,
            out_dir=tmp_path,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        method.estimate(mi)
        per_bin = pd.read_csv(sorted(tmp_path.rglob("*_conditional_by_bin_*.csv"))[0])
        assert (~per_bin["covered"]).all()  # nothing clears an impossibly-high floor
        assert per_bin["p50_uplift"].notna().all()  # all imputed, still never NaN

    def test_ws_only_when_no_sd_column_configured(self) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            era5_hourly_df=_toy_era5(idx),  # conditional uplift (default on) matches on ERA5 weather
            model_params=_FAST_PARAMS,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        out = method.estimate(mi)
        # power does not depend on the SD column, so it is still emitted; only ti drops out.
        assert set(out.p50_by_condition["condition"]) == {"ws", "power"}


def _toy_era5(scada_idx: pd.DatetimeIndex, *, seed: int = 0) -> pd.DataFrame:
    """Hourly ERA5 covering the toy window with the three F6 matching columns, i.i.d. over the window.

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
    """The surviving feature config (Issue 11 reference stats, era5_exclude, availability): columns
    reach the model and estimates stay sound."""

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
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            conditional_uplift=False,
            model_params=_FAST_PARAMS,
            reference_stat_cols=(_POWER_MAX, _POWER_MIN, _POWER_SD),
            out_dir=tmp_path,
        )
        out = method.estimate(mi)
        assert out.p50_overall == pytest.approx(0.05, abs=0.02)
        fitted = self._fitted_feature_names(tmp_path)
        assert {f"{_POWER_SD} @ R1", f"{_POWER_MAX} @ R2", f"{_POWER_MIN} @ R3"} <= fitted
        assert not any(name.endswith(" @ T1") for name in fitted)

    def test_era5_exclude_drops_column_and_direction_companions(self, tmp_path: Path) -> None:
        mi = self._prepost_mi()
        method = PowerModelMethod(
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            era5_hourly_df=_toy_era5(pd.DatetimeIndex(mi.scada_df.index)),
            conditional_uplift=False,
            model_params=_FAST_PARAMS,
            era5_exclude=("wind_speed_10m", "wind_direction_10m"),
            out_dir=tmp_path,
        )
        out = method.estimate(mi)
        assert out.p50_overall == pytest.approx(0.05, abs=0.02)
        fitted = self._fitted_feature_names(tmp_path)
        assert (
            not {
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_direction_10m_sin",
                "wind_direction_10m_cos",
            }
            & fitted
        )
        assert "wind_speed_100m" in fitted

    def test_era5_exclude_of_matching_var_raises_with_conditional_on(self) -> None:
        mi = self._prepost_mi(n=300)
        method = PowerModelMethod(
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            era5_hourly_df=_toy_era5(pd.DatetimeIndex(mi.scada_df.index)),
            era5_exclude=("wind_gusts_10m",),
        )
        with pytest.raises(ValueError, match="matching_vars"):
            method.estimate(mi)

    def test_availability_feature_off_removes_availability_columns(self, tmp_path: Path) -> None:
        mi = self._prepost_mi()
        method = PowerModelMethod(
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            conditional_uplift=False,
            model_params=_FAST_PARAMS,
            availability_feature=False,
            out_dir=tmp_path,
        )
        out = method.estimate(mi)
        assert out.p50_overall == pytest.approx(0.05, abs=0.02)
        fitted = self._fitted_feature_names(tmp_path)
        assert not any(name.startswith(_AVAIL) for name in fitted)
        assert f"{_POWER} @ R1" in fitted


class TestPromotedDefaults:
    def test_effective_lgbm_params_include_tuned_min_child_samples(self) -> None:
        m = PowerModelMethod(active_power_col="p", availability_col="a", baseline_rated_power_kw=2300.0)
        assert m._make_model().get_params()["min_child_samples"] == 50  # noqa: SLF001

    def test_explicit_model_params_override_the_tuned_default(self) -> None:
        m = PowerModelMethod(
            active_power_col="p",
            availability_col="a",
            baseline_rated_power_kw=2300.0,
            model_params={"min_child_samples": 123},
        )
        assert m._make_model().get_params()["min_child_samples"] == 123  # noqa: SLF001

    def test_availability_feature_defaults_off(self) -> None:
        m = PowerModelMethod(active_power_col="p", availability_col="a", baseline_rated_power_kw=2300.0)
        assert m.availability_feature is False

    def test_era5_exclude_defaults_to_curated_set(self) -> None:
        m = PowerModelMethod(active_power_col="p", availability_col="a", baseline_rated_power_kw=2300.0)
        assert m.era5_exclude == CURATED_ERA5_EXCLUDE


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
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
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
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            wind_speed_sd_col=_WS_SD,
            era5_hourly_df=_toy_era5(idx),
            model_params=_FAST_PARAMS,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        out = method.estimate(mi)
        # matched i.i.d. weather -> shrinkage ~1, forward-only overall recovers the true uplift
        assert out.p50_overall == pytest.approx(0.05, abs=0.02)
        assert set(out.p50_by_condition["condition"]) == {"ws", "ti", "power"}
        assert list(out.p50_by_condition.columns) == ["condition", "condition_bin", "p50_uplift"]

    def test_overall_matches_conditional_off_and_bins_aggregate_to_it(self, tmp_path: Path) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.03, treated=treated)
        config = {
            "active_power_col": _POWER,
            "availability_col": _AVAIL,
            "baseline_rated_power_kw": 2300.0,
            "wind_speed_col": _WS,
            "wind_speed_sd_col": _WS_SD,
            "era5_hourly_df": _toy_era5(idx),  # same features both ways, so the headline is comparable
            "model_params": _FAST_PARAMS,
        }
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        overall_only = PowerModelMethod(**config, conditional_uplift=False).estimate(mi).p50_overall
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
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=2300.0,
            wind_speed_col=_WS,
            wind_speed_sd_col=_WS_SD,
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
    the placebo (the F5 mechanism). Weather is i.i.d. across the window, so baseline and upgraded are
    distribution-matched and the shrinkage is common to both cross-predict directions -> it cancels.
    """
    idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC", name="timestamp")
    rng = np.random.default_rng(seed)
    w = rng.uniform(3.0, 12.0, n)  # latent wind speed, i.i.d. -> matched across periods
    curve = 20.0 * w**2  # steep power curve (≈180..2880 kW), so per-ws-bin compression is visible
    test_power = np.where(treated, curve * (1.0 + uplift), curve) + rng.normal(0.0, 20.0, n)
    parts = [pd.DataFrame({_TURBINE: "T1", _POWER: test_power, _AVAIL: 600.0, _WS: w, _WS_SD: 0.05 * w}, index=idx)]
    for i in range(1, 4):
        ref_power = curve + rng.normal(0.0, 500.0, n)  # noisy proxy of the curve -> attenuation shrinkage
        parts.append(
            pd.DataFrame({_TURBINE: f"R{i}", _POWER: ref_power, _AVAIL: 600.0, _WS: w, _WS_SD: 0.05 * w}, index=idx)
        )
    return pd.concat(parts)


def _ws_bin_bias(by_condition: pd.DataFrame) -> pd.Series:
    """Per-ws-bin uplift indexed by bin (truth is 0 at placebo, so the value *is* the bias)."""
    ws = by_condition[by_condition["condition"] == "ws"]
    return ws.set_index("condition_bin")["p50_uplift"]


class TestConditionalRegression:
    def test_conditional_flat_at_shrinkage_placebo(self) -> None:
        # Bias guard (design note §8-analog): on a placebo whose references are noisy proxies of a steep
        # power curve, a single counterfactual fit shrinks and reads a spurious per-ws-bin uplift tilt
        # (the F5 mechanism). The two-direction matched conditional cancels that common shrinkage, so the
        # (default) conditional uplift must read ~flat-zero in every bin against the flat-0 truth.
        n = 5000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _shrinkage_scada(n, uplift=0.0, treated=treated)  # placebo: true uplift 0 in every bin
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        method = PowerModelMethod(
            active_power_col=_POWER,
            availability_col=_AVAIL,
            baseline_rated_power_kw=6000.0,
            wind_speed_col=_WS,
            wind_speed_sd_col=_WS_SD,
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
