"""Recovery / correctness tests for ``PowerModelMethod`` (the §8-analog bias guard).

Builds a toy dataset where the test turbine's power is a known function of the references plus a
known multiplicative uplift in the upgraded window, and asserts the counterfactual power model
recovers the uplift — for both prepost and toggle. Also checks the reference-only rule end-to-end
(a leak-bait test-turbine column cannot change the estimate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.power_model import PowerModelMethod
from benchmarking.harness.method import MethodInput
from benchmarking.synthetic import ToggleSchedule

_TURBINE = "TurbineName"
_POWER = "wtc_ActPower_mean"
_AVAIL = "wtc_ScReToOp_timeon"
_WS = "wtc_AcWindSp_mean"
_WS_SD = "wtc_AcWindSp_stddev"

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
            {_TURBINE: name, _POWER: power, _AVAIL: 600.0, _WS: power / 100.0, _WS_SD: power / 1000.0}, index=idx
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
            active_power_col=_POWER, availability_col=_AVAIL, wind_speed_col=_WS, model_params=_FAST_PARAMS
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
            active_power_col=_POWER, availability_col=_AVAIL, wind_speed_col=_WS, model_params=_FAST_PARAMS
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
            active_power_col=_POWER, availability_col=_AVAIL, wind_speed_col=_WS, model_params=_FAST_PARAMS
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
            wind_speed_col="not_a_real_column",
            era5_hourly_df=pd.DataFrame({"wind_speed_100m": [1.0]}),
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(idx[n // 2]), turbine_col=_TURBINE)
        with pytest.raises(ValueError, match="not in scada_df"):
            method.estimate(mi)


class TestReferenceOnly:
    def test_leak_bait_test_column_does_not_change_estimate(self) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            active_power_col=_POWER, availability_col=_AVAIL, wind_speed_col=_WS, model_params=_FAST_PARAMS
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


class TestConditionalUplift:
    def test_emits_conditional_uplift_by_ws_and_ti(self) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)  # now includes _WS_SD
        method = PowerModelMethod(
            active_power_col=_POWER,
            availability_col=_AVAIL,
            wind_speed_col=_WS,
            wind_speed_sd_col=_WS_SD,
            model_params=_FAST_PARAMS,
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        out = method.estimate(mi)
        bc = out.p50_by_condition
        assert list(bc.columns) == ["condition", "condition_bin", "p50_uplift"]
        assert set(bc["condition"]) == {"ws", "ti"}

    def test_ws_only_when_no_sd_column_configured(self) -> None:
        n = 4000
        idx = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
        changeover = idx[n // 2]
        treated = np.asarray(idx >= changeover)
        scada = _toy_scada(n, uplift=0.05, treated=treated)
        method = PowerModelMethod(
            active_power_col=_POWER, availability_col=_AVAIL, wind_speed_col=_WS, model_params=_FAST_PARAMS
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=pd.Timestamp(changeover), turbine_col=_TURBINE)
        out = method.estimate(mi)
        assert set(out.p50_by_condition["condition"]) == {"ws"}
