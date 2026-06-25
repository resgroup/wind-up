"""End-to-end tests for RLearnerMethod behind the harness Method seam.

Small, weather-driven SCADA where the reference turbines predict the test turbine's power, so
the R-learner can recover an injected uplift. Covers prepost and toggle recovery, the headline
aggregation, the written diagnostics, and the ERA5-free path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.rlearner.method import RLearnerMethod
from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.synthetic import ToggleSchedule, treated_mask

_TURBINE = "TurbineName"
_POWER = "wtc_ActPower_mean"
_WS = "wtc_AcWindSp_mean"
_SMALL = {"n_estimators": 120, "num_leaves": 15, "min_child_samples": 20, "verbose": -1}


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC", name="timestamp")


def _weather_scada(idx: pd.DatetimeIndex, *, treated: np.ndarray, uplift: float) -> pd.DataFrame:
    """Long SCADA: a shared wind drives every turbine; the test turbine is lifted when treated."""
    rng = np.random.default_rng(0)
    w = rng.uniform(4.0, 12.0, len(idx))  # shared free-stream wind
    frames = []
    for name in ("T1", "R1", "R2"):
        ws = w + rng.normal(0, 0.2, len(idx))
        power = 80.0 * w + rng.normal(0, 5.0, len(idx))
        if name == "T1":
            power = np.where(treated, power * (1.0 + uplift), power)
        frames.append(pd.DataFrame({_TURBINE: name, _POWER: power, _WS: ws}, index=idx))
    return pd.concat(frames)


def _method(tmp_path: Path) -> RLearnerMethod:
    return RLearnerMethod(
        active_power_col=_POWER,
        wind_speed_col=_WS,
        out_dir=tmp_path,
        n_folds=4,
        model_params=_SMALL,
        seed=0,
    )


class TestRecovery:
    def test_prepost_recovers_uplift(self, tmp_path: Path) -> None:
        idx = _index(3000)
        upgrade = idx[1500]
        treated = np.asarray(idx >= upgrade)
        scada = _weather_scada(idx, treated=treated, uplift=0.03)
        out = _method(tmp_path).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=upgrade, turbine_col=_TURBINE)
        )
        assert isinstance(out, MethodOutput)
        assert out.p50_overall == pytest.approx(0.03, abs=0.012)
        assert out.p50_by_condition is None

    def test_toggle_recovers_uplift(self, tmp_path: Path) -> None:
        idx = _index(3000)
        schedule = ToggleSchedule(period=pd.Timedelta(minutes=20), start=idx[0])
        treated = np.asarray(treated_mask(idx, schedule))
        scada = _weather_scada(idx, treated=treated, uplift=0.03)
        out = _method(tmp_path).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE)
        )
        assert out.p50_overall == pytest.approx(0.03, abs=0.012)

    def test_placebo_reports_near_zero(self, tmp_path: Path) -> None:
        idx = _index(3000)
        upgrade = idx[1500]
        treated = np.asarray(idx >= upgrade)
        scada = _weather_scada(idx, treated=treated, uplift=0.0)
        out = _method(tmp_path).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=upgrade, turbine_col=_TURBINE)
        )
        assert out.p50_overall == pytest.approx(0.0, abs=0.01)


def _read_one(folder: Path, kind: str) -> pd.DataFrame:
    run_dirs = [p for p in Path(folder).iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, f"expected one run dir, found {run_dirs}"
    matches = list(run_dirs[0].glob(f"*_{kind}_*.csv"))
    assert len(matches) == 1, f"expected one {kind} csv, found {matches}"
    return pd.read_csv(matches[0])


class TestDiagnostics:
    def _run(self, tmp_path: Path, *, save_plots: bool = False) -> MethodOutput:
        idx = _index(2000)
        upgrade = idx[1000]
        treated = np.asarray(idx >= upgrade)
        scada = _weather_scada(idx, treated=treated, uplift=0.03)
        method = RLearnerMethod(
            active_power_col=_POWER,
            wind_speed_col=_WS,
            out_dir=tmp_path,
            n_folds=4,
            model_params=_SMALL,
            save_plots=save_plots,
        )
        return method.estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=upgrade, turbine_col=_TURBINE))

    def test_writes_results_and_stats_and_importance(self, tmp_path: Path) -> None:
        out = self._run(tmp_path)
        results = _read_one(tmp_path, "results")
        stats = _read_one(tmp_path, "data_stats")
        importance = _read_one(tmp_path, "feature_importance")
        assert results["uplift_frc"].iloc[0] == pytest.approx(out.p50_overall)
        assert sorted(stats["segment"]) == ["all", "baseline", "upgraded"]
        # feature importance names the original reference tags (no test turbine columns)
        assert importance["feature"].str.contains("R1").any()
        assert not importance["feature"].str.endswith("T1").any()

    def test_save_plots_writes_pngs(self, tmp_path: Path) -> None:
        self._run(tmp_path, save_plots=True)
        run_dir = next(p for p in Path(tmp_path).iterdir() if p.is_dir())
        pngs = {p.name for p in (run_dir / "plots").glob("*.png")}
        assert pngs  # at least the feature-importance plot
