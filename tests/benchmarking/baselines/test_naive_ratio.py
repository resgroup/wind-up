"""Tests for the NaiveRatioMethod energy-ratio baseline.

The method is light (no wind_up pipeline), so these run the real thing on small
hand-built SCADA frames. They cover the estimator (prepost + toggle recovery),
complete-case data handling, the timebase variable, the active-power-only rule, the
written diagnostics, and the error/degenerate paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.naive_ratio import NaiveRatioMethod, _infer_timebase
from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.synthetic import ToggleSchedule, treated_mask
from wind_up.constants import TIMESTAMP_COL, DataColumns


def _index(n: int, *, freq: str = "10min", start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC", name=TIMESTAMP_COL)


def _scada(power_by_turbine: dict[str, np.ndarray], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Long-format SCADA: one block of rows per turbine, sharing ``index``."""
    frames = [
        pd.DataFrame(
            {DataColumns.turbine_name: name, DataColumns.active_power_mean: np.asarray(vals, dtype=float)},
            index=index,
        )
        for name, vals in power_by_turbine.items()
    ]
    return pd.concat(frames)


def _recovery_scada(
    index: pd.DatetimeIndex, *, treated: np.ndarray, k: float = 0.8, uplift: float = 0.05
) -> pd.DataFrame:
    """Build SCADA where test = k*ref_total in baseline and (1+uplift)*k*ref_total when treated."""
    ref1 = np.linspace(100.0, 1000.0, len(index))
    ref2 = np.linspace(50.0, 500.0, len(index))
    ref_total = ref1 + ref2
    test = k * ref_total
    test = np.where(treated, test * (1.0 + uplift), test)
    return _scada({"T1": test, "R1": ref1, "R2": ref2}, index)


class TestInferTimebase:
    def test_infers_ten_minutes(self) -> None:
        assert _infer_timebase(_index(20)) == pd.Timedelta(minutes=10)

    def test_infers_thirty_minutes(self) -> None:
        assert _infer_timebase(_index(20, freq="30min")) == pd.Timedelta(minutes=30)

    def test_infers_from_duplicated_long_index(self) -> None:
        # long format repeats each timestamp once per turbine
        idx = _index(10)
        doubled = idx.append(idx)
        assert _infer_timebase(doubled) == pd.Timedelta(minutes=10)


class TestRecovery:
    def test_prepost_recovers_known_uplift(self) -> None:
        idx = _index(20)
        upgrade = idx[10]
        treated = np.asarray(idx >= upgrade)
        scada = _recovery_scada(idx, treated=treated, uplift=0.07)
        out = NaiveRatioMethod().estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=upgrade))
        assert isinstance(out, MethodOutput)
        assert out.p50_overall == pytest.approx(0.07)
        assert out.p50_by_condition is None

    def test_toggle_recovers_known_uplift(self) -> None:
        idx = _index(40)
        schedule = ToggleSchedule(period=pd.Timedelta(minutes=20), start=idx[0])
        treated = treated_mask(idx, schedule)
        scada = _recovery_scada(idx, treated=treated, uplift=0.03)
        out = NaiveRatioMethod().estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule))
        assert out.p50_overall == pytest.approx(0.03)


class TestCompleteCase:
    def test_drops_timestamp_when_a_reference_is_nan(self) -> None:
        idx = _index(20)
        upgrade = idx[10]
        treated = np.asarray(idx >= upgrade)
        scada = _recovery_scada(idx, treated=treated, uplift=0.05)
        clean = NaiveRatioMethod().estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=upgrade))

        # NaN one reference at a used baseline timestamp: that timestamp must be excluded, but
        # since the ratio is identical at every row the estimate is unchanged.
        corrupted = scada.copy()
        mask = (corrupted[DataColumns.turbine_name] == "R1") & (corrupted.index == idx[3])
        corrupted.loc[mask, DataColumns.active_power_mean] = np.nan
        out = NaiveRatioMethod().estimate(MethodInput(scada_df=corrupted, test_wtg="T1", upgrade_timing=upgrade))
        assert out.p50_overall == pytest.approx(clean.p50_overall)

    def test_nan_at_unused_timestamp_does_not_change_estimate(self) -> None:
        idx = _index(20)
        upgrade = idx[10]
        treated = np.asarray(idx >= upgrade)
        scada = _recovery_scada(idx, treated=treated, uplift=0.05)
        # make idx[2] already unused (test NaN), then add a second NaN at the same timestamp
        base = scada.copy()
        base.loc[(base[DataColumns.turbine_name] == "T1") & (base.index == idx[2]), DataColumns.active_power_mean] = (
            np.nan
        )
        before = NaiveRatioMethod().estimate(MethodInput(scada_df=base, test_wtg="T1", upgrade_timing=upgrade))
        after_df = base.copy()
        after_df.loc[
            (after_df[DataColumns.turbine_name] == "R1") & (after_df.index == idx[2]), DataColumns.active_power_mean
        ] = np.nan
        after = NaiveRatioMethod().estimate(MethodInput(scada_df=after_df, test_wtg="T1", upgrade_timing=upgrade))
        assert after.p50_overall == pytest.approx(before.p50_overall)


class TestTimebaseInvariance:
    def test_estimate_invariant_to_timebase(self) -> None:
        treated10 = None
        results = []
        for freq in ("10min", "30min"):
            idx = _index(20, freq=freq)
            upgrade = idx[10]
            treated = np.asarray(idx >= upgrade)
            if treated10 is None:
                treated10 = treated
            scada = _recovery_scada(idx, treated=treated, uplift=0.04)
            results.append(
                NaiveRatioMethod()
                .estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=upgrade))
                .p50_overall
            )
        assert results[0] == pytest.approx(results[1])

    def test_override_timebase_used_for_mwh(self, tmp_path) -> None:  # noqa: ANN001
        idx = _index(20)
        upgrade = idx[10]
        treated = np.asarray(idx >= upgrade)
        scada = _recovery_scada(idx, treated=treated, uplift=0.05)
        method = NaiveRatioMethod(out_dir=tmp_path, timebase=pd.Timedelta(minutes=30))
        method.estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=upgrade))
        stats = _read_only_csv(tmp_path, "data_stats")
        all_row = stats[stats["segment"] == "all"].iloc[0]
        # MWh = sum(power_kw) * timebase_hours / 1000; check it used 0.5h not 1/6h
        expected = all_row["used_test_mean_power_kw"] * all_row["n_used_timestamps"] * 0.5 / 1000.0
        assert all_row["used_test_mwh"] == pytest.approx(expected)


class TestActivePowerOnly:
    def test_extra_columns_ignored(self) -> None:
        idx = _index(20)
        upgrade = idx[10]
        treated = np.asarray(idx >= upgrade)
        scada = _recovery_scada(idx, treated=treated, uplift=0.05)
        plain = NaiveRatioMethod().estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=upgrade))

        with_extra = scada.copy()
        rng = np.random.default_rng(0)
        with_extra[DataColumns.wind_speed_mean] = rng.normal(size=len(with_extra))
        with_extra[DataColumns.gen_rpm_mean] = rng.normal(size=len(with_extra))
        out = NaiveRatioMethod().estimate(MethodInput(scada_df=with_extra, test_wtg="T1", upgrade_timing=upgrade))
        assert out.p50_overall == pytest.approx(plain.p50_overall)


class TestErrors:
    def test_no_reference_raises(self) -> None:
        idx = _index(20)
        upgrade = idx[10]
        scada = _scada({"T1": np.ones(len(idx))}, idx)
        with pytest.raises(ValueError, match="reference"):
            NaiveRatioMethod().estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=upgrade))

    def test_no_used_baseline_returns_nan(self) -> None:
        idx = _index(20)
        upgrade = idx[10]
        treated = np.asarray(idx >= upgrade)
        scada = _recovery_scada(idx, treated=treated, uplift=0.05)
        # NaN the test turbine across the whole baseline -> no used baseline timestamps
        is_baseline_test = (scada[DataColumns.turbine_name] == "T1") & np.asarray(scada.index < upgrade)
        scada.loc[is_baseline_test, DataColumns.active_power_mean] = np.nan
        out = NaiveRatioMethod().estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=upgrade))
        assert np.isnan(out.p50_overall)


def _read_only_csv(folder: Path, kind: str) -> pd.DataFrame:
    run_dirs = [p for p in Path(folder).iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, f"expected exactly one run dir, found {run_dirs}"
    matches = list(run_dirs[0].glob(f"*_{kind}_*.csv"))
    assert len(matches) == 1, f"expected one {kind} csv, found {matches}"
    return pd.read_csv(matches[0])


class TestDiagnostics:
    def _run(self, tmp_path, *, save_plots: bool = False):  # noqa: ANN001, ANN202
        idx = _index(20)
        upgrade = idx[10]
        treated = np.asarray(idx >= upgrade)
        scada = _recovery_scada(idx, treated=treated, uplift=0.06)
        method = NaiveRatioMethod(out_dir=tmp_path, save_plots=save_plots)
        out = method.estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=upgrade))
        return out, idx, upgrade

    def test_writes_both_csvs(self, tmp_path) -> None:  # noqa: ANN001
        self._run(tmp_path)
        stats = _read_only_csv(tmp_path, "data_stats")
        results = _read_only_csv(tmp_path, "results")
        assert sorted(stats["segment"]) == ["all", "baseline", "upgraded"]
        assert len(results) == 1

    def test_uplift_rederivable_from_stats(self, tmp_path) -> None:  # noqa: ANN001
        out, _, _ = self._run(tmp_path)
        stats = _read_only_csv(tmp_path, "data_stats").set_index("segment")
        rho_base = stats.loc["baseline", "used_test_mwh"] / stats.loc["baseline", "used_ref_total_mwh"]
        rho_up = stats.loc["upgraded", "used_test_mwh"] / stats.loc["upgraded", "used_ref_total_mwh"]
        assert (rho_up / rho_base - 1.0) == pytest.approx(out.p50_overall)

    def test_results_csv_matches_estimate(self, tmp_path) -> None:  # noqa: ANN001
        out, _, _ = self._run(tmp_path)
        results = _read_only_csv(tmp_path, "results").iloc[0]
        assert results["uplift_frc"] == pytest.approx(out.p50_overall)
        assert results["mode"] == "prepost"
        assert results["n_refs"] == 2

    def test_stats_coverage_full_for_complete_data(self, tmp_path) -> None:  # noqa: ANN001
        self._run(tmp_path)
        stats = _read_only_csv(tmp_path, "data_stats").set_index("segment")
        assert stats.loc["all", "rows_data_coverage"] == pytest.approx(1.0)
        assert stats.loc["all", "used_data_coverage"] == pytest.approx(1.0)
        assert stats.loc["all", "n_used_timestamps"] == 20

    def test_save_plots_writes_pngs(self, tmp_path) -> None:  # noqa: ANN001
        self._run(tmp_path, save_plots=True)
        run_dir = next(p for p in Path(tmp_path).iterdir() if p.is_dir())
        pngs = list((run_dir / "plots").glob("*.png"))
        assert len(pngs) == 2

    def test_no_plots_by_default(self, tmp_path) -> None:  # noqa: ANN001
        self._run(tmp_path)
        run_dir = next(p for p in Path(tmp_path).iterdir() if p.is_dir())
        assert not (run_dir / "plots").exists()
