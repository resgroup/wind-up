"""Tests for the ToggleSpecialistMethod energy-ratio baseline.

The method is light (no wind_up pipeline), so these run the real thing on small hand-built
SCADA frames. It is a toggle-only specialist: it rejects prepost inputs and always fits on the
interleaved campaign on/off blocks. The tests cover the estimator's toggle recovery, the
prepost rejection, complete-case data handling, the timebase variable, the active-power-only
rule, the written diagnostics, the always-on campaign-only restriction, and the error paths.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines import toggle_specialist
from benchmarking.baselines.toggle_specialist import (
    ToggleSpecialistMethod,
    _daily_segment_coverage,
    _daily_segment_ratio,
    _expected_per_day,
    _infer_timebase,
)
from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.synthetic import ColumnSchema, ToggleSchedule, treated_mask

# Deliberately non-v0 column names: the method is source-agnostic, so the active-power column
# is configured and the turbine column comes from the seam. Using names that are nothing like
# v0's ``DataColumns`` proves the method never reaches for wind_up's vocabulary.
_TURBINE_COL = "asset_id"
_POWER_COL = "kw"
_AVAIL_COL = "secs_avail"
# Larger than any test timebase's full period, so the (required) availability filter keeps
# every row unless a test deliberately sets a lower value.
_FULLY_AVAILABLE_SECS = 3600.0
# The method reads active_power + availability from the schema; the other required roles are
# unused by the ratio, so name them with placeholders.
_COLUMNS = ColumnSchema(
    turbine=_TURBINE_COL,
    active_power=_POWER_COL,
    wind_speed="ws",
    wind_speed_sd="ws_sd",
    gen_rpm="rpm",
    availability=_AVAIL_COL,
)


def _index(n: int, *, freq: str = "10min", start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC", name="timestamp")


def _scada(power_by_turbine: dict[str, np.ndarray], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Long-format SCADA: one block of rows per turbine, sharing ``index`` (fully available)."""
    frames = [
        pd.DataFrame(
            {_TURBINE_COL: name, _POWER_COL: np.asarray(vals, dtype=float), _AVAIL_COL: _FULLY_AVAILABLE_SECS},
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


def _toggle_case(
    n: int = 40, *, period_min: int = 20, uplift: float = 0.05
) -> tuple[pd.DataFrame, ToggleSchedule, np.ndarray]:
    """A toggle campaign starting at the first row (so campaign-only drops nothing) with a known uplift."""
    idx = _index(n)
    schedule = ToggleSchedule(period=pd.Timedelta(minutes=period_min), start=idx[0])
    treated = np.asarray(treated_mask(idx, schedule))
    return _recovery_scada(idx, treated=treated, uplift=uplift), schedule, treated


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


def test_toggle_specialist_shares_no_wind_up_code() -> None:
    """The method is an independent, source-native baseline: it must not import wind_up."""
    tree = ast.parse(Path(toggle_specialist.__file__).read_text())
    modules = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    offenders = {m for m in modules if m == "wind_up" or m.startswith("wind_up.")}
    assert not offenders, f"toggle_specialist must not depend on wind_up, found imports: {offenders}"


class TestRecovery:
    def test_toggle_recovers_known_uplift(self) -> None:
        scada, schedule, _ = _toggle_case(uplift=0.03)
        out = ToggleSpecialistMethod(columns=_COLUMNS).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert isinstance(out, MethodOutput)
        assert out.p50_overall == pytest.approx(0.03)
        assert out.p50_by_condition is None


class TestPrepostRejected:
    """The specialist only supports toggle campaigns; a prepost changeover must raise."""

    def test_prepost_timestamp_raises(self) -> None:
        idx = _index(20)
        upgrade = idx[10]  # a bare Timestamp is a prepost changeover, not a toggle
        treated = np.asarray(idx >= upgrade)
        scada = _recovery_scada(idx, treated=treated, uplift=0.05)
        with pytest.raises(ValueError, match="toggle"):
            ToggleSpecialistMethod(columns=_COLUMNS).estimate(
                MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=upgrade, turbine_col=_TURBINE_COL)
            )


class TestDowntimeFilter:
    """Downtime filtering is required and applies to the test turbine and every reference."""

    def test_columns_is_required(self) -> None:
        with pytest.raises(TypeError):
            ToggleSpecialistMethod()  # type: ignore[call-arg]

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_availability_role_raises(self, blank: str) -> None:
        # A schema that leaves the availability role blank (empty or whitespace) would silently skip
        # downtime filtering, so construction must reject it.
        with pytest.raises(ValueError, match="availability"):
            ToggleSpecialistMethod(columns=replace(_COLUMNS, availability=blank))

    def test_missing_availability_column_raises(self) -> None:
        scada, schedule, _ = _toggle_case()
        scada = scada.drop(columns=[_AVAIL_COL])
        with pytest.raises(ValueError, match="availability"):
            ToggleSpecialistMethod(columns=_COLUMNS).estimate(
                MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
            )

    def test_down_reference_timestamps_are_excluded(self) -> None:
        scada, schedule, _ = _toggle_case()
        idx = scada.index.unique().sort_values()
        # Corrupt a reference's power at two timestamps AND mark it unavailable there. The downtime
        # filter must drop those timestamps, so the wild power never biases the ratio.
        corrupted = scada.copy()
        down = (corrupted[_TURBINE_COL] == "R1") & corrupted.index.isin([idx[3], idx[4]])
        corrupted.loc[down, _POWER_COL] = 1e6
        corrupted.loc[down, _AVAIL_COL] = 0.0
        out = ToggleSpecialistMethod(columns=_COLUMNS).estimate(
            MethodInput(scada_df=corrupted, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert out.p50_overall == pytest.approx(0.05)


class TestCompleteCase:
    def test_drops_timestamp_when_a_reference_is_nan(self) -> None:
        scada, schedule, _ = _toggle_case()
        idx = scada.index.unique().sort_values()
        clean = ToggleSpecialistMethod(columns=_COLUMNS).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )

        # NaN one reference at a used timestamp: that timestamp must be excluded, but since the
        # ratio is identical within each segment the estimate is unchanged.
        corrupted = scada.copy()
        mask = (corrupted[_TURBINE_COL] == "R1") & (corrupted.index == idx[3])
        corrupted.loc[mask, _POWER_COL] = np.nan
        out = ToggleSpecialistMethod(columns=_COLUMNS).estimate(
            MethodInput(scada_df=corrupted, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert out.p50_overall == pytest.approx(clean.p50_overall)

    def test_nan_at_unused_timestamp_does_not_change_estimate(self) -> None:
        scada, schedule, _ = _toggle_case()
        idx = scada.index.unique().sort_values()
        # make idx[2] already unused (test NaN), then add a second NaN at the same timestamp
        base = scada.copy()
        base.loc[(base[_TURBINE_COL] == "T1") & (base.index == idx[2]), _POWER_COL] = np.nan
        before = ToggleSpecialistMethod(columns=_COLUMNS).estimate(
            MethodInput(scada_df=base, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        after_df = base.copy()
        after_df.loc[(after_df[_TURBINE_COL] == "R1") & (after_df.index == idx[2]), _POWER_COL] = np.nan
        after = ToggleSpecialistMethod(columns=_COLUMNS).estimate(
            MethodInput(scada_df=after_df, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert after.p50_overall == pytest.approx(before.p50_overall)


class TestTimebaseInvariance:
    def test_estimate_invariant_to_timebase(self) -> None:
        results = []
        for freq in ("10min", "30min"):
            idx = _index(40, freq=freq)
            schedule = ToggleSchedule(period=pd.Timedelta(minutes=20) * (3 if freq == "30min" else 1), start=idx[0])
            treated = np.asarray(treated_mask(idx, schedule))
            scada = _recovery_scada(idx, treated=treated, uplift=0.04)
            results.append(
                ToggleSpecialistMethod(columns=_COLUMNS)
                .estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL))
                .p50_overall
            )
        assert results[0] == pytest.approx(results[1])

    def test_override_timebase_used_for_mwh(self, tmp_path) -> None:  # noqa: ANN001
        scada, schedule, _ = _toggle_case()
        method = ToggleSpecialistMethod(
            columns=_COLUMNS,
            out_dir=tmp_path,
            timebase=pd.Timedelta(minutes=30),
        )
        method.estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL))
        stats = _read_only_csv(tmp_path, "data_stats")
        all_row = stats[stats["segment"] == "all"].iloc[0]
        # MWh = sum(power_kw) * timebase_hours / 1000; check it used 0.5h not 1/6h
        expected = all_row["used_test_mean_power_kw"] * all_row["n_used_timestamps"] * 0.5 / 1000.0
        assert all_row["used_test_mwh"] == pytest.approx(expected)


class TestActivePowerOnly:
    def test_extra_columns_ignored(self) -> None:
        scada, schedule, _ = _toggle_case()
        plain = ToggleSpecialistMethod(columns=_COLUMNS).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )

        with_extra = scada.copy()
        rng = np.random.default_rng(0)
        with_extra["ws"] = rng.normal(size=len(with_extra))
        with_extra["rpm"] = rng.normal(size=len(with_extra))
        out = ToggleSpecialistMethod(columns=_COLUMNS).estimate(
            MethodInput(scada_df=with_extra, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert out.p50_overall == pytest.approx(plain.p50_overall)


class TestErrors:
    def test_no_reference_raises(self) -> None:
        idx = _index(40)
        schedule = ToggleSchedule(period=pd.Timedelta(minutes=20), start=idx[0])
        scada = _scada({"T1": np.ones(len(idx))}, idx)
        with pytest.raises(ValueError, match="reference"):
            ToggleSpecialistMethod(columns=_COLUMNS).estimate(
                MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
            )

    def test_no_used_baseline_returns_nan(self) -> None:
        scada, schedule, treated = _toggle_case()
        idx = scada.index.unique().sort_values()
        # NaN the test turbine across the whole off (baseline) class -> no used baseline timestamps
        off_ts = idx[~treated]
        is_baseline_test = (scada[_TURBINE_COL] == "T1") & scada.index.isin(off_ts)
        scada.loc[is_baseline_test, _POWER_COL] = np.nan
        out = ToggleSpecialistMethod(columns=_COLUMNS).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert np.isnan(out.p50_overall)


class TestPlotSeries:
    """The per-segment daily series that back the ratio plot and the coverage plot."""

    def test_ratio_series_split_by_segment(self) -> None:
        # one day off (ratio 0.8) then one day on (ratio 0.8*1.05); daily sum-based ratio.
        idx = _index(288, freq="10min")  # two full days
        treated = np.asarray(idx >= idx[144])
        scada = _recovery_scada(idx, treated=treated, uplift=0.05)
        wide = scada.pivot_table(index=scada.index, columns=_TURBINE_COL, values=_POWER_COL)
        test_pw = wide["T1"].to_numpy()
        ref_total = wide[["R1", "R2"]].sum(axis=1).to_numpy()
        used = np.ones(len(wide), dtype=bool)

        base = _daily_segment_ratio(wide.index, test_pw, ref_total, used & ~treated)
        up = _daily_segment_ratio(wide.index, test_pw, ref_total, used & treated)
        # each segment only has data on its own day; the other day is NaN.
        assert base.dropna().to_numpy() == pytest.approx(0.8)
        assert up.dropna().to_numpy() == pytest.approx(0.8 * 1.05)
        assert base.index.equals(up.index)

    def test_toggle_coverage_capped_near_duty_cycle(self) -> None:
        # 20-on/20-off toggle on 10-min data -> each segment can occupy at most ~50% of a day.
        idx = _index(288, freq="10min")
        schedule = ToggleSchedule(period=pd.Timedelta(minutes=40), start=idx[0])
        treated = np.asarray(treated_mask(idx, schedule))
        scada = _recovery_scada(idx, treated=treated, uplift=0.05)
        wide = scada.pivot_table(index=scada.index, columns=_TURBINE_COL, values=_POWER_COL)
        used = np.ones(len(wide), dtype=bool)

        expected = _expected_per_day(wide.index, _infer_timebase(wide.index))
        base = _daily_segment_coverage(wide.index, used, ~treated, expected)
        up = _daily_segment_coverage(wide.index, used, treated, expected)
        assert base.to_numpy() == pytest.approx(0.5)
        assert up.to_numpy() == pytest.approx(0.5)


def _read_only_csv(folder: Path, kind: str) -> pd.DataFrame:
    run_dirs = [p for p in Path(folder).iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, f"expected exactly one run dir, found {run_dirs}"
    matches = list(run_dirs[0].glob(f"*_{kind}_*.csv"))
    assert len(matches) == 1, f"expected one {kind} csv, found {matches}"
    return pd.read_csv(matches[0])


class TestDiagnostics:
    def _run(self, tmp_path, *, save_plots: bool = False):  # noqa: ANN001, ANN202
        scada, schedule, _ = _toggle_case(uplift=0.06)
        method = ToggleSpecialistMethod(columns=_COLUMNS, out_dir=tmp_path, save_plots=save_plots)
        out = method.estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        return out, schedule

    def test_writes_both_csvs(self, tmp_path) -> None:  # noqa: ANN001
        self._run(tmp_path)
        stats = _read_only_csv(tmp_path, "data_stats")
        results = _read_only_csv(tmp_path, "results")
        assert sorted(stats["segment"]) == ["all", "baseline", "upgraded"]
        assert len(results) == 1

    def test_uplift_rederivable_from_stats(self, tmp_path) -> None:  # noqa: ANN001
        out, _ = self._run(tmp_path)
        stats = _read_only_csv(tmp_path, "data_stats").set_index("segment")
        rho_base = stats.loc["baseline", "used_test_mwh"] / stats.loc["baseline", "used_ref_total_mwh"]
        rho_up = stats.loc["upgraded", "used_test_mwh"] / stats.loc["upgraded", "used_ref_total_mwh"]
        assert (rho_up / rho_base - 1.0) == pytest.approx(out.p50_overall)

    def test_results_csv_matches_estimate(self, tmp_path) -> None:  # noqa: ANN001
        out, _ = self._run(tmp_path)
        results = _read_only_csv(tmp_path, "results").iloc[0]
        assert results["uplift_frc"] == pytest.approx(out.p50_overall)
        assert results["mode"] == "toggle"
        assert results["n_refs"] == 2

    def test_stats_coverage_full_for_complete_data(self, tmp_path) -> None:  # noqa: ANN001
        self._run(tmp_path)
        stats = _read_only_csv(tmp_path, "data_stats").set_index("segment")
        assert stats.loc["all", "rows_data_coverage"] == pytest.approx(1.0)
        assert stats.loc["all", "used_data_coverage"] == pytest.approx(1.0)
        assert stats.loc["all", "n_used_timestamps"] == 40

    def test_save_plots_writes_pngs(self, tmp_path) -> None:  # noqa: ANN001
        self._run(tmp_path, save_plots=True)
        run_dir = next(p for p in Path(tmp_path).iterdir() if p.is_dir())
        names = {p.name for p in (run_dir / "plots").rglob("*.png")}
        # the method's own plots, plus the shared cross-method diagnostics it now emits
        assert {"T1_scatter.png", "T1_ratio_timeseries.png", "T1_coverage_timeseries.png"} <= names
        # a run-config YAML is written alongside the plots
        assert any(run_dir.glob("config_*.yaml"))

    def test_no_plots_by_default(self, tmp_path) -> None:  # noqa: ANN001
        self._run(tmp_path)
        run_dir = next(p for p in Path(tmp_path).iterdir() if p.is_dir())
        assert not (run_dir / "plots").exists()


class TestCampaignOnly:
    """The campaign-only restriction is mandatory: a pre-campaign window never enters the baseline."""

    def test_precampaign_dropped_from_baseline(self, tmp_path) -> None:  # noqa: ANN001
        idx = _index(300)
        start = idx[100]  # 100 pre-campaign rows, then 200 rows of interleaved on/off
        schedule = ToggleSchedule(period=pd.Timedelta(minutes=20), start=start)
        treated = np.asarray(treated_mask(idx, schedule))
        scada = _recovery_scada(idx, treated=treated, uplift=0.05)
        ToggleSpecialistMethod(columns=_COLUMNS, out_dir=tmp_path).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        baseline = _read_only_csv(tmp_path, "data_stats").set_index("segment").loc["baseline"]
        # the baseline class starts at/after the campaign start; the pre-campaign rows are excluded.
        assert pd.Timestamp(baseline["first_timestamp"]) >= start
