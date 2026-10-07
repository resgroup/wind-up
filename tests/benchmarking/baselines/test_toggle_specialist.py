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
    gap_to_other_segment,
    pair_within,
    restrict_to_campaign,
)
from benchmarking.harness.conditions import condition_bins
from benchmarking.harness.context import CampaignContext
from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.harness.toggle import build_toggle_df, resolve_toggle
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
# One toggle cycle of the 20-minute ``ToggleSchedule`` most tests use (``reference_block`` is required).
_CYCLE = pd.Timedelta(minutes=20)
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
    """The method is an independent, source-native baseline: it must not import wind_up or wind_up_v0."""
    tree = ast.parse(Path(toggle_specialist.__file__).read_text())
    modules = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    offenders = {m for m in modules if m.split(".", 1)[0] in {"wind_up", "wind_up_v0"}}
    assert not offenders, f"toggle_specialist must not depend on wind_up or wind_up_v0, found imports: {offenders}"


class TestRecovery:
    def test_toggle_recovers_known_uplift(self) -> None:
        scada, schedule, _ = _toggle_case(uplift=0.03)
        out = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert isinstance(out, MethodOutput)
        assert out.p50_overall == pytest.approx(0.03)
        assert out.p50_by_condition is None


# --- per-power-bin conditional reporting -------------------------------------------------------

_RATED_KW = 1200.0
# Rows a bin needs before its estimate is precise enough to test for bias rather than for luck.
_MIN_RECORDS_FOR_BIAS_TEST = 50


def _varying_rho_scada(
    index: pd.DatetimeIndex,
    *,
    treated: np.ndarray,
    uplift: float = 0.0,
    noise_frac: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """SCADA whose test/reference ratio **varies with power** — the case that separates the estimators.

    ``k`` (the untreated test/ref_total ratio) falls from 0.9 at low power to 0.7 at high power, as a
    real test-vs-reference ratio does (different turbines, different wakes, saturation near rated).
    Any per-bin estimator that assumes one global ratio will read this structure as uplift.
    """
    ref1 = np.linspace(100.0, 1000.0, len(index))
    ref2 = np.linspace(50.0, 500.0, len(index))
    ref_total = ref1 + ref2
    span = (ref_total - ref_total.min()) / (ref_total.max() - ref_total.min())
    k = 0.9 - 0.2 * span
    test = k * ref_total * np.where(treated, 1.0 + uplift, 1.0)
    if noise_frac:
        rng = np.random.default_rng(seed)
        test = test * (1.0 + rng.normal(0.0, noise_frac, len(index)))
    return _scada({"T1": test, "R1": ref1, "R2": ref2}, index)


def _varying_rho_case(
    n: int = 600, *, uplift: float = 0.0, noise_frac: float = 0.0
) -> tuple[pd.DataFrame, ToggleSchedule]:
    idx = _index(n)
    schedule = ToggleSchedule(period=pd.Timedelta(minutes=20), start=idx[0])
    treated = np.asarray(treated_mask(idx, schedule))
    return _varying_rho_scada(idx, treated=treated, uplift=uplift, noise_frac=noise_frac), schedule


def _per_bin(scada: pd.DataFrame, schedule: ToggleSchedule) -> pd.DataFrame:
    """Run the method with power conditioning on and return its populated per-bin rows."""
    out = ToggleSpecialistMethod(
        columns=_COLUMNS, reference_block=_CYCLE, conditions=("power",), rated_power_kw=_RATED_KW
    ).estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL))
    assert out.p50_by_condition is not None
    frame = out.p50_by_condition
    return frame[frame["n_records"] > 0]


class TestPerBinIsNotBiasedByTheBaselineRatio:
    """The two tests that earn the per-bin `rho_base` design.

    Both rejected estimators fail here:

    - a **global** ``rho_base`` denominator reads the power-dependence of ``k`` as uplift, tilting
      the per-bin curve (positive where ``k`` is above its average, negative where below);
    - binning on the **test turbine's own power** labels each bin with a treated quantity, so with a
      real uplift the treated rows in a bin correspond to *lower* untreated power than the baseline
      rows in it — against a varying ``k`` that mismatch becomes bias.
    """

    def test_flat_zero_truth_reads_zero_in_every_bin(self) -> None:
        # the placebo: k varies strongly with power but no treatment is applied, so a correct
        # estimator reports 0 everywhere. A global-rho_base estimator reports a slope instead.
        scada, schedule = _varying_rho_case(uplift=0.0)
        per_bin = _per_bin(scada, schedule)
        assert len(per_bin) >= 3  # several populated bins, or the test proves little
        assert per_bin["p50_uplift"].abs().max() < 0.005

    def test_constant_uplift_reads_that_uplift_in_every_bin(self) -> None:
        # strictly stronger than the placebo: a real uplift shifts the test turbine's power, so an
        # estimator that bins on that power mis-assigns treated rows relative to baseline rows. With
        # k varying, that mismatch shows up as a per-bin error rather than cancelling.
        scada, schedule = _varying_rho_case(uplift=0.05)
        per_bin = _per_bin(scada, schedule)
        assert len(per_bin) >= 3
        assert per_bin["p50_uplift"].to_numpy() == pytest.approx(0.05, abs=0.005)

    def test_holds_with_noise(self) -> None:
        # this checks bias, not variance, so it needs enough rows that per-bin sampling noise cannot
        # explain a miss: at 2% noise over 6 bins x 2 segments, 2000 rows puts the per-bin standard
        # error near 0.2 pp, so the 1 pp bound is a genuine bias test rather than a coin flip.
        # That standard error assumes a *populated* bin. The extreme bins of this fixture hold a
        # handful of rows (the lowest, one), where the method itself reports a sigma of ~14 pp -- a
        # 1 pp bound there tests the random number generator, not the estimator.
        scada, schedule = _varying_rho_case(n=2000, uplift=0.05, noise_frac=0.02)
        per_bin = _per_bin(scada, schedule)
        populated = per_bin[per_bin["n_records"] >= _MIN_RECORDS_FOR_BIAS_TEST]
        assert len(populated) >= 3
        assert populated["p50_uplift"].to_numpy() == pytest.approx(0.05, abs=0.01)


class TestPerBinLocalisesUplift:
    def test_uplift_confined_to_high_power_shows_only_there(self) -> None:
        # a bin-local treatment must not smear across bins: only the treated bins move.
        idx = _index(600)
        schedule = ToggleSchedule(period=pd.Timedelta(minutes=20), start=idx[0])
        treated = np.asarray(treated_mask(idx, schedule))
        ref1 = np.linspace(100.0, 1000.0, len(idx))
        ref2 = np.linspace(50.0, 500.0, len(idx))
        ref_total = ref1 + ref2
        test = 0.8 * ref_total
        # +10% only where the untreated test power is high
        high = test > 700.0
        test = np.where(treated & high, test * 1.10, test)
        scada = _scada({"T1": test, "R1": ref1, "R2": ref2}, idx)

        per_bin = _per_bin(scada, schedule)
        low_bins = per_bin[per_bin["sum_counterfactual"] > 0].iloc[:2]
        assert low_bins["p50_uplift"].abs().max() < 0.005
        assert per_bin["p50_uplift"].max() == pytest.approx(0.10, abs=0.01)


class TestConditionsConfiguration:
    def test_default_reports_no_conditions(self) -> None:
        # back-compat: existing callers get exactly today's behaviour and need no rating
        scada, schedule, _ = _toggle_case(uplift=0.03)
        out = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert out.p50_by_condition is None

    def test_power_conditioning_does_not_move_the_headline(self) -> None:
        # the per-bin decomposition is additional reporting, never a change to the estimate
        scada, schedule, _ = _toggle_case(uplift=0.03)
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        plain = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(mi)
        conditioned = ToggleSpecialistMethod(
            columns=_COLUMNS, reference_block=_CYCLE, conditions=("power",), rated_power_kw=_RATED_KW
        ).estimate(mi)
        assert conditioned.p50_overall == plain.p50_overall

    def test_frame_is_labelled_with_the_power_condition(self) -> None:
        scada, schedule = _varying_rho_case(uplift=0.05)
        out = ToggleSpecialistMethod(
            columns=_COLUMNS, reference_block=_CYCLE, conditions=("power",), rated_power_kw=_RATED_KW
        ).estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL))
        assert out.p50_by_condition is not None
        assert set(out.p50_by_condition["condition"]) == {"power"}
        for col in ("condition_bin", "p50_uplift", "n_records", "sum_actual", "sum_counterfactual"):
            assert col in out.p50_by_condition.columns

    def test_ws_condition_raises_citing_the_method_limit(self) -> None:
        with pytest.raises(ValueError, match="does not support"):
            ToggleSpecialistMethod(
                columns=_COLUMNS, reference_block=_CYCLE, conditions=("ws",), rated_power_kw=_RATED_KW
            )

    def test_unknown_condition_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown condition"):
            ToggleSpecialistMethod(
                columns=_COLUMNS, reference_block=_CYCLE, conditions=("bogus",), rated_power_kw=_RATED_KW
            )

    def test_power_without_a_rating_raises(self) -> None:
        with pytest.raises(ValueError, match="rated_power_kw"):
            ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE, conditions=("power",))


class TestPerBinSparseData:
    def test_bins_with_no_data_are_nan_not_imputed(self) -> None:
        # a sparse bin must read "no answer", not an invented one: downstream consumers decide what to
        # do about it, and an imputed prior would manufacture false confidence. Rating the turbine far
        # above the data's power range guarantees empty upper bins rather than hoping for them.
        scada, schedule = _varying_rho_case(uplift=0.05)
        out = ToggleSpecialistMethod(
            columns=_COLUMNS, reference_block=_CYCLE, conditions=("power",), rated_power_kw=4000.0
        ).estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL))
        assert out.p50_by_condition is not None
        empty = out.p50_by_condition[out.p50_by_condition["n_records"] == 0]
        assert not empty.empty
        assert empty["p50_uplift"].isna().all()

    def test_every_bin_is_represented(self) -> None:
        scada, schedule = _varying_rho_case(uplift=0.05)
        out = ToggleSpecialistMethod(
            columns=_COLUMNS, reference_block=_CYCLE, conditions=("power",), rated_power_kw=_RATED_KW
        ).estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL))
        assert out.p50_by_condition is not None
        assert len(out.p50_by_condition) == len(condition_bins("power", rated_power_kw=_RATED_KW)) - 1


class TestPerBinDiagnostics:
    def test_per_bin_csv_is_written(self, tmp_path: Path) -> None:
        scada, schedule = _varying_rho_case(uplift=0.05)
        ToggleSpecialistMethod(
            columns=_COLUMNS, reference_block=_CYCLE, conditions=("power",), rated_power_kw=_RATED_KW, out_dir=tmp_path
        ).estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL))
        assert list(tmp_path.rglob("*_by_power_bin_*.csv"))

    def test_no_per_bin_csv_when_conditioning_is_off(self, tmp_path: Path) -> None:
        scada, schedule, _ = _toggle_case(uplift=0.03)
        ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE, out_dir=tmp_path).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert not list(tmp_path.rglob("*_by_power_bin_*.csv"))

    def test_per_bin_plot_written_with_save_plots(self, tmp_path: Path) -> None:
        scada, schedule = _varying_rho_case(uplift=0.05)
        ToggleSpecialistMethod(
            columns=_COLUMNS,
            reference_block=_CYCLE,
            conditions=("power",),
            rated_power_kw=_RATED_KW,
            out_dir=tmp_path,
            save_plots=True,
        ).estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL))
        assert list(tmp_path.rglob("*per_bin_uplift.png"))


class TestPrepostRejected:
    """The specialist only supports toggle campaigns; a prepost changeover must raise."""

    def test_prepost_timestamp_raises(self) -> None:
        idx = _index(20)
        upgrade = idx[10]  # a bare Timestamp is a prepost changeover, not a toggle
        treated = np.asarray(idx >= upgrade)
        scada = _recovery_scada(idx, treated=treated, uplift=0.05)
        with pytest.raises(ValueError, match="toggle"):
            ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
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
            ToggleSpecialistMethod(columns=replace(_COLUMNS, availability=blank), reference_block=_CYCLE)

    def test_missing_availability_column_raises(self) -> None:
        scada, schedule, _ = _toggle_case()
        scada = scada.drop(columns=[_AVAIL_COL])
        with pytest.raises(ValueError, match="availability"):
            ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
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
        out = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(scada_df=corrupted, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert out.p50_overall == pytest.approx(0.05)


class TestCompleteCase:
    def test_drops_timestamp_when_a_reference_is_nan(self) -> None:
        scada, schedule, _ = _toggle_case()
        idx = scada.index.unique().sort_values()
        clean = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )

        # NaN one reference at a used timestamp: that timestamp must be excluded, but since the
        # ratio is identical within each segment the estimate is unchanged.
        corrupted = scada.copy()
        mask = (corrupted[_TURBINE_COL] == "R1") & (corrupted.index == idx[3])
        corrupted.loc[mask, _POWER_COL] = np.nan
        out = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(scada_df=corrupted, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert out.p50_overall == pytest.approx(clean.p50_overall)

    def test_nan_at_unused_timestamp_does_not_change_estimate(self) -> None:
        scada, schedule, _ = _toggle_case()
        idx = scada.index.unique().sort_values()
        # make idx[2] already unused (test NaN), then add a second NaN at the same timestamp
        base = scada.copy()
        base.loc[(base[_TURBINE_COL] == "T1") & (base.index == idx[2]), _POWER_COL] = np.nan
        before = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(scada_df=base, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        after_df = base.copy()
        after_df.loc[(after_df[_TURBINE_COL] == "R1") & (after_df.index == idx[2]), _POWER_COL] = np.nan
        after = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
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
                ToggleSpecialistMethod(columns=_COLUMNS, reference_block=schedule.period)
                .estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL))
                .p50_overall
            )
        assert results[0] == pytest.approx(results[1])

    def test_override_timebase_used_for_mwh(self, tmp_path) -> None:  # noqa: ANN001
        scada, schedule, _ = _toggle_case()
        method = ToggleSpecialistMethod(
            columns=_COLUMNS,
            reference_block=pd.Timedelta(hours=1),
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
        plain = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )

        with_extra = scada.copy()
        rng = np.random.default_rng(0)
        with_extra["ws"] = rng.normal(size=len(with_extra))
        with_extra["rpm"] = rng.normal(size=len(with_extra))
        out = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(scada_df=with_extra, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert out.p50_overall == pytest.approx(plain.p50_overall)


class TestErrors:
    def test_no_reference_is_the_block_leg_alone(self, caplog: pytest.LogCaptureFixture) -> None:
        idx = _index(40)
        schedule = ToggleSchedule(period=pd.Timedelta(minutes=20), start=idx[0])
        scada = _scada({"T1": np.ones(len(idx))}, idx)
        with caplog.at_level("WARNING"):
            out = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
                MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
            )
        assert _legs(out) == {"block_mean"}
        assert "no reference turbines" in caplog.text

    def test_no_used_baseline_returns_nan(self) -> None:
        scada, schedule, treated = _toggle_case()
        idx = scada.index.unique().sort_values()
        # NaN the test turbine across the whole off (baseline) class -> no used baseline timestamps
        off_ts = idx[~treated]
        is_baseline_test = (scada[_TURBINE_COL] == "T1") & scada.index.isin(off_ts)
        scada.loc[is_baseline_test, _POWER_COL] = np.nan
        out = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
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
        method = ToggleSpecialistMethod(
            columns=_COLUMNS, reference_block=_CYCLE, out_dir=tmp_path, save_plots=save_plots
        )
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
        assert results["n_refs_available"] == 2
        assert results["n_refs"] == len(str(results["refs_used"]).split(";"))

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
        ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE, out_dir=tmp_path).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        baseline = _read_only_csv(tmp_path, "data_stats").set_index("segment").loc["baseline"]
        # the baseline class starts at/after the campaign start; the pre-campaign rows are excluded.
        assert pd.Timestamp(baseline["first_timestamp"]) >= start


# --- uncertainty (non-optional, computed after the uplift) --------------------------------------


def _noisy_toggle_case(n: int = 2016, *, uplift: float = 0.03, noise_frac: float = 0.05) -> tuple:
    """A campaign long enough to bootstrap, with per-record noise so sigma has something to find."""
    idx = _index(n)
    schedule = ToggleSchedule(period=pd.Timedelta(minutes=20), start=idx[0])
    treated = np.asarray(treated_mask(idx, schedule))
    scada = _varying_rho_scada(idx, treated=treated, uplift=uplift, noise_frac=noise_frac)
    return scada, schedule


_LEG_NAMES = ("sum", "block_mean", "combined")


def _long_diagnostics(out: MethodOutput) -> pd.DataFrame:
    """The harness frame back in per-leg rows: one row per (component, condition, condition_bin)."""
    wide = out.uncertainty_diagnostics
    assert wide is not None
    frames = []
    for leg in _LEG_NAMES:
        renamed = {c: c.removesuffix(f"_{leg}") for c in wide.columns if c.endswith(f"_{leg}")}
        if renamed:
            leg_rows = wide[["condition", "condition_bin", *renamed]].rename(columns=renamed)
            frames.append(leg_rows.dropna(subset=list(renamed.values()), how="all").assign(component=leg))
    return pd.concat(frames, ignore_index=True)


def _estimate(scada: pd.DataFrame, schedule: ToggleSchedule, **kwargs: object) -> MethodOutput:
    kwargs.setdefault("reference_block", _CYCLE)
    return ToggleSpecialistMethod(columns=_COLUMNS, **kwargs).estimate(
        MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
    )


class TestUncertaintyIsAlwaysReported:
    def test_overall_sigma_is_reported_without_being_asked_for(self) -> None:
        """Uncertainty is not an opt-in: an uplift with no sigma is not a usable answer."""
        out = _estimate(*_noisy_toggle_case())
        assert out.sigma_overall is not None
        assert np.isfinite(out.sigma_overall)
        assert out.sigma_overall > 0

    def test_every_reported_bin_carries_a_sigma(self) -> None:
        scada, schedule = _noisy_toggle_case()
        out = _estimate(scada, schedule, conditions=("power",), rated_power_kw=_RATED_KW)
        assert out.p50_by_condition is not None
        assert "sigma_uplift" in out.p50_by_condition.columns
        populated = out.p50_by_condition[out.p50_by_condition["n_records"] > 0]
        assert len(populated) > 0
        assert populated["sigma_uplift"].notna().all()

    def test_the_harness_frame_has_one_row_per_cell_and_no_reserved_column(self) -> None:
        scada, schedule = _noisy_toggle_case()
        diag = _estimate(scada, schedule, conditions=("power",), rated_power_kw=_RATED_KW).uncertainty_diagnostics
        assert diag is not None
        assert not diag.duplicated(["condition", "condition_bin"]).any()
        assert not set(diag.columns) & {"sigma", "estimate", "truth"}
        assert {"sigma_sum", "sigma_block_mean", "sigma_combined"} <= set(diag.columns)

    def test_diagnostics_cover_the_headline_and_every_bin(self) -> None:
        scada, schedule = _noisy_toggle_case()
        out = _estimate(scada, schedule, conditions=("power",), rated_power_kw=_RATED_KW)
        diag = _long_diagnostics(out)
        assert diag is not None
        assert (diag["condition"] == "overall").sum() == diag["component"].nunique()
        assert set(diag.columns) >= {
            "condition",
            "condition_bin",
            "n_upgraded_records",
            "n_baseline_records",
            "n_blocks",
            "sigma_robust",
            "frac_resamples_finite",
        }
        overall = diag[(diag["condition"] == "overall") & (diag["component"] == "sum")].iloc[0]
        assert overall["n_upgraded_records"] > 0
        assert overall["n_baseline_records"] > 0


class TestUncertaintyDoesNotChangeAnUplift:
    """Every candidate's uplift is fixed before any bootstrap runs.

    The bootstraps decide which reference subset is kept and how the legs are weighted, so the
    headline may move with them; no subset's uplift and not the block leg's may.
    """

    @staticmethod
    def _candidates(tmp_path: Path, **kwargs: object) -> tuple[pd.Series, float, pd.DataFrame]:
        scada, schedule = _noisy_toggle_case()
        out = _estimate(scada, schedule, out_dir=tmp_path, **kwargs)
        subsets = _read_only_csv(tmp_path, "reference_subsets").set_index("refs")["uplift_frc"]
        block = _read_only_csv(tmp_path, "results").loc[0, "uplift_frc_block_mean"]
        diag = _long_diagnostics(out).set_index(["component", "condition_bin"])
        return subsets, block, diag

    def test_block_length_moves_the_bootstrap_and_no_uplift(self, tmp_path: Path) -> None:
        """Asserted on the *bootstrap* sigma, not the reported one: the reported value is
        ``max(bootstrap, fallback)`` and the fallback does not depend on block length.
        """
        subsets_a, block_a, da = self._candidates(tmp_path / "a", block_hours=6.0)
        subsets_b, block_b, db = self._candidates(tmp_path / "b", block_hours=96.0)
        pd.testing.assert_series_equal(subsets_a, subsets_b)
        assert block_a == block_b
        for leg in ("sum", "block_mean"):
            assert da.loc[(leg, "overall"), "sigma_bootstrap"] != db.loc[(leg, "overall"), "sigma_bootstrap"]

    def test_no_uplift_moves_when_the_bootstrap_is_reduced_to_nothing(self, tmp_path: Path) -> None:
        subsets_full, block_full, _ = self._candidates(tmp_path / "full", n_resamples=1000)
        subsets_min, block_min, _ = self._candidates(tmp_path / "minimal", n_resamples=2)
        pd.testing.assert_series_equal(subsets_full, subsets_min)
        assert block_full == block_min


class TestUncertaintyRunsOnlyWhenThereIsAnUpliftToQualify:
    def test_a_nan_uplift_reports_a_nan_sigma(self) -> None:
        """No baseline rows means no uplift; there is nothing for an uncertainty to describe."""
        idx = _index(40)
        # every row treated -> no off rows -> rho_base is NaN
        scada = _recovery_scada(idx, treated=np.ones(len(idx), dtype=bool))
        out = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(
                scada_df=scada,
                test_wtg="T1",
                upgrade_timing=pd.DataFrame({"toggle_on": True, "toggle_off": False}, index=idx),
                turbine_col=_TURBINE_COL,
            )
        )
        assert np.isnan(out.p50_overall)
        assert out.sigma_overall is not None
        assert np.isnan(out.sigma_overall)

    def test_the_bootstrap_is_not_run_when_the_uplift_is_nan(self) -> None:
        """Diagnostics still report the counts: they are what explains why there was no answer."""
        idx = _index(40)
        scada = _recovery_scada(idx, treated=np.ones(len(idx), dtype=bool))
        out = ToggleSpecialistMethod(
            columns=_COLUMNS, reference_block=_CYCLE, conditions=("power",), rated_power_kw=_RATED_KW
        ).estimate(
            MethodInput(
                scada_df=scada,
                test_wtg="T1",
                upgrade_timing=pd.DataFrame({"toggle_on": True, "toggle_off": False}, index=idx),
                turbine_col=_TURBINE_COL,
            )
        )
        diag = _long_diagnostics(out)
        assert diag is not None
        legs = diag[diag["component"] != "combined"]
        assert (legs["n_blocks"] == 0).all()
        # no resample ran, so no leg reports a finite fraction (the harness frame drops the empty column)
        assert "frac_resamples_finite" not in legs or legs["frac_resamples_finite"].isna().all()
        assert (legs["n_baseline_records"] == 0).all()


class TestUncertaintyReproducibility:
    def test_the_same_seed_gives_the_same_sigma(self) -> None:
        scada, schedule = _noisy_toggle_case()
        a = _estimate(scada, schedule, bootstrap_seed=3)
        b = _estimate(scada, schedule, bootstrap_seed=3)
        assert a.sigma_overall == b.sigma_overall

    def test_a_longer_campaign_reports_a_smaller_sigma(self) -> None:
        short = _estimate(*_noisy_toggle_case(n=1008))
        long = _estimate(*_noisy_toggle_case(n=8064))
        assert long.sigma_overall < short.sigma_overall  # type: ignore[operator]


class TestUncertaintyInTheWrittenOutputs:
    def test_results_csv_records_the_sigma_and_the_bootstrap_config(self, tmp_path: Path) -> None:
        scada, schedule = _noisy_toggle_case()
        out = _estimate(scada, schedule, out_dir=tmp_path, block_hours=24.0)
        results = pd.concat([pd.read_csv(p) for p in tmp_path.glob("*/*_results_*.csv")])
        assert results["uplift_sigma_frc"].iloc[0] == pytest.approx(out.sigma_overall)
        assert results["block_hours"].iloc[0] == 24.0
        assert results["n_resamples"].iloc[0] == 1000

    def test_an_uncertainty_csv_is_written(self, tmp_path: Path) -> None:
        scada, schedule = _noisy_toggle_case()
        _estimate(scada, schedule, out_dir=tmp_path, conditions=("power",), rated_power_kw=_RATED_KW)
        written = list(tmp_path.glob("*/*_uncertainty_*.csv"))
        assert len(written) == 1
        frame = pd.read_csv(written[0])
        assert "n_upgraded_records" in frame.columns
        # overall + six power bins, per leg and for the blend
        assert frame.groupby("component").size().to_dict() == {"sum": 7, "block_mean": 7, "combined": 7}


# --- labeled_rows: the row selection the estimate actually used ---------------------------------


class TestLabeledRows:
    """``labeled_rows`` exposes the method's own used/segment/bin labels, per test-turbine record.

    A consumer that wants a per-bin quantity the method does not report (a mean pitch, say) must be
    able to compute it over *exactly* the rows and bins the uplift used. These tests pin that the
    labels reproduce the method's internal masks rather than a plausible re-derivation of them.
    """

    def test_labeled_rows_is_populated_for_a_toggle_campaign(self) -> None:
        scada, schedule = _noisy_toggle_case()
        out = _estimate(scada, schedule, conditions=("power",), rated_power_kw=_RATED_KW)
        assert out.labeled_rows is not None
        assert set(out.labeled_rows.columns) >= {"used", "segment", "power_bin"}

    def test_carries_the_original_test_turbine_scada_columns(self) -> None:
        """The consumer aggregates the source columns, so they must survive unmodified."""
        scada, schedule = _noisy_toggle_case()
        out = _estimate(scada, schedule, conditions=("power",), rated_power_kw=_RATED_KW)
        assert out.labeled_rows is not None
        test_rows = scada[scada[_TURBINE_COL] == "T1"]
        for col in test_rows.columns:
            assert col in out.labeled_rows.columns
        pd.testing.assert_series_equal(out.labeled_rows[_POWER_COL], test_rows[_POWER_COL])

    def test_one_row_per_test_turbine_record(self) -> None:
        """Not the long frame, and not the references: exactly the test turbine's own rows."""
        scada, schedule = _noisy_toggle_case()
        out = _estimate(scada, schedule, conditions=("power",), rated_power_kw=_RATED_KW)
        assert out.labeled_rows is not None
        assert len(out.labeled_rows) == (scada[_TURBINE_COL] == "T1").sum()
        assert out.labeled_rows.index.is_unique

    def test_used_reproduces_the_methods_own_used_mask(self) -> None:
        scada, schedule = _noisy_toggle_case()
        method = ToggleSpecialistMethod(
            columns=_COLUMNS, reference_block=_CYCLE, conditions=("power",), rated_power_kw=_RATED_KW
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        out = method.estimate(mi)
        assert out.labeled_rows is not None

        mi_restricted = restrict_to_campaign(mi)
        wide = toggle_specialist._wide_column(  # noqa: SLF001
            mi_restricted.scada_df, turbine_col=_TURBINE_COL, value_col=_POWER_COL
        )
        expected = method._selection(  # noqa: SLF001
            mi_restricted,
            wide=wide,
            test="T1",
            refs=[c for c in wide.columns if c != "T1"],
            timebase=pd.Timedelta(minutes=10),
            rows=resolve_toggle(mi_restricted.upgrade_timing, wide.index),
        ).paired
        assert out.labeled_rows["used"].to_numpy().tolist() == expected.to_numpy().tolist()

    def test_a_downtime_row_is_labelled_unused(self) -> None:
        """The label tracks the filter: knock one record's availability out and it drops out of `used`."""
        scada, schedule = _noisy_toggle_case()
        victim = scada.index[scada[_TURBINE_COL] == "T1"][100]
        scada.loc[(scada.index == victim) & (scada[_TURBINE_COL] == "T1"), _AVAIL_COL] = 0.0
        out = _estimate(scada, schedule, conditions=("power",), rated_power_kw=_RATED_KW)
        assert out.labeled_rows is not None
        assert not out.labeled_rows.loc[victim, "used"]
        assert out.labeled_rows["used"].sum() > 0

    def test_segment_partitions_rows_into_baseline_upgraded_excluded(self) -> None:
        scada, schedule = _noisy_toggle_case()
        out = _estimate(scada, schedule, conditions=("power",), rated_power_kw=_RATED_KW)
        assert out.labeled_rows is not None
        assert set(out.labeled_rows["segment"]) <= {"baseline", "upgraded", "excluded"}

    def test_segment_reproduces_the_resolved_toggle_rows(self) -> None:
        scada, schedule = _noisy_toggle_case()
        out = _estimate(scada, schedule, conditions=("power",), rated_power_kw=_RATED_KW)
        assert out.labeled_rows is not None
        rows = resolve_toggle(schedule, pd.DatetimeIndex(out.labeled_rows.index))
        assert (out.labeled_rows["segment"] == "upgraded").to_numpy().tolist() == list(rows.upgraded)
        assert (out.labeled_rows["segment"] == "baseline").to_numpy().tolist() == list(rows.campaign_baseline)

    def test_power_bin_uses_the_same_edges_as_the_uplift(self) -> None:
        scada, schedule = _noisy_toggle_case()
        out = _estimate(scada, schedule, conditions=("power",), rated_power_kw=_RATED_KW)
        assert out.labeled_rows is not None
        assert out.p50_by_condition is not None
        labelled = set(out.labeled_rows["power_bin"].dropna().astype(str))
        reported = set(out.p50_by_condition["condition_bin"].astype(str))
        assert labelled <= reported

    def test_upgraded_rows_per_bin_match_the_reported_counts(self) -> None:
        """The point of the frame: aggregate it and you land on the method's own population.

        ``n_records`` counts the *upgraded* rows a bin's ratio consumed, so that is what the labels
        must reproduce -- for every bin the method returned a finite uplift for.
        """
        scada, schedule = _noisy_toggle_case()
        out = _estimate(scada, schedule, conditions=("power",), rated_power_kw=_RATED_KW)
        assert out.labeled_rows is not None
        assert out.p50_by_condition is not None

        rows = out.labeled_rows
        counted = (
            rows[rows["used"] & (rows["segment"] == "upgraded")]
            .groupby(rows["power_bin"].astype(str), observed=True)
            .size()
        )
        reported = out.p50_by_condition[np.isfinite(out.p50_by_condition["p50_uplift"])]
        assert not reported.empty
        for _, row in reported.iterrows():
            assert counted.get(str(row["condition_bin"]), 0) == row["n_records"]

    def test_rows_outside_the_bin_edges_have_no_bin(self) -> None:
        """A row off the end of the bins is NaN, not silently folded into the edge bin.

        The bin edges scale with the declared rating, so rating the turbine well below the power the
        fixture actually reaches pushes its top rows past the outermost edge.
        """
        scada, schedule = _noisy_toggle_case()
        out = _estimate(scada, schedule, conditions=("power",), rated_power_kw=_RATED_KW / 4)
        assert out.labeled_rows is not None
        assert out.labeled_rows["power_bin"].isna().any()
        assert out.labeled_rows["power_bin"].notna().any()


# --- reversal symmetry -------------------------------------------------------------------------


def _swap_states(toggle_df: pd.DataFrame) -> pd.DataFrame:
    """Relabel which state is 'on': the same campaign, described from the other side."""
    return toggle_df.rename(columns={"toggle_on": "toggle_off", "toggle_off": "toggle_on"})


class TestReversalSymmetry:
    """Which of two toggle states is *called* the baseline is a naming choice, not a measurement.

    The bin label must therefore not depend on it. It used to: the label was the baseline state's
    predicted power, so renaming the states shifted every label by the uplift and migrated rows
    across bin edges -- moving per-bin estimates by an appreciable fraction of their own sigma for
    no physical reason.
    """

    def _pair(self) -> tuple[MethodOutput, MethodOutput]:
        scada, schedule = _varying_rho_case(n=600, uplift=0.05)
        index = pd.DatetimeIndex(pd.unique(scada.index)).sort_values()
        toggle_df = build_toggle_df(index, schedule)
        forward = _estimate(scada, toggle_df, conditions=("power",), rated_power_kw=_RATED_KW)
        reversed_ = _estimate(scada, _swap_states(toggle_df), conditions=("power",), rated_power_kw=_RATED_KW)
        return forward, reversed_

    def test_every_row_keeps_its_power_bin_when_the_states_are_swapped(self) -> None:
        forward, reversed_ = self._pair()
        assert forward.labeled_rows is not None
        assert reversed_.labeled_rows is not None
        fwd_bins = forward.labeled_rows["power_bin"].astype(str)
        rev_bins = reversed_.labeled_rows["power_bin"].astype(str)
        assert (fwd_bins == rev_bins).all(), f"{(fwd_bins != rev_bins).sum()} rows changed bin under reversal"

    def test_the_segments_simply_trade_places(self) -> None:
        """Corroborates the bins are fixed: each bin's baseline rows become its upgraded rows."""
        forward, reversed_ = self._pair()
        assert forward.labeled_rows is not None
        assert reversed_.labeled_rows is not None
        fwd, rev = forward.labeled_rows, reversed_.labeled_rows
        assert (fwd["used"] == rev["used"]).all()
        assert (fwd["segment"] == "upgraded").sum() == (rev["segment"] == "baseline").sum()
        assert ((fwd["segment"] == "upgraded") == (rev["segment"] == "baseline")).all()

    def test_the_same_bins_are_reported_either_way(self) -> None:
        forward, reversed_ = self._pair()
        assert forward.p50_by_condition is not None
        assert reversed_.p50_by_condition is not None
        fwd = forward.p50_by_condition.set_index(forward.p50_by_condition["condition_bin"].astype(str))
        rev = reversed_.p50_by_condition.set_index(reversed_.p50_by_condition["condition_bin"].astype(str))
        assert set(fwd.index) == set(rev.index)


# --- caller-supplied row exclusion (ColumnSchema.exclude_row) -----------------------------------

_EXCLUDE_COL = "special_mode"
_EXCLUDE_COLUMNS = replace(_COLUMNS, exclude_row=_EXCLUDE_COL)


def _flag(scada: pd.DataFrame, *, turbine: str, every: int, value: object = True) -> pd.DataFrame:
    """Return a copy of ``scada`` carrying an all-False exclude column with every Nth ``turbine`` row set."""
    out = scada.copy()
    # object dtype so a non-bool ``value`` (the NaN case) can be written without an upcast warning
    out[_EXCLUDE_COL] = pd.Series(data=False, index=out.index, dtype=object if value is not True else bool)
    is_turbine = (out[_TURBINE_COL] == turbine).to_numpy()
    position = np.cumsum(is_turbine) - 1
    out.loc[is_turbine & (position % every == 0), _EXCLUDE_COL] = value
    return out


class TestExcludeRow:
    """``columns.exclude_row`` drops caller-flagged **test** turbine rows from the estimate."""

    def test_flagged_test_rows_are_not_used(self) -> None:
        scada, schedule = _noisy_toggle_case()
        flagged = _flag(scada, turbine="T1", every=5)
        out = ToggleSpecialistMethod(columns=_EXCLUDE_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(scada_df=flagged, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert out.labeled_rows is not None
        excluded = out.labeled_rows[_EXCLUDE_COL].to_numpy(dtype=bool)
        assert excluded.any()
        assert not out.labeled_rows.loc[excluded, "used"].to_numpy().any()
        assert out.labeled_rows.loc[~excluded, "used"].to_numpy().any()

    def test_excluding_corrupted_rows_restores_the_known_uplift(self) -> None:
        """The point of the feature: flagged special-mode rows stop biasing the ratio."""
        scada, schedule, _ = _toggle_case(n=400, uplift=0.05)
        corrupted = _flag(scada, turbine="T1", every=4)
        spoiled = corrupted[_EXCLUDE_COL].to_numpy(dtype=bool) & (corrupted[_TURBINE_COL] == "T1").to_numpy()
        corrupted.loc[spoiled, _POWER_COL] *= 0.5

        naive = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(scada_df=corrupted, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        filtered = ToggleSpecialistMethod(columns=_EXCLUDE_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(scada_df=corrupted, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert filtered.p50_overall == pytest.approx(0.05, abs=1e-9)
        assert abs(naive.p50_overall - 0.05) > abs(filtered.p50_overall - 0.05)

    def test_flagged_reference_rows_leave_the_reference_leg_like_an_outage(self, tmp_path: Path) -> None:
        """A flagged reference row (curtailed, iced, parked) removes the timestamp from the reference leg.

        The block leg uses no references, so it keeps the timestamp and the headline ``used`` stays.
        """
        scada, schedule = _noisy_toggle_case()
        # flag both references, so the flags reach whichever subset the reference leg keeps
        flagged = _flag(scada, turbine="R1", every=3)
        flagged[_EXCLUDE_COL] = (
            flagged[_EXCLUDE_COL].to_numpy() | _flag(scada, turbine="R2", every=3)[_EXCLUDE_COL].to_numpy()
        )
        out = ToggleSpecialistMethod(columns=_EXCLUDE_COLUMNS, reference_block=_CYCLE, out_dir=tmp_path).estimate(
            MethodInput(scada_df=flagged, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert out.labeled_rows is not None
        is_ref = (flagged[_TURBINE_COL] != "T1").to_numpy()
        flagged_ts = flagged.index[is_ref & flagged[_EXCLUDE_COL].to_numpy(dtype=bool)].unique()
        used = out.labeled_rows["used"].astype(bool)
        assert used.loc[flagged_ts].all()
        kept = _read_only_csv(tmp_path, "selection").set_index(["component", "stage", "segment"])["n_kept"]
        for segment in ("baseline", "upgraded"):
            assert kept[("sum", "exclude_row", segment)] < kept[("sum", "filters", segment)]
            assert kept[("block_mean", "exclude_row", segment)] == kept[("block_mean", "filters", segment)]

    def test_nan_in_the_exclude_column_is_rejected(self) -> None:
        """NaN is not "excluded": the schema contract says the column is never NaN, so say so loudly."""
        scada, schedule = _noisy_toggle_case()
        flagged = _flag(scada, turbine="T1", every=7, value=np.nan)
        with pytest.raises(ValueError, match=_EXCLUDE_COL):
            ToggleSpecialistMethod(columns=_EXCLUDE_COLUMNS, reference_block=_CYCLE).estimate(
                MethodInput(scada_df=flagged, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
            )

    def test_unset_role_excludes_nothing(self) -> None:
        scada, schedule = _noisy_toggle_case()
        flagged = _flag(scada, turbine="T1", every=5)
        out = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(scada_df=flagged, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert out.labeled_rows is not None
        excluded = out.labeled_rows[_EXCLUDE_COL].to_numpy(dtype=bool)
        assert out.labeled_rows.loc[excluded, "used"].to_numpy().any()

    def test_absent_column_excludes_nothing(self) -> None:
        """The role names a column the frame does not carry: skip, do not raise."""
        scada, schedule = _noisy_toggle_case()
        out = ToggleSpecialistMethod(columns=_EXCLUDE_COLUMNS, reference_block=_CYCLE).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        assert out.labeled_rows is not None
        assert out.labeled_rows["used"].to_numpy().any()

    def test_exclusions_reach_the_shared_diagnostics(self, tmp_path: Path) -> None:
        """The exclusion mask is handed to the shared diagnostics rather than plotted bespokely.

        The 2x3 operating-curve view of the same mask needs a wind-speed column this fixture does
        not carry, so it is exercised in the diagnostics tests; here the timeline is the evidence
        that ``excluded_ts`` is threaded through.
        """
        scada, schedule = _noisy_toggle_case()
        flagged = _flag(scada, turbine="T1", every=5)
        ToggleSpecialistMethod(
            columns=_EXCLUDE_COLUMNS, reference_block=_CYCLE, out_dir=tmp_path, save_plots=True
        ).estimate(MethodInput(scada_df=flagged, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL))
        run_dir = next(p for p in Path(tmp_path).iterdir() if p.is_dir())
        names = {p.name for p in (run_dir / "plots").rglob("*.png")}
        assert "excluded_row_fraction.png" in names
        assert "T1_excluded_rows.png" not in names, "the bespoke plot was replaced by the shared views"

    def test_no_exclusion_plots_when_nothing_is_excluded(self, tmp_path: Path) -> None:
        scada, schedule = _noisy_toggle_case()
        ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE, out_dir=tmp_path, save_plots=True).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        run_dir = next(p for p in Path(tmp_path).iterdir() if p.is_dir())
        names = {p.name for p in (run_dir / "plots").rglob("*.png")}
        assert "ops_curves_excluded.png" not in names
        assert "excluded_row_fraction.png" not in names


class TestCampaignContext:
    """The method takes reference membership and row validity from the campaign context."""

    @staticmethod
    def _fixture() -> tuple[pd.DataFrame, ToggleSchedule]:
        # Time-varying power throughout: with constant power the ratio-of-ratios cancels and
        # neither manipulation below could change the answer.
        index = _index(96)
        schedule = ToggleSchedule(period=pd.Timedelta(hours=4), start=index[0])
        treated = treated_mask(index, schedule)
        rng = np.random.default_rng(0)
        counterfactual = 100.0 + rng.uniform(0.0, 50.0, len(index))
        test = np.where(treated, counterfactual * 1.05, counterfactual)
        return (
            _scada(
                {
                    "T1": test,
                    "T2": 100.0 + rng.uniform(0.0, 50.0, len(index)),
                    "T3": 400.0 + rng.uniform(0.0, 200.0, len(index)),
                },
                index,
            ),
            schedule,
        )

    @staticmethod
    def _estimate(scada: pd.DataFrame, **kwargs: object) -> float:
        method = ToggleSpecialistMethod(columns=_COLUMNS, reference_block=_CYCLE)
        return method.estimate(MethodInput(scada_df=scada, **kwargs)).p50_overall

    def test_only_offered_references_are_used(self) -> None:
        scada, schedule = self._fixture()
        context = CampaignContext.from_frame(scada, test_wtg="T1", timing=schedule, turbine_col=_TURBINE_COL)
        object.__setattr__(context, "candidate_references", ["T2"])
        offered_t2_only = self._estimate(scada, test_wtg="T1", campaign_context=context)

        assert offered_t2_only == pytest.approx(
            self._estimate(
                scada[scada[_TURBINE_COL] != "T3"], test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL
            )
        )
        with_t3 = self._estimate(scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        assert offered_t2_only != pytest.approx(with_t3)

    def test_rows_a_reference_may_not_contribute_are_dropped(self) -> None:
        scada, schedule = self._fixture()
        context = CampaignContext.from_frame(scada, test_wtg="T1", timing=schedule, turbine_col=_TURBINE_COL)
        valid = context.valid_for_uplift.copy()
        valid.loc[valid.index[:24], "T2"] = False
        object.__setattr__(context, "valid_for_uplift", valid)
        with_holes = self._estimate(scada, test_wtg="T1", campaign_context=context)

        holed = scada.copy()
        holed.loc[(holed.index < valid.index[24]) & (holed[_TURBINE_COL] == "T2"), _POWER_COL] = np.nan
        assert with_holes == pytest.approx(
            self._estimate(holed, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )
        untouched = self._estimate(scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        assert with_holes != pytest.approx(untouched)


# --- pairing filter (pairing_max_gap) -----------------------------------------------------------


class TestPairWithin:
    """``pair_within`` keeps a row only when the other segment has a row within the gap, inclusive."""

    @staticmethod
    def _one_each(gap_rows: int) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
        index = _index(gap_rows + 1)
        baseline = np.zeros(len(index), dtype=bool)
        upgraded = np.zeros(len(index), dtype=bool)
        baseline[0] = True
        upgraded[-1] = True
        return index, baseline, upgraded

    def test_a_gap_exactly_equal_to_the_limit_is_kept(self) -> None:
        index, baseline, upgraded = self._one_each(gap_rows=2)
        paired = pair_within(index, baseline=baseline, upgraded=upgraded, max_gap=pd.Timedelta(minutes=20))
        assert paired.baseline[0]
        assert paired.upgraded[-1]

    def test_a_gap_one_row_beyond_the_limit_is_dropped(self) -> None:
        index, baseline, upgraded = self._one_each(gap_rows=3)
        paired = pair_within(index, baseline=baseline, upgraded=upgraded, max_gap=pd.Timedelta(minutes=20))
        assert not paired.baseline.any()
        assert not paired.upgraded.any()

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_the_index_resolution_does_not_change_the_gap(self, unit: str) -> None:
        """Real SCADA often arrives at microsecond resolution; a 10-min gap must still read as 10 min."""
        index, baseline, upgraded = self._one_each(gap_rows=1)
        index = index.as_unit(unit)
        assert gap_to_other_segment(index, baseline=baseline, upgraded=upgraded)[0] == pytest.approx(600.0)
        paired = pair_within(index, baseline=baseline, upgraded=upgraded, max_gap=pd.Timedelta(minutes=5))
        assert not paired.baseline.any()

    def test_an_empty_other_segment_drops_everything(self) -> None:
        index = _index(6)
        baseline = np.ones(len(index), dtype=bool)
        paired = pair_within(
            index, baseline=baseline, upgraded=np.zeros(len(index), dtype=bool), max_gap=pd.Timedelta(hours=1)
        )
        assert not paired.baseline.any()

    @pytest.mark.parametrize("gap_minutes", [10, 30, 60])
    def test_matches_v0_any_within_timedelta(self, gap_minutes: int) -> None:
        """Parity with wind-up v0's ``_toggle_pairing_filter`` on a gappy toggle frame."""
        v0 = pytest.importorskip("wind_up_v0.main_analysis")
        rng = np.random.default_rng(3)
        full = _index(400)
        index = full[rng.random(len(full)) > 0.3]  # drop rows so gaps vary
        state = rng.random(len(index)) > 0.5
        cols = {"ws": 1.0, "pw": 1.0, "wd": 1.0}
        pre_df = pd.DataFrame(cols, index=index[~state])
        post_df = pd.DataFrame(cols, index=index[state])
        v0_pre, v0_post = v0._toggle_pairing_filter(  # noqa: SLF001 - the v0 behaviour being matched
            pre_df=pre_df,
            post_df=post_df,
            pairing_filter_method="any_within_timedelta",
            pairing_filter_timedelta_seconds=gap_minutes * 60,
            detrend_ws_col="ws",
            test_pw_col="pw",
            ref_wd_col="wd",
            timebase_s=600,
        )

        paired = pair_within(index, baseline=~state, upgraded=state, max_gap=pd.Timedelta(minutes=gap_minutes))
        assert set(index[paired.baseline]) == set(v0_pre.index)
        assert set(index[paired.upgraded]) == set(v0_post.index)


class TestPairingFilter:
    """``pairing_max_gap`` on the method: validation, the rows it drops, and the selection accounting."""

    @staticmethod
    def _block_excluded_case() -> tuple[pd.DataFrame, ToggleSchedule, pd.DatetimeIndex]:
        """A noisy toggle campaign whose upgraded test rows are all flagged over one 1-day block."""
        scada, schedule = _noisy_toggle_case()
        idx = pd.DatetimeIndex(pd.unique(scada.index))
        treated = np.asarray(treated_mask(idx, schedule))
        block = (idx >= idx[500]) & (idx < idx[644])
        flagged = pd.Series(block & treated, index=idx)
        out = scada.copy()
        is_test = out[_TURBINE_COL] == "T1"
        out[_EXCLUDE_COL] = False
        out.loc[is_test, _EXCLUDE_COL] = flagged.reindex(out.index[is_test]).to_numpy()
        return out, schedule, idx[block]

    @staticmethod
    def _run(scada: pd.DataFrame, schedule: ToggleSchedule, **kwargs: object) -> MethodOutput:
        kwargs.setdefault("reference_block", _CYCLE)
        return ToggleSpecialistMethod(columns=_EXCLUDE_COLUMNS, **kwargs).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )

    @pytest.mark.parametrize("gap", [pd.Timedelta(0), pd.Timedelta(minutes=5), pd.Timedelta(minutes=25)])
    def test_rejects_a_gap_that_is_not_a_positive_timebase_multiple(self, gap: pd.Timedelta) -> None:
        scada, schedule = _noisy_toggle_case(n=200)
        with pytest.raises(ValueError, match="pairing_max_gap"):
            _estimate(scada, schedule, pairing_max_gap=gap)

    def test_baseline_rows_far_from_any_used_upgraded_row_are_dropped(self) -> None:
        scada, schedule, block = self._block_excluded_case()
        out = self._run(scada, schedule, pairing_max_gap=pd.Timedelta(minutes=20))
        assert out.labeled_rows is not None
        rows = out.labeled_rows
        inner = block[6:-6]  # more than 20 min from the block's edges
        assert not rows.loc[inner, "used"].astype(bool).any()
        outside = rows.index < block[0]
        assert rows.loc[outside, "used"].astype(bool).all()

    def test_without_pairing_the_block_keeps_its_baseline_rows(self) -> None:
        scada, schedule, block = self._block_excluded_case()
        out = self._run(scada, schedule)
        assert out.labeled_rows is not None
        rows = out.labeled_rows.loc[block]
        assert rows.loc[rows["segment"] == "baseline", "used"].astype(bool).all()

    def test_accounting_shows_each_stage_per_segment(self) -> None:
        scada, schedule, block = self._block_excluded_case()
        out = self._run(scada, schedule, pairing_max_gap=pd.Timedelta(minutes=20))
        assert out.selection_accounting is not None
        kept = out.selection_accounting.set_index(["stage", "segment"])["n_kept"]
        n_block = len(block) // 2
        assert kept["exclude_row", "upgraded"] == kept["filters", "upgraded"] - n_block
        assert kept["exclude_row", "baseline"] == kept["filters", "baseline"]
        assert kept["pairing", "baseline"] < kept["exclude_row", "baseline"]
        assert kept["pairing", "upgraded"] <= kept["exclude_row", "upgraded"]

    def test_accounting_is_reported_without_pairing(self) -> None:
        scada, schedule = _noisy_toggle_case(n=200)
        out = _estimate(scada, schedule)
        assert out.selection_accounting is not None
        kept = out.selection_accounting.set_index(["stage", "segment"])["n_kept"]
        assert kept["pairing", "baseline"] == kept["exclude_row", "baseline"]

    def test_an_uneven_stage_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        scada, schedule, _ = self._block_excluded_case()
        with caplog.at_level("WARNING", logger=toggle_specialist.__name__):
            self._run(scada, schedule, segment_imbalance_warning=0.05)
        assert any("exclude_row" in record.getMessage() for record in caplog.records)

    def test_an_even_selection_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        scada, schedule = _noisy_toggle_case(n=200)
        with caplog.at_level("WARNING", logger=toggle_specialist.__name__):
            _estimate(scada, schedule, pairing_max_gap=pd.Timedelta(minutes=20))
        assert not [r for r in caplog.records if r.name == toggle_specialist.__name__]


@pytest.mark.parametrize("gap", [None, pd.Timedelta(minutes=20)], ids=["no-pairing", "pairing"])
def test_pairing_gap_plot_is_written(tmp_path: Path, gap: pd.Timedelta | None) -> None:
    """The gap histogram is drawn with or without a pairing filter, so a user can judge whether one is needed."""
    scada, schedule = _noisy_toggle_case(n=200)
    _estimate(scada, schedule, out_dir=tmp_path, save_plots=True, pairing_max_gap=gap)
    assert len(list(tmp_path.rglob("*_pairing_gap.png"))) == 1


# --- the two legs, their blend and the decisions on permuted labels -----------------------------

_BLOCK = pd.Timedelta(minutes=40)  # one full 20-min on / 20-min off cycle = 4 rows of 10 min


def _block_case(
    n_blocks: int = 120,
    *,
    uplift: float = 0.04,
    ramp: float = 0.0,
    ref_scale: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, ToggleSchedule]:
    """A toggle campaign whose power level jumps wildly between 40-min blocks (two on/off cycles each).

    Test power is ``level_b * shape_i``, times ``1 + uplift`` when treated, where ``shape`` is a
    within-block ramp of slope ``ramp`` per row (flat by default: a ramp in phase with the toggle
    is exactly the drift the block leg cannot cancel). With ``ref_scale`` (one factor per block) a
    reference ``R1 = ref_scale_b * level_b * shape_i`` is added: it tracks the test turbine inside
    every block but its ratio to the test changes from block to block.
    """
    rng = np.random.default_rng(seed)
    idx = _index(4 * n_blocks)
    schedule = ToggleSchedule(period=pd.Timedelta(minutes=20), start=idx[0])
    treated = np.asarray(treated_mask(idx, schedule))
    level = np.repeat(rng.uniform(200.0, 2000.0, n_blocks), 4)
    shape = np.tile(1.0 + ramp * np.arange(4), n_blocks)
    base = level * shape
    test = np.where(treated, base * (1.0 + uplift), base)
    turbines = {"T1": test}
    if ref_scale is not None:
        turbines["R1"] = np.repeat(ref_scale, 4) * base
    return _scada(turbines, idx), schedule


def _two_reference_case(noise_frac: float = 0.3, seed: int = 5) -> tuple[pd.DataFrame, ToggleSchedule]:
    """``_block_case`` with a perfect tracker ``R1`` plus a noisy tracker ``R2``: the union is worse than R1 alone."""
    scada, schedule = _block_case(uplift=0.04, ref_scale=np.full(120, 0.5), seed=seed)
    rng = np.random.default_rng(seed)
    r2 = scada[scada[_TURBINE_COL] == "R1"].copy()
    r2[_TURBINE_COL] = "R2"
    r2[_POWER_COL] = r2[_POWER_COL].to_numpy() * (1.0 + noise_frac * rng.standard_normal(len(r2)))
    return pd.concat([scada, r2]), schedule


def _legs(out: MethodOutput) -> set[str]:
    return set(_long_diagnostics(out)["component"])


class TestReferenceBlock:
    """``reference_block`` is the caller's statement of one toggle cycle: required, never inferred."""

    def test_it_is_required(self) -> None:
        with pytest.raises(TypeError, match="reference_block"):
            ToggleSpecialistMethod(columns=_COLUMNS)  # type: ignore[call-arg]

    def test_it_is_independent_of_the_pairing_gap(self) -> None:
        scada, schedule = _block_case(uplift=0.04, ref_scale=np.full(120, 0.5))
        out = _estimate(scada, schedule, reference_block=_BLOCK, pairing_max_gap=pd.Timedelta(minutes=10))
        assert out.p50_overall == pytest.approx(0.04, abs=1e-9)

    def test_a_block_off_the_timebase_grid_is_refused(self) -> None:
        scada, schedule = _block_case(n_blocks=10, ref_scale=np.ones(10))
        with pytest.raises(ValueError, match="reference_block"):
            _estimate(scada, schedule, reference_block=pd.Timedelta(minutes=25))


class TestBlockLeg:
    """The block leg: the test turbine against its own block's mean power, needing no references."""

    def test_recovers_the_uplift_with_no_references(self) -> None:
        scada, schedule = _block_case(uplift=0.04)
        out = _estimate(scada, schedule, reference_block=_BLOCK)
        assert out.p50_overall == pytest.approx(0.04, abs=1e-9)
        assert np.isfinite(out.sigma_overall)
        assert _legs(out) == {"block_mean"}

    def test_ignores_a_reference_outage(self) -> None:
        """A reference the leg does not use must not cost it rows; the estimate is then this leg alone."""
        scada, schedule = _block_case(uplift=0.04, ref_scale=np.full(120, 0.5))
        is_ref = (scada[_TURBINE_COL] == "R1").to_numpy()
        scada.loc[is_ref, _POWER_COL] = np.nan
        out = _estimate(scada, schedule, reference_block=_BLOCK)
        assert out.p50_overall == pytest.approx(0.04, abs=1e-9)
        assert out.labeled_rows is not None
        assert out.labeled_rows["used"].astype(bool).all()
        assert _legs(out) == {"block_mean"}

    def test_a_block_with_one_state_is_dropped_and_accounted(self) -> None:
        scada, schedule = _block_case(uplift=0.04)
        idx = pd.DatetimeIndex(pd.unique(scada.index))
        block_rows = idx[40:44]  # the 11th block: two on/off cycles
        off_rows = block_rows[~np.asarray(treated_mask(block_rows, schedule))]
        scada.loc[scada.index.isin(off_rows), _POWER_COL] = np.nan
        out = _estimate(scada, schedule, reference_block=_BLOCK)
        assert out.labeled_rows is not None
        # NaN (a timestamp absent from the pivot) must read as unused, so compare rather than cast
        used = out.labeled_rows["used"] == True  # noqa: E712
        assert not used.loc[block_rows].any()
        assert used.sum() == len(idx) - 4
        kept = out.selection_accounting.set_index(["stage", "segment"])["n_kept"]
        assert kept["block", "upgraded"] == kept["pairing", "upgraded"] - 2
        assert kept["block", "baseline"] == kept["pairing", "baseline"]

    def test_the_reference_leg_keeps_every_block_without_flags(self) -> None:
        scada, schedule = _noisy_toggle_case(n=200)
        out = _estimate(scada, schedule, pairing_max_gap=pd.Timedelta(minutes=20))
        kept = out.selection_accounting.set_index(["stage", "segment"])["n_kept"]
        stages = ["segment", "filters", "exclude_row", "pairing", "block"]
        assert list(out.selection_accounting["stage"].unique()) == stages
        assert kept["block", "baseline"] == kept["pairing", "baseline"]


class TestBlend:
    """The estimate blends the reference and block legs by minimum-variance weights."""

    def test_recovers_the_uplift_when_both_legs_are_exact(self) -> None:
        scada, schedule = _block_case(uplift=0.04, ref_scale=np.full(120, 0.5))
        out = _estimate(scada, schedule, reference_block=_BLOCK)
        assert out.p50_overall == pytest.approx(0.04, abs=1e-9)
        assert _legs(out) == {"sum", "block_mean", "combined"}

    def test_used_rows_are_the_union_of_the_legs(self) -> None:
        scada, schedule = _block_case(uplift=0.04, ref_scale=np.full(120, 0.5))
        is_ref = (scada[_TURBINE_COL] == "R1").to_numpy()
        scada.loc[is_ref & (np.arange(len(scada)) % 7 == 0), _POWER_COL] = np.nan
        out = _estimate(scada, schedule, reference_block=_BLOCK)
        used = out.labeled_rows["used"] == True  # noqa: E712 - NaN (absent timestamp) must read as unused
        assert used.all()
        # The accounting is the reference leg's, which the reference outage did cost rows.
        kept = out.selection_accounting.set_index(["stage", "segment"])["n_kept"]
        assert kept["block", "baseline"] + kept["block", "upgraded"] < used.sum()

    def test_diagnostics_carry_both_legs_and_the_blend(self) -> None:
        scada, schedule = _noisy_toggle_case(n=400)
        out = _estimate(scada, schedule, pairing_max_gap=pd.Timedelta(minutes=20))
        assert _legs(out) == {"sum", "block_mean", "combined"}
        diag = _long_diagnostics(out)
        head = diag[(diag["component"] == "combined") & (diag["condition_bin"] == "overall")].iloc[0]
        assert 0.0 <= head["weight_sum"] <= 1.0
        assert np.isfinite(head["decision_sigma_sum"])
        assert np.isfinite(head["decision_sigma_block_mean"])

    def test_the_blend_is_the_weighted_legs_and_its_sigma_comes_from_the_actual_bootstraps(
        self, tmp_path: Path
    ) -> None:
        scada, schedule = _noisy_toggle_case(n=600)
        out = _estimate(scada, schedule, pairing_max_gap=pd.Timedelta(minutes=20), out_dir=tmp_path)
        results = _read_only_csv(tmp_path, "results").iloc[0]
        diag = _long_diagnostics(out)
        head = diag[(diag["component"] == "combined") & (diag["condition_bin"] == "overall")].iloc[0]
        legs = {
            m: diag[(diag["component"] == m) & (diag["condition_bin"] == "overall")].iloc[0]
            for m in ("sum", "block_mean")
        }
        w = head["weight_sum"]
        assert out.p50_overall == pytest.approx(
            w * results["uplift_frc_sum"] + (1 - w) * results["uplift_frc_block_mean"]
        )
        sa, sb, r = legs["sum"]["sigma"], legs["block_mean"]["sigma"], head["correlation"]
        assert sa == pytest.approx(results["uplift_sigma_frc_sum"])
        assert sb == pytest.approx(results["uplift_sigma_frc_block_mean"])
        variance = w**2 * sa**2 + (1 - w) ** 2 * sb**2 + 2 * w * (1 - w) * r * sa * sb
        assert out.sigma_overall == pytest.approx(np.sqrt(variance))
        # The weights were judged on the permuted sigmas, which are not the reported ones.
        assert results["decision_sigma_frc_sum"] != results["uplift_sigma_frc_sum"]
        assert results["decision_sigma_frc_block_mean"] != results["uplift_sigma_frc_block_mean"]


_BLOCK_COL = "cycle_bad"
_BLOCK_COLUMNS = replace(_COLUMNS, exclude_block=_BLOCK_COL)


def _flag_block(scada: pd.DataFrame, *, turbine: str, at: pd.Timestamp) -> pd.DataFrame:
    """Return a copy of ``scada`` with an all-False ``exclude_block`` column set on one ``turbine`` row."""
    out = scada.copy()
    out[_BLOCK_COL] = False
    out.loc[((out[_TURBINE_COL] == turbine) & (out.index == at)).to_numpy(), _BLOCK_COL] = True
    return out


class TestExcludeBlock:
    """``columns.exclude_block``: a flagged row removes its whole reference block, in both legs."""

    @staticmethod
    def _run(scada: pd.DataFrame, schedule: ToggleSchedule, **kwargs: object) -> MethodOutput:
        return ToggleSpecialistMethod(columns=_BLOCK_COLUMNS, **kwargs).estimate(
            MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )

    def test_a_flagged_test_row_removes_its_whole_block(self) -> None:
        scada, schedule = _block_case(uplift=0.04, ref_scale=np.full(120, 0.5))
        idx = pd.DatetimeIndex(pd.unique(scada.index))
        block_5 = idx[20:24]
        out = self._run(_flag_block(scada, turbine="T1", at=block_5[1]), schedule, reference_block=_BLOCK)
        assert out.labeled_rows is not None
        used = out.labeled_rows["used"].astype(bool)
        assert not used.loc[block_5].any()
        assert used.drop(block_5).all()
        assert out.p50_overall == pytest.approx(0.04, abs=1e-9)
        acc = out.selection_accounting.set_index(["stage", "segment"])["n_kept"]
        for segment in ("baseline", "upgraded"):
            assert acc.loc[("block", segment)] == acc.loc[("pairing", segment)] - 2

    def test_a_flagged_reference_row_removes_the_block_from_the_reference_leg(self) -> None:
        """The block leg uses no references, so it keeps the block and the headline ``used`` stays."""
        scada, schedule = _block_case(uplift=0.04, ref_scale=np.full(120, 0.5))
        idx = pd.DatetimeIndex(pd.unique(scada.index))
        block_7 = idx[28:32]
        out = self._run(_flag_block(scada, turbine="R1", at=block_7[3]), schedule, reference_block=_BLOCK)
        assert out.labeled_rows is not None
        assert out.labeled_rows["used"].astype(bool).all()
        acc = out.selection_accounting.set_index(["stage", "segment"])["n_kept"]
        for segment in ("baseline", "upgraded"):
            assert acc.loc[("block", segment)] == acc.loc[("pairing", segment)] - 2
        assert out.p50_overall == pytest.approx(0.04, abs=1e-9)

    def test_no_flags_change_nothing(self) -> None:
        scada, schedule = _block_case(uplift=0.04, ref_scale=np.full(120, 0.5))
        scada[_BLOCK_COL] = False
        with_role = self._run(scada, schedule, reference_block=_BLOCK)
        without = _estimate(scada.drop(columns=_BLOCK_COL), schedule, reference_block=_BLOCK)
        assert with_role.p50_overall == pytest.approx(without.p50_overall)
        assert with_role.sigma_overall == pytest.approx(without.sigma_overall)


class TestReferenceSubsets:
    """The reference leg takes the reference subset with the smallest label-permuted sigma."""

    def test_picks_the_subset_with_the_smallest_permuted_sigma(self, tmp_path: Path) -> None:
        scada, schedule = _two_reference_case()
        out = _estimate(scada, schedule, out_dir=tmp_path)
        results = _read_only_csv(tmp_path, "results")
        assert results.loc[0, "refs_used"] == "R1"
        assert results.loc[0, "n_refs"] == 1
        assert results.loc[0, "n_refs_available"] == 2
        assert out.p50_overall == pytest.approx(0.04, abs=1e-9)
        subsets = _read_only_csv(tmp_path, "reference_subsets")
        assert subsets["decision_sigma_frc"].notna().all()
        chosen = subsets.loc[subsets["chosen"], "refs"].item()
        assert chosen == subsets.loc[subsets["decision_sigma_frc"].idxmin(), "refs"]
        assert chosen == "R1"  # the perfect tracker is the least noisy on any basis

    def test_every_subset_is_written_and_only_the_winner_has_an_actual_sigma(self, tmp_path: Path) -> None:
        scada, schedule = _two_reference_case()
        _estimate(scada, schedule, out_dir=tmp_path)
        subsets = _read_only_csv(tmp_path, "reference_subsets").set_index("refs")
        assert sorted(subsets.index) == ["R1", "R1;R2", "R2"]
        assert subsets["chosen"].sum() == 1
        assert (subsets["n_used_timestamps"] > 0).all()
        assert subsets["uplift_sigma_frc"].notna().sum() == 1
        assert np.isfinite(subsets.loc["R1", "uplift_sigma_frc"])

    def test_the_reported_sigma_is_the_actual_one_of_the_chosen_subset(self) -> None:
        scada, schedule = _two_reference_case()
        chosen = _estimate(scada, schedule)
        only_r1 = _estimate(scada[scada[_TURBINE_COL] != "R2"], schedule)
        assert chosen.p50_overall == pytest.approx(only_r1.p50_overall)
        assert chosen.sigma_overall == pytest.approx(only_r1.sigma_overall)

    def test_the_permuted_sigma_measures_the_same_noise_as_the_actual_one(self, tmp_path: Path) -> None:
        """For the noisy reference the two sigmas agree; they differ only in whether the on/off split is real."""
        scada, schedule = _two_reference_case()
        _estimate(scada, schedule, out_dir=tmp_path / "both")
        _estimate(scada[scada[_TURBINE_COL] != "R1"], schedule, out_dir=tmp_path / "r2")
        permuted = _read_only_csv(tmp_path / "both", "reference_subsets").set_index("refs")["decision_sigma_frc"]
        actual_r2 = _read_only_csv(tmp_path / "r2", "results").loc[0, "uplift_sigma_frc_sum"]
        assert permuted["R2"] == pytest.approx(actual_r2, rel=0.3)
        assert permuted["R1"] < permuted["R2"]


class TestPermutedLabels:
    """``permute_labels_within_blocks``: the same campaign with the on/off labels shuffled per block."""

    def test_every_block_keeps_its_on_off_counts(self) -> None:
        rng = np.random.default_rng(0)
        n = 40
        upgraded = np.arange(n) % 4 < 2  # two on, two off per block of four
        baseline = ~upgraded
        used = np.ones(n, dtype=bool)
        used[[3, 10, 11]] = False
        ids = np.arange(n) // 4
        up_p, base_p = toggle_specialist.permute_labels_within_blocks(
            upgraded=upgraded, baseline=baseline, used=used, ids=ids, rng=rng
        )
        assert not np.array_equal(up_p, upgraded)
        assert np.array_equal(up_p & used, ~base_p & used)  # every used row is exactly one state
        assert np.array_equal(up_p[~used], upgraded[~used])  # unused rows untouched
        for block in np.unique(ids):
            rows = (ids == block) & used
            assert (up_p & rows).sum() == (upgraded & rows).sum()
            assert (base_p & rows).sum() == (baseline & rows).sum()

    def test_without_blocks_the_labels_are_shuffled_globally(self) -> None:
        rng = np.random.default_rng(1)
        upgraded = np.arange(12) % 2 == 0
        up_p, base_p = toggle_specialist.permute_labels_within_blocks(
            upgraded=upgraded, baseline=~upgraded, used=np.ones(12, dtype=bool), ids=None, rng=rng
        )
        assert up_p.sum() == upgraded.sum()
        assert np.array_equal(up_p, ~base_p)
