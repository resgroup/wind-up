"""``PowerModelMethod``: the simplest-possible ML uplift method behind the harness ``Method`` seam.

A single supervised **counterfactual power model**: learn the test turbine's normal power as a
function of *reference-only* (upgrade-invariant) features over the baseline period, predict the
counterfactual power over the upgraded period, and take the energy ratio
``uplift = sum(actual) / sum(counterfactual) - 1`` over the upgraded rows. No propensity, no
cross-fitting — fit on the baseline, predict on the disjoint upgraded window, so there is no
in-sample leakage to correct for.

The feature set is curated to the *causes* of test-turbine power — weather (all raw ERA5) and
wakes (each reference's active power + availability). Expressing expected power *through the
references* forms the test-vs-reference contrast that cancels common-mode seasonal/long-term
drift (the lever the R-learner lacks; findings F1).

It is v0-independent — ERA5 is supplied as a plain hourly DataFrame (the driver fetches it), so
this package imports nothing from ``wind_up``.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from benchmarking.baselines.era5_sync import sync_era5
from benchmarking.baselines.filtering import NormalOperationFilter
from benchmarking.baselines.naive_ratio import restrict_to_campaign
from benchmarking.baselines.power_model import diagnostics as diag
from benchmarking.baselines.power_model.features import (
    build_reference_features,
    check_reference_only,
    era5_feature_frame,
    extract_outcome,
    reference_mean_wind_speed,
    test_condition_signals,
)
from benchmarking.baselines.power_model.matching import coarsened_exact_match
from benchmarking.baselines.rlearner.nuisance import make_outcome_model
from benchmarking.diagnostics import DiagnosticContext, write_common_diagnostics, write_run_config
from benchmarking.harness.conditions import CONDITION_BINS, energy_ratio_by_bin
from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule, treated_mask

if TYPE_CHECKING:
    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)

_MIN_BASELINE_ROWS = 10
_MIN_HOLDOUT_ROWS = 20  # below this, report the in-sample fit (no point splitting off a tiny valid set)

# F6 matching set + bin widths for the bias-cancellation correction, verified on real HoT by the CEM
# coverage sweep (docs/v1/findings.md F6). wind_speed_100m (dominant) is binned finest, wind_gusts_10m
# coarser, and wind_direction_100m in 20° sectors (a reanalysis direction — finer is finer than the
# signal). Fixed sectors, no wraparound (adjacent sectors are just separate cells, per the CEM utility).
_DEFAULT_MATCHING_VARS: tuple[str, ...] = ("wind_speed_100m", "wind_gusts_10m", "wind_direction_100m")
_DEFAULT_MATCHING_BIN_EDGES: dict[str, list[float]] = {
    "wind_speed_100m": [float(x) for x in np.arange(0.0, 34.0, 2.0)],  # 0,2,…,32
    "wind_gusts_10m": [float(x) for x in np.arange(0.0, 48.0, 3.0)],  # 0,3,…,45
    "wind_direction_100m": [float(x) for x in np.arange(0.0, 380.0, 20.0)],  # 0,20,…,360
}


def _ratio(actual: np.ndarray, counterfactual: np.ndarray) -> float:
    """Energy-ratio ``Σactual / Σcounterfactual - 1`` over finite pairs; NaN if the denominator is 0."""
    finite = np.isfinite(actual) & np.isfinite(counterfactual)
    denom = float(counterfactual[finite].sum())
    return float(actual[finite].sum()) / denom - 1.0 if denom != 0 else float("nan")


def _combine_uplift(r_fwd: np.ndarray, r_rev: np.ndarray) -> np.ndarray:
    """Two-direction uplift ``sqrt((1+r_fwd)/(1+r_rev)) - 1`` (shrinkage cancels); NaN if either 1+r ≤ 0.

    Under a common per-bin multiplicative shrinkage the forward/reverse ratios are ``(1+u)/s`` and
    ``1/(s(1+u))``, so their geometric contrast recovers ``u`` with ``s`` cancelled (design/F5).
    """
    a = 1.0 + np.asarray(r_fwd, dtype=float)
    b = 1.0 + np.asarray(r_rev, dtype=float)
    valid = (a > 0) & (b > 0)
    frac = np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=valid)
    return np.sqrt(frac, out=np.full_like(frac, np.nan), where=np.isfinite(frac)) - 1.0


def _relevel_conditional(sum_actual_b: np.ndarray, one_plus_u_b: np.ndarray, *, one_plus_overall: float) -> np.ndarray:
    """Rescale a per-bin corrected shape by one factor so its energy aggregation equals the overall.

    The per-bin two-direction combine gives the *shape* of the uplift across bins but not a level tied
    to the headline. This scales every bin's ``(1+u_b)`` by ``λ = one_plus_overall / (1+u_agg)``, where
    ``1+u_agg = Σactual / Σ(actual/(1+u_b))`` is the ratio-of-sums the shape currently aggregates to, so
    the re-leveled bins' total MWh uplift equals ``one_plus_overall`` exactly (the "overall = aggregation"
    self-consistency target). Bins with non-finite ``(1+u_b)`` are left NaN and excluded from the sums.
    """
    a = np.asarray(sum_actual_b, dtype=float)
    u1 = np.asarray(one_plus_u_b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(u1) & (u1 != 0)
    cf = np.divide(a, u1, out=np.full(a.shape, np.nan), where=finite)
    denom = float(cf[finite].sum()) if finite.any() else 0.0
    one_plus_u_agg = float(a[finite].sum()) / denom if denom != 0 else float("nan")
    lam = one_plus_overall / one_plus_u_agg if np.isfinite(one_plus_u_agg) and one_plus_u_agg != 0 else float("nan")
    return lam * u1


def _implied_shrinkage(r_fwd: np.ndarray, r_rev: np.ndarray) -> np.ndarray:
    """Implied shrinkage ``s = 1/sqrt((1+r_fwd)(1+r_rev))`` — the bias the correction cancelled (diagnostic)."""
    a = 1.0 + np.asarray(r_fwd, dtype=float)
    b = 1.0 + np.asarray(r_rev, dtype=float)
    valid = (a > 0) & (b > 0)
    prod = np.multiply(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=valid)
    root = np.sqrt(prod, out=np.full_like(prod, np.nan), where=np.isfinite(prod))
    return np.divide(1.0, root, out=np.full_like(root, np.nan), where=np.isfinite(root) & (root != 0))


def _infer_timebase(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Infer the analysis timebase as the median spacing of the sorted unique timestamps."""
    unique = pd.DatetimeIndex(pd.unique(index)).sort_values()
    if len(unique) < 2:  # noqa: PLR2004
        return pd.Timedelta(minutes=10)
    return pd.Timedelta(np.median(np.diff(unique.to_numpy())))


def _upgrade_start(upgrade_timing: pd.Timestamp | ToggleSchedule, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the upgrade-start timestamp (changeover for prepost; toggle origin for toggle)."""
    if isinstance(upgrade_timing, ToggleSchedule):
        return upgrade_timing.start if upgrade_timing.start is not None else index.min()
    return pd.Timestamp(upgrade_timing)


def _clip_predictions(pred: np.ndarray, *, y_train: np.ndarray, rated_power_kw: float) -> np.ndarray:
    """Clip boosted predictions to the physically plausible range of the fitted-on outcome.

    Tree boosting sums trees, so a prediction can drift slightly past the training ``y`` range; the
    clip binds only at those extremes. ``lower = min(0, min(y_train))`` floors at 0 for non-negative
    training data but never pulls a genuinely-negative observation up; ``upper`` is the rated ceiling
    but never below the largest observed training outcome. ``y_train`` is the outcome of the rows the
    predicting model was fitted on; ``rated_power_kw`` is that turbine's rating for those rows.
    """
    lower = min(0.0, float(np.min(y_train)))
    upper = max(float(rated_power_kw), float(np.max(y_train)))
    return np.clip(pred, lower, upper)


@dataclass
class PowerModelMethod:
    """Pluggable counterfactual power-model uplift estimator (prepost and toggle).

    :param active_power_col: the test turbine's active-power column (the outcome ``Y`` and the
        reference active-power feature)
    :param availability_col: **required** "ready to operate" counter; drives the test-turbine
        downtime filter and is itself a reference feature (whether a reference is waking)
    :param baseline_rated_power_kw: **required** rated power of the turbine over the data the model is
        fitted on — today every fit is on baseline rows, so this is the baseline rating. It caps the
        clipped counterfactual predictions. (A future cross-predict direction that trains on upgraded
        data would need the upgraded rating; that is out of scope here.)
    :param wind_speed_col: the wind-speed tag; the reference mean feeds the ERA5 lag sync and the
        stuck-filter calm exemption. Required if ``era5_hourly_df`` is given.
    :param era5_hourly_df: optional raw hourly ERA5 (Open-Meteo columns); added as features when given
    :param columns: source-native column schema, used only by the shared diagnostics (not estimation)
    :param name: method name shown in the leaderboard
    :param out_dir: where per-run folders are written; a temp dir when ``None``
    :param save_plots: also write the diagnostic plots
    :param seed: seed for the baseline holdout split and the LightGBM ``random_state`` (a caller-supplied
        ``random_state`` in ``model_params`` still wins)
    :param model_params: LightGBM overrides passed to the outcome model
    :param timebase: analysis timebase; inferred from the data when ``None``
    :param toggle_campaign_only: for a toggle campaign, fit only on the interleaved on/off blocks
        (drop the pre-campaign baseline) so on and off share a wind distribution; no-op for prepost
    :param bias_correct: opt-in two-direction bias-cancellation (Issue 8). When ``True``, match the
        baseline and upgraded periods on ERA5 weather (CEM) and cancel the counterfactual model's
        conditional shrinkage between two symmetric train/predict directions; **requires ERA5**. When
        ``False`` (default) the single-direction path is used, completely unchanged.
    :param matching_vars: ERA5 columns matched on when ``bias_correct`` (default: the F6 set)
    :param matching_bin_edges: per-variable CEM bin edges; the F6 defaults are used when ``None``
    """

    active_power_col: str
    availability_col: str
    baseline_rated_power_kw: float
    wind_speed_col: str | None = None
    wind_speed_sd_col: str | None = None
    era5_hourly_df: pd.DataFrame | None = None
    columns: ColumnSchema = HOT_COLUMNS
    name: str = "power_model"
    out_dir: Path | None = None
    save_plots: bool = False
    seed: int = 0
    model_params: dict[str, Any] = field(default_factory=dict)
    timebase: pd.Timedelta | None = None
    toggle_campaign_only: bool = True
    bias_correct: bool = False
    matching_vars: tuple[str, ...] = _DEFAULT_MATCHING_VARS
    matching_bin_edges: dict[str, list[float]] | None = None

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Estimate the test turbine's P50 uplift for one campaign and write diagnostics."""
        mi = restrict_to_campaign(mi, toggle_campaign_only=self.toggle_campaign_only)
        scada = mi.scada_df
        if self.availability_col not in scada.columns:
            msg = (
                f"availability_col {self.availability_col!r} is not in scada_df; the downtime filter is "
                f"required for the power model and cannot be skipped."
            )
            raise ValueError(msg)
        index = pd.DatetimeIndex(pd.unique(scada.index)).sort_values()
        timebase = self.timebase if self.timebase is not None else _infer_timebase(scada.index)
        n_refs = scada[mi.turbine_col].nunique() - 1

        y = extract_outcome(
            scada, test_wtg=mi.test_wtg, turbine_col=mi.turbine_col, active_power_col=self.active_power_col
        )
        features = build_reference_features(
            scada,
            test_wtg=mi.test_wtg,
            turbine_col=mi.turbine_col,
            active_power_col=self.active_power_col,
            availability_col=self.availability_col,
        )
        features, era5 = self._add_era5(scada, features, mi=mi, index=index, timebase=timebase)
        check_reference_only(features.columns.tolist(), test_wtg=mi.test_wtg)

        t = np.asarray(treated_mask(index, mi.upgrade_timing)).astype(bool)
        selected = self._select_rows(scada, mi=mi, index=index, y=y, timebase=timebase)
        baseline_sel = selected & ~t
        upgraded_sel = selected & t
        if int(baseline_sel.sum()) < _MIN_BASELINE_ROWS:
            msg = f"too few normally-operating baseline rows ({int(baseline_sel.sum())}) to fit the power model."
            raise ValueError(msg)
        if not upgraded_sel.any():
            msg = "no normally-operating upgraded rows to estimate uplift over."
            raise ValueError(msg)

        if self.bias_correct:
            return self._estimate_bias_corrected(
                scada, mi=mi, index=index, features=features, y=y, baseline_sel=baseline_sel, upgraded_sel=upgraded_sel
            )

        fit = self._fit_predict(
            features, y=y.to_numpy(dtype=float), baseline_sel=baseline_sel, upgraded_sel=upgraded_sel
        )
        sum_actual = float(fit["y_upgraded"].sum())
        sum_counter = float(fit["pred_upgraded"].sum())
        uplift = sum_actual / sum_counter - 1.0 if np.isfinite(sum_counter) and sum_counter != 0 else float("nan")

        by_condition: pd.DataFrame | None = None
        cond_upgraded: pd.DataFrame | None = None
        cond_baseline_valid: pd.DataFrame | None = None
        if self.wind_speed_col is not None:
            conditions = test_condition_signals(
                scada,
                test_wtg=mi.test_wtg,
                turbine_col=mi.turbine_col,
                wind_speed_col=self.wind_speed_col,
                wind_speed_sd_col=self.wind_speed_sd_col,
            )
            by_condition = self._conditional_uplift(conditions, upgraded_sel=upgraded_sel, fit=fit)
            # ws/TI for each segment's residuals, row-aligned to the fit arrays for the diagnostics.
            cond_upgraded = conditions.iloc[upgraded_sel].reset_index(drop=True)
            cond_baseline_valid = conditions.iloc[fit["baseline_valid_pos"]].reset_index(drop=True)

        self._write(
            mi,
            index=index,
            timebase=timebase,
            t=t,
            selected=selected,
            upgraded_sel=upgraded_sel,
            y=y.to_numpy(dtype=float),
            features=features,
            fit=fit,
            uplift=uplift,
            sum_actual=sum_actual,
            sum_counter=sum_counter,
            n_refs=n_refs,
            era5=era5,
            cond_upgraded=cond_upgraded,
            cond_baseline_valid=cond_baseline_valid,
        )
        return MethodOutput(p50_overall=uplift, p50_by_condition=by_condition)

    def _estimate_bias_corrected(
        self,
        scada: pd.DataFrame,
        *,
        mi: MethodInput,
        index: pd.DatetimeIndex,
        features: pd.DataFrame,
        y: pd.Series,
        baseline_sel: np.ndarray,
        upgraded_sel: np.ndarray,
    ) -> MethodOutput:
        """Two-direction (CEM-matched) estimate that cancels the counterfactual model's shrinkage (Issue 8).

        Match the baseline and upgraded periods on ERA5 weather, then fit/predict in both directions and
        combine so the common per-bin shrinkage cancels (design/F5). Requires ERA5 — the matching axis is
        the synced ERA5 columns, which live in ``features`` (``era5_feature_frame`` passes them through).
        """
        if self.era5_hourly_df is None:
            msg = "bias_correct=True requires ERA5 (era5_hourly_df): the matching axis is the ERA5 weather columns."
            raise ValueError(msg)
        y_arr = y.to_numpy(dtype=float)

        # Overall = the uncorrected full-data estimate: train on ALL baseline, predict ALL upgraded, one energy
        # ratio (identical to the bias_correct=False headline). The whole-window shrinkage integrates to ≈ 0
        # (F5) so this is already the cleanest overall; the matched two-direction correction is spent only on
        # the per-bin *decomposition*, which is re-leveled back onto this headline so the MWh adds up (F8).
        fit_full = self._fit_predict(features, y=y_arr, baseline_sel=baseline_sel, upgraded_sel=upgraded_sel)
        sum_actual = float(fit_full["y_upgraded"].sum())
        sum_counter = float(fit_full["pred_upgraded"].sum())
        uplift = sum_actual / sum_counter - 1.0 if np.isfinite(sum_counter) and sum_counter != 0 else float("nan")

        # Matched two-direction fits, for the per-bin shape only. Matching is required: without it the reverse
        # model (train upgraded, predict baseline) would extrapolate out-of-distribution across the prepost
        # weather shift.
        edges = self.matching_bin_edges if self.matching_bin_edges is not None else self._default_bin_edges()
        match = coarsened_exact_match(
            features[list(self.matching_vars)],
            baseline_sel=baseline_sel,
            upgraded_sel=upgraded_sel,
            bin_edges=edges,
            seed=self.seed,
        )
        mb, mu = match.baseline_positions, match.upgraded_positions
        pred_up = self._fit_direction(features, y_arr, train=mb, predict=mu)  # forward: 1+r_fwd = (1+u)/s
        # Reverse (train matched-upgraded, predict matched-baseline): 1+r_rev = 1/(s(1+u)). Clip reuses
        # baseline_rated_power_kw — its upper bound max(rated, max(y_train)) already lifts the ceiling for an
        # uprate, so no separate upgraded-rating field is needed.
        pred_base = self._fit_direction(features, y_arr, train=mu, predict=mb)
        r_fwd = _ratio(y_arr[mu], pred_up)
        r_rev = _ratio(y_arr[mb], pred_base)

        per_bin = self._corrected_conditional(
            scada,
            mi=mi,
            y=y_arr,
            mb=mb,
            mu=mu,
            pred_up=pred_up,
            pred_base=pred_base,
            upgraded_pos=np.flatnonzero(upgraded_sel),
            actual_full=fit_full["y_upgraded"],
            overall_ratio=uplift,
        )
        by_condition = per_bin[["condition", "condition_bin", "p50_uplift"]].copy() if per_bin is not None else None
        self._write_bias_correction(
            mi, index=index, match=match, r_fwd=r_fwd, r_rev=r_rev, uplift=uplift, per_bin=per_bin
        )
        logger.info(
            "%s %s (bias_correct): uplift=%+.3f%% (uncorrected headline)  implied_s=%.3f  "
            "(r_fwd=%+.3f, r_rev=%+.3f, matched=%d/side)",
            self.name,
            mi.test_wtg,
            100 * uplift,
            float(_implied_shrinkage(np.array([r_fwd]), np.array([r_rev]))[0]),
            r_fwd,
            r_rev,
            match.n_matched_per_side,
        )
        return MethodOutput(p50_overall=uplift, p50_by_condition=by_condition)

    def _fit_direction(
        self, features: pd.DataFrame, y: np.ndarray, *, train: np.ndarray, predict: np.ndarray
    ) -> np.ndarray:
        """Fit the outcome model on ``train`` rows and return clipped predictions for ``predict`` rows."""
        model = self._make_model()
        model.fit(features.iloc[train], y[train])
        return _clip_predictions(
            np.asarray(model.predict(features.iloc[predict]), dtype=float),
            y_train=y[train],
            rated_power_kw=self.baseline_rated_power_kw,
        )

    def _corrected_conditional(
        self,
        scada: pd.DataFrame,
        *,
        mi: MethodInput,
        y: np.ndarray,
        mb: np.ndarray,
        mu: np.ndarray,
        pred_up: np.ndarray,
        pred_base: np.ndarray,
        upgraded_pos: np.ndarray,
        actual_full: np.ndarray,
        overall_ratio: float,
    ) -> pd.DataFrame | None:
        """Per-(ws, TI)-bin two-direction uplift *shape*, re-leveled so each marginal aggregates to overall.

        The **shape** ``1+u_b = sqrt((1+r_fwd_b)/(1+r_rev_b))`` comes from the matched forward/reverse fits
        (NaN for a degenerate non-positive-ratio bin). The re-level **weights** are the *full-upgraded* actual
        energy per bin (``actual_full`` over ``upgraded_pos``), so a single per-condition factor pins the
        decomposition's energy aggregation to ``1 + overall_ratio`` — the reported per-bin MWh then partitions
        the full-data headline (F8). Matched forward/reverse stay in-distribution; the full-upgraded energy
        makes the decomposition add up to the full-data overall.
        """
        if self.wind_speed_col is None:
            return None
        conditions = test_condition_signals(
            scada,
            test_wtg=mi.test_wtg,
            turbine_col=mi.turbine_col,
            wind_speed_col=self.wind_speed_col,
            wind_speed_sd_col=self.wind_speed_sd_col,
        )
        cond_up = conditions.iloc[mu].reset_index(drop=True)  # matched-upgraded (forward binning)
        cond_base = conditions.iloc[mb].reset_index(drop=True)  # matched-baseline (reverse binning)
        cond_full = conditions.iloc[upgraded_pos].reset_index(drop=True)  # all upgraded (re-level weights)
        frames = []
        for name in [c for c in ("ws", "ti") if c in conditions.columns]:
            fwd = energy_ratio_by_bin(cond_up[name].to_numpy(), y[mu], pred_up, bins=CONDITION_BINS[name])
            rev = energy_ratio_by_bin(cond_base[name].to_numpy(), y[mb], pred_base, bins=CONDITION_BINS[name])
            # full-upgraded actual energy per bin (counterfactual arg unused — only sum_actual is taken)
            full = energy_ratio_by_bin(cond_full[name].to_numpy(), actual_full, actual_full, bins=CONDITION_BINS[name])
            merged = fwd.merge(rev, on="condition_bin", suffixes=("_fwd", "_rev")).merge(
                full[["condition_bin", "sum_actual"]].rename(columns={"sum_actual": "sum_actual_full"}),
                on="condition_bin",
            )
            r_fwd = merged["p50_uplift_fwd"].to_numpy()  # per-bin energy ratio IS the per-bin r
            r_rev = merged["p50_uplift_rev"].to_numpy()
            one_plus_u = 1.0 + _combine_uplift(r_fwd, r_rev)  # shrinkage-free shape (NaN if degenerate)
            sum_actual_full = merged["sum_actual_full"].to_numpy()
            releveled = _relevel_conditional(sum_actual_full, one_plus_u, one_plus_overall=1.0 + overall_ratio)
            lam = np.divide(releveled, one_plus_u, out=np.full(one_plus_u.shape, np.nan), where=np.isfinite(one_plus_u))
            frames.append(
                pd.DataFrame(
                    {
                        "condition": name,
                        "condition_bin": merged["condition_bin"],
                        "n_records_fwd": merged["n_records_fwd"],
                        "n_records_rev": merged["n_records_rev"],
                        "sum_actual": sum_actual_full,  # full-upgraded energy per bin (the MWh re-level weight)
                        "r_fwd": r_fwd,
                        "r_rev": r_rev,
                        "implied_shrinkage": _implied_shrinkage(r_fwd, r_rev),
                        "u_b": one_plus_u - 1.0,  # per-bin corrected shape, before re-leveling
                        "lambda": lam,  # the condition's re-level factor (constant within a condition)
                        "p50_uplift": releveled - 1.0,  # re-leveled per-bin uplift (aggregates to overall)
                    }
                )
            )
        return pd.concat(frames, ignore_index=True) if frames else None

    def _write_bias_correction(
        self,
        mi: MethodInput,
        *,
        index: pd.DatetimeIndex,
        match: Any,  # noqa: ANN401 - MatchResult
        r_fwd: float,
        r_rev: float,
        uplift: float,
        per_bin: pd.DataFrame | None,
    ) -> None:
        """Write the per-run implied-shrinkage + CEM-balance diagnostics (and the per-bin plot if enabled)."""
        upgrade_start = _upgrade_start(mi.upgrade_timing, index)
        run_name = f"power_model_{mi.test_wtg}_{upgrade_start:%Y%m%d}_{index.max():%Y%m%d}"
        out_root = Path(self.out_dir) if self.out_dir is not None else Path(tempfile.mkdtemp(prefix="power_model_bc_"))
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        overall = {
            "test_wtg": mi.test_wtg,
            "mode": "toggle" if isinstance(mi.upgrade_timing, ToggleSchedule) else "prepost",
            "r_fwd": r_fwd,
            "r_rev": r_rev,
            "uplift_frc": uplift,
            "implied_shrinkage": float(_implied_shrinkage(np.array([r_fwd]), np.array([r_rev]))[0]),
        }
        diag.write_bias_correction_csvs(run_dir, run_name, ts, overall=overall, per_bin=per_bin, match=match)
        if self.save_plots and per_bin is not None:
            diag.plot_implied_shrinkage(run_dir / "plots", per_bin, test_wtg=mi.test_wtg)

    def _default_bin_edges(self) -> dict[str, list[float]]:
        """Per-variable CEM edges for ``matching_vars`` from the F6 defaults; raise on an unknown var."""
        missing = [v for v in self.matching_vars if v not in _DEFAULT_MATCHING_BIN_EDGES]
        if missing:
            msg = (
                f"no default matching bin edges for {missing}; pass matching_bin_edges explicitly for "
                f"non-default matching_vars. Known defaults: {sorted(_DEFAULT_MATCHING_BIN_EDGES)}."
            )
            raise ValueError(msg)
        return {v: _DEFAULT_MATCHING_BIN_EDGES[v] for v in self.matching_vars}

    def _add_era5(
        self,
        scada: pd.DataFrame,
        features: pd.DataFrame,
        *,
        mi: MethodInput,
        index: pd.DatetimeIndex,
        timebase: pd.Timedelta,
    ) -> tuple[pd.DataFrame, Any]:
        """Sync ERA5 (if supplied) and append its features; return the (features, sync-result)."""
        if self.era5_hourly_df is None:
            return features, None
        if self.wind_speed_col is None:
            msg = "wind_speed_col is required to sync ERA5 (it provides the reference wind speed)."
            raise ValueError(msg)
        if self.wind_speed_col not in scada.columns:
            msg = (
                f"wind_speed_col {self.wind_speed_col!r} is not in scada_df; it provides the reference wind "
                f"speed the ERA5 lag sync locks onto. A missing column silently yields an all-NaN reference "
                f"wind speed and a meaningless lag, so this is treated as a configuration error."
            )
            raise ValueError(msg)
        reference_ws = reference_mean_wind_speed(
            scada, test_wtg=mi.test_wtg, turbine_col=mi.turbine_col, wind_speed_col=self.wind_speed_col
        )
        result = sync_era5(self.era5_hourly_df, target_index=index, reference_ws=reference_ws, timebase=timebase)
        return pd.concat([features, era5_feature_frame(result.aligned)], axis=1), result

    def _select_rows(
        self, scada: pd.DataFrame, *, mi: MethodInput, index: pd.DatetimeIndex, y: pd.Series, timebase: pd.Timedelta
    ) -> np.ndarray:
        """Boolean over ``index``: normally-operating test rows (cause-not-effect) with finite outcome."""
        test_rows = scada[scada[mi.turbine_col] == mi.test_wtg].sort_index()
        keep = NormalOperationFilter(
            active_power_col=self.active_power_col,
            wind_speed_col=self.wind_speed_col,
            availability_col=self.availability_col,
        ).keep_mask(test_rows, timebase=timebase)
        keep = keep[~keep.index.duplicated()].reindex(index, fill_value=False)
        return keep.to_numpy() & np.isfinite(y.to_numpy(dtype=float))

    def _fit_predict(
        self, features: pd.DataFrame, *, y: np.ndarray, baseline_sel: np.ndarray, upgraded_sel: np.ndarray
    ) -> dict[str, Any]:
        """Fit on baseline, predict the upgraded counterfactual; also a held-out baseline fit metric.

        The final model (for the counterfactual) is fit on **all** baseline rows. A separate model on
        a baseline train split predicts a held-out baseline slice, giving an honest fit-quality number
        that is not inflated by in-sample optimism.
        """
        x_base = features.iloc[baseline_sel]
        y_base = y[baseline_sel]
        x_up = features.iloc[upgraded_sel]
        y_up = y[upgraded_sel]

        y_valid, pred_valid, valid_local = self._holdout_fit(x_base, y_base)
        baseline_valid_pos = np.flatnonzero(baseline_sel)[valid_local]

        final = self._make_model()
        final.fit(x_base, y_base)
        pred_up = _clip_predictions(
            np.asarray(final.predict(x_up), dtype=float), y_train=y_base, rated_power_kw=self.baseline_rated_power_kw
        )
        return {
            "model": final,
            "pred_upgraded": pred_up,
            "y_upgraded": y_up,
            "y_baseline_valid": y_valid,
            "pred_baseline_valid": pred_valid,
            "baseline_valid_pos": baseline_valid_pos,  # positions over ``index`` of the held-out rows
        }

    def _conditional_uplift(
        self, conditions: pd.DataFrame, *, upgraded_sel: np.ndarray, fit: dict[str, Any]
    ) -> pd.DataFrame | None:
        """Reduce the upgraded actual/counterfactual ledger to per-bin energy-ratio uplift."""
        cond_up = conditions.iloc[upgraded_sel]
        actual = fit["y_upgraded"]
        counterfactual = fit["pred_upgraded"]
        frames = []
        for name in [c for c in ("ws", "ti") if c in cond_up.columns]:
            table = energy_ratio_by_bin(cond_up[name].to_numpy(), actual, counterfactual, bins=CONDITION_BINS[name])
            table.insert(0, "condition", name)
            frames.append(table[["condition", "condition_bin", "p50_uplift"]])
        return pd.concat(frames, ignore_index=True) if frames else None

    def _make_model(self) -> Any:  # noqa: ANN401
        """Outcome model with ``seed`` plumbed into LightGBM's ``random_state`` (caller overrides win)."""
        return make_outcome_model(**{"random_state": self.seed, **self.model_params})

    def _holdout_fit(self, x_base: pd.DataFrame, y_base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Train on a baseline train split, predict a held-out baseline slice (honest fit quality).

        Also returns the held-out rows' positions **within the baseline block** so the caller can
        line the residuals up with their conditions (ws/TI) for the diagnostics.
        """
        n = len(y_base)
        if n < _MIN_HOLDOUT_ROWS:
            model = self._make_model()
            model.fit(x_base, y_base)
            pred = _clip_predictions(
                np.asarray(model.predict(x_base), dtype=float),
                y_train=y_base,
                rated_power_kw=self.baseline_rated_power_kw,
            )
            return y_base, pred, np.arange(n)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n)
        n_valid = n // 5
        valid_idx = order[:n_valid]
        train_idx = order[n_valid:]
        model = self._make_model()
        model.fit(x_base.iloc[train_idx], y_base[train_idx])
        pred_valid = _clip_predictions(
            np.asarray(model.predict(x_base.iloc[valid_idx]), dtype=float),
            y_train=y_base[train_idx],
            rated_power_kw=self.baseline_rated_power_kw,
        )
        return y_base[valid_idx], pred_valid, valid_idx

    def _write(
        self,
        mi: MethodInput,
        *,
        index: pd.DatetimeIndex,
        timebase: pd.Timedelta,
        t: np.ndarray,
        selected: np.ndarray,
        upgraded_sel: np.ndarray,
        y: np.ndarray,
        features: pd.DataFrame,
        fit: dict[str, Any],
        uplift: float,
        sum_actual: float,
        sum_counter: float,
        n_refs: int,
        era5: Any,  # noqa: ANN401
        cond_upgraded: pd.DataFrame | None = None,
        cond_baseline_valid: pd.DataFrame | None = None,
    ) -> None:
        """Assemble the diagnostic data and write the CSVs (+ plots), logging the top features."""
        upgrade_start = _upgrade_start(mi.upgrade_timing, index)
        run_name = f"power_model_{mi.test_wtg}_{upgrade_start:%Y%m%d}_{index.max():%Y%m%d}"
        out_root = Path(self.out_dir) if self.out_dir is not None else Path(tempfile.mkdtemp(prefix="power_model_"))
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S_%f")

        x_sel = features.iloc[selected]
        data = diag.DiagnosticData(
            test_wtg=mi.test_wtg,
            mode="toggle" if isinstance(mi.upgrade_timing, ToggleSchedule) else "prepost",
            index=index,
            treated_all=t,
            selected_all=selected,
            y_all=y,
            timebase=timebase,
            upgraded_ts=index[upgraded_sel],
            y_upgraded=fit["y_upgraded"],
            pred_upgraded=fit["pred_upgraded"],
            y_baseline_valid=fit["y_baseline_valid"],
            pred_baseline_valid=fit["pred_baseline_valid"],
            feature_names=list(features.columns),
            feature_values=x_sel,
            y_selected=y[selected],
            outcome_model=fit["model"],
            overall_uplift=uplift,
            sum_actual_kw=sum_actual,
            sum_counterfactual_kw=sum_counter,
            n_refs=n_refs,
            era5_lag_rows=era5.best_lag_rows if era5 is not None else None,
            era5_corr=era5.best_corr if era5 is not None else None,
            era5_sweep=era5.sweep if era5 is not None else None,
            cond_upgraded=cond_upgraded,
            cond_baseline_valid=cond_baseline_valid,
        )
        importance = diag.write_csvs(run_dir, run_name, ts, data)
        diag.log_top_features(importance)
        logger.info(
            "%s %s: uplift=%+.3f%%  (Σactual=%.1f MWh, Σcounterfactual=%.1f MWh, n_up=%d)",
            self.name,
            mi.test_wtg,
            100 * uplift,
            sum_actual * (timebase / pd.Timedelta(hours=1)) / 1000.0,
            sum_counter * (timebase / pd.Timedelta(hours=1)) / 1000.0,
            len(fit["y_upgraded"]),
        )
        if self.save_plots:
            diag.save_plots(run_dir / "plots", data, importance)
            self._write_shared_diagnostics(mi, run_dir=run_dir, t=t, selected=selected, timebase=timebase, era5=era5)

    def _write_shared_diagnostics(
        self,
        mi: MethodInput,
        *,
        run_dir: Path,
        t: np.ndarray,
        selected: np.ndarray,
        timebase: pd.Timedelta,
        era5: Any,  # noqa: ANN401
    ) -> None:
        """Emit the shared cross-method diagnostics (coverage/curves/histograms) and the run config."""
        columns = replace(self.columns, active_power=self.active_power_col)
        if self.wind_speed_col is not None:
            columns = replace(columns, wind_speed=self.wind_speed_col)
        ctx = DiagnosticContext(
            run_dir=run_dir,
            test_wtg=mi.test_wtg,
            turbine_col=mi.turbine_col,
            columns=columns,
            scada_df=mi.scada_df,
            treated_ts=t.astype(bool),
            used_ts=np.asarray(selected, dtype=bool),
            timebase=timebase,
            mode="toggle" if isinstance(mi.upgrade_timing, ToggleSchedule) else "prepost",
            era5_df=era5.aligned if era5 is not None else None,
        )
        write_common_diagnostics(ctx)
        extra = {
            "era5_lag_rows": era5.best_lag_rows if era5 is not None else None,
            "era5_corr": era5.best_corr if era5 is not None else None,
        }
        write_run_config(ctx, method_name=self.name, method_params=self._config_params(), extra=extra)

    def _config_params(self) -> dict[str, Any]:
        """Return the power-model configuration recorded in the run-config YAML."""
        return {
            "active_power_col": self.active_power_col,
            "availability_col": self.availability_col,
            "baseline_rated_power_kw": self.baseline_rated_power_kw,
            "wind_speed_col": self.wind_speed_col,
            "wind_speed_sd_col": self.wind_speed_sd_col,
            "seed": self.seed,
            "toggle_campaign_only": self.toggle_campaign_only,
            "has_era5": self.era5_hourly_df is not None,
            "model_params": self.model_params,
        }
