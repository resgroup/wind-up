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

from benchmarking.baselines.era5_derived import era5_derived_frame
from benchmarking.baselines.era5_sync import sync_era5
from benchmarking.baselines.filtering import NormalOperationFilter
from benchmarking.baselines.naive_ratio import restrict_to_campaign
from benchmarking.baselines.power_model import diagnostics as diag
from benchmarking.baselines.power_model.conditional import impute_uncovered_bins, relevel_conditional
from benchmarking.baselines.power_model.features import (
    build_reference_features,
    check_reference_only,
    era5_feature_frame,
    extract_outcome,
    reference_mean_wind_speed,
    test_condition_signals,
)
from benchmarking.baselines.power_model.fitting import (
    IDENTITY_CALIBRATION,
    OUTCOME_MODEL_FACTORIES,
    CalibrationLine,
    cell_residual_calibration,
    early_stopped_n_estimators,
    fit_calibration_line,
    time_block_folds,
)
from benchmarking.baselines.power_model.matching import cell_codes, coarsened_exact_match
from benchmarking.baselines.rlearner.nuisance import make_outcome_model
from benchmarking.baselines.time_features import (
    TIME_FEATURE_NAMES,
    days_since_campaign_start,
    season_sin_cos,
    solar_altitude_azimuth,
)
from benchmarking.diagnostics import DiagnosticContext, stages, write_common_diagnostics, write_run_config
from benchmarking.harness.conditions import CONDITION_BINS, energy_ratio_by_bin
from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule, treated_mask

if TYPE_CHECKING:
    from collections.abc import Callable

    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)

_MIN_BASELINE_ROWS = 10
_MIN_HOLDOUT_ROWS = 20  # below this, report the in-sample fit (no point splitting off a tiny valid set)
# Time-blocked split shape shared by the holdout diagnostic, the early-stopping valid set and the
# calibration OOF: 25 contiguous blocks round-robin over 5 folds, so each fold is ~20% of the rows
# spread across the whole window (seasonally balanced) while staying contiguous at the block scale.
_N_FOLDS = 5
_N_BLOCKS = 25
_MIN_EARLY_STOPPING_ROWS = 200  # below this a valid split is too small for early stopping to mean anything
_EARLY_STOPPING_CEILING = 3000  # capacity ceiling when early stopping picks the tree count

# The removal-ablation verdict (findings F13): raw Open-Meteo columns the curated feature set does
# better without — redundant thermodynamic derivatives of temperature/humidity and the precipitation
# trio. HoT drivers pass this (with availability_feature=False) as the accepted default; kept here,
# not baked into ``era5_exclude``'s dataclass default, so a non-Open-Meteo ERA5 frame is not broken
# by exclusions it never had.
CURATED_ERA5_EXCLUDE: tuple[str, ...] = (
    "apparent_temperature",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
)

# The Issue 12 capacity verdict (findings F14): loosening min_child_samples 200 -> 50 materially
# improves prepost spread/score (placebo ALL Δscore -0.62 pp) at neutral overall P50; 20 overshoots
# into overfit. A power_model-specific tuning — the design-note common params in
# ``make_outcome_model`` (shared with the R-learner) are unchanged; drivers pass this instead.
TUNED_MODEL_PARAMS: dict[str, Any] = {"min_child_samples": 50}

# Per-reporting-bin matched-count floor for the two-direction conditional combine (Issue 14). Below this
# many matched rows *per side* in a ws/TI reporting bin, the combine overshoots — the F7/F9 sparse-extreme
# tail (e.g. TI (0.45,0.50] swinging -77 to +93 pp between replicates) — so the bin is marked uncovered and
# filled by the physics-informed imputer instead of trusting its noisy shape. Compared against the raw
# per-side matched count today; if the per-bin balance reweighting (A5) is adopted it becomes the Kish
# ESS. The value is chosen on placebo/benchmark evidence across many bins (findings F17), not tuned to the
# one known bad TI bin.
_MIN_BIN_MATCHED_COUNT = 50

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
    :param model_params: LightGBM overrides passed to the outcome model; merged **over** the tuned
        default ``TUNED_MODEL_PARAMS`` (``min_child_samples=50``, F14), so a bare method already carries
        the accepted capacity and any key here wins. Left empty by default so the ``model_factory`` vs
        ``model_params`` guard still distinguishes "user set params" from "default"
    :param timebase: analysis timebase; inferred from the data when ``None``
    :param toggle_campaign_only: for a toggle campaign, fit only on the interleaved on/off blocks
        (drop the pre-campaign baseline); no-op for prepost. Default ``False`` since Issue 13
        (findings F15): training on the pre-campaign baseline too removes most of the small-fit
        shrinkage bias at short campaigns (3-month placebo +0.35 → +0.12 pp) — made safe for the
        conditional step by matching within the campaign only (see ``estimate``). Set ``True`` to
        restore the strict same-distribution training window.
    :param conditional_uplift: compute the per-(ws, TI)-bin conditional uplift distribution (default
        ``True``). The overall P50 is a single baseline→upgraded fit either way; when this is ``True``
        an extra two-direction, ERA5-weather-matched (CEM) cross-prediction runs **last** to estimate
        the conditional shape (its common per-bin shrinkage cancels), then re-levels onto the headline.
        It **requires ERA5** (the matching axis is the ERA5 columns). Set ``False`` to skip that
        cross-prediction — the expensive part — and return only the overall P50.
    :param matching_vars: ERA5 columns matched on for the conditional step (default: the F6 set)
    :param matching_bin_edges: per-variable CEM bin edges; the F6 defaults are used when ``None``
    :param era5_derivations: derived ERA5 feature columns to add (Issue 9; see
        :data:`~benchmarking.baselines.era5_derived.ERA5_DERIVATIONS`); requires ``era5_hourly_df``
    :param hub_height_m: turbine hub height, required by the ``wind_speed_hub`` derivation
    :param time_features: explicit time features to add (Issue 10; see
        :data:`~benchmarking.baselines.time_features.TIME_FEATURE_NAMES`)
    :param latitude: site latitude in degrees, required by the ``solar`` time feature
    :param longitude: site longitude in degrees (east positive), required by the ``solar`` time feature
    :param reference_stat_cols: extra per-reference value columns to carry as features (Issue 11's
        active-power max/min/SD companion statistics)
    :param era5_exclude: raw ERA5 columns to drop from the model features (removal-ablation knob;
        a dropped direction column also loses its sin/cos companions). Columns used as
        ``matching_vars`` cannot be excluded while ``conditional_uplift`` is on. Defaults to the
        accepted ``CURATED_ERA5_EXCLUDE`` set (F13); the **untouched default** is drop-if-present (so a
        non-Open-Meteo ERA5 frame lacking those columns is not broken), while an **explicitly-set**
        value keeps the strict raise-on-unknown-column typo guard.
    :param availability_feature: when ``False`` (the accepted default, F13), drop the per-reference
        availability *feature*; ``availability_col`` itself stays required for the downtime filter
    :param model_factory: the outcome-model seam (Issue 12): a registered factory name (see
        :data:`~benchmarking.baselines.power_model.fitting.OUTCOME_MODEL_FACTORIES`) or a callable
        ``seed -> unfitted sklearn-style regressor``. ``None`` (default) keeps the design-note
        LightGBM model. A factory bakes in its own hyperparameters, so it cannot be combined with
        ``model_params`` or ``early_stopping``.
    :param n_seed_ensemble: fit this many models (seeds ``seed .. seed+K-1``) per training set and
        average their predictions — variance reduction from the subsample/colsample noise (default 1)
    :param early_stopping: pick the LightGBM tree count by early stopping on a time-blocked
        validation split, then refit on all training rows at that capacity — data-size-adaptive
        capacity instead of one fixed ``n_estimators`` for every fit size (default ``False``)
    :param calibrate_slope: post-hoc calibration-slope correction (Issue 12). **Measured and
        rejected as a default (findings F14): the line is fit on out-of-fold predictions from
        ~80%-sized fits, which shrink more than the final 100% fit it corrects, so it overcorrects
        exactly where the correction is largest (small fits) — keep off unless researching that
        mechanism.** When on: fit ``actual ~ a + b * predicted`` on time-blocked out-of-fold
        baseline predictions and apply the line to the counterfactual predictions before the
        energy ratio. Applies to the overall headline only — the conditional two-direction step
        cancels its per-bin shrinkage by construction. The line is fit with single-seed,
        non-early-stopped models, so it cannot be combined with ``early_stopping`` or
        ``n_seed_ensemble > 1``.
    :param calibrate_residuals: ERA5-cell residual calibration (Issue 13). **Measured and rejected
        as a default (findings F15): out-of-fold residuals transfer to the final refit model in
        neither level nor conditional shape — the estimated correction had the wrong sign vs the
        observed placebo bias — so this knob is retained for research only.** When on: estimate
        the mean out-of-fold baseline residual per coarsened ERA5 cell (the ``matching_vars`` / F6
        CEM coarsening), **centred by the global mean residual** so only the mix-shift
        differential transfers (the OOF *level* artefact, the F14 lesson — necessary but, per F15,
        not sufficient), and add each upgraded row's centred cell mean (0 for cells unseen in
        baseline) to its counterfactual prediction before the energy ratio. Requires
        ``era5_hourly_df``; mutually exclusive with ``calibrate_slope`` and, like it, with
        ``early_stopping`` / ``n_seed_ensemble > 1`` (the out-of-fold basis must match the final
        model configuration).
    :param time_decay_half_life_days: training rows are sample-weighted by their proximity in
        time to the campaign data: ``0.5 ** (days_outside_campaign / half_life)``. Rows inside
        the campaign interval (for toggle, the interleaved on and off rows) weigh 1; rows outside
        decay with their distance to it — so distant history informs the fit without dominating
        it (Issue 13's recency weighting; the alternative to the rejected F11 drift *feature*).
        Default **548 days (1.5 years)**: finite by design so an arbitrarily long SCADA history
        cannot dominate the campaign era (year-old data still weighs ~63%, decade-old ~1%); on
        the 2.5-year benchmark dataset it is neutral-to-slightly-better at every campaign length
        (findings F16). Shorter half-lives (~90 days) measurably help 1-3-month campaigns at a
        small long-prepost bias cost; ``None`` disables the weighting.
    :param toggle_estimator: how a **toggle** headline is estimated (ignored for prepost).
        ``"counterfactual"`` (default): ``Σactual/Σprediction - 1`` — needs the model to be
        absolutely calibrated. ``"double_ratio"``: the naive-adoption hybrid — the model only
        removes condition unfairness and the headline is a ratio of ratios,
        ``(Σy_on/Σpred_on) / (Σy_off/Σpred_off) - 1``, where the off rows are predicted
        out-of-fold and the on rows by the same fold-model ensemble (basis-consistent, the F15
        rule), so the model's shrinkage/level bias cancels between the interleaved on and off
        sides instead of needing to be zero. Implemented as
        ``counterfactual = fold_ensemble_prediction * rho_off`` so every downstream sum and the
        conditional re-level stay self-consistent. Cannot be combined with the calibrations
        (it *is* one) or ``early_stopping`` / ``n_seed_ensemble > 1``.
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
    toggle_campaign_only: bool = False
    conditional_uplift: bool = True
    matching_vars: tuple[str, ...] = _DEFAULT_MATCHING_VARS
    matching_bin_edges: dict[str, list[float]] | None = None
    era5_derivations: tuple[str, ...] = ()
    hub_height_m: float | None = None
    time_features: tuple[str, ...] = ()
    latitude: float | None = None
    longitude: float | None = None
    reference_stat_cols: tuple[str, ...] = ()
    era5_exclude: tuple[str, ...] = CURATED_ERA5_EXCLUDE
    availability_feature: bool = False
    model_factory: str | Callable[[int], Any] | None = None
    n_seed_ensemble: int = 1
    early_stopping: bool = False
    calibrate_slope: bool = False
    calibrate_residuals: bool = False
    time_decay_half_life_days: float | None = 548.0
    toggle_estimator: str = "counterfactual"
    # Not configuration: latches the missing-sample_weight warning so it fires once per instance.
    _warned_no_sample_weight: bool = field(default=False, init=False, repr=False, compare=False)

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Estimate the test turbine's P50 uplift for one campaign and write diagnostics."""
        self._validate_model_config()
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
            extra_cols=self.reference_stat_cols,
            include_availability=self.availability_feature,
        )
        features, era5 = self._add_era5(scada, features, mi=mi, index=index, timebase=timebase)
        if self.time_features:
            time_frame = self._time_feature_frame(index, campaign_start=_upgrade_start(mi.upgrade_timing, index))
            features = pd.concat([features, time_frame], axis=1)
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

        y_arr = y.to_numpy(dtype=float)
        weights = self._time_decay_weights(
            index, campaign_start=_upgrade_start(mi.upgrade_timing, index), campaign_end=index.max()
        )
        double_ratio = self.toggle_estimator == "double_ratio" and isinstance(mi.upgrade_timing, ToggleSchedule)
        fit = self._fit_predict(
            features,
            y=y_arr,
            baseline_sel=baseline_sel,
            upgraded_sel=upgraded_sel,
            weights=weights,
            double_ratio=double_ratio,
        )
        sum_actual = float(fit["y_upgraded"].sum())
        sum_counter = float(fit["pred_upgraded"].sum())
        uplift = sum_actual / sum_counter - 1.0 if np.isfinite(sum_counter) and sum_counter != 0 else float("nan")

        # ws/TI row-aligned to each segment's residuals, for the overall shrinkage-check diagnostics (cheap,
        # plots only). Computed here so the run folder's step-5 residual plots are drawn whether or not the
        # optional conditional step runs.
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
            cond_upgraded = conditions.iloc[upgraded_sel].reset_index(drop=True)
            cond_baseline_valid = conditions.iloc[fit["baseline_valid_pos"]].reset_index(drop=True)

        run_dir = self._run_dir(mi, index)
        self._write(
            mi,
            run_dir=run_dir,
            index=index,
            timebase=timebase,
            t=t,
            selected=selected,
            upgraded_sel=upgraded_sel,
            y=y_arr,
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

        # The conditional uplift distribution is the optional, expensive last step: nothing above depends on
        # it (eventually AEP extrapolation will). Skipped when conditional_uplift is off, or when there is no
        # wind_speed_col — the per-(ws, TI) decomposition needs it, so without it the expensive ERA5 match +
        # two-direction fits would run only for _conditional_by_bin to return None (and could raise a spurious
        # missing-ERA5 error).
        by_condition: pd.DataFrame | None = None
        if self.conditional_uplift and self.wind_speed_col is not None:
            # The conditional two-direction step matches untreated against treated rows and relies on
            # them sharing a distribution *and era* — for a toggle trained on pre-campaign data too
            # (toggle_campaign_only=False), only the interleaved campaign off rows qualify: matching
            # pre-campaign rows against campaign on rows would read reference/era drift as per-bin
            # uplift. The extra pre-campaign rows serve only the headline fit's training data.
            # No time-decay weights here either (F16): the matched contrast is already
            # era-insensitive (its common shrinkage cancels), and weighting the direction fits
            # destabilises the sparse extreme-condition bins (a degenerate tail fit in one
            # replicate can read three-digit per-bin uplift).
            by_condition = self._estimate_conditional(
                scada,
                mi=mi,
                features=features,
                y=y_arr,
                baseline_sel=baseline_sel & self._campaign_mask(index, upgrade_timing=mi.upgrade_timing),
                upgraded_sel=upgraded_sel,
                fit=fit,
                overall_ratio=uplift,
                run_dir=run_dir,
            )
        return MethodOutput(p50_overall=uplift, p50_by_condition=by_condition)

    @staticmethod
    def _campaign_mask(index: pd.DatetimeIndex, *, upgrade_timing: pd.Timestamp | ToggleSchedule) -> np.ndarray:
        """Boolean over ``index``: rows at/after the campaign start (all-True for prepost).

        Prepost has no pre-campaign data beyond its baseline (which *is* the training era), so the
        mask only bites for a toggle whose input still carries pre-campaign rows.
        """
        if isinstance(upgrade_timing, ToggleSchedule) and upgrade_timing.start is not None:
            return np.asarray(index >= upgrade_timing.start)
        return np.ones(len(index), dtype=bool)

    def _estimate_conditional(
        self,
        scada: pd.DataFrame,
        *,
        mi: MethodInput,
        features: pd.DataFrame,
        y: np.ndarray,
        baseline_sel: np.ndarray,
        upgraded_sel: np.ndarray,
        fit: dict[str, Any],
        overall_ratio: float,
        run_dir: Path,
    ) -> pd.DataFrame | None:
        """Per-(ws, TI)-bin conditional uplift via a two-direction, ERA5-weather-matched cross-prediction.

        Match the baseline and upgraded periods on ERA5 weather, then fit/predict in both directions and
        combine so the common per-bin shrinkage cancels (design/F5); the decomposition is re-leveled onto the
        already-computed overall headline (``overall_ratio``, from the single full fit ``fit``) so the per-bin
        MWh partitions it (F8). Requires ERA5 — the matching axis is the synced ERA5 columns, which live in
        ``features`` (``era5_feature_frame`` passes them through). Returns the ``[condition, condition_bin,
        p50_uplift]`` frame (or ``None`` when no wind-speed column), and writes the per-run diagnostics.
        """
        if self.era5_hourly_df is None:
            msg = "conditional_uplift requires ERA5 (era5_hourly_df): the matching axis is the ERA5 weather columns."
            raise ValueError(msg)

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
        # Forward (train matched-baseline, predict matched-upgraded) gives ``1+r_fwd = (1+u)/s``.
        pred_up = self._fit_direction(features, y, train=mb, predict=mu)
        # Reverse (train matched-upgraded, predict matched-baseline): 1+r_rev = 1/(s(1+u)). Clip reuses
        # baseline_rated_power_kw — its upper bound max(rated, max(y_train)) already lifts the ceiling for an
        # uprate, so no separate upgraded-rating field is needed.
        pred_base = self._fit_direction(features, y, train=mu, predict=mb)
        r_fwd = _ratio(y[mu], pred_up)
        r_rev = _ratio(y[mb], pred_base)

        per_bin = self._conditional_by_bin(
            scada,
            mi=mi,
            y=y,
            mb=mb,
            mu=mu,
            pred_up=pred_up,
            pred_base=pred_base,
            upgraded_pos=np.flatnonzero(upgraded_sel),
            actual_full=fit["y_upgraded"],
            overall_ratio=overall_ratio,
        )
        by_condition = per_bin[["condition", "condition_bin", "p50_uplift"]].copy() if per_bin is not None else None
        self._write_conditional(
            mi, run_dir=run_dir, match=match, r_fwd=r_fwd, r_rev=r_rev, uplift=overall_ratio, per_bin=per_bin
        )
        logger.info(
            "%s %s conditional uplift: headline=%+.3f%%  implied_s=%.3f  (r_fwd=%+.3f, r_rev=%+.3f, matched=%d/side)",
            self.name,
            mi.test_wtg,
            100 * overall_ratio,
            float(_implied_shrinkage(np.array([r_fwd]), np.array([r_rev]))[0]),
            r_fwd,
            r_rev,
            match.n_matched_per_side,
        )
        return by_condition

    def _fit_direction(
        self, features: pd.DataFrame, y: np.ndarray, *, train: np.ndarray, predict: np.ndarray
    ) -> np.ndarray:
        """Fit the outcome model(s) on ``train`` rows and return clipped predictions for ``predict`` rows.

        No calibration and no time-decay weights here: the two-direction combine cancels the common
        per-bin shrinkage by construction, and weighting the matched fits only destabilises sparse
        extreme-condition bins (F16) — the corrections are spent on the overall headline instead.
        """
        models = self._fit_models(features.iloc[train], y[train])
        return _clip_predictions(
            self._predict_mean(models, features.iloc[predict]),
            y_train=y[train],
            rated_power_kw=self.baseline_rated_power_kw,
        )

    def _conditional_by_bin(
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
        """Per-(ws, TI)-bin two-direction uplift, imputed where uncovered and re-leveled onto the headline.

        The measured **shape** ``1+u_b = sqrt((1+r_fwd_b)/(1+r_rev_b))`` comes from the matched
        forward/reverse fits; a bin is ``covered`` (trusted) only when that shape is finite (Issue 14 adds
        the per-bin matched-count floor to this test). Uncovered bins are filled by
        :func:`~benchmarking.baselines.power_model.conditional.impute_uncovered_bins` (ws: bfill then 0 at
        rated; ti: the overall uplift) so every bin carries a best estimate rather than a bare NaN. The
        re-level **weights** are the *full-upgraded* actual energy per bin (``actual_full`` over
        ``upgraded_pos``): imputed bins are pinned and one λ scales the measured bins so measured + imputed
        together energy-aggregate to ``1 + overall_ratio`` exactly (F8, corrected for uncovered-bin energy).
        The returned frame carries a ``covered`` flag (a per-run diagnostic — the harness seam only sees
        ``p50_uplift``).
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
        one_plus_overall = 1.0 + overall_ratio
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
            # impute_uncovered_bins needs ascending bin order (low ws/TI first); the merges above do not
            # guarantee it, so pin the canonical CONDITION_BINS order before imputing / re-leveling.
            order = pd.cut([], bins=CONDITION_BINS[name]).categories.astype(str)
            merged["condition_bin"] = pd.Categorical(merged["condition_bin"], categories=order, ordered=True)
            merged = merged.sort_values("condition_bin").reset_index(drop=True)

            r_fwd = merged["p50_uplift_fwd"].to_numpy()  # per-bin energy ratio IS the per-bin r
            r_rev = merged["p50_uplift_rev"].to_numpy()
            shape = 1.0 + _combine_uplift(r_fwd, r_rev)  # shrinkage-free shape (NaN if degenerate)
            sum_actual_full = merged["sum_actual_full"].to_numpy()
            # A bin is covered only if its shape is finite AND both directions have enough matched rows;
            # below the floor the two-direction combine overshoots, so the bin is imputed instead (F7/F9).
            per_side = np.minimum(merged["n_records_fwd"].to_numpy(), merged["n_records_rev"].to_numpy())
            measured = np.isfinite(shape) & (per_side >= _MIN_BIN_MATCHED_COUNT)
            imputed_shape = impute_uncovered_bins(
                shape, condition=name, measured=measured, one_plus_overall=one_plus_overall
            )
            releveled = relevel_conditional(
                sum_actual_full, imputed_shape, measured=measured, one_plus_overall=one_plus_overall
            )
            frames.append(
                pd.DataFrame(
                    {
                        "condition": name,
                        "condition_bin": merged["condition_bin"].astype(str),
                        "n_records_fwd": merged["n_records_fwd"],
                        "n_records_rev": merged["n_records_rev"],
                        "sum_actual": sum_actual_full,  # full-upgraded energy per bin (the MWh re-level weight)
                        "r_fwd": r_fwd,
                        "r_rev": r_rev,
                        "implied_shrinkage": _implied_shrinkage(r_fwd, r_rev),
                        "u_b": shape - 1.0,  # measured two-direction shape (NaN where uncovered)
                        "covered": measured,  # per-run diagnostic: measured vs imputed
                        "p50_uplift": releveled - 1.0,  # measured-or-imputed, re-leveled to the headline
                    }
                )
            )
        return pd.concat(frames, ignore_index=True) if frames else None

    def _write_conditional(
        self,
        mi: MethodInput,
        *,
        run_dir: Path,
        match: Any,  # noqa: ANN401 - MatchResult
        r_fwd: float,
        r_rev: float,
        uplift: float,
        per_bin: pd.DataFrame | None,
    ) -> None:
        """Write the conditional-step diagnostics (implied shrinkage + CEM balance) and the per-bin plot.

        CSVs go in ``run_dir/conditional/``; the per-bin implied-shrinkage plot (when ``save_plots``) goes in
        the step-7 plot folder, keeping the optional conditional outputs separate from the always-on overall
        diagnostics in the same run folder.
        """
        conditional_dir = run_dir / "conditional"
        conditional_dir.mkdir(parents=True, exist_ok=True)
        run_name = run_dir.name
        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        overall = {
            "test_wtg": mi.test_wtg,
            "mode": "toggle" if isinstance(mi.upgrade_timing, ToggleSchedule) else "prepost",
            "r_fwd": r_fwd,
            "r_rev": r_rev,
            "uplift_frc": uplift,
            "implied_shrinkage": float(_implied_shrinkage(np.array([r_fwd]), np.array([r_rev]))[0]),
        }
        diag.write_conditional_csvs(conditional_dir, run_name, ts, overall=overall, per_bin=per_bin, match=match)
        if self.save_plots and per_bin is not None:
            diag.plot_implied_shrinkage(run_dir / "plots" / stages.CONDITIONAL_UPLIFT, per_bin, test_wtg=mi.test_wtg)

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
        """Sync ERA5 (if supplied) and append its features (+ requested derived quantities)."""
        if self.era5_hourly_df is None:
            if self.era5_derivations:
                msg = "era5_derivations requested but no era5_hourly_df supplied; the derivations need ERA5."
                raise ValueError(msg)
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
        era5_features = era5_feature_frame(result.aligned)
        if self.era5_exclude:
            missing = sorted(set(self.era5_exclude) - set(era5_features.columns))
            # An explicitly-set era5_exclude keeps the strict typo guard; the promoted class default
            # (CURATED_ERA5_EXCLUDE) is drop-if-present, so a non-Open-Meteo ERA5 frame lacking those
            # columns is not broken by an exclusion it never had. Identity check = "untouched default".
            if missing and self.era5_exclude is not CURATED_ERA5_EXCLUDE:
                msg = f"era5_exclude names columns not in the ERA5 features: {missing}"
                raise ValueError(msg)
            blocked = sorted(set(self.era5_exclude) & set(self.matching_vars))
            if blocked and (self.conditional_uplift or self.calibrate_residuals):
                msg = (
                    f"era5_exclude {blocked} are matching_vars; excluding them as model features would "
                    f"break the CEM matching / residual-calibration cells. Turn conditional_uplift and "
                    f"calibrate_residuals off or re-pick matching_vars first."
                )
                raise ValueError(msg)
            drop = [
                c
                for c in era5_features.columns
                if c in self.era5_exclude or any(c == f"{raw}_{t}" for raw in self.era5_exclude for t in ("sin", "cos"))
            ]
            era5_features = era5_features.drop(columns=drop)
        frames = [features, era5_features]
        if self.era5_derivations:
            frames.append(
                era5_derived_frame(result.aligned, derivations=self.era5_derivations, hub_height_m=self.hub_height_m)
            )
        return pd.concat(frames, axis=1), result

    def _time_feature_frame(self, index: pd.DatetimeIndex, *, campaign_start: pd.Timestamp) -> pd.DataFrame:
        """Build the configured explicit time features (Issue 10) on the analysis index.

        ``days_since_campaign_start`` is anchored to the upgrade start (prepost changeover / toggle
        origin) so the model can absorb slow reference drift relative to the campaign; ``season``
        adds the June-21-anchored sin/cos pair; ``solar`` adds solar altitude + azimuth sin/cos
        (requires ``latitude`` / ``longitude``).
        """
        unknown = [f for f in self.time_features if f not in TIME_FEATURE_NAMES]
        if unknown:
            msg = f"unknown time feature(s) {unknown}; available: {list(TIME_FEATURE_NAMES)}"
            raise ValueError(msg)
        frames: list[pd.DataFrame] = []
        if "days_since_campaign_start" in self.time_features:
            frames.append(days_since_campaign_start(index, campaign_start=campaign_start).to_frame())
        if "season" in self.time_features:
            frames.append(season_sin_cos(index))
        if "solar" in self.time_features:
            if self.latitude is None or self.longitude is None:
                msg = "the solar time feature requires latitude and longitude."
                raise ValueError(msg)
            frames.append(solar_altitude_azimuth(index, latitude=self.latitude, longitude=self.longitude))
        return pd.concat(frames, axis=1)

    def _time_decay_weights(
        self, index: pd.DatetimeIndex, *, campaign_start: pd.Timestamp, campaign_end: pd.Timestamp
    ) -> np.ndarray | None:
        """Exponential campaign-proximity sample weights over the analysis index (``None`` = knob off).

        ``0.5 ** (days_outside_campaign / half_life)`` where the distance is to the campaign
        *interval* ``[campaign_start, campaign_end]``: every row inside the campaign (for toggle,
        the interleaved on **and** off rows) weighs exactly 1, and rows outside decay with their
        distance to it — so distant history still informs the fit without dominating it. With
        today's flows only pre-campaign rows exist outside the interval, but the definition is
        two-sided on purpose.
        """
        if self.time_decay_half_life_days is None:
            return None
        seconds_outside = np.maximum((campaign_start - index).total_seconds(), (index - campaign_end).total_seconds())
        days_outside = np.maximum(seconds_outside / 86400.0, 0.0)
        return np.asarray(0.5 ** (days_outside / self.time_decay_half_life_days), dtype=float)

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
        self,
        features: pd.DataFrame,
        *,
        y: np.ndarray,
        baseline_sel: np.ndarray,
        upgraded_sel: np.ndarray,
        weights: np.ndarray | None = None,
        double_ratio: bool = False,
    ) -> dict[str, Any]:
        """Fit on baseline, predict the upgraded counterfactual; also a held-out baseline fit metric.

        The final model(s) (for the counterfactual) are fit on **all** baseline rows. A separate model
        on a baseline train split predicts a held-out baseline slice, giving an honest fit-quality
        number that is not inflated by in-sample optimism. When ``calibrate_slope`` /
        ``calibrate_residuals`` is on, the counterfactual predictions pass through the out-of-fold
        correction (line / per-cell residual means) before clipping. ``double_ratio`` switches to the
        toggle ratio-of-ratios estimator (see :attr:`toggle_estimator`).
        """
        x_base = features.iloc[baseline_sel]
        y_base = y[baseline_sel]
        x_up = features.iloc[upgraded_sel]
        y_up = y[upgraded_sel]
        w_base = weights[baseline_sel] if weights is not None else None
        if double_ratio:
            if len(y_base) < _MIN_HOLDOUT_ROWS:
                # Fail loudly rather than silently estimating with a different estimator than requested.
                msg = (
                    f"toggle_estimator='double_ratio' needs at least {_MIN_HOLDOUT_ROWS} off rows for its "
                    f"out-of-fold basis, got {len(y_base)}."
                )
                raise ValueError(msg)
            return self._fit_predict_double_ratio(
                x_base, y_base=y_base, x_up=x_up, y_up=y_up, baseline_sel=baseline_sel, w_base=w_base
            )

        y_valid, pred_valid, valid_local = self._holdout_fit(x_base, y_base, w_base=w_base)
        baseline_valid_pos = np.flatnonzero(baseline_sel)[valid_local]

        models = self._fit_models(x_base, y_base, weights=w_base)
        pred_raw = self._predict_mean(models, x_up)
        calibration = IDENTITY_CALIBRATION
        if self.calibrate_slope:
            calibration = self._fit_calibration(x_base, y_base, w_base=w_base)
            pred_raw = calibration.apply(pred_raw)
        elif self.calibrate_residuals:
            pred_raw = pred_raw + self._residual_corrections(
                features, y=y, baseline_sel=baseline_sel, upgraded_sel=upgraded_sel, w_base=w_base
            )
        pred_up = _clip_predictions(pred_raw, y_train=y_base, rated_power_kw=self.baseline_rated_power_kw)
        return {
            "model": models[0],
            "pred_upgraded": pred_up,
            "y_upgraded": y_up,
            "y_baseline_valid": y_valid,
            "pred_baseline_valid": pred_valid,
            "baseline_valid_pos": baseline_valid_pos,  # positions over ``index`` of the held-out rows
            "calibration": calibration,
        }

    def _fit_predict_double_ratio(
        self,
        x_base: pd.DataFrame,
        *,
        y_base: np.ndarray,
        x_up: pd.DataFrame,
        y_up: np.ndarray,
        baseline_sel: np.ndarray,
        w_base: np.ndarray | None,
    ) -> dict[str, Any]:
        """Estimate via the toggle ratio-of-ratios: a model-normalised naive comparison (Issue 13 hybrid).

        Fit one model per time-blocked fold on the off rows; each off row is predicted out-of-fold
        and every on row by the mean of the fold models — both sides therefore share the ~80%-fit
        basis (the F15 consistency rule), so the model's shrinkage/level bias cancels in
        ``rho_on / rho_off``. The returned counterfactual is ``clip(fold_ensemble_prediction) *
        rho_off`` (``rho_off = Σy_off / Σpred_off``, the model's observed miscalibration on the off
        rows): the physical clip applies to the *prediction* and the calibration scale on top, so
        ``Σactual / Σcounterfactual - 1`` equals the double ratio of the clipped predictions
        exactly — downstream sums, diagnostics and the conditional re-level stay self-consistent.
        (The scaled counterfactual may exceed the clip ceiling by the ~0-3 % calibration factor;
        that is deliberate — re-clipping after scaling would break the identity.) The full
        out-of-fold off-row predictions double as the held-out fit-quality diagnostic.
        """
        n = len(y_base)
        folds = time_block_folds(n, n_folds=_N_FOLDS, n_blocks=_N_BLOCKS)
        oof = np.full(n, np.nan)
        preds_on = []
        models = []
        for f in range(_N_FOLDS):
            hold = folds == f
            model = self._make_model()
            model.fit(
                x_base.iloc[~hold],
                y_base[~hold],
                **self._fit_kwargs(model, w_base[~hold] if w_base is not None else None),
            )
            oof[hold] = np.asarray(model.predict(x_base.iloc[hold]), dtype=float)
            preds_on.append(np.asarray(model.predict(x_up), dtype=float))
            models.append(model)
        pred_off = _clip_predictions(oof, y_train=y_base, rated_power_kw=self.baseline_rated_power_kw)
        rho_off = _ratio(y_base, pred_off) + 1.0
        pred_up = (
            _clip_predictions(np.mean(preds_on, axis=0), y_train=y_base, rated_power_kw=self.baseline_rated_power_kw)
            * rho_off
        )
        logger.info(
            "%s double-ratio toggle estimator: rho_off=%.5f over %d off rows (%d folds)",
            self.name,
            rho_off,
            n,
            _N_FOLDS,
        )
        return {
            "model": models[0],
            "pred_upgraded": pred_up,
            "y_upgraded": y_up,
            "y_baseline_valid": y_base,
            "pred_baseline_valid": pred_off,
            "baseline_valid_pos": np.flatnonzero(baseline_sel),
            "calibration": IDENTITY_CALIBRATION,
        }

    def _validate_model_config(self) -> None:
        """Fail loudly on model-fundamentals config combinations that would silently misbehave."""
        if isinstance(self.model_factory, str) and self.model_factory not in OUTCOME_MODEL_FACTORIES:
            msg = f"unknown model_factory {self.model_factory!r}; registered: {sorted(OUTCOME_MODEL_FACTORIES)}"
            raise ValueError(msg)
        if self.model_factory is not None and self.model_params:
            msg = "model_factory bakes in its own hyperparameters; model_params would be silently ignored."
            raise ValueError(msg)
        if self.model_factory is not None and self.early_stopping:
            msg = "early_stopping picks a LightGBM tree count; it cannot be combined with model_factory."
            raise ValueError(msg)
        if self.n_seed_ensemble < 1:
            msg = f"n_seed_ensemble must be >= 1, got {self.n_seed_ensemble}"
            raise ValueError(msg)
        if self.n_seed_ensemble > 1 and "random_state" in self.model_params:
            msg = "a fixed random_state in model_params would make every ensemble member identical."
            raise ValueError(msg)
        if self.calibrate_slope and self.calibrate_residuals:
            msg = "calibrate_slope and calibrate_residuals are competing headline corrections; enable at most one."
            raise ValueError(msg)
        if (self.calibrate_slope or self.calibrate_residuals) and (self.early_stopping or self.n_seed_ensemble > 1):
            msg = (
                "calibrate_slope / calibrate_residuals fit their out-of-fold basis with single-seed, "
                "non-early-stopped models, so combining them with early_stopping or n_seed_ensemble > 1 "
                "would calibrate against a different model configuration than the predictions they correct."
            )
            raise ValueError(msg)
        if self.calibrate_residuals and self.era5_hourly_df is None:
            msg = "calibrate_residuals requires ERA5 (era5_hourly_df): the residual cells are the ERA5 matching_vars."
            raise ValueError(msg)
        if self.time_decay_half_life_days is not None and self.time_decay_half_life_days <= 0:
            msg = f"time_decay_half_life_days must be positive, got {self.time_decay_half_life_days}"
            raise ValueError(msg)
        if self.toggle_estimator not in ("counterfactual", "double_ratio"):
            msg = f"unknown toggle_estimator {self.toggle_estimator!r}; expected 'counterfactual' or 'double_ratio'"
            raise ValueError(msg)
        if self.toggle_estimator == "double_ratio" and (
            self.calibrate_slope or self.calibrate_residuals or self.early_stopping or self.n_seed_ensemble > 1
        ):
            msg = (
                "toggle_estimator='double_ratio' is itself a fold-basis calibration; it cannot be combined "
                "with calibrate_slope, calibrate_residuals, early_stopping or n_seed_ensemble > 1."
            )
            raise ValueError(msg)

    def _fit_kwargs(self, model: Any, weights: np.ndarray | None) -> dict[str, Any]:  # noqa: ANN401
        """Return the fit kwargs for the time-decay ``weights``, empty for models that cannot take them.

        A ``model_factory`` learner may not accept ``sample_weight`` (e.g. an sklearn ``Pipeline``);
        with the weights on by default, such learners fit unweighted — warned once per method
        instance (many fits happen per estimate: folds, holdout, ensemble) rather than crashing
        the factory seam.
        """
        if weights is None:
            return {}
        from sklearn.utils.validation import has_fit_parameter  # noqa: PLC0415

        if not has_fit_parameter(model, "sample_weight"):
            if not self._warned_no_sample_weight:
                self._warned_no_sample_weight = True
                logger.warning(
                    "%s: outcome model %s does not accept sample_weight; time-decay weights are ignored "
                    "for its fits (set time_decay_half_life_days=None to silence this).",
                    self.name,
                    type(model).__name__,
                )
            return {}
        return {"sample_weight": weights}

    def _make_model(self, *, seed: int | None = None, extra_params: dict[str, Any] | None = None) -> Any:  # noqa: ANN401
        """One unfitted outcome model: the ``model_factory`` seam, or LightGBM with ``seed`` plumbed in.

        For the default LightGBM path a caller-supplied ``random_state`` in ``model_params`` still
        wins over ``seed``; ``extra_params`` (the early-stopped capacity) wins over everything.
        """
        s = self.seed if seed is None else seed
        if self.model_factory is not None:
            factory = (
                OUTCOME_MODEL_FACTORIES[self.model_factory]
                if isinstance(self.model_factory, str)
                else self.model_factory
            )
            return factory(s)
        return make_outcome_model(
            **{"random_state": s, **TUNED_MODEL_PARAMS, **self.model_params, **(extra_params or {})}
        )

    def _fit_models(
        self, x_train: pd.DataFrame, y_train: np.ndarray, *, weights: np.ndarray | None = None
    ) -> list[Any]:
        """Fit the outcome model(s) on one training set, honouring ``early_stopping`` / ``n_seed_ensemble``.

        Early stopping runs once (a probe fit on a time-blocked split picks the tree count), then every
        ensemble member is refit on **all** training rows at that capacity with its own seed. ``weights``
        (the time-decay sample weights, aligned to the training rows) are passed to every fit when given.
        """
        extra: dict[str, Any] = {}
        if self.early_stopping and len(y_train) >= _MIN_EARLY_STOPPING_ROWS:
            ceiling = int(self.model_params.get("n_estimators", _EARLY_STOPPING_CEILING))
            extra["n_estimators"] = early_stopped_n_estimators(
                lambda: self._make_model(extra_params={"n_estimators": ceiling}),
                x_train,
                y_train,
                valid_mask=time_block_folds(len(y_train), n_folds=_N_FOLDS, n_blocks=_N_BLOCKS) == 0,
            )
        models = []
        for k in range(self.n_seed_ensemble):
            model = self._make_model(seed=self.seed + k, extra_params=extra)
            model.fit(x_train, y_train, **self._fit_kwargs(model, weights))
            models.append(model)
        return models

    @staticmethod
    def _predict_mean(models: list[Any], x: pd.DataFrame) -> np.ndarray:
        """Seed-ensemble prediction: the mean over the fitted models (unclipped)."""
        return np.mean([np.asarray(m.predict(x), dtype=float) for m in models], axis=0)

    def _oof_baseline_predictions(
        self, x_base: pd.DataFrame, y_base: np.ndarray, *, w_base: np.ndarray | None = None
    ) -> np.ndarray:
        """Out-of-fold prediction for every baseline row via time-blocked folds (single-seed fits).

        Every baseline row gets a prediction from a model not trained on its fold, so residuals /
        calibration fits on this basis are not flattered by in-sample optimism (a shuffled or
        in-sample basis would read unbiased by construction).
        """
        n = len(y_base)
        folds = time_block_folds(n, n_folds=_N_FOLDS, n_blocks=_N_BLOCKS)
        oof = np.full(n, np.nan)
        for f in range(_N_FOLDS):
            hold = folds == f
            model = self._make_model()
            model.fit(
                x_base.iloc[~hold],
                y_base[~hold],
                **self._fit_kwargs(model, w_base[~hold] if w_base is not None else None),
            )
            oof[hold] = np.asarray(model.predict(x_base.iloc[hold]), dtype=float)
        return oof

    def _fit_calibration(
        self, x_base: pd.DataFrame, y_base: np.ndarray, *, w_base: np.ndarray | None = None
    ) -> CalibrationLine:
        """Fit the calibration line on time-blocked out-of-fold baseline predictions."""
        n = len(y_base)
        if n < _MIN_HOLDOUT_ROWS:
            return IDENTITY_CALIBRATION
        line = fit_calibration_line(y_base, self._oof_baseline_predictions(x_base, y_base, w_base=w_base))
        logger.info("%s calibration line: slope=%.4f intercept=%.2f (n=%d)", self.name, line.slope, line.intercept, n)
        return line

    def _residual_corrections(
        self,
        features: pd.DataFrame,
        *,
        y: np.ndarray,
        baseline_sel: np.ndarray,
        upgraded_sel: np.ndarray,
        w_base: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-upgraded-row correction from centred cell-mean out-of-fold baseline residuals (Issue 13).

        The cells are the F6 CEM coarsening of ``matching_vars``; adding each upgraded row's centred
        cell-mean baseline residual to its prediction removes the headline bias that the
        baseline→upgraded covariate-mix shift converts from the model's conditional bias (the global
        OOF level is centred out — see :func:`~benchmarking.baselines.power_model.fitting.
        cell_residual_calibration`). Returns zeros when the baseline is too small to split.
        """
        x_base = features.iloc[baseline_sel]
        y_base = y[baseline_sel]
        n_up = int(np.asarray(upgraded_sel).sum())
        if len(y_base) < _MIN_HOLDOUT_ROWS:
            return np.zeros(n_up)
        residuals = y_base - self._oof_baseline_predictions(x_base, y_base, w_base=w_base)
        edges = self.matching_bin_edges if self.matching_bin_edges is not None else self._default_bin_edges()
        codes = cell_codes(features[list(self.matching_vars)], edges)
        calib = cell_residual_calibration(
            codes_baseline=codes[baseline_sel], residuals=residuals, codes_target=codes[upgraded_sel]
        )
        logger.info(
            "%s residual calibration: mean centred correction %+.2f kW over %d upgraded rows "
            "(%d cells; %d rows unseen -> 0; centred-out OOF level %+.2f kW)",
            self.name,
            float(calib.per_row_residual.mean()) if n_up else float("nan"),
            n_up,
            calib.n_cells,
            calib.n_target_unseen,
            calib.global_mean_residual,
        )
        return calib.per_row_residual

    def _holdout_fit(
        self, x_base: pd.DataFrame, y_base: np.ndarray, *, w_base: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Train on a baseline train split, predict a held-out baseline slice (honest fit quality).

        The held-out slice is **time-blocked** (fold 0 of the shared split shape), not shuffled — a
        shuffled holdout sits minutes from its training rows, so autocorrelation makes its residuals
        optimistic. Also returns the held-out rows' positions **within the baseline block** so the
        caller can line the residuals up with their conditions (ws/TI) for the diagnostics.
        """
        n = len(y_base)
        if n < _MIN_HOLDOUT_ROWS:
            models = self._fit_models(x_base, y_base, weights=w_base)
            pred = _clip_predictions(
                self._predict_mean(models, x_base),
                y_train=y_base,
                rated_power_kw=self.baseline_rated_power_kw,
            )
            return y_base, pred, np.arange(n)
        valid = time_block_folds(n, n_folds=_N_FOLDS, n_blocks=_N_BLOCKS) == 0
        models = self._fit_models(
            x_base.iloc[~valid], y_base[~valid], weights=w_base[~valid] if w_base is not None else None
        )
        pred_valid = _clip_predictions(
            self._predict_mean(models, x_base.iloc[valid]),
            y_train=y_base[~valid],
            rated_power_kw=self.baseline_rated_power_kw,
        )
        return y_base[valid], pred_valid, np.flatnonzero(valid)

    def _run_dir(self, mi: MethodInput, index: pd.DatetimeIndex) -> Path:
        """Return the per-run output folder ``<out_dir>/power_model_<wtg>_<start>_<end>`` (a temp dir when unset).

        Computed once per ``estimate`` and shared by the overall diagnostics and the optional conditional
        step so both write into the *same* run folder.
        """
        upgrade_start = _upgrade_start(mi.upgrade_timing, index)
        run_name = f"power_model_{mi.test_wtg}_{upgrade_start:%Y%m%d}_{index.max():%Y%m%d}"
        out_root = Path(self.out_dir) if self.out_dir is not None else Path(tempfile.mkdtemp(prefix="power_model_"))
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write(
        self,
        mi: MethodInput,
        *,
        run_dir: Path,
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
        run_name = run_dir.name
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

    def _model_factory_label(self) -> str | None:
        """Return a stable, diffable label for ``model_factory`` in the run-config YAML.

        Registry names pass through; a callable is identified by ``module.qualname`` rather than
        ``repr`` (whose memory address would make the YAML nondeterministic across runs).
        """
        if self.model_factory is None or isinstance(self.model_factory, str):
            return self.model_factory
        module = getattr(self.model_factory, "__module__", None)
        qualname = getattr(self.model_factory, "__qualname__", None)
        if qualname is None:  # e.g. functools.partial or a callable instance
            return f"<callable {type(self.model_factory).__name__}>"
        return f"{module}.{qualname}" if module else qualname

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
            "era5_derivations": list(self.era5_derivations),
            "hub_height_m": self.hub_height_m,
            "time_features": list(self.time_features),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "reference_stat_cols": list(self.reference_stat_cols),
            "era5_exclude": list(self.era5_exclude),
            "availability_feature": self.availability_feature,
            "model_params": {**TUNED_MODEL_PARAMS, **self.model_params} if self.model_factory is None else {},
            "model_factory": self._model_factory_label(),
            "n_seed_ensemble": self.n_seed_ensemble,
            "early_stopping": self.early_stopping,
            "calibrate_slope": self.calibrate_slope,
            "calibrate_residuals": self.calibrate_residuals,
            "time_decay_half_life_days": self.time_decay_half_life_days,
            "toggle_estimator": self.toggle_estimator,
        }
