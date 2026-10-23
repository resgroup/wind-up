"""``PowerModelMethod``: the simplest-possible ML uplift method behind the harness ``Method`` seam.

A single supervised **counterfactual power model**: learn the test turbine's normal power as a
function of *reference-only* (upgrade-invariant) features over the baseline period, predict the
counterfactual power over the upgraded period, and take the energy ratio
``uplift = sum(actual) / sum(counterfactual) - 1`` over the upgraded rows. No propensity, no
cross-fitting — fit on the baseline, predict on the disjoint upgraded window, so there is no
in-sample leakage to correct for.

The features are the power references' active power and nacelle direction, the waking scenarios,
and the sun's elevation at the site. ERA5 is not a feature; it feeds only the conditional step.

It is v0-independent — ERA5 is supplied as a plain hourly DataFrame (the driver fetches it), so
this package imports nothing from ``wind_up``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

from benchmarking.baselines.block_bootstrap import bootstrap_energy_ratio
from benchmarking.baselines.power_model import diagnostics as diag
from benchmarking.baselines.power_model.features import (
    SOLAR_ELEVATION,
    build_reference_features,
    check_reference_only,
    extract_outcome,
    reference_mean_wind_speed,
    solar_elevation_frame,
    waking_scenario_features,
    without_constant_scenarios,
)
from benchmarking.baselines.power_model.fitting import make_outcome_model, model_safe_features, time_block_folds
from benchmarking.baselines.power_model.screening import (
    ScreenResult,
    screen_references,
)
from benchmarking.baselines.power_model.screening import (
    empty_screen_passes as _empty_screen_passes,
)
from benchmarking.baselines.power_model.shape import (
    PREDICTED_POWER,
    ShapeResult,
    freestream_scale,
    power_deviations,
    reference_wind_speed_drift,
    uplift_shape,
)
from benchmarking.diagnostics import DiagnosticContext, stages, write_common_diagnostics, write_run_config
from benchmarking.diagnostics.context import ERA5_UNLOCATED
from benchmarking.harness.conditions import (
    CONDITION_BINS,
    condition_bins,
    label_column,
    validate_conditions,
)
from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.harness.operating_state import VALID_UPLIFT_COL, OperatingStateConfig, label_operating_states
from benchmarking.harness.reanalysis import interpolate_era5
from benchmarking.harness.record_selection import (
    RARE_CONFIGURATION,
    UNPAIRED,
    USABLE,
    loss_reasons,
    rare_configurations,
    record_losses,
)
from benchmarking.harness.toggle import is_toggle, pair_within, resolve_toggle, toggle_cycle, toggle_upgrade_start
from benchmarking.harness.waking import DIRECTION_COL, NONE, UNKNOWN, UPWIND_COL, WAKING_SCENARIO_COL
from wind_up.farm import TurbineUplift, farm_uplift

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)

# The fewest usable hours a measurement needs in each period (method step 11 i).
_MIN_PERIOD_HOURS = 96.0
_MIN_HOLDOUT_ROWS = 20  # below this, report the in-sample fit (no point splitting off a tiny valid set)
# Time-blocked split shape for the baseline holdout diagnostic (``_holdout_fit`` takes fold 0 as the
# held-out slice): 25 contiguous blocks round-robin over 5 folds, so each fold is ~20% of the rows
# spread across the whole window (seasonally balanced) while staying contiguous at the block scale.
_N_FOLDS = 5
_N_BLOCKS = 25
# Adaptive time-decay half-life = this multiple of the campaign's own duration (Issue 15): a
# scale-free "trust pre-campaign data within ~k campaign-durations" rule that gives a short half-life
# for a short campaign (drift protection) and a long one for a long campaign (use the plentiful recent
# data), reproducing the earlier regime map with one mechanism-anchored constant. k=2 -> 1mo~60d, 12mo~730d.
_TIME_DECAY_CAMPAIGN_MULTIPLE = 2.0
_MIN_TIME_DECAY_DURATION_DAYS = 1.0  # floor the campaign duration so the half-life can't degenerate to 0

# Capacity verdict: loosening min_child_samples 200 -> 50 materially
# improves prepost spread/score (placebo ALL Δscore -0.62 pp) at neutral overall P50; 20 overshoots
# into overfit. A power_model-specific tuning — the common params in ``make_outcome_model`` are
# unchanged; drivers pass this instead.
TUNED_MODEL_PARAMS: dict[str, Any] = {"min_child_samples": 50}

# Method step 15: a predicted-power bin needs this many after-period hours to be measured.
_SHAPE_MIN_HOURS = 24.0
# The electrical Cp a turbine is taken to reach between 40 % and 60 % of rated, which sets the
# free-stream scale factor on the references' wind speed.
_DEFAULT_TARGET_CP = 0.46
_DEFAULT_AIR_DENSITY = 1.225
# A reference's monthly wind speed over the others' may move this much before it is flagged.
_ANEMOMETER_DRIFT_THRESHOLD = 0.03
# The predicted-power bin counts a subset's shape may use, tried from the most.
_SUBSET_POWER_BINS = (6, 5, 4)
# A measurement's deviation of each predicted-power bin from its overall uplift, one column per bin
# numbered from the lowest, carried with its reading so a test turbine can read its references' shape.
SHAPE_PREFIX = "power_shape_"

# The headline's block bootstrap (method step 14): the block length its sigma is reported at, and
# the lengths it is also recorded at.
_DEFAULT_BLOCK_HOURS = 24.0
_DEFAULT_BLOCK_HOURS_SWEEP: tuple[float, ...] = (6.0, 24.0, 168.0, 672.0)
_DEFAULT_N_RESAMPLES = 1000
_UNCERTAINTY_COLUMNS = ("block_hours", "sigma", "sigma_robust", "n_blocks", "frac_resamples_finite")
# The reverse-direction reading ``record_reversal`` adds to the uncertainty row (method step 13 option B).
REVERSAL_COLUMNS: tuple[str, ...] = ("uplift_forward", "uplift_reverse", "sigma_reverse")
# What the reference-corrected headline adds to the uncertainty row (method step 13 option C): the kept
# references' mean reading it removed, the forward reading's sampling sigma, the correction's standard
# error and how many references it averaged.
CORRECTION_COLUMNS: tuple[str, ...] = (
    "reference_correction",
    "sigma_sampling",
    "sigma_reference",
    "n_correction_references",
)
# What the held-out before period adds to a prepost uncertainty row (method step 14): the systematic
# sigma and how many windows it was read over.
HOLDOUT_COLUMNS: tuple[str, ...] = ("sigma_holdout", "n_holdout_blocks")
_DEFAULT_HOLDOUT_MAX_BLOCKS = 12
_DEFAULT_SYSTEMATIC_CORRELATION = 0.5
HeadlineEstimator = Literal["forward", "reversal", "reference_corrected"]
HEADLINE_ESTIMATORS: tuple[str, ...] = get_args(HeadlineEstimator)

# The condition axes this method reports, all upgrade-invariant: the test turbine's predicted power,
# the references' wind speed scaled to free stream, the step 5 direction, the sun's elevation and
# whether a turbine is upwind of the test turbine.
_SUPPORTED_CONDITIONS: tuple[str, ...] = (PREDICTED_POWER, "freestream_ws", "direction", "solar_elevation", "waked")
_SUBSET_CONDITIONS: tuple[str, ...] = ("direction", "solar_elevation", "waked")


# Reference-validity screen (R3). A reference whose own performance shifts across the campaign
# boundary biases the counterfactual, so each candidate is estimated against the others and clear
# outliers are made power-free. The floor is a deviation from the pool's median, in uplift
# fraction, calibrated on clean placebo pools rather than chosen: those spread up to 1.19 pp with
# nothing injected, so 2.5 pp leaves roughly a factor of two before a healthy reference is at risk.
# Set high deliberately -- ruling out a good reference costs more than leaving a mild bad one in
# (on the clean fixture a false positive moved the estimate 0.88 pp).
_DEFAULT_SCREEN_FLOOR = 0.025
# A screening estimate is only as good as the campaign it is measured over, and the benchmark sweep
# set this rather than a hand-picked calibration. At 90 days the sweep still produced false
# positives on 3-month campaigns -- a reference read -2.9%, about 3.1 pp from its pool median, and
# ruling it out made the benchmark *worse* (score +0.10 pp, spread +0.13 pp). The same three-
# reference pool is clean at 6 and 12 months, where the screen runs and correctly finds nothing, so
# the driver is campaign length rather than pool size. 150 days sits clear of both: a 3-month
# campaign (~90 days of data) is not screened, a 6-month one (~182) is.
_DEFAULT_SCREEN_MIN_CAMPAIGN_DAYS = 150.0


# The plot stages a reference measurement leaves to the test turbine's run.
_SHARED_STAGES = frozenset({stages.CHANGES, stages.OPERATING_STATES, stages.NORTHING})


def reference_measurement_dirname(reference: str, *, test_wtg: str) -> str:
    """Return the folder of one reference's measurement for one test turbine."""
    return f"{reference}_reference_for_{test_wtg}"


def sigma_column(block_hours: float) -> str:
    """Return the column name of the sigma at ``block_hours``, for example ``sigma_block_24h``."""
    return f"sigma_block_{block_hours:g}h"


def is_reading_column(name: object) -> bool:
    """Whether ``name`` is a column a measurement's reading carries beside its uplift.

    These are ``sigma``, the block-length sweep ``sigma_block_<h>h``, :data:`REVERSAL_COLUMNS`,
    :data:`CORRECTION_COLUMNS`, :data:`HOLDOUT_COLUMNS` and the ``power_shape_<i>`` deviations.
    """
    return (
        name == "sigma"
        or str(name).startswith(("sigma_block_", SHAPE_PREFIX))
        or name in REVERSAL_COLUMNS
        or name in CORRECTION_COLUMNS
        or name in HOLDOUT_COLUMNS
    )


def headline_detail(output: MethodOutput) -> dict[str, float]:
    """Return the :data:`CORRECTION_COLUMNS` and :data:`HOLDOUT_COLUMNS` a headline reports, as one row's columns.

    Empty for an output that reports none of them.
    """
    diagnostics = output.uncertainty_diagnostics
    if diagnostics is None or diagnostics.empty:
        return {}
    row = diagnostics.iloc[0]
    return {c: float(row[c]) for c in (*CORRECTION_COLUMNS, *HOLDOUT_COLUMNS) if c in diagnostics.columns}


def empty_uncertainty() -> pd.DataFrame:
    """Return the empty block-length sweep, typed as a populated one would be."""
    return pd.DataFrame({c: pd.Series(dtype=int if c == "n_blocks" else float) for c in _UNCERTAINTY_COLUMNS})


def uncertainty_summary(sweep: pd.DataFrame, *, block_hours: float) -> dict[str, float]:
    """Return the headline's sigma at ``block_hours`` and at every length of ``sweep``, as one row's columns.

    ``sigma`` is the sigma at ``block_hours``; ``sigma_block_<h>h`` the sigma at each length.
    """
    by_hours = sweep.set_index("block_hours") if not sweep.empty else sweep
    row: dict[str, float] = {"block_hours": float(block_hours)}
    for column in ("sigma", "sigma_robust", "n_blocks", "frac_resamples_finite"):
        row[column] = float(by_hours.loc[block_hours, column]) if block_hours in by_hours.index else float("nan")
    for hours in by_hours.index:
        row[sigma_column(float(hours))] = float(by_hours.loc[hours, "sigma"])
    return row


def reference_overall_uplift(
    reference_uplifts: pd.DataFrame, *, rated_power_kw: float, after_screen: bool = True
) -> float:
    """Combine the references into one energy-weighted uplift, the campaign's check of bias (method step 16).

    After the screen, the references the screen ruled out or did not judge are left out; before it,
    every reference with a finite uplift counts. Combined through ``farm_uplift``, the path the test
    turbines take. NaN when no reference counts.
    """
    if reference_uplifts.empty:
        return float("nan")
    rows = reference_uplifts[np.isfinite(reference_uplifts["uplift"].to_numpy(dtype=float))]
    if after_screen:
        out = rows["screened"].astype(bool).to_numpy()
        if "unjudged" in rows.columns:
            out |= rows["unjudged"].astype(bool).to_numpy()
        rows = rows[~out]
    if rows.empty:
        return float("nan")
    turbines = [
        TurbineUplift(
            turbine=str(row.turbine),
            uplift=float(row.uplift),
            actual_energy=float(row.actual_energy),
            n_records=int(row.n_records),
            rated_power_kw=rated_power_kw,
        )
        for row in rows.itertuples()
    ]
    return float(farm_uplift(turbines).uplift)


def kept_reference_correction(reference_uplifts: pd.DataFrame | None) -> dict[str, float]:
    """Return the kept references' mean reading in log space and its standard error (method step 13 option C).

    A reference is kept when it has a finite reading above -100 % and was neither ruled out nor left
    unjudged by the screen. ``reference_correction`` is ``exp(mean(ln(1 + u))) - 1`` over the kept
    readings and ``n_correction_references`` their number. ``sigma_reference_log`` is the standard
    deviation of ``ln(1 + u)`` over the root of their number, in quadrature with half the mean's size,
    so a larger correction is less certain; NaN below two readings. The correction is NaN when none
    is kept.
    """
    logs = np.array([], dtype=float)
    if reference_uplifts is not None and not reference_uplifts.empty:
        rows = reference_uplifts
        out = rows["screened"].astype(bool).to_numpy()
        if "unjudged" in rows.columns:
            out |= rows["unjudged"].astype(bool).to_numpy()
        uplift = rows["uplift"].to_numpy(dtype=float)[~out]
        uplift = uplift[np.isfinite(uplift) & (uplift > -1.0)]
        logs = np.log1p(uplift)
    k = logs.size
    return {
        "reference_correction": float(np.expm1(logs.mean())) if k else float("nan"),
        "sigma_reference_log": float(np.hypot(logs.std(ddof=1) / np.sqrt(k), logs.mean() / 2))
        if k > 1
        else float("nan"),
        "n_correction_references": float(k),
    }


def holdout_windows(
    before_times: pd.DatetimeIndex, *, after_length: pd.Timedelta, max_blocks: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the ``(start, end]`` windows the before period is held out in, latest first (method step 14).

    Each window is as long as the after period or half the before period's span, whichever is shorter.
    They tile back from the last before record without overlap, at most ``max_blocks`` of them, while
    a window still ends after the first record.
    """
    if before_times.empty or max_blocks <= 0:
        return []
    first, last = before_times.min(), before_times.max()
    length = min(after_length, (last - first) / 2)
    if length <= pd.Timedelta(0):
        return []
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    end = last
    while len(windows) < max_blocks and end > first:
        windows.append((end - length, end))
        end -= length
    return windows


def holdout_excess(readings: np.ndarray, *, sampling_sigmas: np.ndarray) -> float:
    """Return the held-out readings' spread beyond their own sampling sigma, inflated for few readings.

    ``sqrt(max(mean(reading²) - mean(sigma²), 0))`` over the finite pairs, times the one-sigma
    quantile of Student's t with as many degrees of freedom as readings. NaN without a reading.
    """
    readings = np.asarray(readings, dtype=float)
    sampling_sigmas = np.asarray(sampling_sigmas, dtype=float)
    ok = np.isfinite(readings) & np.isfinite(sampling_sigmas)
    n = int(np.count_nonzero(ok))
    if n == 0:
        return float("nan")
    excess = max(float(np.mean(readings[ok] ** 2) - np.mean(sampling_sigmas[ok] ** 2)), 0.0)
    return float(np.sqrt(excess) * student_t.ppf(norm.cdf(1.0), n))


def combined_sigma(sampling: float, *, reference: float, holdout: float, correlation: float) -> float:
    """Return ``sqrt(sampling² + reference² + holdout² - 2 * correlation * reference * holdout)``.

    The reference correction's error and the held-out before period's error are taken as correlated
    by ``correlation``; their combined variance is floored at 0. NaN when any input is.
    """
    systematic = np.maximum(reference**2 + holdout**2 - 2.0 * correlation * reference * holdout, 0.0)
    return float(np.sqrt(sampling**2 + systematic))


def empty_reference_uplifts() -> pd.DataFrame:
    """Return the empty reference-uplift frame, typed as a populated one would be.

    A campaign whose pool leaves no reference estimable still reports the documented columns, so a
    consumer can mask on ``screened`` without special-casing.
    """
    return pd.DataFrame(
        {
            "turbine": pd.Series(dtype=object),
            "uplift": pd.Series(dtype=float),
            "actual_energy": pd.Series(dtype=float),
            "n_records": pd.Series(dtype=int),
            "screened": pd.Series(dtype=bool),
            "unjudged": pd.Series(dtype=bool),
            "sigma": pd.Series(dtype=float),
        }
    )


def _reading(output: MethodOutput) -> dict[str, float]:
    """Return one measurement's uplift with its sigma and the block-length sweep, as columns."""
    reading = {"uplift": float(output.p50_overall), "sigma": float(output.sigma_overall or float("nan"))}
    if output.uncertainty_diagnostics is not None and not output.uncertainty_diagnostics.empty:
        row = output.uncertainty_diagnostics.iloc[0]
        reading.update({str(c): float(row[c]) for c in row.index if is_reading_column(c) and c != "sigma"})
    return reading


def _kept_shape_deviations(reference_uplifts: pd.DataFrame | None) -> np.ndarray | None:
    """Return the ``power_shape_<i>`` deviations of the references the headline correction keeps, one row each."""
    if reference_uplifts is None or reference_uplifts.empty:
        return None
    columns = sorted(
        (c for c in reference_uplifts.columns if str(c).startswith(SHAPE_PREFIX)),
        key=lambda c: int(str(c).removeprefix(SHAPE_PREFIX)),
    )
    if not columns:
        return None
    rows = reference_uplifts
    uplift = rows["uplift"].to_numpy(dtype=float)
    kept = np.isfinite(uplift) & (uplift > -1.0)
    for flag in ("screened", "unjudged"):
        if flag in rows.columns:
            kept &= ~rows[flag].astype(bool).to_numpy()
    return rows.loc[kept, columns].to_numpy(dtype=float)


def _with_sigmas(passes: pd.DataFrame, readings: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Return ``passes`` with each turbine's sigma columns from its latest reading."""
    if passes.empty or not readings:
        return passes
    columns = sorted({c for reading in readings.values() for c in reading if c != "uplift"})
    out = passes.copy()
    for column in columns:
        out[column] = [readings.get(str(t), {}).get(column, float("nan")) for t in out["turbine"]]
    return out


def _ratio(actual: np.ndarray, counterfactual: np.ndarray) -> float:
    """Energy-ratio ``Σactual / Σcounterfactual - 1`` over finite pairs; NaN if the denominator is 0."""
    finite = np.isfinite(actual) & np.isfinite(counterfactual)
    denom = float(counterfactual[finite].sum())
    return float(actual[finite].sum()) / denom - 1.0 if denom != 0 else float("nan")


def _combine_uplift(r_fwd: np.ndarray, r_rev: np.ndarray) -> np.ndarray:
    """Two-direction uplift ``sqrt((1+r_fwd)/(1+r_rev)) - 1`` (shrinkage cancels); NaN if either 1+r <= 0.

    Under a common per-bin multiplicative shrinkage the forward/reverse ratios are ``(1+u)/s`` and
    ``1/(s(1+u))``, so their geometric contrast recovers ``u`` with ``s`` cancelled (by design).
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


def _reference_input(
    mi: MethodInput, *, target: str, references: Sequence[str], timing: pd.Timestamp | None = None
) -> MethodInput:
    """Return the input for estimating reference ``target`` against ``references``.

    Every turbine the campaign's estimate keeps stays in, so the test turbine and any reference left
    out of ``references`` join the wake contributors. ``target``'s records inside the test turbine's
    works window are not valid for uplift, so its before period ends where the test turbine's does.

    :param timing: the contrast to estimate across; the campaign's own when ``None``
    """
    context = mi.context
    kept = [mi.test_wtg, *context.candidate_references, *context.wake_contributors]
    valid = context.valid_for_uplift
    if context.works is not None and target in valid.columns:
        start, end = context.works
        valid = valid.copy()
        valid.loc[(valid.index >= start) & (valid.index < end), target] = False
    sub_context = dataclasses.replace(
        context,
        valid_for_uplift=valid,
        test_wtg=target,
        candidate_references=list(references),
        # Sorted, not in the order the campaign's own estimate happened to hold them: that order
        # starts at the test turbine, so the same reference would be screened against the same
        # turbines laid out differently for each test turbine, and the model is not invariant to
        # column order. Sorted, every test turbine screens a reference identically.
        wake_contributors=sorted(w for w in kept if w != target and w not in set(references)),
        timing=context.timing if timing is None else timing,
        # A single reference's estimate neither refills nor reads its own references.
        reserve_references=[],
    )
    return MethodInput(scada_df=mi.scada_df, test_wtg=target, campaign_context=sub_context)


@dataclass
class PowerModelMethod:
    """Pluggable counterfactual power-model uplift estimator (prepost and toggle).

    :param columns: **required** source-native column schema — the single source of the method's
        column names. It reads the ``active_power`` (the outcome ``Y`` and the reference active-power
        feature), ``availability`` (drives the test-turbine downtime filter) and ``wind_speed`` (its
        reference mean feeds the stuck-filter calm exemption) roles; the remaining roles feed only
        the shared diagnostics.
    :param baseline_rated_power_kw: **required** rated power of the turbine over the data the model is
        fitted on — today every fit is on baseline rows, so this is the baseline rating. It caps the
        clipped counterfactual predictions. (A future cross-predict direction that trains on upgraded
        data would need the upgraded rating; that is out of scope here.)
    :param era5_hourly_df: optional ERA5 (Open-Meteo columns), hourly or already aligned to the SCADA
        periods; used only by the conditional step, never as a model feature
    :param site_location: the site's ``(latitude, longitude)`` in degrees; when given the sun's elevation
        there is a model feature
    :param name: method name shown in the leaderboard
    :param out_dir: where per-run folders are written; a temp dir when ``None``
    :param run_subdir: write each run into its own ``power_model_<wtg>_<start>_<end>`` folder under
        ``out_dir``, with run-named, timestamped CSVs, so a study's runs sit side by side. ``False``
        writes straight into ``out_dir`` with plain names (``results.csv``, ...), for a caller that
        already gives every run its own folder, as a campaign does per test turbine.
    :param save_plots: also write the diagnostic plots
    :param reference_plots_dir: when set, each reference measurement of the reference-uplift report
        writes its own plots and diagnostics, as a test turbine's would, under
        ``reference_plots_dir / reference_measurement_dirname(reference, test_wtg=...)``; its
        estimate is then always refitted rather than taken from the screen
    :param repeat_shared_plots: also write the step 1, 2 and 4 plots; a reference measurement leaves
        them out, as the test turbine's run already draws the same turbines
    :param seed: seed for the baseline holdout split and the LightGBM ``random_state`` (a caller-supplied
        ``random_state`` in ``model_params`` still wins)
    :param model_params: LightGBM overrides passed to the outcome model; merged **over** the tuned
        default ``TUNED_MODEL_PARAMS`` (``min_child_samples=50``), so a bare method already carries
        the accepted capacity and any key here wins
    :param timebase: analysis timebase; inferred from the data when ``None``
    :param conditions: the axes of the uplift distributions (method step 15): ``"predicted_power"``,
        the measured shape, and the axes it is cast to, ``"freestream_ws"`` (the power references'
        mean wind speed times the factor that meets ``target_cp``), ``"direction"`` (step 5's
        direction), ``"solar_elevation"`` and ``"waked"`` (a turbine upwind of the test turbine). An
        axis whose input is missing is left out with a warning. ``()`` skips the step. It never
        moves the headline
    :param subset_conditions: the axes whose subsets each get their own predicted-power shape,
        written to the run's ``conditional/uplift_subsets.csv`` and flagged where they differ
    :param rotor_diameters_m: rotor diameter per turbine, for the free-stream wind speed
    :param target_cp: the electrical Cp the test turbine is taken to reach between 40 % and 60 % of
        rated in the before period, which sets the free-stream scale factor
    :param air_density_kg_m3: the air density of that Cp
    :param shape_min_hours: the after-period hours a predicted-power bin needs to be measured
    :param pairing_max_gap: for a toggle campaign, a record of one state is used only when a usable
        record of the other state lies within this gap; ``None`` (default) means one toggle cycle
    :param block_hours: block length of the headline's circular block bootstrap over the after
        period's (actual, counterfactual) records; its sigma is ``sigma_overall``
    :param block_hours_sweep: block lengths the sigma is also recorded at, in the
        ``uncertainty_diagnostics`` row and the run's ``uncertainty.csv``
    :param n_resamples: bootstrap resamples
    :param bootstrap_seed: RNG seed of the bootstrap
    :param reference_stat_cols: extra per-reference value columns to carry as features beside active power
    :param direction_feature: when ``True``, each reference contributes its **north-calibrated**
        direction as ``sin``/``cos``. Requires the shared northing step to have written
        ``columns.northed("nacelle_position")`` into the frame, and raises naming that column when
        it has not; the raw nacelle position is never used, northed or not. The test turbine's own
        direction stays barred (design-note §3): northing does not make a post-treatment signal safe.
    :param adaptive_time_decay: set the campaign-proximity weights' half-life from the campaign's
        own duration, ``_TIME_DECAY_CAMPAIGN_MULTIPLE * campaign_duration_days``. ``False``
        (**default**) with ``time_decay_half_life_days`` left ``None`` means every training row
        weighs the same, whenever it happened. When ``True``, ``time_decay_half_life_days`` must be
        left ``None``.
    :param time_decay_half_life_days: a *fixed* half-life for the campaign-proximity training
        weights ``0.5 ** (days_outside_campaign / half_life)``, used only when
        ``adaptive_time_decay=False``. Rows inside the campaign interval (for toggle, the
        interleaved on and off rows) weigh 1; rows outside decay with their distance to it.
        ``None`` (**default**) leaves the weighting off.
    :param reference_screen: when ``True`` (**default**), estimate each candidate reference as if it
        were a test turbine against the others, and make clear outliers **power-free**: they contribute
        no features. Needs at least 3 candidate references to form a majority; below that it is skipped.
    :param screen_floor: deviation from the screened pool's median, in uplift fraction, at which a
        reference is ruled out
    :param report_reference_uplifts: report each candidate reference's own uplift, the standard
        campaign sanity check (a healthy campaign's references read near 0%). Costs one extra model
        fit per reference, so a method sweep that scores estimators rather than reporting campaigns
        turns it off. The reference-corrected headline needs it; under the other headlines it is a
        report, not an input, and turning it off never moves the headline
    :param headline_estimator: how the prepost headline is formed. ``"forward"`` is the
        counterfactual energy ratio ``Σactual/Σprediction - 1`` from the baseline-trained model.
        ``"reference_corrected"`` (default) divides ``1 +`` the forward ratio by ``1 +`` the kept
        references' mean reading (:func:`kept_reference_correction`), so the headline and every bin of
        the uplift distributions move together, and its sigma is the forward sampling sigma in
        quadrature with the correction's standard error; the uncertainty row gains
        :data:`CORRECTION_COLUMNS`. It needs ``report_reference_uplifts``, and the headline is NaN
        when no reference is kept. ``"reversal"`` (the reversal correction) also fits the reverse
        direction (train the upgraded rows, predict the baseline rows) and reports
        ``sqrt((1+r_fwd)/(1+r_rev)) - 1``, so a shrinkage common to both directions cancels; it costs
        one extra model fit per estimate. Toggle is always the forward ratio
    :param holdout_max_blocks: the most windows of the before period a prepost estimate holds out in
        turn (:func:`holdout_windows`), refitting the model on the rest and reading each window as if
        it were the after period. Their spread beyond their own sampling sigma (:func:`holdout_excess`)
        is ``sigma_holdout``, joined to the headline sigma and to each measured bin's
        (see ``systematic_correlation``);
        the uncertainty row gains :data:`HOLDOUT_COLUMNS`. A window or remainder under the usable-hours
        floor is skipped, and ``sigma_holdout`` is NaN when none is left. Costs one model fit per
        window; ``0`` turns it off. A toggle estimate holds nothing out
    :param systematic_correlation: the correlation taken between the reference correction's error and
        the held-out before period's error when ``sigma_reference`` and ``sigma_holdout`` are joined
        (:func:`combined_sigma`); ``0`` joins them in quadrature
    :param record_reversal: also fit the reverse direction of a prepost estimate and record it, without
        changing the headline: the ``uncertainty_diagnostics`` row gains ``uplift_forward``,
        ``uplift_reverse`` and ``sigma_reverse`` (the reverse ratio's block bootstrap over the before
        period at ``block_hours``). NaN for a toggle estimate. Costs one extra model fit per estimate
    :param write_diagnostics: write the per-run diagnostics folder. ``False`` for the clones that
        estimate one reference during screening or reporting: ``out_dir=None`` does not suppress
        them, it makes a temp directory per run, and there are O(N) such runs per estimate
    :param screen_min_campaign_days: leave a candidate reference out of the screen when it holds
        less than this much upgraded data; when no candidate holds enough, the screen does not run.
        A screening estimate over a short campaign is too noisy to separate a bad reference from a
        good one at any floor, and ruling out a good reference costs more than leaving a mild bad
        one in
    :param row_dump_dir: when set, write the estimate's per-row frame (actual, counterfactual, row
        selection and the full feature matrix under its original names) to
        ``<row_dump_dir>/<test_wtg>/rows.parquet`` with a ``rows.json`` sidecar, for the level probe.
        A reference reading of the reference-uplift report dumps under
        ``<row_dump_dir>/<test_wtg>/references/<reference>/``; the screening clone does not dump.
        ``None`` (**default**) writes nothing

    The reference pool is the campaign's candidate references (the context's
    ``candidate_references``), not whichever turbines the frame holds; the screen makes some of
    them power-free rather than removing them from it. When the context offers reserve references,
    a reference the screen rules out, or cannot judge for want of campaign data, becomes a wake
    contributor instead and the nearest reserve takes its place, and the refilled pool is screened again.

    The toggle headline is always the counterfactual energy ratio ``Σactual/Σprediction - 1``.
    """

    columns: ColumnSchema
    baseline_rated_power_kw: float
    era5_hourly_df: pd.DataFrame | None = None
    site_location: tuple[float, float] | None = None
    name: str = "power_model"
    out_dir: Path | None = None
    save_plots: bool = False
    seed: int = 0
    model_params: dict[str, Any] = field(default_factory=dict)
    timebase: pd.Timedelta | None = None
    conditions: tuple[str, ...] = _SUPPORTED_CONDITIONS
    subset_conditions: tuple[str, ...] = _SUBSET_CONDITIONS
    rotor_diameters_m: Mapping[str, float] | None = None
    target_cp: float = _DEFAULT_TARGET_CP
    air_density_kg_m3: float = _DEFAULT_AIR_DENSITY
    shape_min_hours: float = _SHAPE_MIN_HOURS
    pairing_max_gap: pd.Timedelta | None = None
    block_hours: float = _DEFAULT_BLOCK_HOURS
    block_hours_sweep: tuple[float, ...] = _DEFAULT_BLOCK_HOURS_SWEEP
    n_resamples: int = _DEFAULT_N_RESAMPLES
    bootstrap_seed: int = 0
    reference_stat_cols: tuple[str, ...] = ()
    direction_feature: bool = True
    adaptive_time_decay: bool = False
    time_decay_half_life_days: float | None = None
    reference_screen: bool = True
    screen_floor: float = _DEFAULT_SCREEN_FLOOR
    screen_min_campaign_days: float = _DEFAULT_SCREEN_MIN_CAMPAIGN_DAYS
    report_reference_uplifts: bool = True
    headline_estimator: HeadlineEstimator = "reference_corrected"
    holdout_max_blocks: int = _DEFAULT_HOLDOUT_MAX_BLOCKS
    systematic_correlation: float = _DEFAULT_SYSTEMATIC_CORRELATION
    record_reversal: bool = False
    write_diagnostics: bool = True
    row_dump_dir: Path | None = None
    # How reanalysis is named in this run's plots, CSVs and logs. A campaign passes
    # era5_source_label() of the point it fetched, so a reader can tell which series was used.
    era5_label: str = ERA5_UNLOCATED
    run_subdir: bool = True
    reference_plots_dir: Path | None = None
    repeat_shared_plots: bool = True
    # A campaign's screen verdict is one answer for the whole campaign: the same pool judged across
    # the same contrast. Every test turbine would otherwise re-run the identical round-robin. A
    # campaign passes one dict to every turbine's method; None means screen per estimate. Every
    # method sharing a cache must share its configuration, since the key does not carry it.
    screen_cache: dict[tuple[tuple[str, ...], str, str, str, tuple[str, ...], str], ScreenResult] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate ``columns`` names every role this method reads, and the requested ``conditions``."""
        self.columns.require_roles(("active_power", "availability", "wind_speed"))
        if self.direction_feature:
            self.columns.require_roles(("nacelle_position",))
        validate_conditions(self.conditions, supported=_SUPPORTED_CONDITIONS, method_name=self.name)

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Estimate the test turbine's P50 uplift for one campaign and write diagnostics."""
        self._validate_model_config()
        # The campaign decides which turbines are references and which rows they may contribute;
        # everything below -- outcome, features, ERA5, diagnostics -- works from this selection.
        scada = mi.context.select(mi.scada_df)
        if self.columns.availability not in scada.columns:
            msg = (
                f"the availability column {self.columns.availability!r} (columns.availability) is not in "
                f"scada_df; the downtime filter is required for the power model and cannot be skipped."
            )
            raise ValueError(msg)
        # Checked before the screen, whose gate reads the same normal-operation mask, so a missing
        # column is reported as the configuration error it is rather than as a bare KeyError.
        index = pd.DatetimeIndex(pd.unique(scada.index)).sort_values()
        timebase = self.timebase if self.timebase is not None else _infer_timebase(scada.index)
        references = self._candidate_references(scada, mi=mi)
        n_refs = len(references)

        y = extract_outcome(
            scada, test_wtg=mi.test_wtg, turbine_col=mi.turbine_col, active_power_col=self.columns.active_power
        )
        screen = self.screen_references(mi) if self.reference_screen else None
        if screen is not None and screen.power_free and mi.context.reserve_references:
            mi, screen = self.refill_screened(mi, screen)
            scada = mi.context.select(mi.scada_df)
            references = self._candidate_references(scada, mi=mi)
            n_refs = len(references)
        # Without reserves the screen does not shrink the pool: a reference it ruled out, or could
        # not judge for want of campaign data, stays in the pool and contributes no features. With
        # reserves it has already left the pool for the wake contributors.
        power_free = tuple(r for r in screen.power_free if r in references) if screen is not None else ()
        features = self._reference_features(scada, mi=mi, references=references, power_free=power_free)
        features = self._add_site_features(features, index=index, timebase=timebase)
        era5 = self._aligned_era5(index=index, timebase=timebase)
        check_reference_only(features.columns.tolist(), test_wtg=mi.test_wtg)

        toggle_rows = resolve_toggle(mi.upgrade_timing, index)
        t = toggle_rows.upgraded
        power_references = [r for r in references if r not in power_free]
        reasons = self._select_records(
            scada,
            mi=mi,
            index=index,
            references=power_references,
            timebase=timebase,
            before=toggle_rows.campaign_baseline,
            after=t,
        )
        selected = np.asarray(reasons == USABLE) & np.isfinite(y.to_numpy(dtype=float))
        features, constant_scenarios = without_constant_scenarios(features, rows=selected)
        # The counterfactual is fitted on the lenient training baseline (pre-campaign U off-blocks);
        # more upgrade-invariant data helps. The conditional matching step below instead uses the
        # strict campaign_baseline (off-blocks only), so it never matches pre-campaign against on rows.
        baseline_sel = selected & toggle_rows.training_baseline
        upgraded_sel = selected & t
        hours_per_record = timebase / pd.Timedelta(hours=1)
        for period, rows in (("before", selected & toggle_rows.campaign_baseline), ("after", upgraded_sel)):
            hours = float(np.count_nonzero(rows)) * hours_per_record
            if hours < _MIN_PERIOD_HOURS:
                msg = (
                    f"{mi.test_wtg}: {hours:.1f} usable hours in the {period} period, under the {_MIN_PERIOD_HOURS:g} "
                    f"needed in each period to measure an uplift."
                )
                raise ValueError(msg)

        y_arr = y.to_numpy(dtype=float)
        weights = self._time_decay_weights(
            index, campaign_start=toggle_upgrade_start(mi.upgrade_timing, index), campaign_end=index.max()
        )
        fit = self._fit_predict(
            features,
            y=y_arr,
            baseline_sel=baseline_sel,
            upgraded_sel=upgraded_sel,
            weights=weights,
        )
        sum_actual = float(fit["y_upgraded"].sum())
        sum_counter = float(fit["pred_upgraded"].sum())
        uplift = sum_actual / sum_counter - 1.0 if np.isfinite(sum_counter) and sum_counter != 0 else float("nan")
        uncertainty = self._headline_uncertainty(
            index[upgraded_sel],
            actual=fit["y_upgraded"],
            counterfactual=fit["pred_upgraded"],
            period_start=toggle_upgrade_start(mi.upgrade_timing, index),
            period_end=index.max(),
            timebase=timebase,
        )
        summary = uncertainty_summary(uncertainty, block_hours=self.block_hours)
        if self.row_dump_dir is not None:
            self._write_row_dump(
                scada,
                mi=mi,
                index=index,
                timebase=timebase,
                references=references,
                power_free=power_free,
                features=features,
                y=y_arr,
                selected=selected,
                reasons=reasons,
                baseline_sel=baseline_sel,
                upgraded_sel=upgraded_sel,
                fit=fit,
                uplift=uplift,
            )

        # The reversal correction to the prepost headline. The forward ratio above is 1+r_fwd; the
        # reverse fit trains the upgraded rows and predicts the baseline rows for 1+r_rev, and
        # _combine_uplift returns sqrt((1+r_fwd)/(1+r_rev))-1 so a shrinkage common to both directions
        # cancels. Toggle keeps the forward ratio: its off-blocks already interleave, so no shift.
        reversal = self._reverse_direction(
            mi,
            features=features,
            y=y_arr,
            baseline_sel=baseline_sel,
            upgraded_sel=upgraded_sel,
            forward=uplift,
            index=index,
            timebase=timebase,
        )
        if self.headline_estimator == "reversal" and np.isfinite(reversal.get("uplift_reverse", np.nan)):
            uplift = float(_combine_uplift(np.array([uplift]), np.array([reversal["uplift_reverse"]]))[0])
        if not self.record_reversal:
            reversal = {}

        # Reported post-screen: the sanity check asks what the *final* analysis says its references
        # did, so a ruled-out reference is listed but does not drag the headline.
        reference_uplifts = (
            self.reference_uplifts(mi, power_free=power_free, screen=screen) if self.report_reference_uplifts else None
        )
        uplift, _, correction = self._reference_corrected(
            mi, forward=uplift, sigma_sampling=summary["sigma"], reference_uplifts=reference_uplifts
        )
        holdout = self._holdout_sigma(
            mi, features=features, y=y_arr, baseline_sel=baseline_sel, upgraded_sel=upgraded_sel, index=index
        )
        sigma = combined_sigma(
            summary["sigma"],
            reference=correction.get("sigma_reference", 0.0),
            holdout=holdout.get("sigma_holdout", 0.0),
            correlation=self.systematic_correlation,
        )

        # A clone that estimates one reference during screening or reporting writes nothing:
        # out_dir=None does not suppress diagnostics, it makes a temp directory per run.
        run_dir = self._run_dir(mi, index) if self.write_diagnostics else None
        if run_dir is not None:
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
                constant_features=constant_scenarios,
                losses=record_losses(
                    reasons, before=toggle_rows.campaign_baseline, after=t, hours_per_record=hours_per_record
                ),
                uncertainty=uncertainty,
            )
            if self.save_plots:
                self._write_shared_diagnostics(
                    mi,
                    run_dir=run_dir,
                    t=t,
                    selected=selected,
                    timebase=timebase,
                    era5=era5,
                    power_references=[r for r in references if r not in power_free],
                )

        # The uplift distributions are the optional last step; nothing above depends on them.
        shape: ShapeResult | None = None
        if self.conditions:
            shape = self._uplift_shape(
                mi,
                scada=scada,
                features=features,
                y=y_arr,
                baseline_sel=baseline_sel,
                upgraded_sel=upgraded_sel,
                fit=fit,
                headline=uplift,
                power_references=power_references,
                index=index,
                timebase=timebase,
                level_sigma=combined_sigma(
                    0.0,
                    reference=correction.get("sigma_reference", 0.0),
                    holdout=holdout.get("sigma_holdout", 0.0),
                    correlation=self.systematic_correlation,
                ),
                reference_deviations=_kept_shape_deviations(reference_uplifts),
                run_dir=run_dir,
            )
        return MethodOutput(
            p50_overall=uplift,
            p50_by_condition=None if shape is None else self._reported_shape(shape),
            sigma_overall=sigma,
            uncertainty_diagnostics=pd.DataFrame(
                [
                    {
                        "condition": "overall",
                        "condition_bin": "overall",
                        **{k: v for k, v in summary.items() if k != "sigma"},
                        **reversal,
                        **correction,
                        **holdout,
                        **self._shape_deviations(fit, timebase=timebase),
                    }
                ]
            ),
            reference_uplifts=reference_uplifts,
            screen_passes=screen.passes if screen is not None else None,
            labeled_rows=None if shape is None else shape.labels,
        )

    def _headline_uncertainty(
        self,
        times: pd.DatetimeIndex,
        *,
        actual: np.ndarray,
        counterfactual: np.ndarray,
        period_start: pd.Timestamp,
        period_end: pd.Timestamp,
        timebase: pd.Timedelta,
    ) -> pd.DataFrame:
        """Bootstrap the headline's sigma at ``block_hours`` and every length of ``block_hours_sweep``.

        One row per block length, with :data:`_UNCERTAINTY_COLUMNS`. The model is fixed: the
        after period's (actual, counterfactual) records are resampled in circular wall-clock blocks.
        """
        rows = []
        for hours in sorted({float(self.block_hours), *(float(h) for h in self.block_hours_sweep)}):
            result = bootstrap_energy_ratio(
                times=times,
                actual=np.asarray(actual, dtype=float),
                counterfactual=np.asarray(counterfactual, dtype=float),
                period_start=period_start,
                period_end=period_end,
                timebase=timebase,
                block_hours=hours,
                n_resamples=self.n_resamples,
                seed=self.bootstrap_seed,
            )
            rows.append(
                {
                    "block_hours": hours,
                    "sigma": result.sigma,
                    "sigma_robust": result.sigma_robust,
                    "n_blocks": result.n_blocks,
                    "frac_resamples_finite": result.frac_resamples_finite,
                }
            )
        return pd.DataFrame(rows, columns=list(_UNCERTAINTY_COLUMNS)) if rows else empty_uncertainty()

    def _reference_corrected(
        self,
        mi: MethodInput,
        *,
        forward: float,
        sigma_sampling: float,
        reference_uplifts: pd.DataFrame | None,
    ) -> tuple[float, float, dict[str, float]]:
        """Return the reference-corrected headline, its sigma and the :data:`CORRECTION_COLUMNS` row.

        Another ``headline_estimator`` keeps ``forward`` and ``sigma_sampling`` and records nothing; a
        toggle estimate keeps them and records no correction, with a zero standard error.
        """
        if self.headline_estimator != "reference_corrected":
            return forward, sigma_sampling, {}
        if is_toggle(mi.upgrade_timing):
            return (
                forward,
                sigma_sampling,
                {
                    "reference_correction": float("nan"),
                    "sigma_sampling": sigma_sampling,
                    "sigma_reference": 0.0,
                    "n_correction_references": 0.0,
                },
            )
        kept = kept_reference_correction(reference_uplifts)
        uplift = (1.0 + forward) / (1.0 + kept["reference_correction"]) - 1.0
        sigma_reference = (1.0 + uplift) * kept["sigma_reference_log"]
        if not np.isfinite(uplift):
            logger.warning("%s: %s has no kept reference reading to correct by", self.name, mi.test_wtg)
        logger.info(
            "%s %s: forward %+.3f%%, kept references' mean %+.3f%% (%d), corrected %+.3f%%",
            self.name,
            mi.test_wtg,
            100 * forward,
            100 * kept["reference_correction"],
            int(kept["n_correction_references"]),
            100 * uplift,
        )
        return (
            uplift,
            float(np.hypot(sigma_sampling, sigma_reference)),
            {
                "reference_correction": kept["reference_correction"],
                "sigma_sampling": sigma_sampling,
                "sigma_reference": sigma_reference,
                "n_correction_references": kept["n_correction_references"],
            },
        )

    def _holdout_sigma(
        self,
        mi: MethodInput,
        *,
        features: pd.DataFrame,
        y: np.ndarray,
        baseline_sel: np.ndarray,
        upgraded_sel: np.ndarray,
        index: pd.DatetimeIndex,
    ) -> dict[str, float]:
        """Return the :data:`HOLDOUT_COLUMNS` of a prepost estimate (see ``holdout_max_blocks``).

        Empty when ``holdout_max_blocks`` is 0; no window and a zero sigma for a toggle estimate.
        """
        if self.holdout_max_blocks <= 0:
            return {}
        if is_toggle(mi.upgrade_timing):
            return {"sigma_holdout": 0.0, "n_holdout_blocks": 0.0}
        timebase = self.timebase if self.timebase is not None else _infer_timebase(index)
        hours_per_record = timebase / pd.Timedelta(hours=1)
        before_pos = np.flatnonzero(baseline_sel)
        before_times = index[before_pos]
        after_times = index[upgraded_sel]
        windows = holdout_windows(
            before_times,
            after_length=after_times.max() - after_times.min() + timebase,
            max_blocks=self.holdout_max_blocks,
        )
        readings, sigmas = [], []
        for start, end in windows:
            held = np.asarray((before_times > start) & (before_times <= end))
            if min(np.count_nonzero(held), np.count_nonzero(~held)) * hours_per_record < _MIN_PERIOD_HOURS:
                continue
            train, predict = before_pos[~held], before_pos[held]
            pred = self._fit_direction(features, y, train=train, predict=predict)
            readings.append(_ratio(y[predict], pred))
            sigmas.append(
                bootstrap_energy_ratio(
                    times=index[predict],
                    actual=y[predict],
                    counterfactual=pred,
                    period_start=index[predict].min(),
                    period_end=index[predict].max(),
                    timebase=timebase,
                    block_hours=float(self.block_hours),
                    n_resamples=self.n_resamples,
                    seed=self.bootstrap_seed,
                ).sigma
            )
        logger.info(
            "%s %s: %d held-out windows of the before period read %s",
            self.name,
            mi.test_wtg,
            len(readings),
            ", ".join(f"{100 * r:+.2f}%" for r in readings),
        )
        return {
            "sigma_holdout": holdout_excess(np.array(readings), sampling_sigmas=np.array(sigmas)),
            "n_holdout_blocks": float(len(readings)),
        }

    def _reverse_direction(
        self,
        mi: MethodInput,
        *,
        features: pd.DataFrame,
        y: np.ndarray,
        baseline_sel: np.ndarray,
        upgraded_sel: np.ndarray,
        forward: float,
        index: pd.DatetimeIndex,
        timebase: pd.Timedelta,
    ) -> dict[str, float]:
        """Return the :data:`REVERSAL_COLUMNS` of a prepost estimate: the reverse fit's ratio and its sigma.

        NaN for a toggle estimate, a non-finite forward ratio, or when neither ``headline_estimator``
        nor ``record_reversal`` asks for the reverse direction.
        """
        out = dict.fromkeys(REVERSAL_COLUMNS, float("nan"))
        wanted = self.headline_estimator == "reversal" or self.record_reversal
        if not wanted or is_toggle(mi.upgrade_timing) or not np.isfinite(forward):
            return out
        pred_baseline = self._fit_direction(features, y, train=upgraded_sel, predict=baseline_sel)
        return {
            "uplift_forward": forward,
            "uplift_reverse": _ratio(y[baseline_sel], pred_baseline),
            "sigma_reverse": self._reverse_sigma(
                index[baseline_sel], actual=y[baseline_sel], counterfactual=pred_baseline, timebase=timebase
            ),
        }

    def _reverse_sigma(
        self,
        times: pd.DatetimeIndex,
        *,
        actual: np.ndarray,
        counterfactual: np.ndarray,
        timebase: pd.Timedelta,
    ) -> float:
        """Bootstrap the reverse ratio's sigma at ``block_hours`` over the before period, the model fixed."""
        if times.empty:
            return float("nan")
        return bootstrap_energy_ratio(
            times=times,
            actual=np.asarray(actual, dtype=float),
            counterfactual=np.asarray(counterfactual, dtype=float),
            period_start=times.min(),
            period_end=times.max(),
            timebase=timebase,
            block_hours=float(self.block_hours),
            n_resamples=self.n_resamples,
            seed=self.bootstrap_seed,
        ).sigma

    def _uplift_shape(
        self,
        mi: MethodInput,
        *,
        scada: pd.DataFrame,
        features: pd.DataFrame,
        y: np.ndarray,
        baseline_sel: np.ndarray,
        upgraded_sel: np.ndarray,
        fit: dict[str, Any],
        headline: float,
        power_references: Sequence[str],
        index: pd.DatetimeIndex,
        timebase: pd.Timedelta,
        level_sigma: float,
        reference_deviations: np.ndarray | None,
        run_dir: Path | None,
    ) -> ShapeResult | None:
        """Return the uplift distributions of method step 15 and write their diagnostics.

        ``level_sigma`` is the headline's reference and holdout terms; ``reference_deviations`` the kept
        references' power-bin deviations the shape is gated against. ``None`` without an after-period record.
        """
        times = index[upgraded_sel]
        if times.empty:
            return None
        rated = self.baseline_rated_power_kw
        axes, scale = self._shape_axes(
            mi,
            scada=scada,
            features=features,
            y=y,
            baseline_sel=baseline_sel,
            upgraded_sel=upgraded_sel,
            power_references=power_references,
            index=index,
        )
        result = uplift_shape(
            times=times,
            actual=fit["y_upgraded"],
            counterfactual=fit["pred_upgraded"],
            power_edges=condition_bins(PREDICTED_POWER, rated_power_kw=rated),
            axes={k: v for k, v in axes.items() if k in self.conditions},
            subsets={k: v for k, v in axes.items() if k in self.subset_conditions},
            subset_power_edges={n: [rated * (-0.05 + 1.1 * i / n) for i in range(n + 1)] for n in _SUBSET_POWER_BINS},
            headline=headline,
            reference_deviations=reference_deviations,
            level_sigma=level_sigma,
            min_hours=self.shape_min_hours,
            hours_per_record=timebase / pd.Timedelta(hours=1),
            timebase=timebase,
            block_hours=self.block_hours,
            n_resamples=self.n_resamples,
            seed=self.bootstrap_seed,
        )
        result = ShapeResult(
            frame=result.frame,
            subsets=result.subsets.assign(sigma_uplift=np.hypot(result.subsets["sigma_sampling"], level_sigma)),
            labels=result.labels.rename(columns=label_column),
        )
        flagged = result.subsets.loc[result.subsets["flagged"].astype(bool)]
        for (axis, subset), _ in flagged.groupby(["axis", "subset"]):
            logger.warning(
                "%s %s: the power shape of %s %s differs from the whole after period's",
                self.name,
                mi.test_wtg,
                axis,
                subset,
            )
        drift = self._reference_wind_speed_drift(mi, scada=scada, power_references=power_references, index=index)
        if run_dir is not None:
            self._write_shape(mi, run_dir=run_dir, shape=result, drift=drift, scale=scale)
        return result

    def _shape_deviations(self, fit: dict[str, Any], *, timebase: pd.Timedelta) -> dict[str, float]:
        """Return the after period's :func:`power_deviations` as ``power_shape_<i>`` columns."""
        deviations = power_deviations(
            fit["y_upgraded"],
            fit["pred_upgraded"],
            power_edges=condition_bins(PREDICTED_POWER, rated_power_kw=self.baseline_rated_power_kw),
            min_records=self.shape_min_hours / (timebase / pd.Timedelta(hours=1)),
        )
        return {f"{SHAPE_PREFIX}{i}": float(d) for i, d in enumerate(deviations)}

    def _shape_axes(
        self,
        mi: MethodInput,
        *,
        scada: pd.DataFrame,
        features: pd.DataFrame,
        y: np.ndarray,
        baseline_sel: np.ndarray,
        upgraded_sel: np.ndarray,
        power_references: Sequence[str],
        index: pd.DatetimeIndex,
    ) -> tuple[dict[str, tuple[np.ndarray, list[float]]], dict[str, float]]:
        """Return each axis's after-period values and edges, and the free-stream scale's details.

        An axis whose input is missing is left out with a warning.
        """
        wanted = (set(self.conditions) | set(self.subset_conditions)) - {PREDICTED_POWER}
        values: dict[str, np.ndarray] = {}
        scale: dict[str, float] = {}
        if "freestream_ws" in wanted:
            freestream, scale = self._freestream_wind_speed(
                mi, scada=scada, y=y, baseline_sel=baseline_sel, power_references=power_references, index=index
            )
            if freestream is not None:
                values["freestream_ws"] = freestream
        test_rows = scada[scada[mi.turbine_col] == mi.test_wtg]
        test_rows = test_rows[~test_rows.index.duplicated()].reindex(index)
        if "direction" in wanted and DIRECTION_COL in test_rows.columns:
            values["direction"] = test_rows[DIRECTION_COL].to_numpy(dtype=float)
        if "waked" in wanted and UPWIND_COL in test_rows.columns:
            upwind = test_rows[UPWIND_COL].astype(object)
            unknown = upwind.isna() | (upwind == UNKNOWN)
            values["waked"] = np.where(unknown, np.nan, (upwind != NONE).astype(float))
        if "solar_elevation" in wanted and SOLAR_ELEVATION in features.columns:
            values["solar_elevation"] = features[SOLAR_ELEVATION].to_numpy(dtype=float)
        for missing in sorted(wanted - set(values)):
            logger.warning("%s %s: no %s axis, its input is missing", self.name, mi.test_wtg, missing)
        axes = {name: (v[upgraded_sel], CONDITION_BINS[name]) for name, v in values.items()}
        return axes, scale

    def _freestream_wind_speed(
        self,
        mi: MethodInput,
        *,
        scada: pd.DataFrame,
        y: np.ndarray,
        baseline_sel: np.ndarray,
        power_references: Sequence[str],
        index: pd.DatetimeIndex,
    ) -> tuple[np.ndarray | None, dict[str, float]]:
        """Return the power references' mean wind speed scaled to free stream, and the scale's details.

        The factor is fitted on the before period (:func:`freestream_scale`). ``None`` without the test
        turbine's rotor diameter or a before-period record in the power band.
        """
        diameter = (self.rotor_diameters_m or {}).get(mi.test_wtg)
        if diameter is None or not power_references:
            return None, {}
        ws = reference_mean_wind_speed(
            scada, references=power_references, turbine_col=mi.turbine_col, wind_speed_col=self.columns.wind_speed
        ).reindex(index)
        ws_arr = ws.to_numpy(dtype=float)
        factor = freestream_scale(
            y[baseline_sel],
            ws_arr[baseline_sel],
            rated_power_kw=self.baseline_rated_power_kw,
            rotor_diameter_m=float(diameter),
            target_cp=self.target_cp,
            air_density=self.air_density_kg_m3,
        )
        details = {
            "scale_factor": factor,
            "target_cp": self.target_cp,
            "air_density_kg_m3": self.air_density_kg_m3,
            "rotor_diameter_m": float(diameter),
        }
        if not np.isfinite(factor):
            return None, details
        logger.info(
            "%s %s: free-stream wind speed = %.3f x the power references' mean (target Cp %.2f)",
            self.name,
            mi.test_wtg,
            factor,
            self.target_cp,
        )
        return ws_arr * factor, details

    def _reference_wind_speed_drift(
        self, mi: MethodInput, *, scada: pd.DataFrame, power_references: Sequence[str], index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Return the power references' monthly wind speed drift against each other, warning on a flag."""
        if len(power_references) < 2 or self.columns.wind_speed not in scada.columns:  # noqa: PLR2004 - needs others
            return reference_wind_speed_drift(pd.DataFrame(), threshold=_ANEMOMETER_DRIFT_THRESHOLD)
        wide = pd.DataFrame(
            {
                r: reference_mean_wind_speed(
                    scada, references=[r], turbine_col=mi.turbine_col, wind_speed_col=self.columns.wind_speed
                ).reindex(index)
                for r in power_references
            }
        )
        drift = reference_wind_speed_drift(wide, threshold=_ANEMOMETER_DRIFT_THRESHOLD)
        for reference, rows in drift[drift["flagged"]].groupby("reference"):
            logger.warning(
                "%s %s: reference %s's anemometer moves %.1f %% against the others' over the months (not corrected)",
                self.name,
                mi.test_wtg,
                reference,
                100 * float(rows["spread"].iloc[0]),
            )
        return drift

    def _reported_shape(self, shape: ShapeResult) -> pd.DataFrame:
        """Return the bins of the requested ``conditions`` as the method reports them."""
        frame = shape.frame[shape.frame["condition"].isin(self.conditions)]
        return frame[["condition", "condition_bin", "p50_uplift", "sigma_uplift", "kind", "hours", "thin"]].reset_index(
            drop=True
        )

    def _write_shape(
        self,
        mi: MethodInput,
        *,
        run_dir: Path,
        shape: ShapeResult,
        drift: pd.DataFrame,
        scale: dict[str, float],
    ) -> None:
        """Write the shape, the subsets, the reference wind speed drift and the free-stream scale, and plot them."""
        out = run_dir / "conditional"
        out.mkdir(parents=True, exist_ok=True)
        run_name, ts = self._csv_naming(run_dir)
        shape.frame.to_csv(out / diag.csv_name("uplift_shape", run_name=run_name, ts=ts), index=False)
        shape.subsets.to_csv(out / diag.csv_name("uplift_subsets", run_name=run_name, ts=ts), index=False)
        drift.to_csv(out / diag.csv_name("reference_wind_speed_drift", run_name=run_name, ts=ts), index=False)
        pd.DataFrame([scale]).to_csv(out / diag.csv_name("freestream_scale", run_name=run_name, ts=ts), index=False)
        if self.save_plots:
            diag.plot_uplift_shape(
                run_dir / "plots" / stages.UPLIFT_DISTRIBUTIONS, shape=shape, drift=drift, test_wtg=mi.test_wtg
            )

    def _fit_direction(
        self, features: pd.DataFrame, y: np.ndarray, *, train: np.ndarray, predict: np.ndarray
    ) -> np.ndarray:
        """Fit the outcome model(s) on ``train`` rows and return clipped predictions for ``predict`` rows.

        No calibration and no time-decay weights here: the two-direction combine cancels the common
        per-bin shrinkage by construction, and weighting the matched fits only churns sparse
        extreme-condition bins (re-confirmed post-floor) — the corrections are spent on the
        overall headline instead.
        """
        models = self._fit_models(features.iloc[train], y[train])
        return _clip_predictions(
            self._predict_mean(models, features.iloc[predict]),
            y_train=y[train],
            rated_power_kw=self.baseline_rated_power_kw,
        )

    def _add_site_features(
        self, features: pd.DataFrame, *, index: pd.DatetimeIndex, timebase: pd.Timedelta
    ) -> pd.DataFrame:
        """Append the sun's elevation at the site, when the site's location is known."""
        if self.site_location is None:
            return features
        latitude, longitude = self.site_location
        solar = solar_elevation_frame(index, timebase=timebase, latitude=latitude, longitude=longitude)
        return pd.concat([features, solar], axis=1)

    def _aligned_era5(self, *, index: pd.DatetimeIndex, timebase: pd.Timedelta) -> pd.DataFrame | None:
        """Return ERA5 aligned to the SCADA periods, or ``None`` when none was supplied."""
        if self.era5_hourly_df is None:
            return None
        return interpolate_era5(self.era5_hourly_df, index=index, timebase=timebase)

    def _effective_half_life(self, *, campaign_start: pd.Timestamp, campaign_end: pd.Timestamp) -> float | None:
        """Return the time-decay half-life (days) for this campaign, or ``None`` when decay is off.

        ``adaptive_time_decay`` (the default) sets it to ``_TIME_DECAY_CAMPAIGN_MULTIPLE`` times the
        campaign's own duration (Issue 15); otherwise the fixed ``time_decay_half_life_days``
        expert override is used verbatim.
        """
        if not self.adaptive_time_decay:
            return self.time_decay_half_life_days
        duration_days = max((campaign_end - campaign_start).total_seconds() / 86400.0, _MIN_TIME_DECAY_DURATION_DAYS)
        return _TIME_DECAY_CAMPAIGN_MULTIPLE * duration_days

    def _time_decay_weights(
        self, index: pd.DatetimeIndex, *, campaign_start: pd.Timestamp, campaign_end: pd.Timestamp
    ) -> np.ndarray | None:
        """Exponential campaign-proximity sample weights over the analysis index (``None`` = knob off).

        ``0.5 ** (days_outside_campaign / half_life)`` where the distance is to the campaign
        *interval* ``[campaign_start, campaign_end]``: every row inside the campaign (for toggle,
        the interleaved on **and** off rows) weighs exactly 1, and rows outside decay with their
        distance to it — so distant history still informs the fit without dominating it. With
        today's flows only pre-campaign rows exist outside the interval, but the definition is
        two-sided on purpose. The half-life is the ``adaptive_time_decay`` campaign-proportional
        value by default (:meth:`_effective_half_life`).
        """
        half_life = self._effective_half_life(campaign_start=campaign_start, campaign_end=campaign_end)
        if half_life is None:
            return None
        seconds_outside = np.maximum((campaign_start - index).total_seconds(), (index - campaign_end).total_seconds())
        days_outside = np.maximum(seconds_outside / 86400.0, 0.0)
        return np.asarray(0.5 ** (days_outside / half_life), dtype=float)

    def _select_records(
        self,
        scada: pd.DataFrame,
        *,
        mi: MethodInput,
        index: pd.DatetimeIndex,
        references: Sequence[str],
        timebase: pd.Timedelta,
        before: np.ndarray,
        after: np.ndarray,
    ) -> pd.Series:
        """Return why each record of ``index`` is lost with ``references`` as the power references, or usable.

        The per-turbine rules of method step 11 i and iii come first, then the waking configuration
        rule of step 11 ii over the usable records, then for a toggle campaign the pairing filter of
        step 11 iv. ``scada`` is the frame the campaign selected; ``before`` and ``after`` are the
        two periods over ``index``.
        """
        reasons = self._loss_reasons(mi, index=index, references=references, timebase=timebase)
        usable = np.asarray(reasons == USABLE)
        if WAKING_SCENARIO_COL in scada.columns and references:
            scenarios = waking_scenario_features(
                scada, test_wtg=mi.test_wtg, references=references, turbine_col=mi.turbine_col
            ).reindex(index)
            rare = rare_configurations(
                scenarios,
                before=usable & before,
                after=usable & after,
                hours_per_record=timebase / pd.Timedelta(hours=1),
            )
            reasons[usable & rare] = RARE_CONFIGURATION
            usable &= ~rare
        if is_toggle(mi.upgrade_timing):
            gap = self.pairing_max_gap if self.pairing_max_gap is not None else toggle_cycle(mi.upgrade_timing, index)
            paired = pair_within(index, baseline=usable & before, upgraded=usable & after, max_gap=gap)
            unpaired = usable & (before | after) & ~(paired.baseline | paired.upgraded)
            reasons[unpaired] = UNPAIRED
        return reasons

    def _loss_reasons(
        self, mi: MethodInput, *, index: pd.DatetimeIndex, references: Sequence[str], timebase: pd.Timedelta
    ) -> pd.Series:
        """Return why each record of ``index`` is lost to the per-turbine rules, or usable.

        Generic states are labelled first when the frame has no ``valid_uplift`` column.
        """
        turbines = [mi.test_wtg, *references]
        scada = mi.scada_df[mi.scada_df[mi.turbine_col].isin(turbines)]
        if VALID_UPLIFT_COL not in scada.columns:
            scada = label_operating_states(
                scada,
                columns=self.columns,
                config=OperatingStateConfig(),
                timebase=timebase,
                rated_power_kw=self.baseline_rated_power_kw,
            )
        return loss_reasons(
            scada,
            subject=mi.test_wtg,
            references=references,
            changed=mi.context.changed_turbines,
            turbine_col=mi.turbine_col,
            usable=mi.context.valid_over(index),
            index=index,
        )

    def _valid_uplift(self, rows: pd.DataFrame, *, timebase: pd.Timedelta) -> pd.Series:
        """Return the rows' ``valid_uplift`` column, labelling generic states first when the frame has none."""
        if VALID_UPLIFT_COL not in rows.columns:
            rows = label_operating_states(
                rows,
                columns=self.columns,
                config=OperatingStateConfig(),
                timebase=timebase,
                rated_power_kw=self.baseline_rated_power_kw,
            )
        return rows[VALID_UPLIFT_COL].astype(bool)

    def _fit_predict(
        self,
        features: pd.DataFrame,
        *,
        y: np.ndarray,
        baseline_sel: np.ndarray,
        upgraded_sel: np.ndarray,
        weights: np.ndarray | None = None,
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
        w_base = weights[baseline_sel] if weights is not None else None

        y_valid, pred_valid, valid_local = self._holdout_fit(x_base, y_base, w_base=w_base)
        baseline_valid_pos = np.flatnonzero(baseline_sel)[valid_local]

        models = self._fit_models(x_base, y_base, weights=w_base)
        pred_up = _clip_predictions(
            self._predict_mean(models, x_up), y_train=y_base, rated_power_kw=self.baseline_rated_power_kw
        )
        return {
            "model": models[0],
            "pred_upgraded": pred_up,
            "y_upgraded": y_up,
            "y_baseline_valid": y_valid,
            "pred_baseline_valid": pred_valid,
            "baseline_valid_pos": baseline_valid_pos,  # positions over ``index`` of the held-out rows
        }

    def _write_row_dump(
        self,
        scada: pd.DataFrame,
        *,
        mi: MethodInput,
        index: pd.DatetimeIndex,
        timebase: pd.Timedelta,
        references: Sequence[str],
        power_free: Sequence[str],
        features: pd.DataFrame,
        y: np.ndarray,
        selected: np.ndarray,
        reasons: pd.Series,
        baseline_sel: np.ndarray,
        upgraded_sel: np.ndarray,
        fit: dict[str, Any],
        uplift: float,
    ) -> None:
        """Write the per-row frame and its sidecar under ``row_dump_dir / test_wtg``.

        ``uplift`` is the forward counterfactual ratio, the one the dumped sums reproduce.
        """
        assert self.row_dump_dir is not None  # noqa: S101
        out_dir = Path(self.row_dump_dir) / mi.test_wtg
        out_dir.mkdir(parents=True, exist_ok=True)
        counterfactual = np.full(len(index), np.nan)
        counterfactual[upgraded_sel] = fit["pred_upgraded"]
        rows = pd.DataFrame(
            {
                "timestamp": index,
                "actual_kw": y,
                "counterfactual_kw": counterfactual,
                "selected": np.asarray(selected, dtype=bool),
                "loss_reason": reasons.astype(str).to_numpy(),
                "baseline": np.asarray(baseline_sel, dtype=bool),
                "upgraded": np.asarray(upgraded_sel, dtype=bool),
                "reference_mean_ws": reference_mean_wind_speed(
                    scada, references=references, turbine_col=mi.turbine_col, wind_speed_col=self.columns.wind_speed
                ).to_numpy(dtype=float),
            }
        )
        rows = pd.concat([rows, features.reset_index(drop=True)], axis=1)
        rows.to_parquet(out_dir / "rows.parquet", engine="pyarrow", index=False)
        sidecar = {
            "test_wtg": mi.test_wtg,
            "references": list(references),
            "power_free": list(power_free),
            "uplift": uplift,
            "sum_actual_kw": float(fit["y_upgraded"].sum()),
            "sum_counterfactual_kw": float(fit["pred_upgraded"].sum()),
            "n_baseline_rows": int(np.count_nonzero(baseline_sel)),
            "n_upgraded_rows": int(np.count_nonzero(upgraded_sel)),
            "timebase_s": timebase.total_seconds(),
            "active_power_col": self.columns.active_power,
            "northed_direction_col": self.columns.northed("nacelle_position") if self.direction_feature else None,
            "baseline_rated_power_kw": self.baseline_rated_power_kw,
            "coords": {t: list(c) for t, c in (mi.context.coords or {}).items()},
        }
        (out_dir / "rows.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    def _candidate_references(self, scada: pd.DataFrame, *, mi: MethodInput) -> list[str]:
        """Return the campaign's candidate references that ``scada`` carries data for, sorted.

        The pool starts at what the campaign offers: a turbine present in the frame that the
        campaign does not offer as a reference is not one, and one the campaign offers that the
        frame has no rows for is dropped with a warning rather than estimated as if it were there.
        """
        present = sorted({str(t) for t in scada[mi.turbine_col].unique()})
        references = mi.context.references_among(present)
        absent = sorted(set(mi.context.candidate_references) - set(references))
        if absent:
            logger.warning(
                "%s %s: the campaign offers %d candidate reference(s) but scada_df carries no rows for %s, so "
                "the estimate runs on a pool of %d. Reference count drives accuracy.",
                self.name,
                mi.test_wtg,
                len(mi.context.candidate_references),
                absent,
                len(references),
            )
        if not references:
            msg = (
                f"no candidate references available for test_wtg {mi.test_wtg!r}: the campaign offers "
                f"{sorted(mi.context.candidate_references)} and scada_df carries {present}. The power model "
                f"needs at least one reference turbine."
            )
            raise ValueError(msg)
        return references

    def reference_features(self, mi: MethodInput, *, power_free: Sequence[str] = ()) -> pd.DataFrame:
        """Build this method's reference feature matrix for ``mi``, with ``power_free`` references left out.

        The same matrix :meth:`estimate` fits on, before the solar elevation is added, exposed so a caller can
        inspect what the screen changed.
        """
        scada = mi.context.select(mi.scada_df)
        return self._reference_features(
            scada, mi=mi, references=self._candidate_references(scada, mi=mi), power_free=power_free
        )

    def _reference_features(
        self, scada: pd.DataFrame, *, mi: MethodInput, references: Sequence[str], power_free: Sequence[str]
    ) -> pd.DataFrame:
        """Return the power references' features and the waking scenarios of the test turbine and power references.

        The power references are ``references`` less ``power_free``. Without a waking scenario column
        in ``scada`` the scenarios are left out with a warning.
        """
        power_references = [r for r in references if r not in set(power_free)]
        features = build_reference_features(
            scada,
            test_wtg=mi.test_wtg,
            references=power_references,
            turbine_col=mi.turbine_col,
            active_power_col=self.columns.active_power,
            extra_cols=self.reference_stat_cols,
            direction_col=self.columns.northed("nacelle_position") if self.direction_feature else None,
        )
        if WAKING_SCENARIO_COL not in scada.columns:
            logger.warning(
                "%s %s: no waking scenario features -- scada_df has no %r column (method step 5)",
                self.name,
                mi.test_wtg,
                WAKING_SCENARIO_COL,
            )
            return features
        scenarios = waking_scenario_features(
            scada, test_wtg=mi.test_wtg, references=power_references, turbine_col=mi.turbine_col
        )
        return features.join(scenarios.reindex(features.index), how="left")

    def reference_uplifts(
        self, mi: MethodInput, *, power_free: Sequence[str] = (), screen: ScreenResult | None = None
    ) -> pd.DataFrame:
        """Return what this campaign's own analysis says each candidate reference did.

        Every reference is estimated under the campaign's own timing against the other power-carrying
        references, so the reference headline and the test headline are the same kind of number, on
        the same pool. A healthy campaign reads near 0% once these are combined by energy.

        A reference the screen ruled out, or could not judge, is estimated and reported too, flagged
        so it stays visible without dragging the headline; it is kept out of every reference pool here.

        When nobody lost their power and the screen already ran this exact contrast, its final pass is
        reused rather than refitting the whole pool.
        """
        context = mi.context
        pool = self._candidate_references(context.select(mi.scada_df), mi=mi)
        ruled_out = set(screen.screened if screen is not None else ())
        demoted = set(power_free)
        unjudged = (demoted | set(screen.unjudged if screen is not None else ())) - ruled_out
        surviving = [r for r in pool if r not in demoted]
        clone = self._reference_clone()
        if self.row_dump_dir is not None:
            clone = dataclasses.replace(clone, row_dump_dir=Path(self.row_dump_dir) / mi.test_wtg / "references")
        reusable = (
            {}
            if self.reference_plots_dir is not None
            else self._reusable_screen_estimates(mi, screen=screen, ruled_out=demoted, pool=surviving)
        )
        # A reference the screen moved out of a refilled pool is still read and reported.
        targets = [*pool, *(r for r in (screen.power_free if screen is not None else ()) if r not in pool)]
        rows: list[dict[str, object]] = []
        for target in targets:
            refs = [r for r in surviving if r != target]
            if not refs:
                continue
            sub_input = _reference_input(mi, target=target, references=refs)
            energy, n_records = self._upgraded_energy(sub_input, turbine=target)
            reading = (
                reusable[target]
                if target in reusable
                else self._reference_uplift(self._plotting(clone, target=target, mi=mi), sub_input)
            )
            rows.append(
                {
                    "turbine": target,
                    "uplift": reading["uplift"],
                    "actual_energy": energy,
                    "n_records": n_records,
                    "screened": target in ruled_out,
                    "unjudged": target in unjudged,
                    **{k: v for k, v in reading.items() if k != "uplift"},
                }
            )
        return pd.DataFrame(rows) if rows else empty_reference_uplifts()

    def _plotting(self, clone: PowerModelMethod, *, target: str, mi: MethodInput) -> PowerModelMethod:
        """Return ``clone`` writing its plots and diagnostics for ``target``, when reference plots are on."""
        if self.reference_plots_dir is None:
            return clone
        return dataclasses.replace(
            clone,
            conditions=self.conditions,
            save_plots=True,
            write_diagnostics=True,
            out_dir=self.reference_plots_dir / reference_measurement_dirname(target, test_wtg=mi.test_wtg),
            run_subdir=False,
            repeat_shared_plots=False,
        )

    def _reference_uplift(self, clone: PowerModelMethod, sub_input: MethodInput) -> dict[str, float]:
        """One reference's own uplift and sigmas, or NaN when its data cannot support an estimate.

        This pass runs after the headline is already computed, so a reference too degenerate to
        estimate is reported as unknown rather than taking a good campaign result down with it.
        """
        try:
            return _reading(clone.estimate(sub_input))
        except ValueError as e:
            logger.warning("%s: no reference uplift for %s (%s)", self.name, sub_input.test_wtg, e)
            return {"uplift": float("nan"), "sigma": float("nan")}

    def _reusable_screen_estimates(
        self, mi: MethodInput, *, screen: ScreenResult | None, ruled_out: set[str], pool: Sequence[str] | None = None
    ) -> dict[str, dict[str, float]]:
        """Screen estimates that already answer the reference-uplift question, so no refit is needed.

        Only when nobody lost their power after the screen ran, the screen covered the same pool the
        reference report uses, and it ran the campaign's own contrast -- the case for a healthy
        prepost campaign. Any other combination refits, because the pools or the contrast differ.
        """
        if screen is None or ruled_out or screen.passes.empty:
            return {}
        if is_toggle(mi.context.timing) or self.screening_timing(mi) != mi.context.timing:
            return {}
        reported = self._candidate_references(mi.context.select(mi.scada_df), mi=mi) if pool is None else pool
        final = screen.passes[screen.passes["pass"] == screen.passes["pass"].max()]
        if set(final["turbine"].astype(str)) != set(reported):
            return {}
        reading_columns = [c for c in final.columns if is_reading_column(c)]
        return {
            str(row["turbine"]): {"uplift": float(row["estimate"]), **{c: float(row[c]) for c in reading_columns}}
            for _, row in final.iterrows()
        }

    def _upgraded_energy(self, mi: MethodInput, *, turbine: str) -> tuple[float, int]:
        """Energy a turbine produced over the campaign's upgraded rows, and how many records it covers."""
        scada = mi.context.select(mi.scada_df)
        rows = scada[scada[mi.turbine_col] == turbine]
        index = pd.DatetimeIndex(pd.unique(scada.index)).sort_values()
        upgraded = pd.Series(resolve_toggle(mi.upgrade_timing, index).upgraded, index=index)
        power = rows[self.columns.active_power]
        selected = power[upgraded.reindex(pd.DatetimeIndex(rows.index)).fillna(value=False).to_numpy()]
        finite = selected[np.isfinite(selected)]
        return float(finite.sum()), int(finite.size)

    def _upgraded_days(self, mi: MethodInput, *, turbines: Sequence[str]) -> dict[str, float]:
        """Upgraded days of *fittable* data each of ``turbines`` has.

        Counted from records rather than calendar span, and per turbine rather than over the frame:
        the frame can span a year while one candidate has a fortnight of campaign data, and that
        candidate is then estimated in exactly the short-data regime the gate exists to avoid.

        Rows are counted through the turbine's own valid-uplift mask, so a
        turbine with a year of readings but a fortnight of valid ones is judged on the fortnight.
        """
        scada = mi.context.select(mi.scada_df)
        index = pd.DatetimeIndex(pd.unique(scada.index)).sort_values()
        upgraded = pd.Series(resolve_toggle(mi.upgrade_timing, index).upgraded, index=index)
        timebase = self.timebase if self.timebase is not None else _infer_timebase(scada.index)
        per_day = timebase.total_seconds() / 86400.0
        covered: dict[str, float] = {}
        for wtg in turbines:
            # This turbine's own rows: the long frame repeats each timestamp per turbine.
            rows = scada[scada[mi.turbine_col] == wtg].sort_index()
            rows = rows[~rows.index.duplicated()]
            if rows.empty:
                covered[wtg] = 0.0
                continue
            in_campaign = upgraded.reindex(pd.DatetimeIndex(rows.index)).fillna(value=False).to_numpy()
            normal = self._valid_uplift(rows, timebase=timebase).to_numpy()
            finite = np.isfinite(rows[self.columns.active_power].to_numpy(dtype=float))
            covered[wtg] = float(np.count_nonzero(in_campaign & normal & finite)) * per_day
        return covered

    def _screening_clone(self) -> PowerModelMethod:
        """Return this method as it estimates one candidate reference during screening.

        It never dumps rows: screening fits are not what the level probe analyses.
        """
        return dataclasses.replace(self._pass_clone("screen"), row_dump_dir=None)

    def _reference_clone(self) -> PowerModelMethod:
        """Return this method as it estimates one reference for the reference-uplift report."""
        return self._pass_clone("reference")

    def _pass_clone(self, suffix: str) -> PowerModelMethod:
        """Return a clone for estimating one reference: forward, no holdout or conditional step, no plots, no pass.

        Both passes must be off, not just the screen. A clone that still reports reference uplifts
        runs the whole reference pass itself, and its clones run it again, so the work is
        combinatorial rather than linear.
        """
        return dataclasses.replace(
            self,
            reference_screen=False,
            report_reference_uplifts=False,
            headline_estimator="forward",
            holdout_max_blocks=0,
            conditions=(),
            save_plots=False,
            write_diagnostics=False,
            out_dir=None,
            name=f"{self.name}_{suffix}",
        )

    def screening_timing(self, mi: MethodInput) -> pd.Timestamp:
        """Return the changeover a reference is screened across -- always a prepost contrast.

        A reference is not the thing being toggled, so asking what its power did between a
        campaign's on-blocks and its off-blocks answers nothing: a change of its own sits in both
        alike. The question that matters is whether it changed across a boundary.

        Prepost uses the campaign's own changeover. Toggle splits the test period in half by data
        volume: within a toggle campaign a reference change predating the test is common-mode
        across every block and largely cancels, so what corrupts the result is a change *during*
        the test -- and the test period is the data worth protecting.
        """
        timing = mi.context.timing
        index = pd.DatetimeIndex(pd.unique(mi.context.select(mi.scada_df).index)).sort_values()
        if not is_toggle(timing):
            return pd.Timestamp(timing)  # type: ignore[arg-type]
        campaign = index[index >= toggle_upgrade_start(timing, index)]
        return pd.Timestamp(campaign[len(campaign) // 2])

    def _screenable_pool(self, mi: MethodInput) -> tuple[list[str], list[str]]:
        """Return the candidate references with enough upgraded data to be screened, and those without.

        A candidate holding less than ``screen_min_campaign_days`` is held out of the screen and
        contributes no features; the rest of the pool is still screened. An empty
        pool means no candidate holds enough, and then nobody is held out either.
        """
        days = self._upgraded_days(mi, turbines=self._candidate_references(mi.context.select(mi.scada_df), mi=mi))
        pool = [wtg for wtg, covered in days.items() if covered >= self.screen_min_campaign_days]
        unjudged = sorted(set(days) - set(pool))
        if not pool:
            logger.info(
                "%s %s: reference screen skipped, the best-covered candidate reference holds %.1f days of upgraded "
                "data, under the %.1f needed for a screening estimate to separate a bad reference from a good one",
                self.name,
                mi.test_wtg,
                max(days.values(), default=0.0),
                self.screen_min_campaign_days,
            )
        elif unjudged:
            logger.info(
                "%s %s: %s held out of the reference screen, each holding under %.1f days of upgraded data; they "
                "contribute no features, and the remaining %d candidate(s) are screened",
                self.name,
                mi.test_wtg,
                unjudged,
                self.screen_min_campaign_days,
                len(pool),
            )
        return pool, ([] if not pool else unjudged)

    def screen_references(self, mi: MethodInput) -> ScreenResult:
        """Estimate each candidate reference against the others and rule out clear outliers.

        Each screening estimate uses this same method with the conditional step and plots off and
        the screen itself disabled, so it is the method judging its own references, across the
        prepost contrast :meth:`screening_timing` picks.
        """
        context = mi.context
        if is_toggle(context.timing):
            # Toggle is not vulnerable to this failure mode: a reference change before the test is
            # common-mode across every on and off block. The screen cannot see one there either --
            # the on/off contrast is blind to it, and splitting the test in half leaves ~3 seasonal
            # months a side, whose noise swamps the signal. Screening out a good reference costs
            # more than leaving a mild bad one in, so it does not run.
            logger.info("%s %s: reference screen skipped, campaign is toggle", self.name, mi.test_wtg)
            return ScreenResult(screened=(), passes=_empty_screen_passes(), screenable=False)
        pool, unjudged = self._screenable_pool(mi)
        if not pool:
            return ScreenResult(screened=(), passes=_empty_screen_passes(), screenable=False)
        timing = self.screening_timing(mi)
        cache_key = (
            tuple(pool),
            str(timing),
            str(mi.scada_df.index.min()),
            str(mi.scada_df.index.max()),
            tuple(f"{turbine} {times}" for turbine, times in sorted(context.changed_turbines.items())),
            str(context.works),
        )
        if self.screen_cache is not None and cache_key in self.screen_cache:
            # Re-stamped: the pool decides the screening estimates, but which candidates were held
            # out of it is this test turbine's own, and two of them can share a pool.
            cached = dataclasses.replace(self.screen_cache[cache_key], unjudged=tuple(unjudged))
            logger.info(
                "%s %s: reference screen already run for this campaign; %s",
                self.name,
                mi.test_wtg,
                f"ruled out {list(cached.screened)}" if cached.screened else "it ruled out nobody",
            )
            return cached
        clone = self._screening_clone()
        shared = " (once for the campaign)" if self.screen_cache is not None else ""
        logger.info(
            "%s %s: screening %d candidate reference(s) across %s%s, one estimate each per pass",
            self.name,
            mi.test_wtg,
            len(pool),
            timing.date() if hasattr(timing, "date") else timing,
            shared,
        )

        # Why each candidate could not be estimated, so a screen that gives up can say what stopped
        # it rather than only that it gave up.
        causes: dict[str, str] = {}
        # Each candidate's sigmas from its latest estimate, to go with its uplift in the passes.
        readings: dict[str, dict[str, float]] = {}

        def estimate_one(target: str, refs: list[str]) -> float:
            sub_input = _reference_input(mi, target=target, references=refs, timing=timing)
            try:
                reading = _reading(clone.estimate(sub_input))
                readings[target] = reading
                return reading["uplift"]
            except ValueError as e:
                # NaN ranks worst, so the candidate is ruled out rather than aborting the estimate:
                # surviving a reference this bad is what the screen is for.
                logger.warning("%s: reference screen could not estimate %s (%s)", self.name, target, e)
                causes[target] = str(e)
                return float("nan")

        try:
            result = screen_references(pool, estimate_one=estimate_one, floor=self.screen_floor)
        except ValueError as e:
            if not causes:
                raise
            detail = "; ".join(f"{target}: {cause}" for target, cause in sorted(causes.items()))
            msg = f"{e} What stopped each of them: {detail}"
            raise ValueError(msg) from e
        result = dataclasses.replace(result, passes=_with_sigmas(result.passes, readings))
        if result.screened:
            logger.info(
                "%s %s: reference screen ruled out %s; they contribute a waking feature and nothing else",
                self.name,
                mi.test_wtg,
                list(result.screened),
            )
        if self.screen_cache is not None:
            self.screen_cache[cache_key] = result
        return dataclasses.replace(result, unjudged=tuple(unjudged))

    def refill_screened(self, mi: MethodInput, screen: ScreenResult) -> tuple[MethodInput, ScreenResult]:
        """Replace each reference the screen ruled out or could not judge with the nearest reserve, and screen again.

        A replaced reference becomes a wake contributor. Rounds continue until a screen rules out and
        holds out nobody new or the reserves run out, when the pool shrinks and a warning says so.

        :return: the input with the refilled context, and every round's result combined: every
            ruled-out reference in order, every unjudged one, every round's passes numbered on from the last
        """
        context = mi.context
        ejected: list[str] = []
        ruled_out: list[str] = []
        unjudged: set[str] = set()
        rounds = [screen.passes]
        result = screen
        while True:
            new = [r for r in result.power_free if r not in ejected and r in context.candidate_references]
            if not new:
                break
            ejected.extend(new)
            ruled_out.extend(r for r in new if r in result.screened)
            unjudged |= {r for r in new if r not in result.screened}
            promoted = list(context.reserve_references[: len(new)])
            context = dataclasses.replace(
                context,
                candidate_references=[*(r for r in context.candidate_references if r not in new), *promoted],
                reserve_references=list(context.reserve_references[len(new) :]),
                wake_contributors=sorted((set(context.wake_contributors) - set(promoted)) | set(new)),
            )
            mi = MethodInput(scada_df=mi.scada_df, test_wtg=mi.test_wtg, campaign_context=context)
            if len(promoted) < len(new):
                logger.warning(
                    "%s %s: the screen ruled out or could not judge %s and the reserves ran out; the pool is %d, so "
                    "the pool rule is broken",
                    self.name,
                    mi.test_wtg,
                    ejected,
                    len(context.candidate_references),
                )
                break
            logger.info("%s %s: %s replace %s in the reference pool", self.name, mi.test_wtg, promoted, new)
            result = self.screen_references(mi)
            offset = max((int(r["pass"].max()) for r in rounds if not r.empty), default=0)
            rounds.append(result.passes.assign(**{"pass": result.passes["pass"] + offset}))
        if not ejected:
            return mi, screen
        recorded = [r for r in rounds if not r.empty]
        passes = pd.concat(recorded, ignore_index=True) if recorded else screen.passes
        return mi, dataclasses.replace(
            result,
            screened=tuple(ruled_out),
            unjudged=tuple(sorted(unjudged | set(result.unjudged))),
            passes=passes,
            screenable=True,
        )

    def _validate_model_config(self) -> None:
        """Fail loudly on config combinations that would silently misbehave."""
        if self.headline_estimator not in HEADLINE_ESTIMATORS:
            msg = f"headline_estimator must be one of {HEADLINE_ESTIMATORS}, got {self.headline_estimator!r}"
            raise ValueError(msg)
        if self.headline_estimator == "reference_corrected" and not self.report_reference_uplifts:
            msg = (
                "headline_estimator='reference_corrected' corrects by the references' own readings, so it needs "
                "report_reference_uplifts=True. Turn the readings on, or pass headline_estimator='forward'."
            )
            raise ValueError(msg)
        if self.time_decay_half_life_days is not None and self.time_decay_half_life_days <= 0:
            msg = f"time_decay_half_life_days must be positive, got {self.time_decay_half_life_days}"
            raise ValueError(msg)
        if self.adaptive_time_decay and self.time_decay_half_life_days is not None:
            msg = (
                "adaptive_time_decay sets the half-life from the campaign duration; a fixed "
                "time_decay_half_life_days is only used with adaptive_time_decay=False. Set "
                "adaptive_time_decay=False to use the fixed override, or leave time_decay_half_life_days=None."
            )
            raise ValueError(msg)
        if not self.write_diagnostics and self.conditions:
            msg = (
                "write_diagnostics=False needs conditions=(): the uplift distributions write their CSVs "
                "into the run directory, so there is nowhere for them to go."
            )
            raise ValueError(msg)

    @staticmethod
    def _fit_kwargs(weights: np.ndarray | None) -> dict[str, Any]:
        """Return the fit kwargs for the time-decay ``weights`` (empty when unweighted)."""
        return {} if weights is None else {"sample_weight": weights}

    def _make_model(self, *, seed: int | None = None) -> Any:  # noqa: ANN401
        """One unfitted LightGBM outcome model with ``seed`` plumbed in.

        A caller-supplied ``random_state`` in ``model_params`` still wins over ``seed``.
        """
        s = self.seed if seed is None else seed
        return make_outcome_model(**{"random_state": s, **TUNED_MODEL_PARAMS, **self.model_params})

    def _fit_models(
        self, x_train: pd.DataFrame, y_train: np.ndarray, *, weights: np.ndarray | None = None
    ) -> list[Any]:
        """Fit the outcome model on one training set.

        ``weights`` (the time-decay sample weights, aligned to the training rows) are passed to the
        fit when given. Returns a single-element list so :meth:`_predict_mean` stays uniform.
        """
        model = self._make_model(seed=self.seed)
        model.fit(model_safe_features(x_train), y_train, **self._fit_kwargs(weights))
        return [model]

    @staticmethod
    def _predict_mean(models: list[Any], x: pd.DataFrame) -> np.ndarray:
        """Mean prediction over the fitted model(s) (unclipped)."""
        safe = model_safe_features(x)
        return np.mean([np.asarray(m.predict(safe), dtype=float) for m in models], axis=0)

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

        Without :attr:`run_subdir` it is ``out_dir`` itself.

        Computed once per ``estimate`` and shared by the overall diagnostics and the optional conditional
        step so both write into the *same* run folder.
        """
        upgrade_start = toggle_upgrade_start(mi.upgrade_timing, index)
        run_name = f"power_model_{mi.test_wtg}_{upgrade_start:%Y%m%d}_{index.max():%Y%m%d}"
        out_root = Path(self.out_dir) if self.out_dir is not None else Path(tempfile.mkdtemp(prefix="power_model_"))
        run_dir = out_root / run_name if self.run_subdir or self.out_dir is None else out_root
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _csv_naming(self, run_dir: Path) -> tuple[str, str]:
        """Return the run name and timestamp a run's CSV names carry; both empty without a run folder."""
        if not self.run_subdir and self.out_dir is not None:
            return "", ""
        return run_dir.name, pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S_%f")

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
        constant_features: list[str],
        losses: pd.DataFrame,
        uncertainty: pd.DataFrame,
    ) -> None:
        """Assemble the diagnostic data and write the CSVs (+ plots), logging the top features."""
        run_name, ts = self._csv_naming(run_dir)
        losses.to_csv(run_dir / diag.csv_name("record_losses", run_name=run_name, ts=ts), index=False)
        uncertainty.to_csv(run_dir / diag.csv_name("uncertainty", run_name=run_name, ts=ts), index=False)

        x_sel = features.iloc[selected]
        data = diag.DiagnosticData(
            test_wtg=mi.test_wtg,
            mode="toggle" if is_toggle(mi.upgrade_timing) else "prepost",
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
            constant_features=tuple(constant_features),
        )
        importance = diag.write_csvs(run_dir, run_name, ts, data)
        diag.log_top_features(importance, active_power_col=self.columns.active_power)
        logger.info(
            "%s %s: uplift=%+.3f%%  (sum_actual=%.1f MWh, sum_counterfactual=%.1f MWh, n_up=%d)",
            self.name,
            mi.test_wtg,
            100 * uplift,
            sum_actual * (timebase / pd.Timedelta(hours=1)) / 1000.0,
            sum_counter * (timebase / pd.Timedelta(hours=1)) / 1000.0,
            len(fit["y_upgraded"]),
        )
        if self.save_plots:
            diag.save_plots(run_dir / "plots", data, importance)

    def _write_shared_diagnostics(
        self,
        mi: MethodInput,
        *,
        run_dir: Path,
        t: np.ndarray,
        selected: np.ndarray,
        timebase: pd.Timedelta,
        era5: pd.DataFrame | None,
        power_references: list[str] | None = None,
    ) -> None:
        """Emit the shared cross-method diagnostics (coverage/curves/histograms) and the run config."""
        ctx = DiagnosticContext(
            run_dir=run_dir,
            test_wtg=mi.test_wtg,
            turbine_col=mi.turbine_col,
            columns=self.columns,
            scada_df=mi.scada_df,
            treated_ts=t.astype(bool),
            used_ts=np.asarray(selected, dtype=bool),
            timebase=timebase,
            mode="toggle" if is_toggle(mi.upgrade_timing) else "prepost",
            era5_df=era5,
            era5_label=self.era5_label,
            power_references=power_references,
        )
        write_common_diagnostics(ctx, skip_stages=frozenset() if self.repeat_shared_plots else _SHARED_STAGES)
        write_run_config(ctx, method_name=self.name, method_params=self._config_params())

    def _config_params(self) -> dict[str, Any]:
        """Return the power-model configuration recorded in the run-config YAML."""
        return {
            "active_power_col": self.columns.active_power,
            "availability_col": self.columns.availability,
            "baseline_rated_power_kw": self.baseline_rated_power_kw,
            "wind_speed_col": self.columns.wind_speed,
            "seed": self.seed,
            "has_era5": self.era5_hourly_df is not None,
            "site_location": list(self.site_location) if self.site_location is not None else None,
            "reference_stat_cols": list(self.reference_stat_cols),
            "direction_feature": self.direction_feature,
            "model_params": {**TUNED_MODEL_PARAMS, **self.model_params},
            "adaptive_time_decay": self.adaptive_time_decay,
            "time_decay_half_life_days": self.time_decay_half_life_days,
            "reference_screen": self.reference_screen,
            "screen_floor": self.screen_floor,
            "screen_min_campaign_days": self.screen_min_campaign_days,
            "report_reference_uplifts": self.report_reference_uplifts,
            "pairing_max_gap": None if self.pairing_max_gap is None else str(self.pairing_max_gap),
            "block_hours": self.block_hours,
            "block_hours_sweep": list(self.block_hours_sweep),
            "n_resamples": self.n_resamples,
            "bootstrap_seed": self.bootstrap_seed,
        }
