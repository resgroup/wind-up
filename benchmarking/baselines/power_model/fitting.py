"""Outcome-model fitting strategies for the power model (Issue 12).

Pure helpers behind ``PowerModelMethod``'s model-fundamentals knobs:

* :func:`time_block_folds` — contiguous-block, round-robin fold assignment for time-ordered rows.
  A shuffled split leaks autocorrelation (held-out rows sit minutes from training rows), so its
  residuals are optimistic; contiguous blocks confine the leakage to the block edges, which makes
  the held-out fit-quality diagnostics honest and gives the calibration a usable basis.
* :func:`fit_calibration_line` — the post-hoc calibration-slope correction: a least-squares line
  ``actual ~ a + b * predicted`` fit on out-of-fold baseline predictions, applied to the
  counterfactual predictions (the cheap de-shrinking cousin of residual calibration).
* :func:`early_stopped_n_estimators` — data-size-adaptive capacity: pick the LightGBM tree count
  by early stopping on a time-blocked validation split, then refit on all rows at that capacity
  (the 3-month toggle fit and the 2-year prepost baseline differ ~10x in rows but share one
  capacity today).
* :data:`OUTCOME_MODEL_FACTORIES` — the model-factory seam: named, JSON-addressable factories for
  alternative outcome learners. Every factory maps ``seed -> unfitted sklearn-style regressor``
  whose objective targets the **conditional mean** (the estimand is an energy ratio and energy is
  a sum of conditional means, so median-type objectives are out; design note §2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_MIN_CALIBRATION_ROWS = 100
_MIN_PREDICTION_STD = 1e-12  # below this the predictions are effectively constant; a line fit is degenerate


def time_block_folds(n: int, *, n_folds: int = 5, n_blocks: int = 25) -> np.ndarray:
    """Assign ``n`` time-ordered rows to ``n_folds`` folds as round-robin contiguous blocks.

    Rows must already be in time order (the power model's row arrays are — they follow the sorted
    analysis index). The rows are cut into ``n_blocks`` contiguous, equal-length blocks and block
    ``i`` goes to fold ``i % n_folds``, so every fold samples all seasons while staying contiguous
    at the scale that matters for autocorrelation (only the block edges sit near training rows).
    Returns an int array of fold ids, one per row.
    """
    if n_folds < 2 or n_blocks < n_folds:  # noqa: PLR2004
        msg = f"need n_folds >= 2 and n_blocks >= n_folds, got n_folds={n_folds}, n_blocks={n_blocks}"
        raise ValueError(msg)
    block = np.minimum((np.arange(n) * n_blocks) // max(n, 1), n_blocks - 1)
    return (block % n_folds).astype(int)


@dataclass(frozen=True)
class CalibrationLine:
    """The post-hoc calibration line ``corrected = intercept + slope * predicted``.

    ``slope`` is the predicted-vs-actual calibration slope (target ~= 1); a slope > 1 means the
    model's predictions are compressed toward the mean (shrinkage) and the correction stretches
    them back out.
    """

    intercept: float
    slope: float

    def apply(self, predicted: np.ndarray) -> np.ndarray:
        """Return the calibrated predictions."""
        return self.intercept + self.slope * np.asarray(predicted, dtype=float)


IDENTITY_CALIBRATION = CalibrationLine(intercept=0.0, slope=1.0)


def fit_calibration_line(y: np.ndarray, predicted: np.ndarray) -> CalibrationLine:
    """Least-squares ``y ~ a + b * predicted`` over finite pairs (the calibration-slope fit).

    ``predicted`` must be **out-of-fold** (a prediction of each row from a model not trained on
    it) or the slope is optimistically ~1 by in-sample construction. Falls back to the identity
    line when there are too few pairs or the predictions are degenerate (near-zero variance), so
    applying the result is always safe.
    """
    y = np.asarray(y, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(y) & np.isfinite(predicted)
    if int(finite.sum()) < _MIN_CALIBRATION_ROWS or float(np.std(predicted[finite])) <= _MIN_PREDICTION_STD:
        return IDENTITY_CALIBRATION
    slope, intercept = np.polyfit(predicted[finite], y[finite], deg=1)
    return CalibrationLine(intercept=float(intercept), slope=float(slope))


@dataclass(frozen=True)
class ResidualCellCalibration:
    """Per-target-row residual corrections from **centred** cell-mean baseline residuals (Issue 13).

    :param per_row_residual: one value per target row — its cell's mean baseline residual minus the
        global mean residual (the mix-shift differential), or 0 where the cell was unseen in
        baseline (or the row's cell is invalid)
    :param global_mean_residual: mean residual over all finite baseline residuals (the centring
        level; reported for diagnostics, deliberately **not** applied — see below)
    :param n_target_unseen: target rows that fell back to 0 (no differential information)
    :param n_cells: baseline cells with a residual estimate
    """

    per_row_residual: np.ndarray
    global_mean_residual: float
    n_target_unseen: int
    n_cells: int


def cell_residual_calibration(
    *, codes_baseline: np.ndarray, residuals: np.ndarray, codes_target: np.ndarray
) -> ResidualCellCalibration:
    """Estimate each target row's expected model residual **differential** from baseline residuals.

    The counterfactual model's conditional bias ``b(x)`` integrates to ~0 over the *training*
    (baseline) covariate mix but the headline evaluates it over the *target* (upgraded) mix — any
    shift between the two converts conditional bias into headline bias. This estimates that term on
    untreated data: mean **out-of-fold** residual (``y - prediction``) per coarsened covariate cell
    over the baseline rows, **centred by the global mean residual**, read out under the target
    rows' cell occupancy. Adding the returned per-row values to the target predictions removes the
    mix-shift headline bias.

    The centring is the F14 lesson applied: the out-of-fold *level* (fold models see only ~80% of
    an already-finite fit, so their residuals carry a small global offset the final 100% fit does
    not have) is an artefact of the OOF basis and must not be transferred to the final model's
    predictions; only the between-cell *differential* is the mix-shift term. A model that is
    level-unbiased over its own training mix therefore gets a correction that integrates to ~0
    under the baseline mix, exactly as ``b(x)`` does.

    ``codes_baseline`` / ``codes_target`` are integer cell codes per (row, var) from
    :func:`~benchmarking.baselines.power_model.matching.cell_codes`; a row with any ``-1`` (NaN /
    out-of-range) belongs to no cell. Cells unseen in baseline — and invalid rows — get 0 (no
    differential information), so the correction is always defined.
    """
    import pandas as pd  # noqa: PLC0415 - keep the module import-light like the factories

    residuals = np.asarray(residuals, dtype=float)
    finite = np.isfinite(residuals)
    global_mean = float(residuals[finite].mean()) if finite.any() else 0.0

    n_vars = codes_baseline.shape[1]
    key_cols = list(range(n_vars))
    valid_b = (codes_baseline >= 0).all(axis=1) & finite
    base = pd.DataFrame(codes_baseline[valid_b], columns=key_cols)
    base["residual"] = residuals[valid_b]
    cell_means = base.groupby(key_cols, sort=False)["residual"].mean().rename("cell_mean").reset_index()

    target = pd.DataFrame(codes_target, columns=key_cols)
    mapped = target.merge(cell_means, on=key_cols, how="left")["cell_mean"].to_numpy(dtype=float)
    mapped[~(codes_target >= 0).all(axis=1)] = np.nan
    unseen = ~np.isfinite(mapped)
    return ResidualCellCalibration(
        per_row_residual=np.where(unseen, 0.0, mapped - global_mean),
        global_mean_residual=global_mean,
        n_target_unseen=int(unseen.sum()),
        n_cells=len(cell_means),
    )


def early_stopped_n_estimators(
    make_model: Callable[[], Any],
    x: Any,  # noqa: ANN401 - pd.DataFrame
    y: np.ndarray,
    *,
    valid_mask: np.ndarray,
    stopping_rounds: int = 100,
) -> int:
    """Pick the LightGBM tree count by early stopping on the given validation rows.

    ``make_model`` returns the probe regressor, already configured with the capacity **ceiling**
    as its ``n_estimators``. The probe is fit on ``~valid_mask`` rows with ``valid_mask`` rows as
    the eval set; the returned count is the best iteration (the ceiling itself when early stopping
    never triggers), for the caller to refit on **all** rows at that capacity.
    """
    import lightgbm  # noqa: PLC0415 - optional dependency, imported lazily like the factories

    train = ~np.asarray(valid_mask, dtype=bool)
    model = make_model()
    model.fit(
        x.iloc[train],
        y[train],
        eval_set=[(x.iloc[valid_mask], y[valid_mask])],
        callbacks=[lightgbm.early_stopping(stopping_rounds=stopping_rounds, verbose=False)],
    )
    # With the callback attached, best_iteration_ is a positive int whether or not early stopping
    # triggered (= n_estimators when it never did). None/0 are LightGBM's "no early stopping ran"
    # sentinels (0 is the Booster-level value), so both fall back to the ceiling — a plain
    # `is not None` check would turn the 0 sentinel into a zero-tree model.
    best = getattr(model, "best_iteration_", None)
    chosen = int(best) if best else int(model.n_estimators)
    logger.info(
        "early stopping picked n_estimators=%d (ceiling %d, n_train=%d)", chosen, model.n_estimators, int(train.sum())
    )
    return chosen


def _make_hgb(seed: int) -> Any:  # noqa: ANN401
    """Sklearn ``HistGradientBoostingRegressor`` — a same-family (boosted trees, L2) sanity check.

    Capacity mirrors the LightGBM defaults where the knobs correspond (600 iterations, lr 0.03,
    63 leaves, ``min_samples_leaf=200``); NaN handling is native, like LightGBM's.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415

    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=600,
        learning_rate=0.03,
        max_leaf_nodes=63,
        min_samples_leaf=200,
        early_stopping=False,
        random_state=seed,
    )


def _make_linear(seed: int) -> Any:  # noqa: ANN401 ARG001
    """Deliberately low-variance structured baseline: median-impute -> standardise -> Ridge.

    Lightly regularised so it is nearly shrinkage-free — a cross-check on the tree models'
    conditional bias, valuable even though its overall accuracy is worse. Deterministic, so the
    seed is unused.
    """
    from sklearn.impute import SimpleImputer  # noqa: PLC0415
    from sklearn.linear_model import Ridge  # noqa: PLC0415
    from sklearn.pipeline import make_pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


# The model-factory seam: named factories (seed -> unfitted regressor) selectable by string on
# ``PowerModelMethod.model_factory`` (JSON-addressable for --method-overrides A/B runs). All are
# conditional-mean learners; a callable can be passed directly for anything not registered.
OUTCOME_MODEL_FACTORIES: dict[str, Callable[[int], Any]] = {
    "hgb": _make_hgb,
    "linear": _make_linear,
}
