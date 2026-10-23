"""Model-fitting pieces for the power model: fold assignment and the outcome-model factory.

:func:`time_block_folds` — contiguous-block, round-robin fold assignment for time-ordered rows.
A shuffled split leaks autocorrelation (held-out rows sit minutes from training rows), so its
residuals are optimistic; contiguous blocks confine the leakage to the block edges, which makes
the held-out fit-quality diagnostic honest.

:func:`purged_time_block_folds` — five contiguous blocks with an embargo either side of each
held-out block, for out-of-fold predictions that must not see their neighbours' minutes.

:func:`make_outcome_model` — the L2 LightGBM regressor for the counterfactual power ``E[Y|X]``.

:func:`make_propensity_model` — the LightGBM classifier for ``P(upgraded | X)``, same parameters.

:func:`model_safe_features` — positional column names for the fit, since LightGBM rejects JSON
special characters and real source tags carry them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier, LGBMRegressor


def model_safe_features(features: pd.DataFrame) -> pd.DataFrame:
    """Return ``features`` with columns renamed to positional ``f0``, ``f1``, ... names.

    LightGBM refuses feature names containing JSON special characters, which real source tags carry
    (``Power, Minimum (kW)``, ``Nacelle position (°)``). Renaming positionally sidesteps the whole
    class of them rather than guessing which characters this version objects to.

    Nothing is lost: importances come back positionally, and the diagnostics carry the original
    names alongside. Apply it to the training and prediction frames alike, or their columns will
    not match.
    """
    safe = features.copy(deep=False)
    safe.columns = [f"f{i}" for i in range(features.shape[1])]
    return safe


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


def purged_time_block_folds(
    timestamps: pd.DatetimeIndex,
    *,
    n_folds: int = 5,
    embargo: pd.Timedelta,
    strata: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split time-ordered rows into ``n_folds`` contiguous blocks, each training set purged of its edges.

    Fold ``k`` holds out the ``k``-th contiguous block of rows; its training set is every other row
    whose timestamp lies more than ``embargo`` outside that block's first-to-last timestamp. One
    contiguous block per fold, unlike :func:`time_block_folds`' round-robin, so an out-of-fold
    prediction is genuinely extrapolated in time, and the embargo keeps the autocorrelated minutes
    either side out of its training set. Returns ``(train_positions, test_positions)`` per fold.

    :param strata: a label per row; when given, each stratum is cut into its own ``n_folds`` blocks
        and fold ``k`` holds out block ``k`` of every stratum. A prepost classifier needs this: its
        label *is* time, so an unstratified block at either end holds one class out almost whole
        and trains on the other alone.
    """
    if n_folds < 2:  # noqa: PLR2004
        msg = f"need n_folds >= 2, got n_folds={n_folds}"
        raise ValueError(msg)
    ts = pd.DatetimeIndex(timestamps)
    n = len(ts)
    labels = np.zeros(n, dtype=int) if strata is None else np.unique(np.asarray(strata), return_inverse=True)[1]
    block = np.empty(n, dtype=int)
    for stratum in np.unique(labels):
        rows = np.flatnonzero(labels == stratum)
        block[rows] = np.minimum((np.arange(len(rows)) * n_folds) // len(rows), n_folds - 1)
    folds = []
    for k in range(n_folds):
        test = np.flatnonzero(block == k)
        purged = np.zeros(n, dtype=bool)
        for stratum in np.unique(labels[test]):
            rows = test[labels[test] == stratum]
            purged |= np.asarray((ts >= ts[rows[0]] - embargo) & (ts <= ts[rows[-1]] + embargo))
        folds.append((np.flatnonzero(~purged), test))
    return folds


# Common LightGBM hyperparameters; native NaN handling, seconds to train. Callers (and drivers)
# override via keyword arguments.
#
# deterministic + force_row_wise make a fit reproducible. Without them LightGBM picks row-wise or
# col-wise histograms by timing the machine, so the same estimate moved with the load on the box
# and with the number of threads: measured on a 2.2M-row campaign, the two strategies differ by
# 0.001 pp and thread count by 0.005 pp, while these settings hold six decimal places across 1, 4
# and 12 threads. Training is 20-50% slower. force_row_wise rather than col: this data is many
# rows and few features, and it is what the auto-choice picks on an idle machine.
_COMMON: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def _import_lightgbm() -> Any:  # noqa: ANN401
    """Import lightgbm lazily with a helpful error if the optional ``ml`` group is missing."""
    try:
        import lightgbm  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        msg = "lightgbm is required for the power model; install the 'ml' optional dependency group."
        raise ImportError(msg) from exc
    return lightgbm


def make_outcome_model(**overrides: Any) -> LGBMRegressor:  # noqa: ANN401
    """L2 outcome regressor for the counterfactual power ``E[Y|X]`` (the energy-relevant mean model)."""
    lgb = _import_lightgbm()
    return lgb.LGBMRegressor(objective="regression", **{**_COMMON, **overrides})


def make_propensity_model(**overrides: Any) -> LGBMClassifier:  # noqa: ANN401
    """Binary classifier for ``P(upgraded | X)`` under the outcome model's common parameters."""
    lgb = _import_lightgbm()
    return lgb.LGBMClassifier(objective="binary", **{**_COMMON, **overrides})
