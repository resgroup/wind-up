"""Model-fitting pieces for the power model: fold assignment and the outcome-model factory.

:func:`time_block_folds` — contiguous-block, round-robin fold assignment for time-ordered rows.
A shuffled split leaks autocorrelation (held-out rows sit minutes from training rows), so its
residuals are optimistic; contiguous blocks confine the leakage to the block edges, which makes
the held-out fit-quality diagnostic honest.

:func:`make_outcome_model` — the L2 LightGBM regressor for the counterfactual power ``E[Y|X]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lightgbm import LGBMRegressor


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


# Common LightGBM hyperparameters; native NaN handling, seconds to train. Callers (and drivers)
# override via keyword arguments.
_COMMON: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
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
