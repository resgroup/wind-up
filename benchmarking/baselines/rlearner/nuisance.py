"""LightGBM nuisance/effect model factories for the R-learner.

Three model roles (design note §4, Stage 1/2):

* **outcome** ``m(x)=E[Y|X]`` — an L2 (or Huber) regressor. Energy is the integral of the
  *mean*, so the energy-relevant model targets the mean, not the median (design note §2).
* **propensity** ``e(x)=E[T|X]`` — a classifier for the upgrade flag.
* **effect** ``tau(x)`` — a regressor fit on the R-learner pseudo-outcome.

LightGBM is an optional dependency (the ``ml`` group); it is imported lazily so this package
imports without it installed. Defaults follow the design note's common hyperparameters;
callers (and tests) override via keyword arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier, LGBMRegressor

# Design note §6 "common" hyperparameters; native NaN handling, seconds to train.
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
        msg = "lightgbm is required for the R-learner method; install the 'ml' optional dependency group."
        raise ImportError(msg) from exc
    return lightgbm


def make_outcome_model(**overrides: Any) -> LGBMRegressor:  # noqa: ANN401
    """L2 outcome regressor for ``E[Y|X]`` (the energy-relevant mean model)."""
    lgb = _import_lightgbm()
    return lgb.LGBMRegressor(objective="regression", **{**_COMMON, **overrides})


def make_propensity_model(**overrides: Any) -> LGBMClassifier:  # noqa: ANN401
    """Binary classifier for the propensity ``E[T|X]`` (flat ~0.5 for toggle, real for before/after)."""
    lgb = _import_lightgbm()
    return lgb.LGBMClassifier(objective="binary", **{**_COMMON, **overrides})


def make_effect_model(**overrides: Any) -> LGBMRegressor:  # noqa: ANN401
    """Regressor for the effect ``tau(x)``, fit on the R-learner pseudo-outcome."""
    lgb = _import_lightgbm()
    return lgb.LGBMRegressor(objective="regression", **{**_COMMON, **overrides})
