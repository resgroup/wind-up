"""The pure cross-fit R-learner core (Robinson partialling-out).

No I/O, no source vocabulary: given a feature matrix ``X`` (upgrade-invariant; NaN allowed),
outcome ``y`` and upgrade flag ``t``, it returns per-row ``tau`` (absolute effect), the
cross-fit nuisances ``m_hat``/``e_hat``, the baseline power ``mu0`` and the fitted
outcome/effect models for feature-importance diagnostics.

Method (design note §4):

1. Cross-fit the nuisances: over K folds, fit outcome ``m(x)=E[Y|X]`` and propensity
   ``e(x)=E[T|X]`` on the training folds and predict on the held-out fold, so every row gets
   an out-of-fold prediction it did not help train. Shuffled K-fold is valid because there are
   no timestamp features.
2. Residualise: ``y_res = y - m_hat``, ``t_res = t - e_hat``.
3. Fit the effect model ``tau(x)`` on the pseudo-outcome ``y_res / t_res`` with R-loss weights
   ``t_res**2`` (guarding the ``t_res≈0`` divide).
4. Baseline power ``mu0 = m_hat - e_hat * tau`` (no extra model).

Collapses to plain regression adjustment when the propensity is flat (toggle) and
orthogonalises when it is not (before/after) — one code path for both modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt
    import pandas as pd

# Below this |t_res| the R-learner pseudo-outcome y_res/t_res is numerically unstable; such
# rows carry ~zero R-loss weight anyway, so they are dropped from the effect fit.
_MIN_T_RES = 1e-6


def _import_kfold() -> Any:  # noqa: ANN401
    """Import scikit-learn's ``KFold`` lazily with a helpful error if the optional ``ml`` group is missing.

    Mirrors the lazy LightGBM import so this package imports without the optional ``ml`` dependencies.
    """
    try:
        from sklearn.model_selection import KFold  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        msg = "scikit-learn is required for the R-learner method; install the 'ml' optional dependency group."
        raise ImportError(msg) from exc
    return KFold


@dataclass
class RLearnerFit:
    """Per-row R-learner outputs plus the fitted models for diagnostics.

    :param tau: absolute per-row effect estimate ``tau(x)``
    :param m_hat: cross-fit outcome prediction ``E[Y|X]``
    :param e_hat: cross-fit propensity ``E[T|X]``
    :param mu0: baseline (un-upgraded) expected power ``m_hat - e_hat * tau``
    :param outcome_model: outcome model refit on all rows (for feature importance)
    :param effect_model: the fitted effect model ``tau(x)`` (for feature importance)
    """

    tau: npt.NDArray[np.float64]
    m_hat: npt.NDArray[np.float64]
    e_hat: npt.NDArray[np.float64]
    mu0: npt.NDArray[np.float64]
    outcome_model: Any
    effect_model: Any


def cross_fit_rlearner(
    x: pd.DataFrame,
    *,
    y: npt.ArrayLike,
    t: npt.ArrayLike,
    make_outcome: Callable[[], Any],
    make_propensity: Callable[[], Any],
    make_effect: Callable[[], Any],
    n_folds: int = 5,
    seed: int = 0,
) -> RLearnerFit:
    """Cross-fit R-learner; returns per-row ``tau``/``m_hat``/``e_hat``/``mu0`` and fitted models.

    ``y`` and ``t`` must be finite (the caller filters downtime/NaN rows); ``x`` may contain NaN.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(y)
    if not (np.isfinite(y).all() and np.isfinite(t).all()):
        msg = "cross_fit_rlearner requires finite y and t; filter NaN/downtime rows before calling."
        raise ValueError(msg)

    m_hat = np.empty(n)
    e_hat = np.empty(n)
    folds = _import_kfold()(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(x):
        x_tr, x_te = x.iloc[train_idx], x.iloc[test_idx]
        m_hat[test_idx] = make_outcome().fit(x_tr, y[train_idx]).predict(x_te)
        e_hat[test_idx] = make_propensity().fit(x_tr, t[train_idx]).predict_proba(x_te)[:, 1]

    y_res = y - m_hat
    t_res = t - e_hat
    weights = t_res**2
    usable = np.abs(t_res) >= _MIN_T_RES
    pseudo = np.where(usable, y_res / np.where(usable, t_res, 1.0), 0.0)

    effect_model = make_effect()
    effect_model.fit(x.iloc[usable], pseudo[usable], sample_weight=weights[usable])
    tau = effect_model.predict(x)

    # Refit the outcome on all rows so feature importance reflects one model over the full data.
    outcome_model = make_outcome().fit(x, y)

    mu0 = m_hat - e_hat * tau
    return RLearnerFit(
        tau=tau, m_hat=m_hat, e_hat=e_hat, mu0=mu0, outcome_model=outcome_model, effect_model=effect_model
    )
