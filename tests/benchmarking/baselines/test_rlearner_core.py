"""Tests for the cross-fit R-learner core and its LightGBM nuisance factories.

The core is pure (no I/O): given a feature matrix ``X``, outcome ``y`` and upgrade flag ``t``
it returns per-row ``tau``, ``m_hat``, ``e_hat`` and ``mu0`` plus the fitted outcome/effect
models. These tests use generous synthetic data so the recovered effect is unambiguous,
covering the toggle-like (flat propensity) and confounded (before/after-like) regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.rlearner.nuisance import (
    make_effect_model,
    make_outcome_model,
    make_propensity_model,
)
from benchmarking.baselines.rlearner.rlearner import RLearnerFit, cross_fit_rlearner


def _small_models() -> dict:
    """Fast, quiet LightGBM factories for tests (few trees, no logging)."""
    params = {"n_estimators": 80, "num_leaves": 15, "min_child_samples": 20, "verbose": -1}
    return {
        "make_outcome": lambda: make_outcome_model(**params),
        "make_propensity": lambda: make_propensity_model(**params),
        "make_effect": lambda: make_effect_model(**params),
    }


class TestNuisanceFactories:
    def test_outcome_is_regressor_that_predicts(self) -> None:
        rng = np.random.default_rng(0)
        x = pd.DataFrame({"f": rng.normal(size=200)})
        y = 3.0 * x["f"]
        model = make_outcome_model(n_estimators=50, verbose=-1).fit(x, y)
        assert model.predict(x).shape == (200,)

    def test_propensity_predicts_probability(self) -> None:
        rng = np.random.default_rng(1)
        x = pd.DataFrame({"f": rng.normal(size=200)})
        t = (x["f"] > 0).astype(int)
        model = make_propensity_model(n_estimators=50, verbose=-1).fit(x, t)
        proba = model.predict_proba(x)[:, 1]
        assert ((proba >= 0) & (proba <= 1)).all()


class TestCrossFitRLearner:
    def test_recovers_constant_effect_with_flat_propensity(self) -> None:
        # toggle-like: t is random (propensity ~0.5); y = mu0(x) + tau*t + small noise
        rng = np.random.default_rng(0)
        n = 3000
        x = rng.uniform(0, 1, size=n)
        x_df = pd.DataFrame({"ref_ws": x})
        mu0 = 10.0 * x
        t = rng.binomial(1, 0.5, size=n)
        tau_true = 2.0
        y = mu0 + tau_true * t + rng.normal(0, 0.1, size=n)
        fit = cross_fit_rlearner(x_df, y=y, t=t, n_folds=4, seed=0, **_small_models())
        assert isinstance(fit, RLearnerFit)
        assert float(np.mean(fit.tau)) == pytest.approx(tau_true, abs=0.3)
        assert float(np.mean(fit.e_hat)) == pytest.approx(0.5, abs=0.05)

    def test_recovers_effect_under_confounding(self) -> None:
        # before/after-like: t is confounded with x but stochastic, so overlap (positivity) holds.
        # The upgraded period over-samples high-wind conditions; a naive treated-minus-baseline
        # difference is badly biased, but the R-learner recovers tau by partialling x out of both.
        rng = np.random.default_rng(1)
        n = 5000
        x = rng.uniform(0, 1, size=n)
        x_df = pd.DataFrame({"ref_ws": x})
        mu0 = 10.0 * x
        propensity = 0.2 + 0.6 * x  # in (0.2, 0.8): confounded with x but always overlapping
        t = rng.binomial(1, propensity)
        tau_true = 2.0
        y = mu0 + tau_true * t + rng.normal(0, 0.1, size=n)
        naive_diff = y[t == 1].mean() - y[t == 0].mean()
        assert naive_diff > 3.5  # confirm the naive estimate really is badly biased high
        fit = cross_fit_rlearner(x_df, y=y, t=t, n_folds=5, seed=0, **_small_models())
        assert float(np.mean(fit.tau)) == pytest.approx(tau_true, abs=0.5)

    def test_placebo_reports_zero_effect_flat_propensity(self) -> None:
        # a placebo upgrade (no real effect) must report ~0 uplift, not a spurious one.
        rng = np.random.default_rng(10)
        n = 4000
        x = rng.uniform(0, 1, size=n)
        x_df = pd.DataFrame({"ref_ws": x})
        t = rng.binomial(1, 0.5, size=n)
        y = 10.0 * x + rng.normal(0, 0.1, size=n)  # no tau*t term
        fit = cross_fit_rlearner(x_df, y=y, t=t, n_folds=4, seed=0, **_small_models())
        tau_mean = float(np.mean(fit.tau))
        assert tau_mean == pytest.approx(0.0, abs=0.2)
        # the headline aggregation (sum tau / sum mu0) is also ~0
        assert float(np.sum(fit.tau) / np.sum(fit.mu0)) == pytest.approx(0.0, abs=0.05)

    def test_placebo_reports_zero_effect_under_confounding(self) -> None:
        # placebo with the upgraded period over-sampling high wind: still ~0, no covariate-shift bias.
        rng = np.random.default_rng(11)
        n = 5000
        x = rng.uniform(0, 1, size=n)
        x_df = pd.DataFrame({"ref_ws": x})
        t = rng.binomial(1, 0.2 + 0.6 * x)
        y = 10.0 * x + rng.normal(0, 0.1, size=n)  # no real effect
        naive_diff = y[t == 1].mean() - y[t == 0].mean()
        assert naive_diff > 1.5  # naive would wrongly report a large positive "uplift"
        fit = cross_fit_rlearner(x_df, y=y, t=t, n_folds=5, seed=0, **_small_models())
        assert float(np.sum(fit.tau) / np.sum(fit.mu0)) == pytest.approx(0.0, abs=0.05)

    def test_mu0_identity_holds(self) -> None:
        rng = np.random.default_rng(2)
        n = 1500
        x = rng.uniform(0, 1, size=n)
        x_df = pd.DataFrame({"ref_ws": x})
        t = rng.binomial(1, 0.5, size=n)
        y = 10.0 * x + 2.0 * t + rng.normal(0, 0.1, size=n)
        fit = cross_fit_rlearner(x_df, y=y, t=t, n_folds=4, seed=0, **_small_models())
        # mu0 = m_hat - e_hat * tau by construction
        assert fit.mu0 == pytest.approx(fit.m_hat - fit.e_hat * fit.tau)

    def test_handles_nan_features(self) -> None:
        # LightGBM handles NaN natively; the core must not choke on NaN in X.
        rng = np.random.default_rng(3)
        n = 1500
        x = rng.uniform(0, 1, size=n)
        x_df = pd.DataFrame({"ref_ws": x, "noisy": rng.normal(size=n)})
        x_df.loc[x_df.index[:50], "noisy"] = np.nan
        t = rng.binomial(1, 0.5, size=n)
        y = 10.0 * x + 2.0 * t + rng.normal(0, 0.1, size=n)
        fit = cross_fit_rlearner(x_df, y=y, t=t, n_folds=4, seed=0, **_small_models())
        assert np.isfinite(fit.tau).all()
