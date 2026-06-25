"""The design-note §8 bias-guard regression test.

The upgrade distorts the test turbine's own nacelle wind speed (a post-treatment variable,
design note §3). This test proves the upgrade-invariant reference-only feature rule removes the
resulting bias, and guards against anyone re-adding a test-turbine signal to the feature set:

* a model that (wrongly) conditions on the corrupted test wind speed reports a materially biased
  effect (the post-treatment signal both leaks the treatment into the covariate and destroys
  propensity overlap);
* the reference-only R-learner recovers the known uplift despite the same corrupted signal;
* the feature builder's guard rejects a test-turbine-qualified feature outright.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.rlearner.features import QUALIFIER, build_reference_features, check_upgrade_invariant
from benchmarking.baselines.rlearner.method import RLearnerMethod
from benchmarking.baselines.rlearner.rlearner import cross_fit_rlearner
from benchmarking.harness.method import MethodInput

_TURBINE = "TurbineName"
_POWER = "wtc_ActPower_mean"
_WS = "wtc_AcWindSp_mean"
_SMALL = {"n_estimators": 120, "num_leaves": 15, "min_child_samples": 20, "verbose": -1}
_UPLIFT = 0.05


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC", name="timestamp")


def _corrupted_scada(idx: pd.DatetimeIndex, treated: np.ndarray) -> pd.DataFrame:
    """SCADA with a known uplift AND a test wind speed corrupted by the upgrade (post-treatment).

    The test turbine's measured wind speed is shifted hard whenever it is upgraded, so the signal
    effectively encodes the treatment — the textbook post-treatment trap.
    """
    rng = np.random.default_rng(0)
    w = rng.uniform(4.0, 12.0, len(idx))  # true free-stream wind
    frames = []
    for name in ("T1", "R1", "R2"):
        ws = w + rng.normal(0, 0.2, len(idx))
        power = 80.0 * w + rng.normal(0, 5.0, len(idx))
        if name == "T1":
            power = np.where(treated, power * (1.0 + _UPLIFT), power)
            ws = ws - 100.0 * treated  # upgrade corrupts the test anemometer (post-treatment)
        frames.append(pd.DataFrame({_TURBINE: name, _POWER: power, _WS: ws}, index=idx))
    return pd.concat(frames)


def _headline(x: pd.DataFrame, *, y: np.ndarray, t: np.ndarray) -> float:
    fit = cross_fit_rlearner(x, y=y, t=t, n_folds=4, seed=0, **_factories())
    up = t.astype(bool)
    return float(np.sum(fit.tau[up]) / np.sum(fit.mu0[up]))


def _factories() -> dict:
    from benchmarking.baselines.rlearner.nuisance import (  # noqa: PLC0415
        make_effect_model,
        make_outcome_model,
        make_propensity_model,
    )

    return {
        "make_outcome": lambda: make_outcome_model(**_SMALL),
        "make_propensity": lambda: make_propensity_model(**_SMALL),
        "make_effect": lambda: make_effect_model(**_SMALL),
    }


def test_conditioning_on_test_ws_biases_estimate() -> None:
    # Demonstration: leaking the corrupted test wind speed gives a materially wrong uplift,
    # while the reference-only feature set recovers the true 5%.
    idx = _index(3000)
    upgrade = idx[1500]
    treated = np.asarray(idx >= upgrade)
    scada = _corrupted_scada(idx, treated)

    x_ref = build_reference_features(scada, test_wtg="T1", turbine_col=_TURBINE)
    y = scada.loc[scada[_TURBINE] == "T1", _POWER].to_numpy(dtype=float)
    t = treated.astype(float)

    x_leaky = x_ref.copy()
    x_leaky["LEAKED_test_ws"] = scada.loc[scada[_TURBINE] == "T1", _WS].to_numpy(dtype=float)

    ref_only = _headline(x_ref, y=y, t=t)
    leaky = _headline(x_leaky, y=y, t=t)

    assert ref_only == pytest.approx(_UPLIFT, abs=0.015)  # reference-only is correct
    assert abs(leaky - _UPLIFT) > 0.03  # leaking the post-treatment signal gives a materially wrong answer


def test_method_recovers_uplift_despite_corrupted_test_ws(tmp_path) -> None:  # noqa: ANN001
    # The full method never reads the test turbine's signals, so the corruption is harmless.
    idx = _index(3000)
    upgrade = idx[1500]
    treated = np.asarray(idx >= upgrade)
    scada = _corrupted_scada(idx, treated)
    out = RLearnerMethod(
        active_power_col=_POWER, wind_speed_col=_WS, out_dir=tmp_path, n_folds=4, model_params=_SMALL
    ).estimate(MethodInput(scada_df=scada, test_wtg="T1", upgrade_timing=upgrade, turbine_col=_TURBINE))
    assert out.p50_overall == pytest.approx(_UPLIFT, abs=0.015)


def test_feature_builder_never_includes_test_turbine() -> None:
    idx = _index(200)
    treated = np.asarray(idx >= idx[100])
    scada = _corrupted_scada(idx, treated)
    x = build_reference_features(scada, test_wtg="T1", turbine_col=_TURBINE)
    assert not any(c.endswith(f"{QUALIFIER}T1") for c in x.columns)


def test_guard_rejects_a_test_turbine_feature() -> None:
    with pytest.raises(ValueError, match="upgrade-invariant"):
        check_upgrade_invariant([f"{_WS}{QUALIFIER}T1"], test_wtg="T1")
