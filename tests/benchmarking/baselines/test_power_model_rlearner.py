"""The R-learner prepost headline: its estimator on known inputs and its wiring into ``PowerModelMethod``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.power_model import rlearner
from benchmarking.baselines.power_model.method import RLEARNER_COLUMNS, MethodConfigurationError
from benchmarking.harness.toggle import ToggleSchedule
from tests.benchmarking.baselines.test_power_model_method import (
    _FAST_PARAMS,
    _fundamentals_method,
    _long_case,
    _prepost_case,
    _screen_case,
    _screen_method,
    _shape_case,
    _shape_method,
    _uncertainty_case,
)

_RATED = 2000.0
_EDGES = [_RATED * (-0.05 + 0.11 * i) for i in range(11)]


def _oracle(
    n: int = 20_000, *, theta: float = 0.04, share_after: float = 0.4, noise: float = 0.0, seed: int = 0
) -> dict[str, np.ndarray]:
    """Records with a known baseline, the exact ``m`` and ``e`` of a period independent of the conditions."""
    rng = np.random.default_rng(seed)
    baseline = rng.uniform(100.0, 1900.0, n)
    after = np.arange(n) >= int(n * (1 - share_after))
    y = baseline * (1 + theta * after) + rng.normal(0.0, noise, n)
    return {
        "y": y,
        "m": baseline * (1 + theta * share_after),
        "after": after,
        "e": np.full(n, share_after),
        "baseline": baseline,
    }


class TestTimeFolds:
    def test_blocks_are_contiguous_and_dealt_round_robin(self) -> None:
        times = pd.date_range("2020-01-01", periods=21 * 24, freq="1h", tz="UTC")
        folds = rlearner.time_folds(times, block_days=7.0, n_folds=2)
        assert folds.tolist() == [0] * 168 + [1] * 168 + [0] * 168

    def test_a_gap_does_not_shift_the_later_blocks(self) -> None:
        times = pd.date_range("2020-01-01", periods=21 * 24, freq="1h", tz="UTC")
        kept = np.r_[0:100, 400:504]
        assert rlearner.time_folds(times[kept], block_days=7.0, n_folds=3).tolist() == [0] * 100 + [2] * 104


class TestCrossFit:
    def test_every_record_is_predicted_by_a_model_that_did_not_see_its_fold(self) -> None:
        folds = np.array([0, 0, 1, 1, 2, 2])
        seen: list[tuple[list[int], list[int]]] = []

        def fit_predict(train: np.ndarray, held: np.ndarray) -> np.ndarray:
            seen.append((train.tolist(), held.tolist()))
            return np.full(len(held), float(len(train)))

        out = rlearner.cross_fit(folds, fit_predict=fit_predict)
        assert out.tolist() == [4.0] * 6
        assert all(not set(train) & set(held) for train, held in seen)
        assert sorted(i for _, held in seen for i in held) == list(range(6))


class TestEstimate:
    def test_recovers_the_uplift_from_exact_nuisances(self) -> None:
        d = _oracle()
        assert rlearner.overall_theta(d["y"], d["m"], d["after"], d["e"]) == pytest.approx(0.04, abs=1e-9)
        fitted = rlearner.estimate(
            d["y"], d["m"], d["after"], d["e"], power_edges=_EDGES, rated_power_kw=_RATED, min_records=10
        )
        assert fitted.uplift == pytest.approx(0.04, abs=1e-9)
        assert fitted.baseline == pytest.approx(d["baseline"])
        assert fitted.counterfactual == pytest.approx(d["baseline"][d["after"]])

    def test_a_placebo_reads_zero(self) -> None:
        d = _oracle(theta=0.0)
        fitted = rlearner.estimate(
            d["y"], d["m"], d["after"], d["e"], power_edges=_EDGES, rated_power_kw=_RATED, min_records=10
        )
        assert fitted.uplift == pytest.approx(0.0, abs=1e-12)

    def test_a_model_bias_common_to_both_periods_cancels(self) -> None:
        # A power model 3 % high everywhere would put -3 % into a forward ratio.
        d = _oracle(theta=0.0)
        fitted = rlearner.estimate(
            d["y"], 1.03 * d["m"], d["after"], d["e"], power_edges=_EDGES, rated_power_kw=_RATED, min_records=10
        )
        assert abs(fitted.uplift) < 0.002

    def test_an_uplift_that_varies_with_power_is_combined_by_after_period_energy(self) -> None:
        rng = np.random.default_rng(1)
        n = 40_000
        after = np.arange(n) >= n // 2
        # the after period is windier, so its energy sits at higher power than the before period's
        baseline = np.where(after, rng.uniform(800.0, 1900.0, n), rng.uniform(100.0, 1900.0, n))
        gain = np.where(baseline < 1000.0, 0.06, 0.01)
        y = baseline * (1 + gain * after)
        e = np.where(baseline < 800.0, 0.0, 0.5 * 1800 / (0.5 * 1800 + 0.5 * 1100))
        e = np.clip(e, 0.01, 0.99)
        m = baseline * (1 + gain * e)
        fitted = rlearner.estimate(y, m, after, e, power_edges=_EDGES, rated_power_kw=_RATED, min_records=10)
        truth = y[after].sum() / baseline[after].sum() - 1
        assert fitted.uplift == pytest.approx(truth, abs=0.002)

    def test_a_bin_seen_in_one_period_only_takes_the_overall_uplift(self) -> None:
        d = _oracle()
        low = d["baseline"] < 400.0
        keep = ~(low & ~d["after"])  # the before period never saw low power
        fitted = rlearner.estimate(
            d["y"][keep],
            d["m"][keep],
            d["after"][keep],
            d["e"][keep],
            power_edges=_EDGES,
            rated_power_kw=_RATED,
            min_records=10,
        )
        assert not fitted.measured[1]
        assert fitted.bin_theta[1] == pytest.approx(fitted.theta)
        assert fitted.uplift == pytest.approx(0.04, abs=1e-6)

    def test_no_overlap_is_not_a_number(self) -> None:
        d = _oracle()
        certain = np.where(d["after"], 1.0, 0.0)
        # the clip keeps a sliver of weight, so the estimate exists; with the weight truly zero it does not
        assert np.isnan(rlearner.overall_theta(d["y"], d["m"], d["after"], certain))


class TestBootstrap:
    def _times(self, n: int) -> pd.DatetimeIndex:
        return pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC")

    def _sigmas(self, d: dict[str, np.ndarray], **kwargs: object) -> pd.DataFrame:
        fitted = rlearner.estimate(
            d["y"], d["m"], d["after"], d["e"], power_edges=_EDGES, rated_power_kw=_RATED, min_records=10
        )
        defaults: dict[str, object] = {"block_hours": [6.0, 24.0], "n_resamples": 200, "seed": 0}
        rows = rlearner.bootstrap_sigmas(
            self._times(len(d["y"])),
            y=d["y"],
            m=d["m"],
            after=d["after"],
            e=d["e"],
            fitted=fitted,
            **{**defaults, **kwargs},  # type: ignore[arg-type]
        )
        return pd.DataFrame(rows).set_index("block_hours")

    def test_noise_free_records_have_no_spread(self) -> None:
        sigmas = self._sigmas(_oracle())
        assert sigmas["sigma"].max() < 1e-9
        assert (sigmas["frac_resamples_finite"] == 1.0).all()

    def test_the_sigma_matches_the_spread_over_independent_draws(self) -> None:
        estimates = []
        for seed in range(30):
            d = _oracle(n=6000, noise=150.0, seed=seed)
            estimates.append(
                rlearner.estimate(
                    d["y"], d["m"], d["after"], d["e"], power_edges=_EDGES, rated_power_kw=_RATED, min_records=10
                ).uplift
            )
        sigma = float(self._sigmas(_oracle(n=6000, noise=150.0)).loc[6.0, "sigma"])
        assert sigma == pytest.approx(np.std(estimates, ddof=1), rel=0.5)

    def test_one_block_has_no_sigma(self) -> None:
        sigmas = self._sigmas(_oracle(n=600), block_hours=[6.0, 1000.0])
        assert np.isnan(sigmas.loc[1000.0, "sigma"])
        assert np.isfinite(sigmas.loc[6.0, "sigma"])


class TestPropensity:
    def test_auc_and_overlap_of_an_uninformative_propensity(self) -> None:
        d = _oracle()
        assert rlearner.propensity_overlap(d["after"], d["e"]) == pytest.approx(1.0)
        assert rlearner.propensity_auc(d["after"], np.random.default_rng(0).uniform(size=len(d["e"]))) == pytest.approx(
            0.5, abs=0.02
        )

    def test_auc_is_read_within_folds(self) -> None:
        # Each fold's propensities are level with each other and not with the other fold's.
        after = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=bool)
        e = np.array([0.1, 0.1, 0.2, 0.2, 0.6, 0.6, 0.7, 0.7])
        folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        assert rlearner.propensity_auc(after, e) == pytest.approx(0.75)
        assert rlearner.propensity_auc(after, e, folds=folds) == pytest.approx(1.0)
        assert np.isnan(rlearner.propensity_auc(after, e, folds=after.astype(int)))

    def test_auc_and_overlap_of_a_certain_propensity(self) -> None:
        d = _oracle()
        certain = np.where(d["after"], 0.99, 0.01)
        assert rlearner.propensity_auc(d["after"], certain) == pytest.approx(1.0)
        assert rlearner.propensity_overlap(d["after"], certain) < 0.01


class TestRLearnerHeadline:
    """``headline_estimator="rlearner"`` on ``PowerModelMethod``."""

    def test_recovers_a_known_uplift(self) -> None:
        mi, _ = _prepost_case(uplift=0.05)
        out = _fundamentals_method(model_params=_FAST_PARAMS, headline_estimator="rlearner").estimate(mi)
        assert out.p50_overall == pytest.approx(0.05, abs=0.01)

    def test_a_placebo_reads_near_zero(self) -> None:
        mi, _ = _prepost_case(uplift=0.0)
        out = _fundamentals_method(model_params=_FAST_PARAMS, headline_estimator="rlearner").estimate(mi)
        assert abs(out.p50_overall) < 0.01

    def test_records_the_forward_reading_and_the_propensity(self) -> None:
        mi = _long_case(before_months=4, after_months=4)
        forward = _fundamentals_method(
            model_params=_FAST_PARAMS, headline_estimator="forward", training_period="before"
        ).estimate(mi)
        out = _fundamentals_method(model_params=_FAST_PARAMS, headline_estimator="rlearner").estimate(mi)
        assert out.uncertainty_diagnostics is not None
        row = out.uncertainty_diagnostics.iloc[0]
        assert set(RLEARNER_COLUMNS) <= set(row.index)
        assert row["forward_uplift"] == pytest.approx(forward.p50_overall)
        # A held-out record's propensity comes from the other folds, so on this toy's periodic direction it
        # can lean against the record's own period; the estimate does not depend on which way it leans.
        assert 0.0 <= row["propensity_auc"] <= 1.0
        assert row["propensity_overlap"] > 0.5
        assert out.p50_overall == pytest.approx(0.05, abs=0.01)

    def test_the_sigma_joins_the_bootstrap_and_the_windows_inside_the_periods(self) -> None:
        mi, _ = _prepost_case(uplift=0.05)
        out = _fundamentals_method(model_params=_FAST_PARAMS, headline_estimator="rlearner", n_resamples=200).estimate(
            mi
        )
        assert out.uncertainty_diagnostics is not None
        row = out.uncertainty_diagnostics.iloc[0]
        assert row["sigma_block_24h"] > 0
        assert row["n_holdout_blocks"] >= 1
        assert out.sigma_overall == pytest.approx(np.hypot(row["sigma_block_24h"], row["sigma_holdout"]))

    def test_neither_period_is_exchanged(self) -> None:
        mi, _ = _prepost_case(uplift=0.05)
        before = _fundamentals_method(
            model_params=_FAST_PARAMS, headline_estimator="rlearner", training_period="before"
        ).estimate(mi)
        after = _fundamentals_method(
            model_params=_FAST_PARAMS, headline_estimator="rlearner", training_period="after"
        ).estimate(mi)
        assert after.p50_overall == pytest.approx(before.p50_overall)
        assert after.uncertainty_diagnostics is not None
        assert after.uncertainty_diagnostics.iloc[0]["training_after"] == 1.0

    def test_the_references_are_read_by_the_rlearner_too(self) -> None:
        mi, _ = _screen_case(step=0.0)
        method = _screen_method(headline_estimator="rlearner", reference_screen=False, n_resamples=50)
        out = method.estimate(mi)
        assert out.reference_uplifts is not None
        assert "propensity_auc" in out.reference_uplifts.columns
        assert out.reference_uplifts["propensity_auc"].notna().all()
        # no reference correction is made
        assert out.uncertainty_diagnostics is not None
        assert "reference_correction" not in out.uncertainty_diagnostics.columns

    def test_the_uplift_distributions_combine_to_the_headline(self) -> None:
        mi = _shape_case(uplift=0.05)
        out = _shape_method(headline_estimator="rlearner", report_reference_uplifts=False).estimate(mi)
        assert out.p50_by_condition is not None
        assert out.labeled_rows is not None
        power = out.p50_by_condition[out.p50_by_condition["condition"] == "predicted_power"]
        assert power["p50_uplift"].notna().any()
        assert out.p50_overall == pytest.approx(0.05, abs=0.015)

    def test_a_toggle_estimate_keeps_the_forward_ratio(self) -> None:
        mi = _uncertainty_case(toggle=True)
        assert isinstance(mi.upgrade_timing, ToggleSchedule)
        forward = _fundamentals_method(model_params=_FAST_PARAMS, headline_estimator="forward").estimate(mi)
        out = _fundamentals_method(model_params=_FAST_PARAMS, headline_estimator="rlearner").estimate(mi)
        assert out.p50_overall == pytest.approx(forward.p50_overall)
        assert out.uncertainty_diagnostics is not None
        assert "propensity_auc" not in out.uncertainty_diagnostics.columns

    def test_an_unknown_estimator_is_still_refused(self) -> None:
        mi, _ = _prepost_case()
        with pytest.raises(MethodConfigurationError):
            _fundamentals_method(headline_estimator="r-learner").estimate(mi)
