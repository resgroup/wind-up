"""Unit tests for the power model's fitting strategies (Issue 12): folds, calibration, capacity, factories."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.power_model.fitting import (
    IDENTITY_CALIBRATION,
    OUTCOME_MODEL_FACTORIES,
    CalibrationLine,
    early_stopped_n_estimators,
    fit_calibration_line,
    time_block_folds,
)
from benchmarking.baselines.rlearner.nuisance import make_outcome_model


class TestTimeBlockFolds:
    def test_round_robin_contiguous_blocks(self) -> None:
        folds = time_block_folds(1000, n_folds=5, n_blocks=25)
        assert folds.shape == (1000,)
        assert set(folds) == {0, 1, 2, 3, 4}
        # 25 equal blocks of 40 rows, block i -> fold i % 5
        for i in range(25):
            block = folds[i * 40 : (i + 1) * 40]
            assert (block == i % 5).all()

    def test_every_fold_is_a_fifth(self) -> None:
        folds = time_block_folds(10_000, n_folds=5, n_blocks=25)
        counts = np.bincount(folds)
        assert (counts == 2000).all()

    def test_uneven_n_still_covers_all_folds(self) -> None:
        folds = time_block_folds(103, n_folds=5, n_blocks=25)
        assert len(folds) == 103
        assert set(folds) == {0, 1, 2, 3, 4}

    def test_bad_args_raise(self) -> None:
        with pytest.raises(ValueError, match="n_folds"):
            time_block_folds(100, n_folds=1, n_blocks=25)
        with pytest.raises(ValueError, match="n_folds"):
            time_block_folds(100, n_folds=5, n_blocks=3)


class TestCalibrationLine:
    def test_recovers_known_line(self) -> None:
        rng = np.random.default_rng(0)
        pred = rng.uniform(0, 2000, 5000)
        y = 50.0 + 1.25 * pred + rng.normal(0, 10, 5000)  # the model under-dispersed by 1/1.25
        line = fit_calibration_line(y, pred)
        assert line.slope == pytest.approx(1.25, abs=0.01)
        assert line.intercept == pytest.approx(50.0, abs=5.0)
        assert line.apply(np.array([1000.0]))[0] == pytest.approx(50.0 + 1250.0, rel=0.01)

    def test_nan_pairs_ignored(self) -> None:
        rng = np.random.default_rng(1)
        pred = rng.uniform(0, 100, 1000)
        y = 2.0 * pred
        pred[:100] = np.nan
        y[100:200] = np.nan
        line = fit_calibration_line(y, pred)
        assert line.slope == pytest.approx(2.0, abs=1e-6)

    def test_too_few_rows_gives_identity(self) -> None:
        assert fit_calibration_line(np.arange(10.0), np.arange(10.0)) == IDENTITY_CALIBRATION

    def test_degenerate_predictions_give_identity(self) -> None:
        y = np.random.default_rng(2).normal(0, 1, 500)
        assert fit_calibration_line(y, np.full(500, 7.0)) == IDENTITY_CALIBRATION

    def test_identity_apply_is_noop(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(IDENTITY_CALIBRATION.apply(x), x)
        assert CalibrationLine(intercept=1.0, slope=2.0).apply(np.array([3.0]))[0] == 7.0


class TestEarlyStoppedNEstimators:
    def test_simple_signal_stops_well_below_ceiling(self) -> None:
        rng = np.random.default_rng(0)
        n = 2000
        x = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
        y = 3.0 * x["a"].to_numpy() + rng.normal(0, 0.1, n)
        ceiling = 1500
        valid = time_block_folds(n, n_folds=5, n_blocks=25) == 0
        best = early_stopped_n_estimators(
            lambda: make_outcome_model(n_estimators=ceiling, learning_rate=0.1, min_child_samples=20, random_state=0),
            x,
            y,
            valid_mask=valid,
            stopping_rounds=20,
        )
        assert 1 <= best < ceiling


class TestFactories:
    @pytest.mark.parametrize("name", sorted(OUTCOME_MODEL_FACTORIES))
    def test_factory_fits_and_predicts(self, name: str) -> None:
        rng = np.random.default_rng(0)
        n = 1000
        x = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
        x.loc[x.index[:20], "b"] = np.nan  # both factories must tolerate NaN features
        y = 5.0 + 2.0 * x["a"].to_numpy() + rng.normal(0, 0.1, n)
        model = OUTCOME_MODEL_FACTORIES[name](0)
        model.fit(x, y)
        pred = np.asarray(model.predict(x), dtype=float)
        assert np.isfinite(pred).all()
        # both learners should capture a clean linear signal well
        assert float(np.corrcoef(pred, y)[0, 1]) > 0.95
