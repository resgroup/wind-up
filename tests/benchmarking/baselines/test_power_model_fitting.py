"""Unit tests for the power model's fitting strategies (Issue 12): folds, calibration, capacity, factories."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.power_model.fitting import (
    IDENTITY_CALIBRATION,
    OUTCOME_MODEL_FACTORIES,
    CalibrationLine,
    cell_residual_calibration,
    early_stopped_n_estimators,
    fit_calibration_line,
    time_block_folds,
)
from benchmarking.baselines.power_model.matching import cell_codes
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


class TestCellResidualCalibration:
    def test_reads_out_centred_cell_means_under_target_mix(self) -> None:
        # two cells: cell 0 residual mean +10, cell 1 residual mean -10 (global mean 0);
        # the target sits mostly in cell 1, so the correction reads out the differential there
        codes_base = np.array([[0]] * 50 + [[1]] * 50)
        residuals = np.array([10.0] * 50 + [-10.0] * 50)
        codes_target = np.array([[0]] * 2 + [[1]] * 8)
        calib = cell_residual_calibration(codes_baseline=codes_base, residuals=residuals, codes_target=codes_target)
        np.testing.assert_allclose(calib.per_row_residual, [10.0] * 2 + [-10.0] * 8)
        assert calib.global_mean_residual == pytest.approx(0.0)
        assert calib.n_target_unseen == 0
        assert calib.n_cells == 2

    def test_global_level_offset_is_centred_out(self) -> None:
        # the F14 lesson: a uniform OOF level offset (here +5 everywhere) is an artefact of the
        # out-of-fold basis and must not transfer to the correction
        codes_base = np.array([[0]] * 50 + [[1]] * 50)
        residuals = np.array([15.0] * 50 + [-5.0] * 50)  # cell means +15/-5, global +5
        codes_target = np.array([[0], [1]])
        calib = cell_residual_calibration(codes_baseline=codes_base, residuals=residuals, codes_target=codes_target)
        np.testing.assert_allclose(calib.per_row_residual, [10.0, -10.0])
        assert calib.global_mean_residual == pytest.approx(5.0)

    def test_unseen_cell_and_invalid_row_get_zero(self) -> None:
        codes_base = np.array([[0]] * 10)
        residuals = np.full(10, 3.0)
        codes_target = np.array([[0], [5], [-1]])  # seen, unseen, invalid
        calib = cell_residual_calibration(codes_baseline=codes_base, residuals=residuals, codes_target=codes_target)
        np.testing.assert_allclose(calib.per_row_residual, [0.0, 0.0, 0.0])  # single cell = global mean
        assert calib.n_target_unseen == 2

    def test_nan_and_invalid_baseline_rows_only_count_toward_global_mean(self) -> None:
        codes_base = np.array([[0], [0], [-1], [0]])
        residuals = np.array([2.0, 4.0, 100.0, np.nan])  # invalid-cell row only enters the global mean
        codes_target = np.array([[0], [7]])
        calib = cell_residual_calibration(codes_baseline=codes_base, residuals=residuals, codes_target=codes_target)
        global_mean = (2.0 + 4.0 + 100.0) / 3
        assert calib.per_row_residual[0] == pytest.approx(3.0 - global_mean)  # centred cell mean
        assert calib.per_row_residual[1] == pytest.approx(0.0)  # unseen -> no differential info
        assert calib.n_cells == 1

    def test_multi_var_cells_via_cell_codes(self) -> None:
        frame = pd.DataFrame({"a": [0.5, 0.5, 1.5, 1.5], "b": [0.5, 0.5, 0.5, 1.5]})
        edges = {"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0]}
        codes = cell_codes(frame, edges)
        residuals = np.array([1.0, 3.0, 5.0, 7.0])
        calib = cell_residual_calibration(codes_baseline=codes, residuals=residuals, codes_target=codes)
        global_mean = 4.0
        np.testing.assert_allclose(calib.per_row_residual, np.array([2.0, 2.0, 5.0, 7.0]) - global_mean)
        assert calib.n_cells == 3

    def test_correction_integrates_to_zero_under_baseline_mix(self) -> None:
        rng = np.random.default_rng(3)
        codes = rng.integers(0, 4, size=(500, 1))
        residuals = rng.normal(2.0, 1.0, 500) + 3.0 * codes[:, 0]
        calib = cell_residual_calibration(codes_baseline=codes, residuals=residuals, codes_target=codes)
        assert float(calib.per_row_residual.mean()) == pytest.approx(0.0, abs=1e-9)


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
