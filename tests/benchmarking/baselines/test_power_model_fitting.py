"""Unit tests for the power model's fitting pieces: fold assignment and the outcome-model factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.power_model.fitting import make_outcome_model, time_block_folds


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


class TestMakeOutcomeModel:
    def test_is_l2_regressor_with_default_params(self) -> None:
        params = make_outcome_model().get_params()
        assert params["objective"] == "regression"
        assert params["n_estimators"] == 600
        assert params["min_child_samples"] == 200

    def test_overrides_win(self) -> None:
        params = make_outcome_model(n_estimators=50, random_state=7).get_params()
        assert params["n_estimators"] == 50
        assert params["random_state"] == 7

    def test_fits_and_predicts(self) -> None:
        rng = np.random.default_rng(0)
        x = pd.DataFrame({"f": rng.normal(size=200)})
        y = 3.0 * x["f"]
        model = make_outcome_model(n_estimators=50, verbose=-1).fit(x, y)
        assert model.predict(x).shape == (200,)
