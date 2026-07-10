"""Unit tests for the power model's time-blocked fold assignment (the baseline holdout fit)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarking.baselines.power_model.fitting import time_block_folds


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
