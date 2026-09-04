"""Unit tests for the power model's fitting pieces: fold assignment and the outcome-model factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.power_model.fitting import make_outcome_model, model_safe_features, time_block_folds


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


class TestModelSafeNames:
    """LightGBM rejects JSON-special and non-ASCII feature names; real sources have both."""

    def test_offending_names_are_renamed_positionally(self) -> None:
        frame = pd.DataFrame({"Power, Minimum (kW) @ T01": [1.0], "Nacelle position (°)_sin @ T01": [2.0]})
        safe = model_safe_features(frame)
        assert list(safe.columns) == ["f0", "f1"]

    def test_the_values_and_order_are_untouched(self) -> None:
        """Importances come back positionally, so column order is what carries the original names."""
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        safe = model_safe_features(frame)
        assert safe.to_numpy().tolist() == frame.to_numpy().tolist()

    def test_the_original_frame_is_not_mutated(self) -> None:
        frame = pd.DataFrame({"Power, Minimum (kW)": [1.0]})
        model_safe_features(frame)
        assert list(frame.columns) == ["Power, Minimum (kW)"]

    def test_train_and_predict_frames_get_the_same_names(self) -> None:
        """A mismatch between fit and predict columns is a LightGBM error of its own."""
        train = pd.DataFrame({"°a": [1.0], "b,c": [2.0]})
        predict = pd.DataFrame({"°a": [3.0], "b,c": [4.0]})
        assert list(model_safe_features(train).columns) == list(model_safe_features(predict).columns)

    def test_a_greenbyte_shaped_frame_fits(self) -> None:
        """The Kelmarsh/Penmanshiel column names crashed the road test before this."""
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "Power (kW) @ T01": rng.normal(800, 100, 200),
                "Power, Minimum (kW) @ T01": rng.normal(600, 100, 200),
                "northed_Nacelle position (°)_sin @ T01": rng.normal(0, 1, 200),
            }
        )
        model = make_outcome_model(n_estimators=5, random_state=0)
        model.fit(model_safe_features(frame), rng.normal(1000, 50, 200))
        assert model.predict(model_safe_features(frame)).shape == (200,)
