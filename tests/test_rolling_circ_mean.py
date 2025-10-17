import timeit

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from scipy.stats import circmean

from wind_up.circular_math import rolling_circ_mean


@pytest.mark.parametrize("range_360", [True, False])
def test_rolling_circ_mean(*, range_360: bool) -> None:
    timestamps = pd.date_range(start="2024-01-01", periods=8, freq="600s")
    input_df = pd.DataFrame(
        {
            "wind_direction_1": 2 * [359, 2.1, np.nan, 1],
            "wind_direction_2": 2 * [345, 30, np.nan, np.nan],
            "nacelle_direction": 2 * [359, 3, 358, 1],
        },
        index=timestamps,
    )

    for col in input_df.columns:
        result = rolling_circ_mean(input_df[col], window=4, min_periods=1, center=True, range_360=range_360)
        expected = (
            (
                input_df[col]
                .rolling(window=4, min_periods=1, center=True)
                .apply(lambda x: circmean(x, low=0, high=360, nan_policy="omit"))
            )
            if range_360
            else (
                input_df[col]
                .rolling(window=4, min_periods=1, center=True)
                .apply(lambda x: (circmean(x, low=-180, high=180, nan_policy="omit")))
            )
        )
        assert_series_equal(result, expected)


def test_rolling_circ_mean_all_nans() -> None:
    timestamps = pd.date_range(start="2024-01-01", periods=8, freq="1s")
    input_df = pd.DataFrame(
        {"wind_direction": 8 * [np.nan], "nacelle_direction": 8 * [np.nan]},
        index=timestamps,
    )

    for col in input_df.columns:
        result = rolling_circ_mean(input_df[col], window=4, min_periods=1, center=True)
        expected = (
            input_df[col]
            .rolling(window=4, min_periods=1, center=True)
            .apply(lambda x: circmean(x, low=0, high=360, nan_policy="omit"))
        )
        assert_series_equal(result, expected)


@pytest.mark.slow
def test_rolling_circ_mean_performance() -> None:
    # Generate a large dataset
    n_rows = 10_000
    n_cols = 1
    timestamps = pd.date_range(start="2024-01-01", periods=n_rows, freq="1s")

    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(4, 20, n_rows // 2) % 360, rng.normal(354, 20, n_rows // 2) % 360])
    # Add some NaN values (5% of data)
    nan_rate = 0.05
    nan_mask = rng.random(size=data.shape) < nan_rate
    data[nan_mask] = np.nan

    input_df = pd.DataFrame(data, columns=[f"direction_{i}" for i in range(n_cols)], index=timestamps)

    window_size = 40
    min_periods = 10

    col = "direction_0"

    def new_method() -> float:
        return rolling_circ_mean(input_df[col], window=window_size, min_periods=min_periods, center=True)

    def scipy_method() -> float:
        return (
            input_df[col]
            .rolling(window=window_size, min_periods=min_periods, center=True)
            .apply(lambda x: circmean(x, low=0, high=360, nan_policy="omit"))
        )

    assert_series_equal(new_method(), scipy_method())

    # compare speed of new vs scipy_method
    number_of_runs = 10
    rolling_circ_mean_time = timeit.timeit(new_method, number=number_of_runs)
    apply_scipy_circmean_time = timeit.timeit(
        scipy_method,
        number=number_of_runs,
    )
    minimum_speed_up = 100
    assert rolling_circ_mean_time < apply_scipy_circmean_time / minimum_speed_up, (
        f"New method ({rolling_circ_mean_time:.2f}s) should be at least {minimum_speed_up}x faster than "
        f"old method ({apply_scipy_circmean_time:.2f}s)"
    )
