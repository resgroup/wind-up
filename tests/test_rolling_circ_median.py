import timeit

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from tests.test_math_funcs import circ_median_exact
from wind_up.circular_math import circ_diff, rolling_circ_median_approx


@pytest.mark.parametrize("range_360", [True, False])
def test_rolling_circ_median(*, range_360: bool) -> None:
    timestamps = pd.date_range(start="2024-01-01", periods=8, freq="600s")
    input_df = pd.DataFrame(
        {
            "wind_direction_1": 2 * [359, 2.1, np.nan, 1],
            "wind_direction_2": 2 * [345, 30, np.nan, np.nan],
            "nacelle_direction": 2 * [359, 3, 358, 1],
        },
        index=timestamps,
    )
    window = 4
    min_periods = 3
    for col in input_df.columns:
        result = rolling_circ_median_approx(
            input_df[col], window=window, min_periods=min_periods, center=True, range_360=range_360
        )
        expected = input_df[col].rolling(window=window, min_periods=min_periods, center=True).apply(circ_median_exact)
        if not range_360:
            expected = ((expected + 180) % 360) - 180
        residuals = circ_diff(result, expected)
        if any(~residuals.isna()):
            assert residuals.abs().max() < 1e-1
        assert_series_equal(result, expected, atol=360)  # big atol to avoid wrap error


def test_rolling_circ_median_all_nans() -> None:
    timestamps = pd.date_range(start="2024-01-01", periods=8, freq="1s")
    input_df = pd.DataFrame(
        {"wind_direction": 8 * [np.nan], "nacelle_direction": 8 * [np.nan]},
        index=timestamps,
    )

    for col in input_df.columns:
        result = rolling_circ_median_approx(input_df[col], window=4, min_periods=3, center=True, range_360=True)
        expected = input_df[col].rolling(window=4, min_periods=3, center=True).apply(lambda x: circ_median_exact(x))
        assert_series_equal(result, expected)


@pytest.mark.slow
def test_rolling_circ_median_performance() -> None:
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
    min_periods = 30

    col = "direction_0"

    def new_method() -> float:
        return rolling_circ_median_approx(input_df[col], window=window_size, min_periods=min_periods, center=True)

    def exact_method() -> float:
        return (
            input_df[col]
            .rolling(window=window_size, min_periods=min_periods, center=True)
            .apply(lambda x: circ_median_exact(x))
        )

    new_method_results = new_method()
    exact_method_results = exact_method()
    residuals = circ_diff(new_method_results, exact_method_results)
    # results are not expected to exactly match
    assert abs(residuals.mean()) < 2e-2
    assert residuals.abs().mean() < 3e-1
    assert residuals.abs().max() < 6
    assert_series_equal(new_method_results, exact_method_results, atol=360)  # big atol to avoid wrap error

    # check speed
    number_of_runs = 10
    rolling_circ_median_time = timeit.timeit(new_method, number=number_of_runs)
    apply_exact_time = timeit.timeit(
        exact_method,
        number=number_of_runs,
    )
    minimum_speed_up = 100
    assert rolling_circ_median_time < apply_exact_time / minimum_speed_up, (
        f"New method ({rolling_circ_median_time:.2f}s) should be at least {minimum_speed_up}x faster than "
        f"old method ({apply_exact_time:.2f}s)"
    )
