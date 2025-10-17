from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from scipy.stats import circmean

from wind_up.circular_math import circ_diff, circ_median

test_circ_diff_data = [
    (0, 0, 0),
    (2, 1, 1),
    (359, 1, -2),
    (90, 270, -180),
    (90, 270.1, 179.9),
    ([1, 90, 90], [-1, 270, 270.1], [2, -180, 179.9]),
    (pd.Series([1, 90, 90]), pd.Series([-1, 270, 270.1]), pd.Series([2, -180, 179.9])),
    (pd.Series([1, 359.1, 2.1]), 1, pd.Series([0, -1.9, 1.1])),
    (1, pd.Series([1, 359.1, 2.1]), pd.Series([0, 1.9, -1.1])),
]


@pytest.mark.parametrize(("angle1", "angle2", "expected"), test_circ_diff_data)
def test_circ_diff(angle1: float | np.generic, angle2: float | np.generic, expected: float | np.generic) -> None:
    if isinstance(expected, pd.Series):
        assert_series_equal(circ_diff(angle1, angle2), (expected))
        assert_series_equal(circ_diff(angle1 - 360, angle2 + 360), (expected))
    else:
        assert circ_diff(angle1, angle2) == pytest.approx(expected)


def test_within_bin() -> None:
    # this test replicates logic in detrend.py where an older version of circ_diff gets the wrong result
    d = 242
    dir_bin_width = 10.0
    assert not np.abs(circ_diff(d - 5, d)) < dir_bin_width / 2


def circ_median_exact(angles: np.ndarray | list) -> float:
    """Exact circular median using O(n²) approach for testing.

    In case of a tie, returns the circular mean of the tied angles.
    """

    angles = np.asarray(angles).flatten()
    angles = angles[~np.isnan(angles)]

    if len(angles) == 0:
        return np.nan

    angles = np.mod(angles, 360)

    def _sum_circ_dist(candidate: float) -> float:
        diffs = circ_diff(angles, candidate)
        return np.sum(np.abs(diffs))

    distances = np.array([_sum_circ_dist(angle) for angle in angles])
    min_distance = np.min(distances)

    # Find all angles that have the minimum distance (handle ties)
    tied_indices = np.where(distances == min_distance)[0]

    if len(tied_indices) == 1:
        return angles[tied_indices[0]]
    # Return circular mean of tied angles
    tied_angles = angles[tied_indices]
    return circmean(tied_angles, high=360, low=0) % 360


test_circ_median_data = [
    # Simple cases
    ([0], 0, True),
    ([0, 20], 10, True),
    ([0, 15, 20], 15, True),
    ([350, 0, 15], 0, True),
    ([170, 181, 190], 181, True),
    # Edge cases around 0/360 boundary
    ([355, 0, 15], 0, True),
    ([350, 351, 9, 10], 0, True),
    # Symmetric cases
    ([0, 90, 180, 270], None, True),  # Any answer is valid due to symmetry
    ([0, 120, 240], None, True),
    # Single value
    ([42], 42, True),
    # Two values
    ([10, 20], None, True),  # Either 10 or 20 is valid
    ([350, 10], None, True),  # Could be either
    # Larger datasets
    (list(range(10, 351, 1)), 180, True),  # Should be near middle
    ([i % 360 for i in range(-178, 181, 1)], 1, True),
    # Test range_360=False
    ([170, 180, 190], 180, False),
    ([350, 0, 10], 0, False),
    ([-10, 0, 10], 0, False),
]


@pytest.mark.parametrize(("angles", "expected", "range_360"), test_circ_median_data)
def test_circ_median(angles: list, *, expected: float | None, range_360: bool) -> None:
    result = circ_median(angles, range_360=range_360)
    if expected is None:
        assert not np.isnan(result)
        return

    exact_result = circ_median_exact(angles)
    exact_result = np.mod(exact_result, 360) if range_360 else np.mod(exact_result + 180, 360) - 180

    # Check that fast and exact methods give similar results
    # Allow for small differences due to approximation (within 10 degrees)
    abs_circ_distance = abs(circ_diff(result, exact_result))
    assert abs_circ_distance < 1e-3, (
        f"Fast method result {result} differs from exact {exact_result} by {abs_circ_distance} degrees"
    )

    expected = np.mod(expected, 360) if range_360 else np.mod(expected + 180, 360) - 180
    abs_circ_distance_expected = abs(circ_diff(result, expected))
    assert abs_circ_distance_expected < 1e-3, (
        f"Result {result} differs from expected {expected} by {abs_circ_distance_expected} degrees"
    )


def test_circ_median_with_series() -> None:
    """Test that pandas Series work correctly."""
    angles = pd.Series([350, 0, 10, 5])
    result = circ_median(angles)
    assert isinstance(result, (float, np.floating))
    assert result == pytest.approx(2.5)


def test_circ_median_with_nan() -> None:
    """Test that NaN values are handled correctly."""
    angles = [0, 10, np.nan, 20]
    result = circ_median(angles)
    assert not np.isnan(result)
    assert result == pytest.approx(10)


def test_circ_median_all_nan() -> None:
    """Test that all NaN returns NaN."""
    angles = [np.nan, np.nan, np.nan]
    result = circ_median(angles)
    assert np.isnan(result)


@pytest.mark.parametrize("range_360", [True, False])
def test_circ_median_groupby(*, range_360: bool) -> None:
    """Test usage with pandas groupby."""
    df = pd.DataFrame({"group": ["A", "A", "A", "B", "B", "B"], "angle": [350, 0, 15, 170, 181, 195]})
    result = df.groupby("group")["angle"].apply(lambda x: circ_median(x, range_360=range_360))

    assert len(result) == 2
    if range_360:
        assert result["A"] == pytest.approx(0, abs=1e-3)
        assert result["B"] == pytest.approx(181, abs=1e-3)
    else:
        assert result["A"] == pytest.approx(0, abs=1e-3)
        assert result["B"] == pytest.approx(-179, abs=1e-3)


def test_circ_median_performance_comparison() -> None:
    """Verify that results are consistent between fast and exact methods on larger dataset."""
    rng = np.random.default_rng(0)
    # Generate lots of angles near 0 degrees
    angles = np.concatenate([rng.normal(4, 20, 500) % 360, rng.normal(354, 20, 500) % 360])

    result_fast = circ_median(angles)
    result_exact = circ_median_exact(angles)

    abs_circ_distance = abs(circ_diff(result_fast, result_exact))
    assert abs_circ_distance < 1e-1, f"Fast and exact methods differ by {abs_circ_distance} degrees on large dataset"


def test_circ_median_range_conversion() -> None:
    """Test that range_360 parameter works correctly."""
    angles = [350, 0, 10]

    result_360 = circ_median(angles, range_360=True)
    assert 0 <= result_360 < 360

    result_180 = circ_median(angles, range_360=False)
    assert -180 <= result_180 < 180

    # They should represent the same angle
    abs_circ_distance = abs(circ_diff(result_360, result_180))
    assert abs_circ_distance < 1e-3
