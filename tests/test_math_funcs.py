from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from wind_up.circular_math import circ_diff

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
    else:
        assert circ_diff(angle1, angle2) == pytest.approx(expected)


def test_within_bin() -> None:
    # this test replicates logic in detrend.py where an older version of circ_diff gets the wrong result
    d = 242
    dir_bin_width = 10.0
    assert not np.abs(circ_diff(d - 5, d)) < dir_bin_width / 2
