import pandas as pd
import pytest

from wind_up.conversions import ensure_utc

SAMPLE_NAIVE = pd.date_range(pd.Timestamp("2000-01-01T00:00"), periods=2)
SAMPLE_UTC = pd.date_range(pd.Timestamp("2000-01-01T00:00+00:00"), periods=2)
SAMPLE_OTHER = pd.date_range(pd.Timestamp("2000-01-01T05:00+05:00"), periods=2)


@pytest.mark.parametrize(
    ("t", "exp"),
    [
        (pd.DatetimeIndex(SAMPLE_NAIVE), pd.DatetimeIndex(SAMPLE_UTC)),
        (pd.DatetimeIndex(SAMPLE_UTC), pd.DatetimeIndex(SAMPLE_UTC)),
        (pd.DatetimeIndex(SAMPLE_OTHER), pd.DatetimeIndex(SAMPLE_UTC)),
        (pd.Timestamp("2000-01-01T00:00"), pd.Timestamp("2000-01-01T00:00+00:00")),
        (pd.Timestamp("2000-01-01T00:00+00:00"), pd.Timestamp("2000-01-01T00:00+00:00")),
        (pd.Timestamp("2000-01-01T05:00+05:00"), pd.Timestamp("2000-01-01T00:00+00:00")),
    ],
)
def test_ensure_utc(t: pd.Timestamp, exp: pd.Timestamp) -> None:
    actual = ensure_utc(t)

    if isinstance(t, pd.Timestamp):
        assert actual == exp
    else:
        pd.testing.assert_index_equal(actual, exp)
