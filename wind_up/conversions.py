from typing import TypeVar

import pandas as pd

T = TypeVar("T", pd.Timestamp, pd.DatetimeIndex)


def ensure_utc(t: T) -> T:
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")
