"""Read the Princess Amalia (PAWP) per-variable CSVs into wide frames, cached as parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path("F:/princess-amalia-wind-park-pawp")
WORK_DIR = Path("F:/pawp_work")
WIDE_DIR = WORK_DIR / "wide"


def read_wide(variable: str) -> pd.DataFrame:
    """Return ``PAWP_<variable>.csv`` as a float32 frame indexed by UTC timestamp, one column per turbine."""
    cached = WIDE_DIR / f"{variable}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    frame = pd.read_csv(RAW_DIR / f"PAWP_{variable}.csv", na_values=["NaN"])
    index = pd.to_datetime(frame.pop("TimeUTC"), format="%d-%b-%Y %H:%M:%S", utc=True)
    frame = frame.astype("float32").set_index(index)
    frame.index.name = "TimeStamp_EndFormat"
    WIDE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cached)
    return frame
