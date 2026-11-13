"""The site's wind rose from reanalysis, for ranking references and designing campaigns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.harness.reanalysis import ERA5_WD, ERA5_WD_RAW
from wind_up.wake_reach import wind_rose

if TYPE_CHECKING:
    from wind_up.wake_reach import Rose


def era5_rose(era5_df: pd.DataFrame, *, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> Rose:
    """Return the rose of the reanalysis hub-height direction over ``[start, end)``.

    :param era5_df: hourly or aligned reanalysis carrying :data:`ERA5_WD` or :data:`ERA5_WD_RAW`
    :param start: the first timestamp to count; the frame's start by default
    :param end: the end, exclusive; the frame's end by default
    """
    column = ERA5_WD if ERA5_WD in era5_df.columns else ERA5_WD_RAW
    if column not in era5_df.columns:
        msg = f"the reanalysis has no {ERA5_WD!r} or {ERA5_WD_RAW!r} column to build a wind rose from"
        raise ValueError(msg)
    index = pd.DatetimeIndex(era5_df.index)
    keep = np.ones(len(index), dtype=bool)
    if start is not None:
        keep &= index >= start
    if end is not None:
        keep &= index < end
    return wind_rose(era5_df.loc[keep, column].to_numpy(dtype=float))


def rose_label(era5_df: pd.DataFrame, *, source: str) -> str:
    """Return how a rose built from ``era5_df`` is described in the reports, e.g. ``"ERA5 2018-01..2020-12"``."""
    index = pd.DatetimeIndex(era5_df.index)
    return f"{source} {index.min():%Y-%m}..{index.max():%Y-%m}"
