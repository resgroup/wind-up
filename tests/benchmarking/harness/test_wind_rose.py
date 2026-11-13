"""Tests for the wind rose built from reanalysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.harness.reanalysis import ERA5_WD, ERA5_WD_RAW
from benchmarking.harness.wind_rose import era5_rose, rose_label

INDEX = pd.date_range("2020-01-01", periods=4, freq="1h", tz="UTC")


def test_the_rose_counts_the_hub_height_direction() -> None:
    rose = era5_rose(pd.DataFrame({ERA5_WD_RAW: [10.2, 10.7, 200.0, np.nan]}, index=INDEX))
    assert rose[10] == pytest.approx(2 / 3)
    assert rose[200] == pytest.approx(1 / 3)


def test_the_aligned_alias_is_preferred_and_the_span_cut() -> None:
    frame = pd.DataFrame({ERA5_WD: [90.0, 90.0, 180.0, 180.0], ERA5_WD_RAW: [0.0] * 4}, index=INDEX)
    rose = era5_rose(frame, start=INDEX[1], end=INDEX[3])
    assert rose[90] == pytest.approx(0.5)
    assert rose[180] == pytest.approx(0.5)


def test_a_frame_without_a_direction_is_refused() -> None:
    with pytest.raises(ValueError, match="no 'era5_wd'"):
        era5_rose(pd.DataFrame({"x": [1.0]}, index=INDEX[:1]))


def test_the_label_names_the_source_and_months() -> None:
    frame = pd.DataFrame({ERA5_WD: [1.0, 2.0]}, index=[INDEX[0], pd.Timestamp("2021-03-01", tz="UTC")])
    assert rose_label(frame, source="ERA5 (57.5, -3.1)") == "ERA5 (57.5, -3.1) 2020-01..2021-03"
