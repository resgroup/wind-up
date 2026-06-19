"""Tests for comparison-derived ground-truth uplift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.synthetic.ground_truth import true_uplift
from wind_up.constants import TIMESTAMP_COL, DataColumns


def _paired(
    orig_power: list[float],
    syn_power: list[float],
    *,
    wtg: str = "T01",
    ws: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(orig_power)
    index = pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC")

    def frame(power: list[float]) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                DataColumns.turbine_name: wtg,
                DataColumns.active_power_mean: np.array(power, dtype=float),
                DataColumns.wind_speed_mean: np.array(ws if ws is not None else [8.0] * n, dtype=float),
                DataColumns.wind_speed_sd: 0.8,
            },
            index=index,
        )
        df.index.name = TIMESTAMP_COL
        return df

    return frame(orig_power), frame(syn_power)


def test_overall_uplift_is_energy_ratio_of_synthetic_to_original() -> None:
    """Overall uplift is sum(synthetic)/sum(original) - 1 over the treated records."""
    original, synthetic = _paired([1000.0] * 10, [1050.0] * 10)
    result = true_uplift(synthetic, original, test_wtg="T01")
    assert result.overall == pytest.approx(0.05)


def test_default_mask_is_changed_records_only() -> None:
    """With no mask, untreated (unchanged) records are excluded from the ratio."""
    # first two rows unchanged, last two +10%
    original, synthetic = _paired([1000.0, 1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1100.0, 1100.0])
    result = true_uplift(synthetic, original, test_wtg="T01")
    assert result.overall == pytest.approx(0.10)


def test_uplift_depends_on_record_window() -> None:
    """Passing an explicit mask covering all records dilutes the uplift (record-dependent)."""
    original, synthetic = _paired([1000.0, 1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1100.0, 1100.0])
    full = true_uplift(synthetic, original, test_wtg="T01", mask=np.array([True, True, True, True]))
    assert full.overall == pytest.approx(0.05)


def test_per_condition_breakdown_by_wind_speed() -> None:
    """A by='ws' breakdown reports the true uplift within original wind-speed bins."""
    original, synthetic = _paired(
        [800.0, 800.0, 1500.0, 1500.0],
        [880.0, 880.0, 1530.0, 1530.0],
        ws=[6.0, 6.0, 10.0, 10.0],
    )
    result = true_uplift(synthetic, original, test_wtg="T01", by="ws", bins=[5.0, 8.0, 12.0])
    by_condition = result.by_condition
    assert by_condition is not None
    assert len(by_condition) == 2
    np.testing.assert_allclose(by_condition["true_uplift"].to_numpy(), [0.10, 0.02])
