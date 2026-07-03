"""Tests for the shared condition bins and the binned energy-ratio reducer."""

from __future__ import annotations

import numpy as np

from benchmarking.harness.conditions import (
    CONDITION_BINS,
    CONDITIONS,
    TI_BINS,
    WS_BINS,
    energy_ratio_by_bin,
)


def test_bin_edges_have_expected_width() -> None:
    assert WS_BINS[0] == 0.0
    assert WS_BINS[-1] == 26.0
    assert np.allclose(np.diff(WS_BINS), 2.0)
    assert TI_BINS[0] == 0.0
    assert TI_BINS[-1] == 0.5
    assert np.allclose(np.diff(TI_BINS), 0.05)
    assert CONDITIONS == ("ws", "ti")
    assert CONDITION_BINS == {"ws": WS_BINS, "ti": TI_BINS}


def test_energy_ratio_by_bin_computes_per_bin_ratio() -> None:
    # two rows in (4,6], two in (6,8]; counterfactual constant so uplift = mean(actual)/cf - 1
    cond = np.array([5.0, 5.0, 7.0, 7.0])
    actual = np.array([110.0, 90.0, 150.0, 150.0])
    counterfactual = np.array([100.0, 100.0, 100.0, 100.0])
    out = energy_ratio_by_bin(cond, actual, counterfactual, bins=WS_BINS)
    by = out.set_index("condition_bin")
    assert by.loc["(4.0, 6.0]", "p50_uplift"] == 0.0  # (110+90)/200 - 1
    assert by.loc["(6.0, 8.0]", "p50_uplift"] == 0.5  # 300/200 - 1
    assert by.loc["(4.0, 6.0]", "n_records"] == 2


def test_energy_ratio_by_bin_is_nan_safe_and_covers_all_bins() -> None:
    cond = np.array([5.0, np.nan, 7.0])
    actual = np.array([100.0, 50.0, np.nan])
    counterfactual = np.array([100.0, 50.0, 100.0])
    out = energy_ratio_by_bin(cond, actual, counterfactual, bins=WS_BINS)
    # one row per bin edge interval, no warnings, empty bins -> NaN uplift
    assert len(out) == len(WS_BINS) - 1
    assert out["condition_bin"].is_unique
    assert out.loc[out["condition_bin"] == "(6.0, 8.0]", "p50_uplift"].isna().all()


def test_energy_ratio_by_bin_exposes_per_bin_sums() -> None:
    # the per-bin actual/counterfactual energy sums are needed to re-level a decomposition to an overall
    cond = np.array([5.0, 5.0, 7.0])
    actual = np.array([110.0, 90.0, 150.0])
    counterfactual = np.array([100.0, 100.0, 100.0])
    by = energy_ratio_by_bin(cond, actual, counterfactual, bins=WS_BINS).set_index("condition_bin")
    assert by.loc["(4.0, 6.0]", "sum_actual"] == 200.0
    assert by.loc["(4.0, 6.0]", "sum_counterfactual"] == 200.0
    assert by.loc["(6.0, 8.0]", "sum_actual"] == 150.0
    # empty bins carry a zero energy sum (they contribute nothing to an aggregation)
    assert by.loc["(8.0, 10.0]", "sum_actual"] == 0.0
