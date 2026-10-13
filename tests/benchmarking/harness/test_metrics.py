"""Tests for the accuracy/precision/score metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarking.harness.metrics import ErrorSummary, summarize_errors


def test_bias_is_the_mean_signed_error() -> None:
    summary = summarize_errors([1.0, 3.0])
    assert summary.bias == pytest.approx(2.0)


def test_spread_is_the_population_std() -> None:
    # population std of {1, 3} about mean 2 is sqrt(((1-2)^2 + (3-2)^2)/2) = 1.0
    summary = summarize_errors([1.0, 3.0])
    assert summary.spread == pytest.approx(1.0)


def test_score_is_rmse_and_equals_root_bias_sq_plus_spread_sq() -> None:
    errors = [1.0, 3.0]
    summary = summarize_errors(errors)
    expected_rmse = math.sqrt((1.0**2 + 3.0**2) / 2)
    assert summary.score == pytest.approx(expected_rmse)
    assert summary.score == pytest.approx(math.hypot(summary.bias, summary.spread))


def test_single_error_has_zero_spread_and_abs_score() -> None:
    summary = summarize_errors([-0.4])
    assert summary.bias == pytest.approx(-0.4)
    assert summary.spread == pytest.approx(0.0)
    assert summary.score == pytest.approx(0.4)
    assert summary.n == 1


def test_empty_errors_give_nan_summary_with_zero_n() -> None:
    summary = summarize_errors([])
    assert summary.n == 0
    assert math.isnan(summary.bias)
    assert math.isnan(summary.spread)
    assert math.isnan(summary.score)


def test_nan_errors_are_ignored() -> None:
    # a replicate that produced no estimate (NaN) must not poison the summary
    summary = summarize_errors([1.0, np.nan, 3.0])
    assert summary.n == 2
    assert summary.bias == pytest.approx(2.0)


def test_summary_is_a_frozen_dataclass() -> None:
    summary = summarize_errors([1.0, 3.0])
    assert isinstance(summary, ErrorSummary)
    with pytest.raises(AttributeError):
        summary.bias = 0.0  # type: ignore[misc]
