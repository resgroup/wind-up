"""Tests for the harness campaign-length plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless: no display needed for tests

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from benchmarking.harness.method import MethodOutput
from benchmarking.harness.plots import conditional_estimates, plot_campaign_curves, plot_conditional_uplift

if TYPE_CHECKING:
    from pathlib import Path


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["v0", "v0", "power_model", "power_model"],
            "profile": "p",
            "campaign_months": [3, 6, 3, 6],
            "bias": [0.02, 0.01, 0.0, 0.0],
            "spread": [0.03, 0.02, 0.02, 0.01],
            "score": [0.036, 0.022, 0.02, 0.01],
            "mean_truth": [0.05, 0.05, 0.05, 0.05],
            "mean_estimate": [0.07, 0.06, 0.05, 0.05],
            "n_replicates": 5,
        }
    )


# fig.axes order matches the stacked panels top-to-bottom.
_UPLIFT, _BAND, _SCORE = 0, 1, 2


def test_three_panels() -> None:
    fig = plot_campaign_curves(_summary())
    assert len(fig.axes) == 3


def test_one_line_per_method_in_bias_panel() -> None:
    fig = plot_campaign_curves(_summary())
    _, labels = fig.axes[_BAND].get_legend_handles_labels()
    assert set(labels) == {"v0", "power_model"}


def test_top_panel_shows_true_uplift_and_each_method() -> None:
    fig = plot_campaign_curves(_summary())
    _, labels = fig.axes[_UPLIFT].get_legend_handles_labels()
    assert set(labels) == {"true uplift", "v0", "power_model"}
    true_line = next(line for line in fig.axes[_UPLIFT].get_lines() if line.get_label() == "true uplift")
    assert true_line.get_ydata().tolist() == [5.0, 5.0]  # 0.05 fraction -> 5.0 pp


def test_top_panel_has_spread_band() -> None:
    fig = plot_campaign_curves(_summary())
    # fill_between adds a PolyCollection per method; the band makes the estimate's spread visible.
    assert len(fig.axes[_UPLIFT].collections) >= 1


def test_x_axis_is_campaign_length() -> None:
    fig = plot_campaign_curves(_summary())
    # x-axis is shared; the label lives on the bottom panel.
    assert "campaign" in fig.axes[_SCORE].get_xlabel().lower()


def test_x_axis_starts_at_zero() -> None:
    fig = plot_campaign_curves(_summary())
    assert fig.axes[_SCORE].get_xlim()[0] == 0.0


def test_y_axis_in_percentage_points() -> None:
    fig = plot_campaign_curves(_summary())
    # v0 bias of 0.02/0.01 (fractions) should be plotted as 2.0/1.0 percentage points.
    v0_line = next(line for line in fig.axes[_BAND].get_lines() if line.get_label() == "v0")
    assert v0_line.get_ydata().tolist() == [2.0, 1.0]


def test_score_y_axis_floor_below_zero_so_zero_points_show() -> None:
    summary = _summary()
    summary.loc[summary["method"] == "power_model", "score"] = 0.0  # an oracle-like method at 0
    fig = plot_campaign_curves(summary)
    assert fig.axes[_SCORE].get_ylim()[0] < 0.0


def test_saves_file(tmp_path: Path) -> None:
    save_path = tmp_path / "campaign_curves.png"
    plot_campaign_curves(_summary(), save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_conditional_uplift_writes_png(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "method": ["power_model", "power_model"],
            "condition": ["ws", "ws"],
            "condition_bin": ["(4.0, 6.0]", "(6.0, 8.0]"],
            "mean_estimate": [0.09, 0.05],
            "mean_truth": [0.10, 0.05],
            "bias": [-0.01, 0.0],
            "spread": [0.01, 0.005],
        }
    )
    out = tmp_path / "cond.png"
    fig = plot_conditional_uplift(summary, condition="ws", save_path=out, title="ws_dependent_cp 6mo")
    assert out.exists()
    plt.close(fig)


# --- the estimate-only conditional shape, for a report with no ground truth ---------------------


def _estimate_output() -> MethodOutput:
    """A method output reporting two power bins and nothing else."""
    return MethodOutput(
        p50_overall=0.02,
        p50_by_condition=pd.DataFrame(
            {"condition": "power", "condition_bin": ["(0.0, 0.5]", "(0.5, 1.0]"], "p50_uplift": [0.01, 0.03]}
        ),
    )


def test_conditional_estimates_shapes_an_output_without_truth() -> None:
    frame = conditional_estimates(_estimate_output(), method_name="wind-up")
    assert list(frame.columns) == ["method", "condition", "condition_bin", "mean_estimate"]
    assert "mean_truth" not in frame.columns
    assert frame["mean_estimate"].tolist() == [0.01, 0.03]


def test_conditional_estimates_rejects_an_output_with_no_conditions() -> None:
    with pytest.raises(ValueError, match="p50_by_condition"):
        conditional_estimates(MethodOutput(p50_overall=0.0), method_name="wind-up")


def test_plot_conditional_uplift_draws_estimates_alone_when_there_is_no_truth(tmp_path: Path) -> None:
    out = tmp_path / "estimates.png"
    frame = conditional_estimates(_estimate_output(), method_name="wind-up")
    fig = plot_conditional_uplift(frame, condition="power", save_path=out, title="power")
    assert out.exists()
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert "true uplift" not in labels
    assert "wind-up" in labels
    plt.close(fig)
