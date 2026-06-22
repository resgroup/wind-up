"""Tests for the harness campaign-length plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless: no display needed for tests

import pandas as pd

from benchmarking.harness.plots import plot_campaign_curves

if TYPE_CHECKING:
    from pathlib import Path


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["v0", "v0", "rlearner", "rlearner"],
            "profile": "p",
            "campaign_months": [3, 6, 3, 6],
            "bias": [0.02, 0.01, 0.0, 0.0],
            "spread": [0.03, 0.02, 0.02, 0.01],
            "score": [0.036, 0.022, 0.02, 0.01],
            "n_replicates": 5,
        }
    )


def test_one_score_line_per_method() -> None:
    fig = plot_campaign_curves(_summary())
    ax = fig.axes[0]
    _, labels = ax.get_legend_handles_labels()
    assert set(labels) == {"v0", "rlearner"}


def test_x_axis_is_campaign_length() -> None:
    fig = plot_campaign_curves(_summary())
    assert "campaign" in fig.axes[0].get_xlabel().lower()


def test_saves_file(tmp_path: Path) -> None:
    save_path = tmp_path / "campaign_curves.png"
    plot_campaign_curves(_summary(), save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
