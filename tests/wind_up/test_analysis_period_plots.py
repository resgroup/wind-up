"""Smoke tests for the analysis timeline plot: it must draw a plan, draw an unplanned turbine, and save."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from tests.wind_up.layouts import line_layout
from wind_up.analysis_period import MONTH, PlanSettings, plan_analysis
from wind_up.analysis_period_plots import plot_analysis_timeline
from wind_up.layout import Layout

if TYPE_CHECKING:
    from pathlib import Path

ROTOR_D = 100.0


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def _layout(n: int = 5) -> Layout:
    return Layout.from_frame(line_layout([i * 3.0 * ROTOR_D for i in range(n)], rotor_diameter_m=ROTOR_D))


def names() -> list[str]:
    return [f"T{i}" for i in range(5)]


def _inputs() -> dict:
    return {
        "works": {
            "T0": [(utc("2020-01-01"), utc("2020-01-08"))],
            "T3": [(utc("2020-10-01"), utc("2020-10-15"))],
        },
        "exclusions": [(None, utc("2019-03-01"), utc("2019-04-01")), ("T1", utc("2020-05-01"), utc("2020-06-01"))],
        "extents": {n: (utc("2018-01-01"), utc("2021-01-01")) for n in names()},
    }


class TestPlotAnalysisTimeline:
    def test_draws_and_saves_a_plan(self, tmp_path: Path) -> None:
        layout, inputs = _layout(), _inputs()
        plan = plan_analysis(
            layout, turbine="T0", candidates=names(), settings=PlanSettings(min_side=1 * MONTH), **inputs
        )
        figure = plot_analysis_timeline("T0", layout=layout, plan=plan, out_dir=tmp_path, **inputs)
        try:
            labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]
            assert labels[0].startswith("T0")
            assert len(labels) == 5
            assert "chosen" in figure.axes[0].get_title()
            assert (tmp_path / "analysis_timeline_T0.png").exists()
        finally:
            plt.close(figure)

    def test_draws_an_unplanned_turbine_with_its_reason(self) -> None:
        layout, inputs = _layout(), _inputs()
        figure = plot_analysis_timeline(
            "T0", layout=layout, plan=None, reason="no span gives both sides at least 3 months", **inputs
        )
        try:
            assert "not analysed" in figure.axes[0].get_title()
            assert "3 months" in figure.axes[0].get_title()
        finally:
            plt.close(figure)

    def test_rows_are_nearest_first_after_the_test_turbine(self) -> None:
        layout, inputs = _layout(), _inputs()
        figure = plot_analysis_timeline("T2", layout=layout, plan=None, reason="", **inputs)
        try:
            labels = [t.get_text().split()[0] for t in figure.axes[0].get_yticklabels()]
            assert labels[0] == "T2"
            assert labels[1:3] in (["T1", "T3"], ["T3", "T1"])
        finally:
            plt.close(figure)
