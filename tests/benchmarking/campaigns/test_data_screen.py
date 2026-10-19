"""Tests for the data screen of method step 9 pass 1."""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Circle

from benchmarking.campaigns.data_screen import (
    DataScreenSettings,
    screen_plan,
    write_data_screen,
)
from benchmarking.campaigns.data_screen_plots import plot_correlation_map, plot_reference_map
from benchmarking.campaigns.plans import plans_for
from benchmarking.harness.operating_state import VALID_UPLIFT_COL
from benchmarking.harness.record_selection import ChangedInterval
from benchmarking.harness.waking import DIRECTION_COL, UPWIND_COL
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule
from tests.benchmarking.campaigns.timeline_fixtures import hourly_scada, staggered_spec, utc

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.campaigns.data_screen import DataScreen
    from benchmarking.campaigns.declaration import CampaignSpec
    from wind_up.analysis_period import AnalysisPlan

POWER = HOT_COLUMNS.active_power
TURBINE = HOT_COLUMNS.turbine


def _scada() -> pd.DataFrame:
    """Hourly SCADA whose power follows one shared signal, every record valid with nothing upwind."""
    scada = hourly_scada()
    rng = np.random.default_rng(0)
    hours = (scada.index - scada.index.min()) / pd.Timedelta(hours=1)
    shared = 900.0 + 500.0 * np.sin(2.0 * np.pi * hours.to_numpy() / 37.0)
    return scada.assign(
        **{
            POWER: shared + rng.normal(0.0, 30.0, len(scada)),
            VALID_UPLIFT_COL: True,
            UPWIND_COL: "none",
            DIRECTION_COL: 225.0,
        }
    )


def _plan(scada: pd.DataFrame) -> AnalysisPlan:
    """T0's plan: power references T1-T4, reserve T5."""
    plan = plans_for(staggered_spec(), scada, columns=HOT_COLUMNS).plans["T0"]
    assert plan.power_references == ("T1", "T2", "T3", "T4")
    assert plan.reserves == ("T5",)
    return plan


def _screen(scada: pd.DataFrame, **kwargs: object) -> DataScreen:
    spec = kwargs.pop("spec", staggered_spec())
    plan = _plan(scada)
    frame = scada[(scada.index >= plan.start) & (scada.index < plan.end)]
    return screen_plan(plan, frame, spec=spec, columns=HOT_COLUMNS, **kwargs)  # type: ignore[arg-type]


def _invalid(scada: pd.DataFrame, turbine: str, *, before: str | None = None) -> pd.DataFrame:
    """Mark ``turbine`` invalid for uplift, everywhere or before a date."""
    rows = scada[TURBINE] == turbine
    if before is not None:
        rows &= scada.index < pd.Timestamp(before, tz="UTC")
    out = scada.copy()
    out.loc[rows, VALID_UPLIFT_COL] = False
    return out


def _row(screen: DataScreen, candidate: str) -> pd.Series:
    table = screen.table.set_index("candidate")
    return table.loc[candidate]


class TestACleanPool:
    def test_every_candidate_passes_and_the_plan_is_unchanged(self) -> None:
        scada = _scada()
        screen = _screen(scada)
        assert screen.plan == _plan(scada)
        assert list(screen.table["verdict"]) == ["kept"] * 4 + ["spare"]
        assert (screen.table["reason"] == "").all()

    def test_the_table_has_one_row_per_eligible_candidate_in_rank_order(self) -> None:
        screen = _screen(_scada())
        assert list(screen.table["candidate"]) == ["T1", "T2", "T3", "T4", "T5"]
        assert list(screen.table["rank"]) == [1, 2, 3, 4, 5]
        assert (screen.table["test_wtg"] == "T0").all()


class TestLowHours:
    def test_the_candidate_is_dropped_and_the_next_passer_refills(self) -> None:
        scada = _invalid(_scada(), "T2", before="2019-04-01")
        screen = _screen(scada)
        assert _row(screen, "T2")["verdict"] == "dropped"
        assert "hours before" in _row(screen, "T2")["reason"]
        assert screen.plan.power_references == ("T1", "T3", "T4", "T5")
        assert screen.plan.reserves == ()

    def test_the_medians_are_over_every_eligible_candidate(self) -> None:
        screen = _screen(_invalid(_scada(), "T2", before="2019-04-01"))
        table = screen.table
        assert (table["median_hours_before"] == np.median(table["hours_before"])).all()
        assert (table["median_hours_after"] == np.median(table["hours_after"])).all()
        assert (table["median_correlation"] == np.median(table["correlation"])).all()


class TestLowCorrelation:
    def test_the_candidate_is_dropped(self) -> None:
        scada = _scada()
        rows = scada[TURBINE] == "T3"
        scada.loc[rows, POWER] = np.random.default_rng(1).uniform(0.0, 2000.0, int(rows.sum()))
        screen = _screen(scada)
        assert _row(screen, "T3")["verdict"] == "dropped"
        assert "correlation" in _row(screen, "T3")["reason"]
        assert screen.plan.power_references == ("T1", "T2", "T4", "T5")

    def test_the_margin_is_a_setting(self) -> None:
        scada = _scada()
        rows = scada[TURBINE] == "T3"
        scada.loc[rows, POWER] = scada.loc[rows, POWER] + np.random.default_rng(1).normal(0.0, 150.0, int(rows.sum()))
        assert _row(_screen(scada), "T3")["verdict"] == "kept"
        strict = _screen(scada, settings=DataScreenSettings(correlation_margin=0.01))
        assert _row(strict, "T3")["verdict"] == "dropped"


class TestNoUsableRecords:
    def test_zero_hours_and_no_correlation_fail(self) -> None:
        screen = _screen(_invalid(_scada(), "T4"))
        row = _row(screen, "T4")
        assert row["hours_before"] == 0
        assert row["hours_after"] == 0
        assert np.isnan(row["correlation"])
        assert row["verdict"] == "dropped"


class TestTooFewPassers:
    def test_a_warning_says_the_pool_rule_is_broken(self, caplog: pytest.LogCaptureFixture) -> None:
        scada = _invalid(_invalid(_scada(), "T4"), "T5")
        with caplog.at_level(logging.WARNING):
            screen = _screen(scada)
        assert screen.plan.power_references == ("T1", "T2", "T3")
        assert "pool rule" in caplog.text


class TestChangedUpwind:
    def test_a_candidate_downwind_of_the_test_turbine_loses_records(self) -> None:
        scada = _scada()
        rows = (scada[TURBINE] == "T1") & (scada.index.month.isin([1, 2, 3, 4, 5, 6, 7, 8]))
        scada.loc[rows, UPWIND_COL] = "T0"
        screen = _screen(scada)
        assert _row(screen, "T1")["verdict"] == "dropped"
        losses = screen.losses.set_index(["candidate", "reason"])["records"]
        assert losses["T1", "changed upwind of reference"] > 0


def _far_spec() -> CampaignSpec:
    """The staggered campaign with T5 also worked far before T0's changeover."""
    spec = staggered_spec()
    works = {**spec.works, "T5": [(utc("2018-03-01"), utc("2018-03-05"))]}
    return dataclasses.replace(spec, works=works)


class TestAFarChange:
    def test_it_costs_only_the_records_on_its_far_side(self) -> None:
        scada = _scada()
        scada.loc[scada[TURBINE] == "T1", UPWIND_COL] = "T5"
        screen = _screen(scada, spec=_far_spec())
        assert screen.changed["T5"] == (ChangedInterval(None, utc("2018-03-05")),)
        frame = scada[(scada.index >= screen.plan.start) & (scada.index < screen.plan.end)]
        lost = frame[(frame[TURBINE] == "T1") & (frame.index < utc("2018-03-05"))]
        losses = screen.losses.set_index(["candidate", "reason"])["records"]
        assert losses["T1", "changed upwind of reference"] == len(lost)
        assert _row(screen, "T1")["verdict"] == "kept"


class TestToggle:
    def test_before_and_after_are_off_and_on(self) -> None:
        scada = _scada()
        plan = _plan(scada)
        spec = dataclasses.replace(
            staggered_spec(),
            upgrade_timing=ToggleSchedule(period=pd.Timedelta(days=2), start=plan.works[1]),
            analysis_period=(plan.start, plan.end),
        )
        prepost = _row(_screen(scada), "T1")
        toggle = _row(_screen(scada, spec=spec), "T1")
        assert toggle["hours_after"] == pytest.approx(prepost["hours_after"] / 2, rel=0.02)
        assert toggle["hours_before"] == pytest.approx(prepost["hours_before"] + prepost["hours_after"] / 2, rel=0.02)


class TestReferenceHours:
    def test_one_row_per_reference_count_over_the_screened_order(self) -> None:
        screen = _screen(_invalid(_scada(), "T2", before="2019-04-01"))
        hours = screen.hours
        assert list(hours["n_references"]) == [1, 2, 3, 4]
        assert list(hours["references"]) == ["T1", "T1; T3", "T1; T3; T4", "T1; T3; T4; T5"]

    def test_joint_hours_never_grow_with_more_references(self) -> None:
        scada = _invalid(_scada(), "T3", before="2018-03-01")
        hours = _screen(scada).hours
        assert (np.diff(hours["hours_before"]) <= 0).all()


class TestOutputs:
    def test_the_tables_and_plots_are_written(self, tmp_path: Path) -> None:
        scada = _invalid(_scada(), "T2", before="2019-04-01")
        screen = _screen(scada, spec=_far_spec())
        write_data_screen(
            {"T0": screen}, layout=staggered_spec().layout, scada_df=scada, turbine_col=TURBINE, out_dir=tmp_path
        )
        names = {p.name for p in tmp_path.iterdir()}
        assert {
            "reference_data.csv",
            "reference_hours.csv",
            "data_screen_T0.png",
            "record_losses_T0.png",
            "reference_hours_T0.png",
            "reference_map_T0.png",
            "correlation_map_T0.png",
        } <= names
        table = pd.read_csv(tmp_path / "reference_data.csv")
        assert list(table.columns) == [
            "test_wtg",
            "candidate",
            "rank",
            "distance_d",
            "hours_before",
            "hours_after",
            "correlation",
            "median_hours_before",
            "median_hours_after",
            "median_correlation",
            "verdict",
            "reason",
        ]


def _legend_dashes(fig: plt.Figure) -> list[bool]:
    """Whether each ring in the map's legend is dashed."""
    legend = fig.axes[0].get_legend()
    return [ring.get_linestyle() == "--" for ring in legend.findobj(Circle)]


class TestTheMaps:
    def test_the_correlation_map_shows_a_dashed_ring_for_dropped(self) -> None:
        scada = _invalid(_scada(), "T2", before="2019-04-01")
        fig = plot_correlation_map(_screen(scada), layout=staggered_spec().layout)
        assert any(_legend_dashes(fig))
        assert fig.axes[0].get_title().endswith("for T0")
        plt.close(fig)

    def test_the_reference_map_shows_a_dashed_ring_and_the_span(self) -> None:
        scada = _scada()
        screen = _screen(scada, spec=_far_spec())
        fig = plot_reference_map(screen, layout=staggered_spec().layout, scada_df=scada, turbine_col=TURBINE)
        assert any(_legend_dashes(fig))
        title = fig.axes[0].get_title()
        assert f"{screen.plan.start:%Y-%m-%d}" in title
        assert f"{screen.plan.end:%Y-%m-%d}" in title
        plt.close(fig)
