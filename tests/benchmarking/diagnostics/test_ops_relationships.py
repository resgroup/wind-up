"""Operating relationships over time: binned means of one signal against another, month by month."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.diagnostics.context import DiagnosticContext
from benchmarking.diagnostics.input_data import write_input_data_plots
from benchmarking.diagnostics.ops_relationships import (
    PAIRS,
    monthly_binned_means,
    operating_mask,
    plot_ops_relationships,
    plot_run_ops_relationships,
)
from benchmarking.synthetic import ColumnSchema

if TYPE_CHECKING:
    from pathlib import Path

TIMEBASE = pd.Timedelta(minutes=10)

COLUMNS = ColumnSchema(
    turbine="turbine",
    active_power="power",
    wind_speed="ws",
    wind_speed_sd="ws_sd",
    gen_rpm="rpm",
    pitch="pitch",
    reactive_power="reactive",
    availability="avail",
)
NO_PITCH_OR_REACTIVE = ColumnSchema(
    turbine="turbine", active_power="power", wind_speed="ws", wind_speed_sd="ws_sd", gen_rpm="rpm", availability="avail"
)


def _series(values: np.ndarray, index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(values, index=index, dtype=float)


def _turbine_rows(index: pd.DatetimeIndex, *, turbine: str, rng: np.random.Generator) -> pd.DataFrame:
    ws = rng.uniform(3, 18, len(index))
    power = np.clip(0.5 * ws**3, 0, 2300)
    return pd.DataFrame(
        {
            "turbine": turbine,
            "power": power,
            "ws": ws,
            "ws_sd": 1.0,
            "rpm": np.clip(ws * 90, 0, 1600),
            "pitch": np.clip(ws - 12, 0, None),
            "reactive": rng.normal(0, 50, len(index)),
            "avail": 600.0,
        },
        index=index,
    )


def _farm(turbines: list[str], *, months: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=months * 30 * 144, freq="10min", tz="UTC")
    return pd.concat([_turbine_rows(index, turbine=t, rng=rng) for t in turbines])


class TestMonthlyBinnedMeans:
    """The tidy frame the plots draw: one row per calendar month and bin of x."""

    @pytest.fixture
    def two_months(self) -> tuple[pd.Series, pd.Series]:
        index = pd.date_range("2020-01-01", "2020-03-01", freq="10min", tz="UTC", inclusive="left")
        x = _series(np.tile(np.arange(10.0), len(index) // 10 + 1)[: len(index)], index)
        return x, 2.0 * x

    def test_one_row_per_month_and_bin(self, two_months: tuple[pd.Series, pd.Series]) -> None:
        x, y = two_months
        result = monthly_binned_means(x, y, timebase=TIMEBASE, n_bins=10)
        assert len(result) == 2 * 10
        assert list(result.columns) == ["month", "bin", "x_lo", "x_hi", "x_mean", "y_mean", "hours"]

    def test_the_bin_edges_do_not_move_between_months(self) -> None:
        # x spans 0..10 in January and 10..20 in February, the same number of records each: fixed
        # edges put each month in its own bins, where per-month edges would share them all
        jan = pd.date_range("2020-01-01", "2020-01-29", freq="10min", tz="UTC", inclusive="left")
        feb = pd.date_range("2020-02-01", "2020-02-29", freq="10min", tz="UTC", inclusive="left")
        index = jan.append(feb)
        rng = np.random.default_rng(2)
        x = _series(np.where(index.month == 1, 0.0, 10.0) + rng.uniform(0, 10, len(index)), index)
        result = monthly_binned_means(x, x, timebase=TIMEBASE, n_bins=4)
        jan = result[result["month"].dt.month == 1]
        feb = result[result["month"].dt.month == 2]
        assert set(jan["bin"]).isdisjoint(set(feb["bin"]))

    def test_a_cell_under_the_minimum_hours_is_dropped(self) -> None:
        index = pd.date_range("2020-01-01", "2020-03-01", freq="10min", tz="UTC", inclusive="left")
        x = _series(np.zeros(len(index)), index)
        x[(index.month == 2)] = 1.0
        # February keeps five hours of data only
        keep = (index.month == 1) | (index < pd.Timestamp("2020-02-01 05:00", tz="UTC"))
        result = monthly_binned_means(x[keep], x[keep], timebase=TIMEBASE, n_bins=2, min_hours=6.0)
        assert list(result["month"].dt.month) == [1]

    def test_hours_count_the_records_behind_each_mean(self, two_months: tuple[pd.Series, pd.Series]) -> None:
        x, y = two_months
        result = monthly_binned_means(x, y, timebase=TIMEBASE, n_bins=10)
        assert result["hours"].sum() == pytest.approx(len(x) * TIMEBASE / pd.Timedelta(hours=1))

    def test_a_step_change_shows_in_that_bins_series(self) -> None:
        index = pd.date_range("2020-01-01", "2020-07-01", freq="10min", tz="UTC", inclusive="left")
        rng = np.random.default_rng(1)
        x = _series(rng.uniform(0, 10, len(index)), index)
        y = 2.0 * x + np.where((index >= pd.Timestamp("2020-04-01", tz="UTC")) & (x > 8), 5.0, 0.0)
        result = monthly_binned_means(x, y, timebase=TIMEBASE, n_bins=5)
        top = result[result["bin"] == result["bin"].max()].set_index("month")["y_mean"]
        low = result[result["bin"] == 0].set_index("month")["y_mean"]
        assert top.iloc[-1] - top.iloc[0] > 2.0
        assert abs(low.iloc[-1] - low.iloc[0]) < 0.5

    def test_missing_values_are_ignored(self, two_months: tuple[pd.Series, pd.Series]) -> None:
        x, y = two_months
        y = y.copy()
        y.iloc[::2] = np.nan
        result = monthly_binned_means(x, y, timebase=TIMEBASE, n_bins=10)
        assert result["y_mean"].notna().all()

    def test_no_data_gives_an_empty_frame(self) -> None:
        empty = pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
        assert monthly_binned_means(empty, empty, timebase=TIMEBASE).empty


class TestOperatingMask:
    """Only producing, fully available records are binned."""

    def test_it_keeps_producing_fully_available_rows(self) -> None:
        index = pd.date_range("2020-01-01", periods=4, freq="10min", tz="UTC")
        rows = pd.DataFrame({"power": [100.0, 0.0, -5.0, 100.0], "avail": [600.0, 600.0, 600.0, 300.0]}, index=index)
        assert list(operating_mask(rows, columns=COLUMNS, timebase=TIMEBASE)) == [True, False, False, False]

    def test_missing_availability_is_not_available(self) -> None:
        index = pd.date_range("2020-01-01", periods=2, freq="10min", tz="UTC")
        rows = pd.DataFrame({"power": [100.0, 100.0], "avail": [600.0, np.nan]}, index=index)
        assert list(operating_mask(rows, columns=COLUMNS, timebase=TIMEBASE)) == [True, False]


class TestTheFigure:
    def test_it_is_written(self, tmp_path: Path) -> None:
        rows = _farm(["T1"])
        path = plot_ops_relationships(rows, turbine="T1", columns=COLUMNS, timebase=TIMEBASE, out_dir=tmp_path)
        assert path == tmp_path / "ops_relationships_T1.png"
        assert path.exists()

    def test_a_pair_with_an_unset_signal_is_skipped(self, tmp_path: Path) -> None:
        columns = NO_PITCH_OR_REACTIVE
        path = plot_ops_relationships(_farm(["T1"]), turbine="T1", columns=columns, timebase=TIMEBASE, out_dir=tmp_path)
        assert path is not None

    def test_nothing_is_written_without_producing_rows(self, tmp_path: Path) -> None:
        rows = _farm(["T1"]).assign(power=0.0)
        assert plot_ops_relationships(rows, turbine="T1", columns=COLUMNS, timebase=TIMEBASE, out_dir=tmp_path) is None
        assert not list(tmp_path.iterdir())

    def test_it_covers_the_five_relationships(self) -> None:
        assert [(x, y) for x, y in PAIRS] == [
            ("wind_speed", "active_power"),
            ("active_power", "gen_rpm"),
            ("active_power", "pitch"),
            ("wind_speed", "gen_rpm"),
            ("wind_speed", "pitch"),
        ]


def _context(tmp_path: Path, *, power_references: list[str] | None) -> DiagnosticContext:
    scada = _farm(["T1", "T2", "T3", "T4"], months=2)
    index = pd.DatetimeIndex(pd.unique(scada.index)).sort_values()
    return DiagnosticContext(
        run_dir=tmp_path / "run",
        test_wtg="T1",
        turbine_col="turbine",
        columns=COLUMNS,
        scada_df=scada,
        treated_ts=np.asarray(index >= index[len(index) // 2]),
        used_ts=np.ones(len(index), dtype=bool),
        timebase=TIMEBASE,
        mode="prepost",
        power_references=power_references,
    )


class TestTheRunLevelPlots:
    """Inside a run: the test turbine and its power references, over the span the method sees."""

    def test_the_test_turbine_and_its_power_references(self, tmp_path: Path) -> None:
        paths = plot_run_ops_relationships(_context(tmp_path, power_references=["T3", "T2"]))
        assert sorted(p.name for p in paths) == [
            "ops_relationships_T1.png",
            "ops_relationships_T2.png",
            "ops_relationships_T3.png",
        ]
        assert {p.parent.name for p in paths} == {"1_inputs"}

    def test_every_other_turbine_when_the_method_names_no_power_references(self, tmp_path: Path) -> None:
        paths = plot_run_ops_relationships(_context(tmp_path, power_references=None))
        assert len(paths) == 4


class TestTheFarmWidePlots:
    """Before planning: every turbine, every record provided."""

    def test_every_turbine_gets_a_figure(self, tmp_path: Path) -> None:
        paths = write_input_data_plots(_farm(["T1", "T2", "T3"]), columns=COLUMNS, out_dir=tmp_path)
        names = {p.name for p in paths}
        assert {"ops_relationships_T1.png", "ops_relationships_T2.png", "ops_relationships_T3.png"} <= names

    def test_power_factor_and_coverage_cover_the_farm(self, tmp_path: Path) -> None:
        paths = write_input_data_plots(_farm(["T1", "T2"]), columns=COLUMNS, out_dir=tmp_path)
        assert {"power_factor.png", "input_data_coverage.png"} <= {p.name for p in paths}

    def test_power_factor_is_skipped_without_a_reactive_signal(self, tmp_path: Path) -> None:
        columns = NO_PITCH_OR_REACTIVE
        paths = write_input_data_plots(_farm(["T1"]), columns=columns, out_dir=tmp_path)
        assert "power_factor.png" not in {p.name for p in paths}
