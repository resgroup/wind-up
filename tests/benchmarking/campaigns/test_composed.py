"""Tests for the composed `wind-up`: its configuration, and running one from a declaration."""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import yaml

from benchmarking.campaigns.composed import (
    _LOG_HANDLER_NAME,
    INPUT_PLOTS_DIRNAME,
    LOG_FILENAME,
    WIND_UP,
    default_input_plots_dir,
    default_out_dir,
    run_declaration,
    wind_up_method,
)
from benchmarking.campaigns.methods import carried_forward_methods
from benchmarking.synthetic import HOT_COLUMNS

from .test_loader import load, write_campaign

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

# Configuration that belongs to the run, not to the method, so it is excluded from the comparison.
PER_RUN_FIELDS = frozenset({"name", "out_dir"})


def era5() -> pd.DataFrame:
    """A minimal hourly reanalysis frame, enough for the power model to be built with conditions."""
    index = pd.date_range("2017-01-01", "2019-01-01", freq="1h", tz="UTC", inclusive="left")
    return pd.DataFrame({"wind_speed_100m": 9.0, "wind_direction_100m": 210.0, "temperature_2m": 8.0}, index=index)


def _config(method: object) -> dict:
    """One method's configuration, minus the per-run fields."""
    return {f.name: getattr(method, f.name) for f in dataclasses.fields(method) if f.name not in PER_RUN_FIELDS}


class TestTheComposition:
    def test_it_is_named_wind_up(self, tmp_path: Path) -> None:
        spec = load(tmp_path).spec
        assert (
            wind_up_method(spec, columns=HOT_COLUMNS, out_dir=tmp_path, era5_hourly_df=era5()).name
            == WIND_UP
            == "wind-up"
        )

    def test_its_configuration_is_the_accepted_power_model_defaults(self, tmp_path: Path) -> None:
        # a default drifting on either side is caught here, since wind-up is that configuration
        spec = load(tmp_path).spec
        reanalysis = era5()
        composed = wind_up_method(spec, columns=HOT_COLUMNS, out_dir=tmp_path / "a", era5_hourly_df=reanalysis)
        carried = next(
            m
            for m in carried_forward_methods(spec, out_dir=tmp_path / "b", era5_hourly_df=reanalysis)
            if m.name == "power_model"
        )
        assert _config(composed) == _config(carried)

    def test_it_reports_per_condition_estimates(self, tmp_path: Path) -> None:
        spec = load(tmp_path).spec
        assert wind_up_method(spec, columns=HOT_COLUMNS, out_dir=tmp_path, era5_hourly_df=era5()).conditions

    def test_it_screens_its_references_and_reports_them(self, tmp_path: Path) -> None:
        # the R3 screen and the reference-stability table are part of what wind-up is
        method = wind_up_method(load(tmp_path).spec, columns=HOT_COLUMNS, out_dir=tmp_path, era5_hourly_df=era5())
        assert method.reference_screen
        assert method.report_reference_uplifts


class TestTheOutputDirectory:
    def test_it_defaults_to_the_benchmarking_root_under_the_campaign_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", str(tmp_path))
        assert default_out_dir("demo") == tmp_path / "demo"

    def test_the_campaign_name_is_what_identifies_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", str(tmp_path))
        assert default_out_dir("other").name == "other"


class TestTheRunLog:
    """Some of what a run decides is reported only in the log, so the log has to outlive the run."""

    @pytest.fixture(autouse=True)
    def _detach(self) -> Iterator[None]:
        """Leave the root logger as the test found it."""
        yield
        root = logging.getLogger()
        for handler in [h for h in root.handlers if getattr(h, "name", None) == _LOG_HANDLER_NAME]:
            root.removeHandler(handler)
            handler.close()

    def _run(self, tmp_path: Path, out_dir: Path) -> None:
        """Start a run that dies on the fixture's stub parquet, after the log is opened."""
        with pytest.raises(Exception, match=r"[Pp]arquet"):
            run_declaration(write_campaign(tmp_path), out_dir=out_dir, era5_hourly_df=era5())

    def test_it_is_written_into_the_output_directory(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        out_dir = tmp_path / "out"
        with caplog.at_level(logging.INFO):
            self._run(tmp_path, out_dir)
        assert "Running campaign 'demo'" in (out_dir / LOG_FILENAME).read_text()

    def test_a_second_run_logs_to_its_own_directory_and_not_the_first(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        first, second = tmp_path / "first", tmp_path / "second"
        with caplog.at_level(logging.INFO):
            self._run(tmp_path, first)
            before = (first / LOG_FILENAME).read_text()
            self._run(tmp_path, second)
        assert "Running campaign 'demo'" in (second / LOG_FILENAME).read_text()
        assert (first / LOG_FILENAME).read_text() == before


def test_the_declaration_is_echoed_before_the_run_so_a_failure_still_leaves_it(tmp_path: Path) -> None:
    # visible beats infallible: the resolved UTC values must be readable even when the run dies
    path = write_campaign(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(Exception, match=r"[Pp]arquet"):  # the fixture's scada.parquet is an empty stub
        run_declaration(path, out_dir=out_dir, era5_hourly_df=era5())
    echoed = yaml.safe_load((out_dir / "campaign_resolved.yaml").read_text())
    assert echoed["name"] == "demo"
    assert echoed["analysis_period"]["start"] == "2017-01-01 00:00:00+00:00"
    assert echoed["timing"]["changeover"] == "2018-01-01 00:00:00+00:00"
    assert echoed["reanalysis"]["window"] == ["2017-01-01", "2018-12-31"]


class TestTheReanalysisWindow:
    def test_a_declared_period_sets_it(self, tmp_path: Path) -> None:
        from benchmarking.campaigns.composed import reanalysis_window  # noqa: PLC0415

        index = pd.date_range("2015-01-01", "2020-01-01", freq="1h", tz="UTC")
        assert reanalysis_window(load(tmp_path), index=index) == ("2017-01-01", "2018-12-31")

    def test_without_a_period_it_covers_the_data(self, tmp_path: Path) -> None:
        from benchmarking.campaigns.composed import reanalysis_window  # noqa: PLC0415

        from .test_loader import STAGGERED, load_staggered  # noqa: PLC0415

        index = pd.date_range("2016-03-01", "2019-12-31 23:00", freq="1h", tz="UTC")
        assert reanalysis_window(load_staggered(tmp_path, STAGGERED), index=index) == ("2016-01-01", "2019-12-31")


def _write_small_scada(path: Path) -> None:
    """Two months of hourly SCADA for the fixture's four turbines, every signal the plots read."""
    index = pd.date_range("2017-01-01", "2017-03-01", freq="1h", tz="UTC", inclusive="left")
    rng = np.random.default_rng(0)
    frames = []
    for turbine in ("T01", "T02", "T03", "T04"):
        ws = rng.uniform(3, 18, len(index))
        power = np.clip(0.5 * ws**3, 0, 2050)
        frames.append(
            pd.DataFrame(
                {
                    HOT_COLUMNS.turbine: turbine,
                    HOT_COLUMNS.active_power: power,
                    HOT_COLUMNS.active_power_min: power,
                    HOT_COLUMNS.wind_speed: ws,
                    HOT_COLUMNS.wind_speed_sd: 1.0,
                    HOT_COLUMNS.gen_rpm: np.clip(ws * 90, 0, 1600),
                    HOT_COLUMNS.pitch: np.clip(ws - 12, 0, None),
                    HOT_COLUMNS.reactive_power: 0.1 * power,
                    HOT_COLUMNS.availability: 3600.0,
                },
                index=index,
            )
        )
    pd.concat(frames).to_parquet(path)


class _StopError(Exception):
    """Raised by the stubbed campaign estimate, so a test stops the run where it wants to look."""


class TestTheInputDataPlots:
    """Step 1: every turbine and every record provided, drawn before anything is planned."""

    @pytest.fixture
    def declaration(self, tmp_path: Path) -> Path:
        campaign = tmp_path / "campaign"
        campaign.mkdir()
        path = write_campaign(campaign)
        _write_small_scada(campaign / "scada.parquet")
        return path

    @pytest.fixture(autouse=True)
    def _stop_before_planning(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        def stub(*_args: object, **_kwargs: object) -> None:
            raise _StopError

        monkeypatch.setattr("benchmarking.campaigns.composed.estimate_campaign", stub)
        yield
        root = logging.getLogger()
        for handler in [h for h in root.handlers if getattr(h, "name", None) == _LOG_HANDLER_NAME]:
            root.removeHandler(handler)
            handler.close()

    def test_they_sit_beside_the_campaign_folder(self, declaration: Path) -> None:
        assert default_input_plots_dir(declaration) == declaration.parent.parent / INPUT_PLOTS_DIRNAME

    def test_every_turbine_is_drawn_before_planning(self, declaration: Path, tmp_path: Path) -> None:
        with pytest.raises(_StopError):
            run_declaration(declaration, out_dir=tmp_path / "out", era5_hourly_df=era5())
        names = {p.name for p in (tmp_path / INPUT_PLOTS_DIRNAME).iterdir()}
        # T04 is excluded from the campaign but its data was provided, so it is drawn
        assert {f"ops_relationships_T0{i}.png" for i in range(1, 5)} <= names
        assert {"power_factor.png", "input_data_coverage.png"} <= names

    def test_the_folder_can_be_named(self, declaration: Path, tmp_path: Path) -> None:
        with pytest.raises(_StopError):
            run_declaration(
                declaration, out_dir=tmp_path / "out", era5_hourly_df=era5(), input_plots_dir=tmp_path / "mine"
            )
        assert (tmp_path / "mine" / "input_data_coverage.png").exists()


class TestTheRunFolders:
    @pytest.fixture(autouse=True)
    def _detach(self) -> Iterator[None]:
        """Leave the root logger as the test found it."""
        yield
        root = logging.getLogger()
        for handler in [h for h in root.handlers if getattr(h, "name", None) == _LOG_HANDLER_NAME]:
            root.removeHandler(handler)
            handler.close()

    def test_each_test_turbine_writes_straight_into_its_own_folder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: list[object] = []

        def stub(*_args: object, build_methods: Callable[[str], list[object]], **_kwargs: object) -> None:
            built.extend(build_methods("T01"))
            raise _StopError

        monkeypatch.setattr("benchmarking.campaigns.composed.estimate_campaign", stub)
        campaign = tmp_path / "campaign"
        campaign.mkdir()
        declaration = write_campaign(campaign)
        _write_small_scada(campaign / "scada.parquet")
        with pytest.raises(_StopError):
            run_declaration(declaration, out_dir=tmp_path / "out", era5_hourly_df=era5())
        (method,) = built
        assert method.out_dir == (tmp_path / "out" / "T01").resolve()  # type: ignore[attr-defined]
        assert not method.run_subdir  # type: ignore[attr-defined]

    def test_studies_keep_a_named_folder_per_run(self, tmp_path: Path) -> None:
        method = wind_up_method(load(tmp_path).spec, columns=HOT_COLUMNS, out_dir=tmp_path, era5_hourly_df=era5())
        assert method.run_subdir
