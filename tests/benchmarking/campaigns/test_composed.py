"""Tests for the composed `wind-up`: its configuration, and running one from a declaration."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from benchmarking.campaigns.composed import WIND_UP, default_out_dir, run_declaration, wind_up_method
from benchmarking.campaigns.methods import carried_forward_methods
from benchmarking.synthetic import HOT_COLUMNS

from .test_loader import load, write_campaign

if TYPE_CHECKING:
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


def test_the_declaration_is_echoed_before_the_run_so_a_failure_still_leaves_it(tmp_path: Path) -> None:
    # visible beats infallible: the resolved UTC values must be readable even when the run dies
    path = write_campaign(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(Exception, match=r"[Pp]arquet"):  # the fixture's scada.parquet is an empty stub
        run_declaration(path, out_dir=out_dir, era5_hourly_df=era5())
    echoed = json.loads((out_dir / "resolved_campaign.json").read_text())
    assert echoed["name"] == "demo"
    assert echoed["analysis_period"]["start"] == "2017-01-01 00:00:00+00:00"
    assert echoed["timing"]["changeover"] == "2018-01-01 00:00:00+00:00"
    assert echoed["reanalysis"]["window"] == ["2017-01-01", "2018-12-31"]
