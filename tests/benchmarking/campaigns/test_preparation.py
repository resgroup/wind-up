"""Tests for Part A, data preparation, run once per dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns.loader import Declaration, load_declaration
from benchmarking.campaigns.preparation import PreparedData, prepare_data
from benchmarking.diagnostics import stages
from benchmarking.diagnostics.yaw_changes import POSSIBLE_YAW_CHANGES_CSV, REPORT_COLUMNS
from benchmarking.harness.northing import NORTH_TABLE_YAML
from benchmarking.harness.operating_state import STATE_COLUMNS
from benchmarking.harness.waking import WAKING_COLUMNS
from benchmarking.synthetic import HOT_COLUMNS

from .test_composed import era5
from .test_loader import write_campaign

if TYPE_CHECKING:
    from pathlib import Path

TURBINES = ("T01", "T02", "T03", "T04")


def small_scada() -> pd.DataFrame:
    """Two months of hourly SCADA for the fixture's four turbines, with a nacelle position to north."""
    index = pd.date_range("2017-01-01", "2017-03-01", freq="1h", tz="UTC", inclusive="left")
    rng = np.random.default_rng(0)
    direction = (210.0 + 40.0 * np.sin(np.arange(len(index)) / 50.0)) % 360.0
    frames = []
    for i, turbine in enumerate(TURBINES):
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
                    HOT_COLUMNS.nacelle_position: (direction + 5.0 * i + rng.normal(0, 2, len(index))) % 360.0,
                },
                index=index,
            )
        )
    return pd.concat(frames)


@pytest.fixture(scope="module")
def declaration(tmp_path_factory: pytest.TempPathFactory) -> Declaration:
    return load_declaration(write_campaign(tmp_path_factory.mktemp("campaign")))


@pytest.fixture(scope="module")
def prepared(declaration: Declaration) -> PreparedData:
    return prepare_data(small_scada(), spec=declaration.spec, columns=declaration.columns, era5_hourly_df=era5())


class TestPartA:
    def test_the_records_carry_the_operating_state_and_waking_columns(self, prepared: PreparedData) -> None:
        assert set(STATE_COLUMNS) <= set(prepared.scada_df.columns)
        assert set(WAKING_COLUMNS) <= set(prepared.scada_df.columns)
        assert len(prepared.scada_df) == len(small_scada())

    def test_the_reanalysis_is_aligned_to_the_records(self, prepared: PreparedData) -> None:
        index = pd.DatetimeIndex(prepared.scada_df.index.unique()).sort_values()
        assert prepared.aligned_era5.index.equals(index)

    def test_every_turbine_is_northed_over_the_whole_record(
        self, prepared: PreparedData, declaration: Declaration
    ) -> None:
        assert set(prepared.north_tables) == set(TURBINES)
        offsets = prepared.with_north_offsets(declaration.spec).north_offsets
        assert offsets is not None
        assert {turbine for turbine, _, _ in offsets} == set(TURBINES)

    def test_the_possible_yaw_changes_are_reported(self, prepared: PreparedData) -> None:
        assert list(prepared.possible_changes.columns) == list(REPORT_COLUMNS)
        assert isinstance(prepared.possible_changes_by_turbine(), dict)

    def test_the_timebase_is_read_from_the_records(self, prepared: PreparedData) -> None:
        assert prepared.timebase == pd.Timedelta(hours=1)


class TestWhatIsWritten:
    def test_nothing_without_an_output_folder(self, tmp_path: Path) -> None:
        declaration = load_declaration(write_campaign(tmp_path))
        prepare_data(small_scada(), spec=declaration.spec, columns=declaration.columns, era5_hourly_df=era5())
        assert not list(tmp_path.glob("**/*.png"))
        assert [p.name for p in tmp_path.glob("**/*.yaml")] == ["campaign.yaml"]

    def test_tables_only_without_plots(self, tmp_path: Path) -> None:
        declaration = load_declaration(write_campaign(tmp_path))
        out = tmp_path / "A"
        prepare_data(
            small_scada(),
            spec=declaration.spec,
            columns=declaration.columns,
            era5_hourly_df=era5(),
            out_dir=out,
            plots=False,
        )
        assert (out / stages.NORTHING / NORTH_TABLE_YAML).exists()
        assert (out / stages.NORTHING / POSSIBLE_YAW_CHANGES_CSV).exists()
        assert not list(out.glob("**/*.png"))

    def test_every_step_writes_its_plots(self, tmp_path: Path) -> None:
        declaration = load_declaration(write_campaign(tmp_path))
        out = tmp_path / "A"
        prepare_data(
            small_scada(), spec=declaration.spec, columns=declaration.columns, era5_hourly_df=era5(), out_dir=out
        )
        for stage in (
            stages.CHANGES,
            stages.OPERATING_STATES,
            stages.REANALYSIS,
            stages.NORTHING,
            stages.WAKING,
            stages.FEATURES,
        ):
            assert list((out / stage).glob("**/*.png")), stage
