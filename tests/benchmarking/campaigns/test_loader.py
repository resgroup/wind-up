"""Tests for the campaign declaration: YAML plus a turbines sidecar into a CampaignSpec."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from benchmarking.campaigns.loader import load_declaration
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule

if TYPE_CHECKING:
    from pathlib import Path

TURBINES_CSV = """Name,Latitude,Longitude
T01,57.40,-3.30
T02,57.60,-3.20
T03,57.50,-3.25
T04,57.50,-3.25
"""

PREPOST = """
name: demo

data:
  scada: scada.parquet
  schema: hill_of_towie
  turbines: turbines.csv

turbines:
  upgraded:   [T01]
  references: [T02, T03]
  excluded:   [T04]
  rated_power_kw: 2050

timing:
  mode: prepost
  changeover: 2018-01-01T00:00:00Z

analysis_period:
  start: 2017-01-01T00:00:00Z
  end:   2019-01-01T00:00:00Z
"""


def write_campaign(tmp_path: Path, yaml_text: str = PREPOST) -> Path:
    """Write a campaign declaration plus the sidecar files it names, and return the YAML path."""
    (tmp_path / "turbines.csv").write_text(TURBINES_CSV)
    (tmp_path / "scada.parquet").write_bytes(b"")
    path = tmp_path / "campaign.yaml"
    path.write_text(textwrap.dedent(yaml_text))
    return path


def load(tmp_path: Path, yaml_text: str = PREPOST):  # noqa: ANN201
    """Write and load a declaration in one step."""
    return load_declaration(write_campaign(tmp_path, yaml_text))


class TestTheCampaignFacts:
    def test_the_name_is_carried(self, tmp_path: Path) -> None:
        assert load(tmp_path).name == "demo"

    def test_the_turbine_roles_are_carried(self, tmp_path: Path) -> None:
        spec = load(tmp_path).spec
        assert spec.upgraded_turbines == ["T01"]
        assert spec.candidate_references == ["T02", "T03"]
        assert spec.excluded_turbines == ["T04"]
        assert spec.rated_power_kw == 2050.0

    def test_the_coordinates_come_from_the_turbines_sidecar(self, tmp_path: Path) -> None:
        spec = load(tmp_path).spec
        assert spec.coords["T01"] == (57.40, -3.30)
        assert set(spec.coords) == {"T01", "T02", "T03", "T04"}

    def test_the_named_schema_resolves_to_a_column_schema(self, tmp_path: Path) -> None:
        declaration = load(tmp_path)
        assert declaration.columns == HOT_COLUMNS
        assert declaration.spec.turbine_col == HOT_COLUMNS.turbine

    def test_the_scada_path_resolves_relative_to_the_declaration(self, tmp_path: Path) -> None:
        assert load(tmp_path).scada_path == tmp_path / "scada.parquet"

    def test_the_analysis_period_end_is_exclusive(self, tmp_path: Path) -> None:
        start, end = load(tmp_path).spec.analysis_period
        assert start == pd.Timestamp("2017-01-01", tz="UTC")
        assert end == pd.Timestamp("2019-01-01", tz="UTC")


class TestTiming:
    def test_prepost_carries_the_changeover(self, tmp_path: Path) -> None:
        spec = load(tmp_path).spec
        assert spec.mode == "prepost"
        assert spec.upgrade_timing == pd.Timestamp("2018-01-01", tz="UTC")

    def test_toggle_builds_a_schedule(self, tmp_path: Path) -> None:
        spec = load(
            tmp_path,
            PREPOST.replace(
                "  mode: prepost\n  changeover: 2018-01-01T00:00:00Z",
                "  mode: toggle\n  start: 2018-01-01T00:00:00Z\n  period: 100min",
            ),
        ).spec
        assert spec.mode == "toggle"
        assert spec.upgrade_timing == ToggleSchedule(
            period=pd.Timedelta(minutes=100), start=pd.Timestamp("2018-01-01", tz="UTC")
        )

    def test_toggle_carries_start_on(self, tmp_path: Path) -> None:
        spec = load(
            tmp_path,
            PREPOST.replace(
                "  mode: prepost\n  changeover: 2018-01-01T00:00:00Z",
                "  mode: toggle\n  start: 2018-01-01T00:00:00Z\n  period: 100min\n  start_on: true",
            ),
        ).spec
        assert spec.upgrade_timing.start_on is True

    def test_an_unknown_mode_names_the_modes_there_are(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"prepost.*toggle|toggle.*prepost"):
            load(tmp_path, PREPOST.replace("mode: prepost", "mode: sideways"))


class TestTimezones:
    def test_a_naive_timestamp_is_read_as_utc(self, tmp_path: Path) -> None:
        spec = load(tmp_path, PREPOST.replace("changeover: 2018-01-01T00:00:00Z", "changeover: 2018-01-01")).spec
        assert spec.upgrade_timing == pd.Timestamp("2018-01-01", tz="UTC")

    def test_an_offset_timestamp_is_converted_to_utc(self, tmp_path: Path) -> None:
        # +01:00 is one hour ahead, so midnight there is 23:00 UTC the day before
        spec = load(
            tmp_path, PREPOST.replace("changeover: 2018-01-01T00:00:00Z", "changeover: 2018-01-01T00:00:00+01:00")
        ).spec
        assert spec.upgrade_timing == pd.Timestamp("2017-12-31 23:00", tz="UTC")

    def test_every_resolved_timestamp_is_tz_aware_utc(self, tmp_path: Path) -> None:
        spec = load(tmp_path, PREPOST.replace("start: 2017-01-01T00:00:00Z", "start: 2017-01-01")).spec
        for stamp in (*spec.analysis_period, spec.upgrade_timing):
            assert str(stamp.tz) == "UTC"

    def test_the_resolved_timestamps_are_echoed(self, tmp_path: Path) -> None:
        # visible beats infallible: a mis-declared timezone should be readable in the output
        resolved = load(
            tmp_path, PREPOST.replace("changeover: 2018-01-01T00:00:00Z", "changeover: 2018-01-01")
        ).resolved()
        assert resolved["timing"]["changeover"] == "2018-01-01 00:00:00+00:00"
        assert resolved["analysis_period"]["start"] == "2017-01-01 00:00:00+00:00"


class TestNorthing:
    def test_discover_is_the_default_when_northing_is_not_declared(self, tmp_path: Path) -> None:
        assert load(tmp_path).spec.north_offsets is None

    def test_discover_true_leaves_the_offsets_to_be_found(self, tmp_path: Path) -> None:
        assert load(tmp_path, PREPOST + "\nnorthing:\n  discover: true\n").spec.north_offsets is None

    def test_a_declared_table_is_applied_exactly(self, tmp_path: Path) -> None:
        text = PREPOST + textwrap.dedent("""
            northing:
              discover: false
              table:
                - [T01, 2017-06-01T00:00:00Z, 4.5]
            """)
        assert load(tmp_path, text).spec.north_offsets == [("T01", pd.Timestamp("2017-06-01", tz="UTC"), 4.5)]

    def test_discover_false_with_no_table_applies_nothing_and_discovers_nothing(self, tmp_path: Path) -> None:
        assert load(tmp_path, PREPOST + "\nnorthing:\n  discover: false\n").spec.north_offsets == []


class TestReanalysis:
    def test_the_centroid_is_the_mean_of_the_declared_turbines(self, tmp_path: Path) -> None:
        assert load(tmp_path).centroid == (57.50, -3.25)

    def test_the_fetch_window_is_rounded_out_to_whole_calendar_years(self, tmp_path: Path) -> None:
        # the cache key includes the dates, so exact windows would refetch for every campaign
        assert load(tmp_path).era5_window == ("2017-01-01", "2018-12-31")

    def test_the_exclusive_end_does_not_pull_in_an_extra_year(self, tmp_path: Path) -> None:
        # the period ends at midnight on 1 Jan 2019, so no 2019 record is ever read
        assert load(tmp_path).era5_window[1] == "2018-12-31"


class TestErrors:
    def test_an_unknown_schema_name_lists_the_known_ones(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="hill_of_towie"):
            load(tmp_path, PREPOST.replace("schema: hill_of_towie", "schema: greenbyte_maybe"))

    def test_a_turbine_missing_from_the_sidecar_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="T99"):
            load(tmp_path, PREPOST.replace("upgraded:   [T01]", "upgraded:   [T01, T99]"))

    def test_a_turbine_in_two_roles_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="T01"):
            load(tmp_path, PREPOST.replace("references: [T02, T03]", "references: [T01, T02]"))

    def test_an_end_before_the_start_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="end"):
            load(tmp_path, PREPOST.replace("end:   2019-01-01T00:00:00Z", "end:   2016-01-01T00:00:00Z"))

    def test_a_missing_sidecar_file_is_named(self, tmp_path: Path) -> None:
        path = write_campaign(tmp_path)
        (tmp_path / "turbines.csv").unlink()
        with pytest.raises(FileNotFoundError, match=r"turbines.csv"):
            load_declaration(path)

    def test_no_upgraded_turbines_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="upgraded"):
            load(tmp_path, PREPOST.replace("upgraded:   [T01]", "upgraded:   []"))
