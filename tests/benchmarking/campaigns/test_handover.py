"""Tests for the analyst handover directory, and the isolation it is supposed to guarantee."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.campaigns.handover import GROUND_TRUTH_FILENAME, write_handover
from benchmarking.campaigns.placebo import placebo_instance
from benchmarking.synthetic import HOT_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.campaigns.declaration import SyntheticCampaign
    from benchmarking.synthetic import SyntheticDataset

BRIEF = "# The campaign\n\nSomething may have happened to some turbines.\n"
TURBINES = ("T01", "T02", "T03", "T04", "T05", "T06", "T17")


def _campaign() -> SyntheticCampaign:
    """A randomised placebo over a small slice of the farm."""
    campaign = placebo_instance("prepost", seed=1, turbines=TURBINES)
    campaign.coords = {w: (57.5 + i * 0.01, -3.25) for i, w in enumerate(TURBINES)}
    return campaign


def _dataset(campaign: SyntheticCampaign) -> SyntheticDataset:
    """Generate the campaign's data from a tiny flat-power frame."""
    index = pd.date_range(*campaign.analysis_period, freq="6h", tz="UTC", inclusive="left")
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    HOT_COLUMNS.turbine: wtg,
                    HOT_COLUMNS.active_power: 900.0,
                    HOT_COLUMNS.active_power_min: 850.0,
                    HOT_COLUMNS.wind_speed: 8.0,
                    HOT_COLUMNS.wind_speed_sd: 0.8,
                    HOT_COLUMNS.gen_rpm: 1400.0,
                    HOT_COLUMNS.availability: 3600.0,
                },
                index=index,
            )
            for wtg in TURBINES
        ]
    )
    return campaign.generate(frame)


def _handover(tmp_path: Path, *, docs: tuple[Path, ...] = ()) -> Path:
    """Write a handover and return its root."""
    campaign = _campaign()
    return write_handover(campaign, _dataset(campaign), root=tmp_path, brief=BRIEF, docs=docs)


class TestWhatTheAnalystGets:
    def test_the_analyst_directory_holds_the_brief_the_template_and_the_data(self, tmp_path: Path) -> None:
        root = _handover(tmp_path)
        assert (root / "analyst" / "brief.md").read_text() == BRIEF
        assert (root / "analyst" / "campaign.yaml").exists()
        assert (root / "analyst" / "data" / "scada.parquet").exists()
        assert (root / "analyst" / "data" / "turbines.csv").exists()

    def test_the_scada_is_the_frame_the_campaign_produced(self, tmp_path: Path) -> None:
        campaign = _campaign()
        dataset = _dataset(campaign)
        root = write_handover(campaign, dataset, root=tmp_path, brief=BRIEF)
        written = pd.read_parquet(root / "analyst" / "data" / "scada.parquet")
        pd.testing.assert_frame_equal(written, dataset.synthetic_df)

    def test_the_turbines_file_names_every_participating_turbine(self, tmp_path: Path) -> None:
        turbines = pd.read_csv(_handover(tmp_path) / "analyst" / "data" / "turbines.csv")
        assert set(turbines["Name"]) == set(TURBINES)
        assert {"Name", "Latitude", "Longitude"} <= set(turbines.columns)

    def test_the_documentation_under_test_is_copied_in(self, tmp_path: Path) -> None:
        doc = tmp_path / "how_to.md"
        doc.write_text("read me")
        root = _handover(tmp_path, docs=(doc,))
        assert (root / "analyst" / "docs" / "how_to.md").read_text() == "read me"


class TestTheTemplateIsBlank:
    def test_it_names_none_of_the_campaign_s_turbines(self, tmp_path: Path) -> None:
        # a populated template would answer the question before the analyst starts
        campaign = _campaign()
        template = (_handover(tmp_path) / "analyst" / "campaign.yaml").read_text()
        for turbine in campaign.upgraded_turbines:
            assert turbine not in template

    def test_it_carries_neither_the_changeover_nor_the_analysis_period(self, tmp_path: Path) -> None:
        campaign = _campaign()
        template = (_handover(tmp_path) / "analyst" / "campaign.yaml").read_text()
        assert str(campaign.upgrade_timing.date()) not in template
        assert str(campaign.analysis_period[0].date()) not in template

    def test_it_still_explains_every_field_the_analyst_must_fill(self, tmp_path: Path) -> None:
        template = (_handover(tmp_path) / "analyst" / "campaign.yaml").read_text()
        for field in ("name", "scada", "schema", "turbines", "upgraded", "references", "timing", "analysis_period"):
            assert field in template

    def test_it_does_not_claim_name_makes_an_output_subdirectory(self, tmp_path: Path) -> None:
        # it does not, once --out is given, and the documented run command gives it
        template = (_handover(tmp_path) / "analyst" / "campaign.yaml").read_text()
        assert "names the output subdirectory" not in template
        assert "adds no subdirectory" in template


class TestIsolation:
    def test_the_answer_key_is_outside_the_analyst_directory(self, tmp_path: Path) -> None:
        root = _handover(tmp_path)
        assert (root / "key" / GROUND_TRUTH_FILENAME).exists()
        assert not list((root / "analyst").rglob(GROUND_TRUTH_FILENAME))

    def test_the_analyst_directory_carries_no_original_data_and_no_run_metadata(self, tmp_path: Path) -> None:
        # the second of W1a's two isolation assertions: original_df and run_metadata never leave
        analyst = _handover(tmp_path) / "analyst"
        assert not list(analyst.rglob("original*"))
        assert not list(analyst.rglob("*run_metadata*"))
        assert sorted(p.name for p in analyst.rglob("*.parquet")) == ["scada.parquet"]

    def test_the_scada_carries_no_column_the_campaign_did_not_measure(self, tmp_path: Path) -> None:
        campaign = _campaign()
        dataset = _dataset(campaign)
        root = write_handover(campaign, dataset, root=tmp_path, brief=BRIEF)
        written = pd.read_parquet(root / "analyst" / "data" / "scada.parquet")
        assert set(written.columns) == set(dataset.synthetic_df.columns)


class TestTheAnswerKey:
    def _key(self, tmp_path: Path) -> dict:
        return json.loads((_handover(tmp_path) / "key" / GROUND_TRUTH_FILENAME).read_text())

    def test_it_records_what_was_injected(self, tmp_path: Path) -> None:
        key = self._key(tmp_path)
        assert key["upgrades"] == []  # a placebo injects nothing
        assert key["faults"] == []

    def test_it_records_who_was_treated_and_when(self, tmp_path: Path) -> None:
        campaign = _campaign()
        key = self._key(tmp_path)
        assert key["upgraded_turbines"] == campaign.upgraded_turbines
        assert key["timing"]["mode"] == "prepost"
        assert key["timing"]["changeover"] == str(campaign.upgrade_timing)

    def test_it_records_the_true_farm_uplift_and_the_seed(self, tmp_path: Path) -> None:
        key = self._key(tmp_path)
        assert key["true_farm_uplift"] == 0.0  # placebo: truth is 0 by construction
        assert key["seed"] == _campaign().seed
