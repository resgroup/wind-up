"""Offline tests for how the reference-screen road-test driver reports its results.

The driver itself needs four farms of real SCADA, so what is tested here is the result handling
around it: a farm that fails must not be mistaken for a farm that passed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pandas as pd
import pytest

from benchmarking.campaigns.screen_roadtest import RoadTest, run_roadtest, summarise

if TYPE_CHECKING:
    from pathlib import Path

_TESTS = [
    RoadTest(name="farm_a", source="hot", turbines=("T01", "T02", "T03"), test_wtgs=("T01",)),
    RoadTest(name="farm_b", source="hot", turbines=("T01", "T02", "T03"), test_wtgs=("T01",)),
]


def _row(farm: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "farm": farm,
                "test_wtg": "T01",
                "reference": "T02",
                "screened": False,
                "test_uplift_pct": 0.1,
                "reference_overall_pct": 0.2,
            }
        ]
    )


def _run(tmp_path: Path, *, failing: set[str]) -> pd.DataFrame:
    def fake_run_one(test: RoadTest, *, out_dir: Path) -> pd.DataFrame:  # noqa: ARG001
        if test.name in failing:
            msg = f"{test.name} could not load"
            raise RuntimeError(msg)
        return _row(test.name)

    with (
        patch("benchmarking.campaigns.screen_roadtest.road_tests", return_value=_TESTS),
        patch("benchmarking.campaigns.screen_roadtest.run_one", side_effect=fake_run_one),
    ):
        return run_roadtest(out_root=tmp_path)


def _written(tmp_path: Path) -> pd.DataFrame:
    return pd.read_csv(next(tmp_path.glob("*/roadtest.csv")))


class TestRunRoadtest:
    def test_every_farm_running_returns_a_row_each(self, tmp_path: Path) -> None:
        results = _run(tmp_path, failing=set())
        assert sorted(results["farm"]) == ["farm_a", "farm_b"]
        assert sorted(_written(tmp_path)["farm"]) == ["farm_a", "farm_b"]

    def test_a_failed_farm_raises_rather_than_reading_as_a_clean_run(self, tmp_path: Path) -> None:
        """A partial road test must not be mistaken for a completed validation."""
        with pytest.raises(RuntimeError, match="farm_b"):
            _run(tmp_path, failing={"farm_b"})

    def test_the_farms_that_did_run_are_still_written(self, tmp_path: Path) -> None:
        """Every farm is attempted and its results kept, so one run shows the whole picture."""
        with pytest.raises(RuntimeError):
            _run(tmp_path, failing={"farm_b"})

        assert _written(tmp_path)["farm"].tolist() == ["farm_a"]


class TestSummarise:
    def test_a_farm_with_nothing_screened_reads_as_a_dash(self) -> None:
        assert summarise(_row("farm_a"))["screened"].tolist() == ["-"]

    def test_screened_references_are_listed_in_order(self) -> None:
        rows = pd.concat([_row("farm_a"), _row("farm_a")], ignore_index=True)
        rows.loc[:, "reference"] = ["T09", "T02"]
        rows.loc[:, "screened"] = True
        assert summarise(rows)["screened"].tolist() == ["T02,T09"]
