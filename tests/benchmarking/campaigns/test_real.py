"""Tests for the real campaigns declared from open data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
import yaml

from benchmarking.campaigns.loader import load_declaration
from benchmarking.campaigns.plans import plans_for
from benchmarking.campaigns.real import HOT_AEROUP_T13, write_hot_aeroup_t13
from wind_up.analysis_period import MONTH

if TYPE_CHECKING:
    from pathlib import Path

WORKS = """Turbine,First date of AeroUp works,Last date of AeroUp works
T13,2021-09-23,2021-09-29
T14,2022-08-30,2022-10-03
"""


def test_the_t13_declaration_loads(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "scada.parquet").write_bytes(b"")
    (data / "works.csv").write_text(WORKS)
    (data / "turbines.csv").write_text(
        "name,latitude,longitude,rotor_diameter_m\nT13,57.50,-3.08,82\nT14,57.51,-3.08,82\n"
    )
    (tmp_path / "campaign.yaml").write_text(yaml.safe_dump(HOT_AEROUP_T13))
    spec = load_declaration(tmp_path / "campaign.yaml").spec
    assert spec.timing_for("T13") == pd.Timestamp("2021-09-30", tz="UTC")
    assert spec.analysis_period is None
    assert spec.exclusions == [(None, pd.Timestamp("2022-09-03", tz="UTC"), pd.Timestamp("2023-02-12", tz="UTC"))]


@pytest.mark.slow
def test_the_t13_folder_plans_sensibly(tmp_path: Path) -> None:
    declaration = load_declaration(write_hot_aeroup_t13(tmp_path))
    scada = pd.read_parquet(declaration.scada_path)
    plan = plans_for(declaration.spec, scada, columns=declaration.columns)["T13"]
    assert plan.pool_rule_met
    assert plan.pre >= 3 * MONTH
    assert plan.post >= 3 * MONTH
    for reference in plan.power_references:
        windows = declaration.spec.works.get(reference, [])
        assert all(not (w0 < plan.end and w1 > plan.start) for w0, w1 in windows)
    nearest = plan.table()["turbine"].iloc[0]
    assert nearest in plan.power_references
