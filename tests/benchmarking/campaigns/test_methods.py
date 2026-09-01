"""Tests for the applicable-method rule."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.campaigns.methods import carried_forward_methods
from benchmarking.synthetic import ToggleSchedule

from .test_declaration import CHANGEOVER, campaign

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.campaigns import CampaignSpec

TOGGLE = ToggleSchedule(period=pd.Timedelta(hours=4), start=CHANGEOVER)


def _names(spec: CampaignSpec, out_dir: Path) -> list[str]:
    return [m.name for m in carried_forward_methods(spec, out_dir=out_dir, include_power_model=False)]


def test_prepost_skips_the_toggle_specialist(tmp_path: Path) -> None:
    assert "toggle_specialist" not in _names(campaign().spec(), tmp_path)


def test_toggle_includes_the_toggle_specialist(tmp_path: Path) -> None:
    assert "toggle_specialist" in _names(campaign(upgrade_timing=TOGGLE).spec(), tmp_path)


def test_naive_ratio_runs_in_both_modes(tmp_path: Path) -> None:
    assert "naive_ratio" in _names(campaign().spec(), tmp_path)
    assert "naive_ratio" in _names(campaign(upgrade_timing=TOGGLE).spec(), tmp_path)


def test_power_model_is_included_when_asked(tmp_path: Path) -> None:
    methods = carried_forward_methods(campaign().spec(), out_dir=tmp_path, include_power_model=True)
    assert "power_model" in [m.name for m in methods]


def test_power_model_reports_no_conditions_without_era5(tmp_path: Path) -> None:
    methods = carried_forward_methods(campaign().spec(), out_dir=tmp_path, include_power_model=True)
    power_model = next(m for m in methods if m.name == "power_model")
    assert power_model.conditions == ()


def test_each_method_writes_into_its_own_subfolder(tmp_path: Path) -> None:
    methods = carried_forward_methods(
        campaign(upgrade_timing=TOGGLE).spec(), out_dir=tmp_path, include_power_model=False
    )
    out_dirs = {m.out_dir for m in methods}
    assert len(out_dirs) == len(methods)
    assert all(d.parent == tmp_path for d in out_dirs)
