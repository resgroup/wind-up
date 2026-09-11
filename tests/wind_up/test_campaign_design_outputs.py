"""The files a campaign design writes, and its maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from tests.wind_up.layouts import grid_layout, scatter_layout
from wind_up.campaign_design import check_design, design_campaign, write_compliance, write_design
from wind_up.campaign_design_plots import plot_design_map

if TYPE_CHECKING:
    from pathlib import Path


def _layout_with_a_neighbour() -> pd.DataFrame:
    home = grid_layout(rows=4, cols=4, spacing_m=300).assign(wind_farm="Home")
    neighbour = scatter_layout([(1500, 0), (1500, 300), (1800, 150)]).assign(
        name=["N1", None, "N3"], wind_farm=["Other", "Other", None]
    )
    return pd.concat([home, neighbour], ignore_index=True)


def _design() -> object:
    return design_campaign(_layout_with_a_neighbour(), wind_farm="Home", excluded=["R3C3"], reference_only=["R0C0"])


def test_write_design_writes_every_file(tmp_path: Path) -> None:
    write_design(_design(), out_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "compliance.csv",
        "design_map.png",
        "design_map_latlon.png",
        "roles.yaml",
        "summary.yaml",
        "turbines.csv",
    ]


def test_compliance_csv_is_the_report_table(tmp_path: Path) -> None:
    design = _design()
    write_design(design, out_dir=tmp_path)
    written = pd.read_csv(tmp_path / "compliance.csv")
    assert list(written["test_turbine"]) == list(design.test_turbines)
    assert list(written.columns) == list(design.compliance.table.columns)


def test_turbines_csv_gives_every_farm_turbine_its_role_and_outcome(tmp_path: Path) -> None:
    design = _design()
    write_design(design, out_dir=tmp_path)
    turbines = pd.read_csv(tmp_path / "turbines.csv").set_index("name")
    assert len(turbines) == 16
    assert turbines.loc["R3C3", "role"] == "excluded"
    assert turbines.loc["R0C0", "role"] == "reference-only"
    for test in design.test_turbines:
        assert turbines.loc[test, "role"] == "test"
        assert turbines.loc[test, "outcome"] == "test"
    for test, refs in design.references.items():
        for ref in refs:
            assert test in turbines.loc[ref, "reference_for"].split(", ")
    skipped = [o for o in design.candidates if o.outcome == "skipped"]
    for o in skipped:
        assert turbines.loc[o.name, "reason"] == o.reason
        assert turbines.loc[o.name, "priority_rank"] == o.rank


def test_roles_yaml_is_a_declaration_turbines_block(tmp_path: Path) -> None:
    design = _design()
    write_design(design, out_dir=tmp_path)
    assert yaml.safe_load((tmp_path / "roles.yaml").read_text()) == {"turbines": design.roles()}


def test_summary_yaml_carries_the_maximum_and_the_problems(tmp_path: Path) -> None:
    design = _design()
    write_design(design, out_dir=tmp_path)
    summary = yaml.safe_load((tmp_path / "summary.yaml").read_text())
    assert summary["max_test_turbines"] == design.max_test_turbines
    assert summary["compliant"] is True
    assert summary["problems"] == []
    assert summary["available_turbines"] == 15


def test_write_compliance_for_a_hand_picked_set(tmp_path: Path) -> None:
    report = check_design(grid_layout(rows=4, cols=4, spacing_m=300), test_turbines=["R1C1", "R2C2"])
    write_compliance(report, out_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["compliance.csv", "summary.yaml"]
    summary = yaml.safe_load((tmp_path / "summary.yaml").read_text())
    assert summary["compliant"] is False
    assert summary["problems"] == list(report.problems)


def test_map_draws_in_metres_and_in_degrees() -> None:
    design = _design()
    for latlon in (False, True):
        fig = plot_design_map(design, latlon=latlon)
        ax = fig.axes[0]
        assert ax.get_xlabel() == ("longitude [deg]" if latlon else "east [m]")
        assert ax.get_title().startswith("Home:")
        plt.close(fig)


def test_metre_map_starts_at_zero() -> None:
    fig = plot_design_map(_design(), latlon=False)
    xs = [x for line in fig.axes[0].collections for x, _ in line.get_offsets()]
    assert min(xs) == 0
    plt.close(fig)


def test_map_zooms_to_the_farm_under_design() -> None:
    fig = plot_design_map(_design(), latlon=False)
    left, right = fig.axes[0].get_xlim()
    # the farm spans 0-900 m east; the neighbours at 1500-1800 m fall outside the view
    assert left < 0
    assert 900 < right < 1500
    plt.close(fig)
