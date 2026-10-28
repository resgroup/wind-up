"""Design an upgrade campaign on Hill of Towie's 21 turbines, with T17 reference-only and no test priority.

The upgrade is expected to give 5 % more energy, so its wake change reaches 12 rotor diameters, and the
design weighs the wind each test turbine loses by the site's wind rose. For comparison the same campaign
is also designed with a 20 D reach and with a 0 D reach (a change that leaves the wakes as they were).

Run it from the repository root::

    uv run python examples/v1/design_hill_of_towie.py

It writes each design to ``examples/v1/output/design_hill_of_towie/reach_<reach>/`` and copies its maps
into ``docs/images/designing-a-campaign/``, which ``docs/designing-a-campaign.md`` shows: the 12 D maps
under their own names, and the 20 D and 0 D design maps with a ``_<reach>`` suffix.

Turbine coordinates: Clerc, A. and Lingkan, E. (2026). Hill of Towie wind farm open dataset
(2.1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.22662930, CC BY 4.0.

Wind rose: the share of hours per 1 degree of 100 m wind direction, 2016 to 2020, at (57.51, -3.07),
from ERA5 (Hersbach et al. (2023), ERA5 hourly data on single levels from 1940 to present, Copernicus
Climate Change Service Climate Data Store, https://doi.org/10.24381/cds.adbb2d47) through Open-Meteo.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import pandas as pd

from wind_up.campaign_design import design_campaign, write_design

if TYPE_CHECKING:
    from wind_up.campaign_design import CampaignDesign

HERE = Path(__file__).parent
TURBINES_CSV = HERE / "hill_of_towie_turbines.csv"
WIND_ROSE_CSV = HERE / "hill_of_towie_wind_rose.csv"
EXPECTED_UPLIFT = 0.05
OUTPUT_DIR = HERE / "output" / "design_hill_of_towie"
DOCS_IMAGES_DIR = HERE.parents[1] / "docs" / "images" / "designing-a-campaign"
# Each design: its expected uplift, its wake reach in rotor diameters, and the maps copied to the docs
# with the name each is copied to.
DESIGNS: dict[str, tuple[float | None, float | None, dict[str, str]]] = {
    "12d": (
        EXPECTED_UPLIFT,
        None,
        {
            "design_map.png": "design_map.png",
            "front_row_map.png": "front_row_map.png",
            "coverage_map.png": "coverage_map.png",
        },
    ),
    "20d": (None, 20.0, {"design_map.png": "design_map_20d.png"}),
    "0d": (None, 0.0, {"design_map.png": "design_map_0d.png"}),
}
DOCS_MAPS = tuple(name for _, _, maps in DESIGNS.values() for name in maps.values())


def main(*, out_dir: Path = OUTPUT_DIR, docs_images_dir: Path | None = DOCS_IMAGES_DIR) -> dict[str, CampaignDesign]:
    """Design the campaign at each reach, write each to ``out_dir/reach_<reach>``, and copy the maps.

    The maps go to ``docs_images_dir`` unless it is ``None``. Returns the designs by reach.
    """
    layout = pd.read_csv(TURBINES_CSV)
    rose = pd.read_csv(WIND_ROSE_CSV)["share"].to_numpy(dtype=float)
    designs = {}
    for reach, (expected_uplift, wake_reach_d, maps) in DESIGNS.items():
        design = design_campaign(
            layout,
            reference_only=["T17"],
            wind_rose=rose / rose.sum(),
            rose_label="ERA5 2016-2020",
            expected_uplift=expected_uplift,
            wake_reach_d=wake_reach_d,
        )
        reach_dir = out_dir / f"reach_{reach}"
        write_design(design, out_dir=reach_dir)
        if docs_images_dir is not None:
            docs_images_dir.mkdir(parents=True, exist_ok=True)
            for name, docs_name in maps.items():
                shutil.copyfile(reach_dir / name, docs_images_dir / docs_name)
        _report(design, out_dir=reach_dir)
        designs[reach] = design
    return designs


def _report(design: CampaignDesign, *, out_dir: Path) -> None:
    summary = design.compliance.summary
    furthest = design.compliance.table.filter(like="_distance_d").to_numpy().max()
    print(f"wake reach {design.wake_reach_d:g} D")
    print(
        f"  test turbines ({len(design.test_turbines)} of {summary['available_turbines']}):",
        *sorted(design.test_turbines),
    )
    print(f"  front row: {summary['front_row_test_turbines']} tested, fair share {summary['fair_front_row_share']:.2f}")
    print(f"  furthest reference: {furthest:.1f} rotor diameters")
    print(f"  mean kept wind share {summary['mean_kept_wind_share']:.2f}")
    print(f"  written to {out_dir}")


if __name__ == "__main__":
    mpl.use("Agg")
    main()
