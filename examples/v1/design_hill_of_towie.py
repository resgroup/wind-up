"""Design an upgrade campaign on Hill of Towie's 21 turbines, with T17 reference-only and no test priority.

Run it from the repository root::

    uv run python examples/v1/design_hill_of_towie.py

It writes the design to ``examples/v1/output/design_hill_of_towie/`` and copies its maps into
``docs/images/designing-a-campaign/``, which ``docs/designing-a-campaign.md`` shows.

Turbine coordinates: Clerc, A. and Lingkan, E. (2026). Hill of Towie wind farm open dataset
(2.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20204946, CC BY 4.0.
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
OUTPUT_DIR = HERE / "output" / "design_hill_of_towie"
DOCS_IMAGES_DIR = HERE.parents[1] / "docs" / "images" / "designing-a-campaign"
DOCS_MAPS = ("design_map.png", "front_row_map.png")


def main(*, out_dir: Path = OUTPUT_DIR, docs_images_dir: Path | None = DOCS_IMAGES_DIR) -> CampaignDesign:
    """Design the campaign, write it to ``out_dir``, and copy its maps to ``docs_images_dir`` unless it is ``None``."""
    layout = pd.read_csv(TURBINES_CSV)
    design = design_campaign(layout, reference_only=["T17"])
    write_design(design, out_dir=out_dir)
    if docs_images_dir is not None:
        docs_images_dir.mkdir(parents=True, exist_ok=True)
        for name in DOCS_MAPS:
            shutil.copyfile(out_dir / name, docs_images_dir / name)

    summary = design.compliance.summary
    furthest = design.compliance.table.filter(like="_distance_d").to_numpy().max()
    print(
        f"test turbines ({len(design.test_turbines)} of {summary['available_turbines']}):",
        *sorted(design.test_turbines),
    )
    print(f"front row: {summary['front_row_test_turbines']} tested, fair share {summary['fair_front_row_share']:.2f}")
    print(f"furthest reference: {furthest:.1f} rotor diameters")
    print(f"written to {out_dir}")
    return design


if __name__ == "__main__":
    mpl.use("Agg")
    main()
