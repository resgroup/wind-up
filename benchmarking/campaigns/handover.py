"""Write the analyst handover directory for a dry run: everything they see, and nothing else.

The layout separates the two halves physically::

    <root>/
      analyst/            all the analyst is given
        brief.md
        campaign.yaml     a blank template to fill in
        data/scada.parquet
        data/turbines.csv
        design/           the campaign design, when one is given
        docs/
      key/ground_truth.json

Only ``synthetic_df`` reaches ``analyst/``. The dataset's ``original_df`` and ``run_metadata``
carry the answer, so they stay behind.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

import pandas as pd

from benchmarking.synthetic import ToggleSchedule, treated_mask
from wind_up.campaign_design import write_design

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from benchmarking.campaigns.declaration import SyntheticCampaign
    from benchmarking.synthetic import SyntheticDataset
    from wind_up.campaign_design import CampaignDesign

GROUND_TRUTH_FILENAME = "ground_truth.json"

# Columns of the design's turbines table the analyst does not get: a placebo's priority is a random
# shuffle, and would read as expected uplift.
WITHHELD_DESIGN_COLUMNS = ("priority_rank", "from_test_priority", "outcome", "reason")

# The declaration the analyst fills in. Blank: a populated one would answer the brief on sight.
CAMPAIGN_TEMPLATE = """\
# Fill this in from the brief, then run:
#   python -m benchmarking.campaigns run campaign.yaml --out out
#
# The report is written to --out exactly as given; `name` adds no subdirectory under it,
# so give each run its own --out or the second overwrites the first.
#
# Every time below is UTC. A value without a timezone is read as UTC.

name:                    # identifies the run

data:
  scada: data/scada.parquet
  schema: hill_of_towie  # the column vocabulary the SCADA is keyed by
  turbines: data/turbines.csv

turbines:
  upgraded:   []         # the turbines whose uplift you want
  references: []         # turbines you are willing to compare them against
  excluded:   []         # turbines whose data must not be used at all
  rated_power_kw:

timing:
  mode:                  # prepost or toggle
  # prepost:
  # changeover: YYYY-MM-DDTHH:MM:SSZ
  # toggle:
  # start:  YYYY-MM-DDTHH:MM:SSZ
  # period: 100min

analysis_period:
  start:                 # inclusive
  end:                   # exclusive

northing:
  discover: true         # false plus a `table:` applies exactly that table and discovers nothing
"""


def write_handover(
    campaign: SyntheticCampaign,
    dataset: SyntheticDataset,
    *,
    root: Path,
    brief: str,
    docs: Sequence[Path] = (),
    design: CampaignDesign | None = None,
) -> Path:
    """Write the handover for ``campaign`` under ``root`` and return it.

    :param campaign: the declared campaign, whose injected upgrades are the answer key
    :param dataset: its generated data; only ``synthetic_df`` is handed over
    :param root: the directory to build the handover in, outside the checkout
    :param brief: the campaign brief, as an owner would write it
    :param docs: documentation files to copy into ``analyst/docs/``
    :param design: the campaign design that chose ``campaign``'s upgraded turbines, written to
        ``analyst/design/`` without its priority columns
    """
    if design is not None and set(design.test_turbines) != set(campaign.upgraded_turbines):
        msg = (
            f"the design tests {sorted(design.test_turbines)} but the campaign upgrades "
            f"{sorted(campaign.upgraded_turbines)}; hand over the design that chose the campaign"
        )
        raise ValueError(msg)
    analyst = root / "analyst"
    (analyst / "data").mkdir(parents=True, exist_ok=True)
    (root / "key").mkdir(parents=True, exist_ok=True)

    (analyst / "brief.md").write_text(brief)
    (analyst / "campaign.yaml").write_text(CAMPAIGN_TEMPLATE)
    dataset.synthetic_df.to_parquet(analyst / "data" / "scada.parquet")
    _write_turbines(campaign, path=analyst / "data" / "turbines.csv")
    if design is not None:
        _write_design(design, out_dir=analyst / "design")
    if docs:
        (analyst / "docs").mkdir(parents=True, exist_ok=True)
        for doc in docs:
            shutil.copy(doc, analyst / "docs" / doc.name)

    (root / "key" / GROUND_TRUTH_FILENAME).write_text(json.dumps(_ground_truth(campaign, dataset), indent=2))
    return root


def _write_turbines(campaign: SyntheticCampaign, *, path: Path) -> None:
    """Write the turbines sidecar: name, latitude, longitude for every participating turbine."""
    rows = [{"Name": w, "Latitude": lat, "Longitude": lon} for w, (lat, lon) in sorted(campaign.coords.items())]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_design(design: CampaignDesign, *, out_dir: Path) -> None:
    """Write the design's outputs, with the priority columns of its turbines table withheld."""
    write_design(design, out_dir=out_dir)
    turbines = pd.read_csv(out_dir / "turbines.csv")
    turbines.drop(columns=list(WITHHELD_DESIGN_COLUMNS)).to_csv(out_dir / "turbines.csv", index=False)


def _ground_truth(campaign: SyntheticCampaign, dataset: SyntheticDataset) -> dict[str, Any]:
    """Return the answer key: what was injected, into which turbines, and what it really came to."""
    start, end = campaign.analysis_period
    timing: dict[str, Any] = {"mode": "toggle" if isinstance(campaign.upgrade_timing, ToggleSchedule) else "prepost"}
    if isinstance(campaign.upgrade_timing, ToggleSchedule):
        timing["start"] = str(campaign.upgrade_timing.start)
        timing["period"] = str(campaign.upgrade_timing.period)
        timing["start_on"] = campaign.upgrade_timing.start_on
    else:
        timing["changeover"] = str(campaign.upgrade_timing)
    return {
        "upgraded_turbines": list(campaign.upgraded_turbines),
        "timing": timing,
        "analysis_period": {"start": str(start), "end": str(end)},
        "upgrades": [_describe(u) for u in campaign.upgrades],
        "faults": [_describe(f) for f in campaign.faults],
        "true_farm_uplift": _true_farm_uplift(campaign, dataset),
        "seed": campaign.seed,
    }


def _describe(injected: object) -> dict[str, Any]:
    """Return an injected upgrade's or fault's own description, stringified for JSON."""
    description = getattr(injected, "description", None)
    if description is None:
        return {"kind": type(injected).__name__}
    return {k: str(v) for k, v in dict(description).items()}


def _true_farm_uplift(campaign: SyntheticCampaign, dataset: SyntheticDataset) -> float:
    """Return the exact pooled farm uplift over the upgraded turbines' treated records."""
    masks = {}
    for wtg in campaign.upgraded_turbines:
        rows = dataset.synthetic_df[dataset.synthetic_df[dataset.columns.turbine] == wtg]
        masks[wtg] = treated_mask(pd.DatetimeIndex(rows.index), campaign.upgrade_timing)
    return float(dataset.true_farm_uplift(test_wtgs=list(campaign.upgraded_turbines), masks=masks))
