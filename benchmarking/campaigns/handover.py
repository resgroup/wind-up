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

# The brief an analyst is given: the prose an owner would send, and the menu of what the change and
# the data problems could be. The menu is stated rather than hidden, so the question is which one
# and how big rather than a guess at an unbounded space. Anemometer faults are not on it: CF11
# priced them at 0.21 pp in prepost and zero in toggle, so they are neither injected nor asked about.
BRIEF_TEMPLATE = """\
# {farm} — campaign assessment

We made a change to part of the {farm} fleet and need to know what it was worth.

Work was carried out on **{upgraded}**. {timing} The remaining turbines in the data were not
touched and are yours to use as references.

The campaign was designed before the work began; the design documents are in `design/`.

We have pulled SCADA from **{start}** to **{end}** — see `data/`. Please declare the campaign in
`campaign.yaml`, run it, and tell us:

**(a)** what kind of change this was, and how big it was;
**(b)** whether anything was wrong with the data — any turbine misbehaving, any sensor drifting,
any reference you would not trust.

Be honest about the size of the number relative to the noise. If the answer is "we cannot
distinguish this from zero", that is a useful answer and we would rather have it than a
confident one that is wrong.

## What the change could have been

Our contractors do a limited set of things, so the answer to (a) is one of:

- a **flat efficiency change** across the whole operating range (blade cleaning, fouling, add-ons)
- a **wind-speed-dependent efficiency change**, biggest in mid winds (an aerodynamic blade upgrade)
- an efficiency change that **varies with some other condition** (turbulence, say)
- a change to the turbine's **rated power** (an uprate or a downrate)
- **wake steering**, where one turbine is yawed deliberately to benefit another
- **nothing at all** — sometimes the work is cancelled, or does not do what was promised

For (b), the data problems we have seen before are: a **nacelle-position step** (a turbine's
direction sensor re-zeroed), a **reference turbine that changed** during the period, and
**missing data**.

## Getting started

Read `docs/running-a-campaign.md`, and `docs/designing-a-campaign.md` for how to read `design/`.
Everything you need is in this directory; run from here.
"""


def campaign_brief(campaign: SyntheticCampaign, *, farm: str = "Hill of Towie") -> str:
    """Return the brief for ``campaign``: who was treated, when, and what to report.

    :param campaign: the declared campaign; only facts an analyst would know are used
    :param farm: the wind farm's name, as the brief addresses it
    """
    timing = campaign.upgrade_timing
    start, end = campaign.analysis_period
    if isinstance(timing, ToggleSchedule):
        began = timing.start if timing.start is not None else start
        minutes = int(timing.period.total_seconds() // 60)
        prose = (
            f"The change was run as a **toggle**: from **{began:%d %B %Y}** the turbines alternated "
            f"between changed and unchanged in blocks, on a full cycle of **{minutes} minutes** (so "
            f"half that on, half off). Everything before that date is untouched baseline."
        )
    else:
        prose = f"Treatment began on **{pd.Timestamp(timing):%d %B %Y}** and everything after that date is post-change."
    return BRIEF_TEMPLATE.format(
        farm=farm,
        upgraded=", ".join(campaign.upgraded_turbines),
        timing=prose,
        start=f"{start:%d %B %Y}",
        end=f"{end:%d %B %Y}",
    )


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
  excluded:   []         # turbines never to use as a reference; their wake still counts
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
