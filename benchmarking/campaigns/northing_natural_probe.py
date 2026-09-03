"""Size the naturally occurring northing failure mode, with no fault injected.

T06's nearest neighbour T05 carries real step changes in its reported yaw direction during
2017-2018, so using T05 as a reference exercises the failure mode R1 addresses without
injecting anything. The probe runs the same campaign twice -- northing off, then on -- and
reports how far the uplift estimate moves.

The AeroUp uplift is injected so truth is non-zero and the reported number is an error rather
than placebo drift; the fixture uses the same shape, so the two are comparable.

Run from the repo root::

    uv run python -m benchmarking.campaigns.northing_natural_probe

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``northing_natural_probe``/``<timestamp>/``.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import pandas as pd

from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.campaigns.declaration import SyntheticCampaign
from benchmarking.campaigns.methods import carried_forward_methods
from benchmarking.campaigns.northing_fixture import BASELINE_MONTHS, CAMPAIGN_START, UPLIFT
from benchmarking.campaigns.runner import CampaignRunner
from benchmarking.harness.northing import era5_direction
from benchmarking.synthetic import HOT_RATED_POWER_KW
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.harness import Method

logger = logging.getLogger(__name__)

PROBE_TEST_WTG = "T06"
# T05 is T06's nearest neighbour and carries real northing steps in 2017-2018. Alone it drives the
# reference direction entirely; alongside the fixture's stable references its effect is diluted,
# which is the contrast the probe reports.
REFERENCE_SETS: dict[str, tuple[str, ...]] = {
    "t05_only": ("T05",),
    "t05_plus_stable": ("T05", "T15", "T10", "T08"),
}
CAMPAIGN_MONTHS = 12


def default_output_root() -> Path:
    """Where probe runs are written."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "northing_natural_probe"


def analysis_period() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the whole record the methods see: the baseline year plus the campaign year."""
    return (
        CAMPAIGN_START - pd.DateOffset(months=BASELINE_MONTHS),
        CAMPAIGN_START + pd.DateOffset(months=CAMPAIGN_MONTHS),
    )


def _coords(turbines: Sequence[str]) -> dict[str, tuple[float, float]]:
    metadata = load_hot_metadata()
    return {
        str(row.Name): (float(row.Latitude), float(row.Longitude))
        for row in metadata.itertuples()
        if str(row.Name) in set(turbines)
    }


def probe_campaign(*, references: Sequence[str], northing: bool) -> SyntheticCampaign:
    """Declare one arm: the same real campaign, with the shared northing step on or off."""
    turbines = (PROBE_TEST_WTG, *references)
    return SyntheticCampaign(
        upgraded_turbines=[PROBE_TEST_WTG],
        upgrade_timing=CAMPAIGN_START,
        candidate_references=list(references),
        upgrades=list(UPLIFT),
        faults=[],
        coords=_coords(turbines),
        north_offsets=None if northing else [],
        rated_power_kw=HOT_RATED_POWER_KW,
        analysis_period=analysis_period(),
    )


def _methods_for(
    wtg: str,
    *,
    spec: CampaignSpec,
    out_dir: Path,
    era5_hourly_df: pd.DataFrame | None,
    include_power_model: bool,
) -> list[Method]:
    """Build one turbine's methods into its own subfolder."""
    return carried_forward_methods(
        spec,
        out_dir=out_dir / wtg,
        era5_hourly_df=era5_hourly_df,
        include_power_model=include_power_model,
    )


def run_probe(*, out_root: str | Path | None = None, include_power_model: bool = True) -> pd.DataFrame:
    """Run every reference set with northing off and on; return one row per (set, arm, method)."""
    root = Path(out_root) if out_root is not None else default_output_root()
    run_dir = root / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    period = analysis_period()
    every_turbine = sorted({PROBE_TEST_WTG, *(t for refs in REFERENCE_SETS.values() for t in refs)})
    era5_df = build_hot_v0_context(wtg_names=every_turbine).reanalysis_datasets[0].data
    logger.info("loading Hill of Towie SCADA %s..%s for %s", *period, every_turbine)
    scada_df, _ = load_hot_scada(
        start_dt=period[0],
        end_dt_excl=period[1],
        wtg_numbers=[int(w[1:]) for w in every_turbine],
        wtg_names=every_turbine,
    )

    rows: list[dict[str, object]] = []
    for set_name, references in REFERENCE_SETS.items():
        for northing in (False, True):
            arm = "northed" if northing else "raw"
            logger.info("running %s / %s", set_name, arm)
            campaign = probe_campaign(references=references, northing=northing)
            dataset = campaign.generate(scada_df)
            spec = campaign.spec()
            index = pd.DatetimeIndex(dataset.synthetic_df.index.unique()).sort_values()
            runner = CampaignRunner(
                spec,
                dataset,
                build_methods=partial(
                    _methods_for,
                    spec=spec,
                    out_dir=run_dir / f"{set_name}_{arm}",
                    era5_hourly_df=era5_df if include_power_model else None,
                    include_power_model=include_power_model,
                ),
                era5_wd=era5_direction(era5_df, index),
            )
            result = runner.run()
            rows.extend(
                {
                    "reference_set": set_name,
                    "references": ",".join(references),
                    "northing": northing,
                    "arm": arm,
                    "method": row.method,
                    "estimate": row.estimate,
                    "truth": row.truth,
                    "signed_error": row.signed_error,
                }
                for row in result.farm.itertuples()
            )

    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "natural_probe.csv", index=False)
    sensitivity = sensitivity_table(table)
    sensitivity.to_csv(run_dir / "sensitivity.csv", index=False)
    logger.info("wrote the probe results to %s\n%s", run_dir, sensitivity.to_string(index=False))
    return table


def sensitivity_table(table: pd.DataFrame) -> pd.DataFrame:
    """How far northing moves the answer, per reference set and method, in percentage points."""
    rows = []
    for (set_name, method), group in table.groupby(["reference_set", "method"]):
        cell = {bool(r.northing): float(r.signed_error) * 100 for r in group.itertuples()}
        if len(cell) != 2:  # noqa: PLR2004 - both arms are needed for a shift
            continue
        rows.append(
            {
                "reference_set": set_name,
                "method": method,
                "raw_error_pp": cell[False],
                "northed_error_pp": cell[True],
                "shift_pp": cell[True] - cell[False],
                "improved_pp": abs(cell[False]) - abs(cell[True]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the natural probe over both reference sets."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_probe()


if __name__ == "__main__":
    main()
