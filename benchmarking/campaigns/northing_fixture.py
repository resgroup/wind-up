"""The R1 northing fixture: does a northing step bite, and does the shared step close the gap.

A small campaign -- T06 plus its three nearest turbines whose northing is stable over the
period -- with a known AeroUp-shaped uplift injected, run four ways per mode:

|          | northing off      | northing on       |
|----------|-------------------|-------------------|
| clean    | the reference error | must be no worse (*no harm*) |
| faulted  | must be much worse (*bites*) | must return to ~clean (*fixed*) |

"Northing off" is a campaign declaring ``north_offsets=[]``: the shared step still writes the
``northed_`` column methods read, but as an uncorrected copy of the raw signal. "Northing on"
declares ``None`` and the step discovers the corrections from the data.

Run it::

    uv run python -m benchmarking.campaigns.northing_fixture

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``northing_fixture``/``<timestamp>/``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl

mpl.use("Agg")  # headless: the report writes plots without a display

import numpy as np
import pandas as pd

from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.campaigns.declaration import SyntheticCampaign
from benchmarking.campaigns.methods import carried_forward_methods
from benchmarking.campaigns.runner import CampaignRunner
from benchmarking.harness.northing import era5_direction
from benchmarking.synthetic import HOT_RATED_POWER_KW, NorthingStep, ToggleSchedule, WindSpeedCpChange
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarking.campaigns.runner import CampaignResult

logger = logging.getLogger(__name__)

# T06 is the measured-best fixture turbine; its three nearest neighbours whose northing is stable
# over 2017-2018 are the references (T05 is nearer but carries real northing steps of its own).
FIXTURE_TEST_WTG = "T06"
FIXTURE_REFERENCES = ("T15", "T10", "T08")
FIXTURE_TURBINES = (FIXTURE_TEST_WTG, *FIXTURE_REFERENCES)

CAMPAIGN_START = pd.Timestamp("2018-01-01", tz="UTC")
BASELINE_MONTHS = 12
CAMPAIGN_MONTHS = {"prepost": 12, "toggle": 6}
TOGGLE_PERIOD = pd.Timedelta(minutes=100)

# The AeroUp shape: +10% Cp held below 5 m/s, fading linearly to zero by 12 m/s.
UPLIFT = (WindSpeedCpChange(ws_points=(5.0, 12.0), deltas=(0.10, 0.0)),)

# The injected fault. It lands on the nearest reference (the most influential one) at the moment
# the contrast is measured across -- the changeover in prepost, mid-campaign in toggle -- which is
# where a direction corruption does most damage.
FAULT_TURBINE = "T15"
FAULT_OFFSET_DEG = 40.0


def analysis_period(mode: Literal["prepost", "toggle"]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the whole record the methods see for ``mode``: the baseline plus the campaign."""
    return (
        CAMPAIGN_START - pd.DateOffset(months=BASELINE_MONTHS),
        CAMPAIGN_START + pd.DateOffset(months=CAMPAIGN_MONTHS[mode]),
    )


def fault_time(mode: Literal["prepost", "toggle"]) -> pd.Timestamp:
    """When the injected step lands: at the changeover (prepost) or mid-campaign (toggle)."""
    if mode == "prepost":
        return CAMPAIGN_START
    _, end = analysis_period(mode)
    return CAMPAIGN_START + (end - CAMPAIGN_START) / 2


def default_output_root() -> Path:
    """Return the directory this driver writes under (``WIND_UP_BENCHMARKING_OUTPUT_DIR`` overrides)."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "northing_fixture"


def _coords(turbines: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Hill of Towie coordinates for ``turbines``."""
    metadata = load_hot_metadata()
    return {
        str(row.Name): (float(row.Latitude), float(row.Longitude))
        for row in metadata.itertuples()
        if str(row.Name) in set(turbines)
    }


def fixture_campaign(
    mode: Literal["prepost", "toggle"],
    *,
    faulted: bool,
    northing: bool,
    coords: dict[str, tuple[float, float]] | None = None,
) -> SyntheticCampaign:
    """Declare one cell of the fixture's 2x2.

    :param mode: ``"prepost"`` or ``"toggle"``
    :param faulted: inject the northing step into :data:`FAULT_TURBINE`
    :param northing: ``True`` leaves ``north_offsets`` undeclared so the shared step discovers
        them; ``False`` declares an empty list, so the northed column is an uncorrected copy
    :param coords: turbine coordinates; a placeholder is used when omitted
    """
    if mode == "prepost":
        timing: pd.Timestamp | ToggleSchedule = CAMPAIGN_START
    elif mode == "toggle":
        timing = ToggleSchedule(period=TOGGLE_PERIOD, start=CAMPAIGN_START)
    else:
        msg = f"unknown mode {mode!r}; expected 'prepost' or 'toggle'"
        raise ValueError(msg)
    faults = [NorthingStep(turbine=FAULT_TURBINE, at=fault_time(mode), offset_deg=FAULT_OFFSET_DEG)] if faulted else []
    return SyntheticCampaign(
        upgraded_turbines=[FIXTURE_TEST_WTG],
        upgrade_timing=timing,
        candidate_references=list(FIXTURE_REFERENCES),
        upgrades=list(UPLIFT),
        faults=faults,
        coords=coords if coords is not None else dict.fromkeys(FIXTURE_TURBINES, (0.0, 0.0)),
        north_offsets=None if northing else [],
        rated_power_kw=HOT_RATED_POWER_KW,
        analysis_period=analysis_period(mode),
    )


def run_cell(
    *,
    mode: Literal["prepost", "toggle"],
    faulted: bool,
    northing: bool,
    scada_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    out_dir: Path,
    include_power_model: bool = True,
) -> CampaignResult:
    """Run one cell of the 2x2 and return its result."""
    campaign = fixture_campaign(mode, faulted=faulted, northing=northing, coords=_coords(FIXTURE_TURBINES))
    dataset = campaign.generate(scada_df)
    spec = campaign.spec()
    index = pd.DatetimeIndex(dataset.synthetic_df.index.unique()).sort_values()
    runner = CampaignRunner(
        spec,
        dataset,
        build_methods=lambda wtg: carried_forward_methods(
            spec,
            out_dir=out_dir / wtg,
            era5_hourly_df=era5_df if include_power_model else None,
            include_power_model=include_power_model,
        ),
        era5_wd=era5_direction(era5_df, index),
    )
    return runner.run()


def run_fixture(
    *,
    modes: Sequence[str] = ("prepost", "toggle"),
    include_power_model: bool = True,
    out_root: str | Path | None = None,
) -> pd.DataFrame:
    """Run the whole 2x2 for each mode and return the bites/fixed table.

    :return: one row per ``(mode, method, arm)`` with the estimate, truth and signed error
    """
    root = Path(out_root) if out_root is not None else default_output_root()
    run_dir = root / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    era5_df = build_hot_v0_context(wtg_names=list(FIXTURE_TURBINES)).reanalysis_datasets[0].data
    rows: list[dict[str, object]] = []
    for mode in modes:
        period = analysis_period(mode)  # type: ignore[arg-type]
        logger.info("loading Hill of Towie SCADA %s..%s for %s", *period, list(FIXTURE_TURBINES))
        scada_df, _ = load_hot_scada(
            start_dt=period[0],
            end_dt_excl=period[1],
            wtg_numbers=[int(w[1:]) for w in FIXTURE_TURBINES],
            wtg_names=list(FIXTURE_TURBINES),
        )
        for faulted in (False, True):
            for northing in (False, True):
                arm = f"{'faulted' if faulted else 'clean'}/{'northed' if northing else 'raw'}"
                logger.info("running %s %s", mode, arm)
                result = run_cell(
                    mode=mode,  # type: ignore[arg-type]
                    faulted=faulted,
                    northing=northing,
                    scada_df=scada_df,
                    era5_df=era5_df,
                    out_dir=run_dir / f"{mode}_{'faulted' if faulted else 'clean'}_{'northed' if northing else 'raw'}",
                    include_power_model=include_power_model,
                )
                rows.extend(
                    {
                        "mode": mode,
                        "method": row.method,
                        "faulted": faulted,
                        "northing": northing,
                        "arm": arm,
                        "estimate": row.estimate,
                        "truth": row.truth,
                        "signed_error": row.signed_error,
                    }
                    for row in result.farm.itertuples()
                )
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "bites_and_fixed.csv", index=False)
    verdicts = verdict_table(table)
    verdicts.to_csv(run_dir / "verdicts.csv", index=False)
    logger.info("wrote the fixture results to %s", run_dir)
    return table


def verdict_table(table: pd.DataFrame) -> pd.DataFrame:
    """Turn the 2x2 of errors into the bites / fixed / no-harm verdicts, per mode and method.

    ``bites`` is the degradation the fault causes with no northing; ``fixed`` is what the fault
    still costs once northing runs; ``no_harm`` is what northing costs on clean data. All in
    percentage points of energy-ratio error.
    """
    rows = []
    for (mode, method), group in table.groupby(["mode", "method"]):
        cell = {(bool(r.faulted), bool(r.northing)): abs(float(r.signed_error)) * 100 for r in group.itertuples()}
        if len(cell) != 4:  # noqa: PLR2004 - the 2x2 needs all four arms
            continue
        reference = cell[False, False]
        rows.append(
            {
                "mode": mode,
                "method": method,
                "clean_raw_pp": reference,
                "faulted_raw_pp": cell[True, False],
                "clean_northed_pp": cell[False, True],
                "faulted_northed_pp": cell[True, True],
                "bites_pp": cell[True, False] - reference,
                "fixed_pp": cell[True, True] - reference,
                "no_harm_pp": cell[False, True] - reference,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["bites"] = out["bites_pp"] >= 1.0
        out["fixed"] = out["fixed_pp"] <= 0.25  # noqa: PLR2004 - the spec's acceptance threshold
        out["no_harm"] = out["no_harm_pp"] <= 0.25  # noqa: PLR2004 - the spec's acceptance threshold
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = run_fixture()
    summary = verdict_table(results)
    print(summary.to_string(index=False))  # noqa: T201 - a driver's whole point is its printed summary
    print(f"\n{np.count_nonzero(summary['bites'])} of {len(summary)} cases bite")  # noqa: T201
