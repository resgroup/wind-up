"""The placebo campaigns: a whole farm with nothing injected, run end-to-end.

Both modes are declared once here. With no upgrade the synthetic data equals the original, so
every method's per-turbine and farm estimate should read ~0 and the truth is 0 by construction.

Run it::

    uv run python -m benchmarking.campaigns.placebo

Both campaigns start at the beginning of 2018 on a full year of 2017 baseline. Prepost changes
over at the campaign start and runs a full year, so its baseline and treated periods hold the
same seasons; toggle alternates in 50-minute blocks over six months.

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``placebo``/``<mode>_<timestamp>/``. The
first run downloads and caches the Hill of Towie SCADA (Zenodo) and ERA5 (Open-Meteo).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl

mpl.use("Agg")  # headless: the report writes plots without a display

import pandas as pd

from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.campaigns.declaration import SyntheticCampaign
from benchmarking.campaigns.methods import carried_forward_methods
from benchmarking.campaigns.report import write_campaign_report
from benchmarking.campaigns.runner import CampaignRunner
from benchmarking.harness.northing import era5_direction
from benchmarking.synthetic import HOT_RATED_POWER_KW, ToggleSchedule
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarking.campaigns.runner import CampaignResult
    from benchmarking.synthetic import Fault

logger = logging.getLogger(__name__)

# Every Hill of Towie turbine; the campaign draws its test and reference turbines from these.
HOT_WTG_NUMBERS = tuple(range(1, 22))
HOT_TURBINES = tuple(f"T{number:02d}" for number in HOT_WTG_NUMBERS)

# Turbines eligible to be test turbines, spread across the site so test and reference turbines
# have roughly equal exposure to wake-free weather, ties going to the references.
PLACEBO_TEST_CANDIDATES = ("T07", "T11", "T12", "T06", "T16", "T19")

PLACEBO_UPGRADED = PLACEBO_TEST_CANDIDATES
PLACEBO_TURBINES = HOT_TURBINES
PLACEBO_WTG_NUMBERS = HOT_WTG_NUMBERS
PLACEBO_EXCLUDED: tuple[str, ...] = ()

# Treatment starts at the beginning of 2018, on a full year of baseline. Prepost runs a full year
# so baseline and treated periods hold the same seasons; toggle needs only six months, since its
# on and off blocks interleave within whatever period it is given.
PLACEBO_CAMPAIGN_START = pd.Timestamp("2018-01-01", tz="UTC")
PLACEBO_CAMPAIGN_MONTHS = {"prepost": 12, "toggle": 6}
PLACEBO_BASELINE_MONTHS = 12
# A full on/off cycle; ``ToggleSchedule`` halves it, so blocks are 50 minutes of 10-minute records.
PLACEBO_TOGGLE_PERIOD = pd.Timedelta(minutes=100)


def placebo_analysis_period(mode: Literal["prepost", "toggle"]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the whole record the methods see for ``mode``: the baseline plus the campaign."""
    return (
        PLACEBO_CAMPAIGN_START - pd.DateOffset(months=PLACEBO_BASELINE_MONTHS),
        PLACEBO_CAMPAIGN_START + pd.DateOffset(months=PLACEBO_CAMPAIGN_MONTHS[mode]),
    )


def default_output_root() -> Path:
    """Return the directory this driver writes under (``WIND_UP_BENCHMARKING_OUTPUT_DIR`` overrides)."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "placebo"


def _coords(turbines: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Hill of Towie coordinates for ``turbines``."""
    metadata = load_hot_metadata()
    return {
        str(row.Name): (float(row.Latitude), float(row.Longitude))
        for row in metadata.itertuples()
        if str(row.Name) in set(turbines)
    }


def placebo_campaign(
    mode: Literal["prepost", "toggle"],
    *,
    upgraded: Sequence[str] | None = None,
    turbines: Sequence[str] | None = None,
    excluded: Sequence[str] | None = None,
    coords: dict[str, tuple[float, float]] | None = None,
    faults: Sequence[Fault] | None = None,
) -> SyntheticCampaign:
    """Declare the placebo campaign for ``mode``: a whole farm with no upgrade injected.

    :param mode: ``"prepost"`` or ``"toggle"``
    :param upgraded: the test turbines; defaults to :data:`PLACEBO_UPGRADED`
    :param turbines: every participating turbine; defaults to :data:`PLACEBO_TURBINES`. Those
        that are neither upgraded nor excluded become the candidate references.
    :param excluded: turbines whose data must not be used; defaults to :data:`PLACEBO_EXCLUDED`
    :param coords: turbine coordinates; a placeholder is used when omitted, since no declared
        upgrade reads them
    :param faults: measurement corruptions to inject; none by default, so the placebo stays a
        clean-data campaign. The R-series fixtures inject one and compare against that.
    """
    upgraded = tuple(PLACEBO_UPGRADED if upgraded is None else upgraded)
    participating = tuple(PLACEBO_TURBINES if turbines is None else turbines)
    excluded = tuple(PLACEBO_EXCLUDED if excluded is None else excluded)
    if mode == "prepost":
        timing: pd.Timestamp | ToggleSchedule = PLACEBO_CAMPAIGN_START
    elif mode == "toggle":
        timing = ToggleSchedule(period=PLACEBO_TOGGLE_PERIOD, start=PLACEBO_CAMPAIGN_START)
    else:
        msg = f"unknown mode {mode!r}; expected 'prepost' or 'toggle'"
        raise ValueError(msg)
    return SyntheticCampaign(
        upgraded_turbines=list(upgraded),
        upgrade_timing=timing,
        candidate_references=[w for w in participating if w not in upgraded and w not in excluded],
        excluded_turbines=list(excluded),
        upgrades=[],
        faults=list(faults) if faults is not None else [],
        coords=coords if coords is not None else dict.fromkeys(participating, (0.0, 0.0)),
        # discovered by the shared northing step, not supplied: the placebo exercises the norther
        north_offsets=None,
        rated_power_kw=HOT_RATED_POWER_KW,
        analysis_period=placebo_analysis_period(mode),
    )


def run_placebo(
    *,
    mode: Literal["prepost", "toggle"],
    upgraded: Sequence[str] | None = None,
    turbines: Sequence[str] | None = None,
    include_power_model: bool = True,
    out_root: str | Path | None = None,
) -> CampaignResult:
    """Run one placebo campaign end-to-end on real Hill of Towie SCADA and write its report.

    :param mode: ``"prepost"`` or ``"toggle"``
    :param upgraded: the test turbines; defaults to :data:`PLACEBO_UPGRADED`
    :param turbines: every participating turbine; defaults to :data:`PLACEBO_TURBINES`
    :param include_power_model: run the power model, the method under test. Off only for a quick
        look or to avoid the ``ml`` dependency; the result is then not about v1 wind-up
    :param out_root: base output dir; defaults to :func:`default_output_root`
    :return: the campaign result, whose estimates should all read ~0
    """
    participating = list(PLACEBO_TURBINES if turbines is None else turbines)
    root = Path(out_root) if out_root is not None else default_output_root()
    run_dir = root / f"{mode}_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    period = placebo_analysis_period(mode)
    logger.info("Loading Hill of Towie SCADA %s..%s for %s", *period, participating)
    scada_df, _ = load_hot_scada(
        start_dt=period[0],
        end_dt_excl=period[1],
        wtg_numbers=[int(w[1:]) for w in participating],
        wtg_names=participating,
    )
    campaign = placebo_campaign(mode, upgraded=upgraded, turbines=participating, coords=_coords(participating))
    dataset = campaign.generate(scada_df)
    spec = campaign.spec()
    # ERA5 is needed whether or not the power model runs: it is the anchor the shared northing
    # step discovers against.
    era5 = build_hot_v0_context(wtg_names=participating).reanalysis_datasets[0].data
    index = pd.DatetimeIndex(dataset.synthetic_df.index.unique()).sort_values()

    runner = CampaignRunner(
        spec,
        dataset,
        build_methods=lambda wtg: carried_forward_methods(
            spec,
            out_dir=run_dir / wtg,
            era5_hourly_df=era5 if include_power_model else None,
            include_power_model=include_power_model,
        ),
        era5_wd=era5_direction(era5, index),
        northing_out_dir=run_dir / "northing",
    )
    result = runner.run()
    write_campaign_report(result, dataset, out_dir=run_dir)
    logger.info("Wrote the %s placebo report to %s", mode, run_dir)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for placebo_mode in ("prepost", "toggle"):
        run_placebo(mode=placebo_mode)
