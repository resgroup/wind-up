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

import numpy as np
import pandas as pd

from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.campaigns.declaration import SyntheticCampaign
from benchmarking.campaigns.methods import carried_forward_methods
from benchmarking.campaigns.report import write_campaign_report
from benchmarking.campaigns.runner import CampaignRunner
from benchmarking.harness.northing import era5_direction
from benchmarking.synthetic import HOT_RATED_POWER_KW, HOT_ROTOR_DIAMETER_M, ToggleSchedule
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada
from wind_up.campaign_design import design_campaign

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarking.campaigns.runner import CampaignResult
    from benchmarking.synthetic import Fault
    from wind_up.campaign_design import CampaignDesign

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


def placebo_analysis_period(
    mode: Literal["prepost", "toggle"], *, campaign_start: pd.Timestamp = PLACEBO_CAMPAIGN_START
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the whole record the methods see for ``mode``: the baseline plus the campaign."""
    return (
        campaign_start - pd.DateOffset(months=PLACEBO_BASELINE_MONTHS),
        campaign_start + pd.DateOffset(months=PLACEBO_CAMPAIGN_MONTHS[mode]),
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
    campaign_start: pd.Timestamp = PLACEBO_CAMPAIGN_START,
    seed: int = 0,
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
    :param campaign_start: when treatment begins; defaults to :data:`PLACEBO_CAMPAIGN_START`
    :param seed: recorded on the campaign, so an answer key names the draw that made it
    """
    upgraded = tuple(PLACEBO_UPGRADED if upgraded is None else upgraded)
    participating = tuple(PLACEBO_TURBINES if turbines is None else turbines)
    excluded = tuple(PLACEBO_EXCLUDED if excluded is None else excluded)
    if mode == "prepost":
        timing: pd.Timestamp | ToggleSchedule = campaign_start
    elif mode == "toggle":
        timing = ToggleSchedule(period=PLACEBO_TOGGLE_PERIOD, start=campaign_start)
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
        analysis_period=placebo_analysis_period(mode, campaign_start=campaign_start),
        seed=seed,
    )


# A randomised instance draws its treatment start from these post-years, and never upgrades
# these turbines. The window stays before the site's real blade-upgrade installs, and before the
# thin-baseline year.
PLACEBO_INSTANCE_YEARS = (2018,)
PLACEBO_INSTANCE_KEEP_AS_REFERENCE = ("T17",)
PLACEBO_INSTANCE_LAST_CLEAN = pd.Timestamp("2021-01-01", tz="UTC")
# A treated period reaching past this rests on a 2019-only baseline, which reads far worse.
PLACEBO_INSTANCE_LAST_GOOD_END = pd.Timestamp("2020-01-01", tz="UTC")
PLACEBO_WIND_FARM = "Hill of Towie"
# An instance tests this many turbines fewer than a compliant design allows (never fewer than one),
# so instances differ in their test turbines.
PLACEBO_INSTANCE_BELOW_MAX = 1


def placebo_layout(coords: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Return a campaign-design layout of the Hill of Towie turbines at ``coords``."""
    return pd.DataFrame(
        {
            "name": list(coords),
            "latitude": [lat for lat, _ in coords.values()],
            "longitude": [lon for _, lon in coords.values()],
            "rotor_diameter_m": HOT_ROTOR_DIAMETER_M,
            "wind_farm": PLACEBO_WIND_FARM,
        }
    )


def placebo_design(
    *,
    seed: int,
    coords: dict[str, tuple[float, float]],
    turbines: Sequence[str] | None = None,
) -> CampaignDesign:
    """Return the campaign design behind :func:`placebo_instance` for the same ``seed`` and turbines.

    :data:`PLACEBO_INSTANCE_BELOW_MAX` fewer test turbines than a compliant design allows, every
    candidate listed in a seeded random priority.
    """
    return _instance_design(np.random.default_rng(seed), coords=coords, turbines=turbines)


def placebo_instance(
    mode: Literal["prepost", "toggle"],
    *,
    seed: int,
    coords: dict[str, tuple[float, float]],
    turbines: Sequence[str] | None = None,
) -> SyntheticCampaign:
    """Return a seeded random placebo: which turbines are upgraded, and when treatment starts.

    A handover built from the checked-in defaults would identify its own campaign, so both are
    drawn. Nothing is injected, so the truth stays 0. The upgraded turbines are the test turbines of
    :func:`placebo_design` for the same seed.

    :param mode: ``"prepost"`` or ``"toggle"``
    :param seed: the draw's seed; the same seed gives the same campaign
    :param coords: turbine name to ``(latitude, longitude)``, which the campaign design reads
    :param turbines: every participating turbine; defaults to :data:`PLACEBO_TURBINES`
    """
    participating = list(PLACEBO_TURBINES if turbines is None else turbines)
    rng = np.random.default_rng(seed)
    design = _instance_design(rng, coords=coords, turbines=participating)
    year = int(rng.choice(PLACEBO_INSTANCE_YEARS))
    month = int(rng.integers(1, 13))
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    return placebo_campaign(
        mode,
        upgraded=sorted(design.test_turbines),
        turbines=participating,
        coords={w: coords[w] for w in participating},
        campaign_start=start,
        seed=seed,
    )


def _instance_design(
    rng: np.random.Generator,
    *,
    coords: dict[str, tuple[float, float]],
    turbines: Sequence[str] | None,
) -> CampaignDesign:
    """Design a placebo instance, drawing its priority from ``rng``."""
    participating = list(PLACEBO_TURBINES if turbines is None else turbines)
    reference_only = [w for w in PLACEBO_INSTANCE_KEEP_AS_REFERENCE if w in participating]
    candidates = [w for w in participating if w not in reference_only]
    layout = placebo_layout({w: coords[w] for w in participating})
    most = design_campaign(layout, reference_only=reference_only).max_test_turbines
    return design_campaign(
        layout,
        test_priority=[str(w) for w in rng.permutation(candidates)],
        reference_only=reference_only,
        n_test=max(1, most - PLACEBO_INSTANCE_BELOW_MAX),
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
