"""Calibrate the reference screen's floor against how far apart *clean* references read.

The screen rules out a reference whose estimate sits ``screen_floor`` or more from its pool's
median. That constant is only meaningful against the spread a healthy farm already shows, so this
driver runs a **single screening pass** over prepost placebo campaigns -- nothing injected, so
every deviation observed is the farm's own noise and drift -- and reports the distribution. Prepost
only, because the screen does not run for toggle campaigns.

The floor has to sit above that null spread (or the screen fires on good references) and below the
deviation a bad reference produces, which the R3 fixture measures at roughly 4.5 pp for a 3% Cp
change on one of three references.

Run it::

    uv run python -m benchmarking.campaigns.screen_calibration

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``screen_calibration``/``<timestamp>/``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl

mpl.use("Agg")  # headless

import pandas as pd

from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.power_model import CURATED_ERA5_EXCLUDE, TUNED_MODEL_PARAMS, PowerModelMethod
from benchmarking.campaigns.context import context_for
from benchmarking.campaigns.placebo import placebo_analysis_period, placebo_campaign
from benchmarking.campaigns.reference_fixture import REFERENCES_3, REFERENCES_5
from benchmarking.harness.method import MethodInput
from benchmarking.harness.northing import DEFAULT_NORTHING_ROLES, era5_direction, north_scada
from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# The fixture's own pools, plus the same turbine set seen from two other test turbines, so the
# spread is not read off one turbine's neighbourhood alone.
POOLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "T06_3ref": ("T06", REFERENCES_3),
    "T06_5ref": ("T06", REFERENCES_5),
    "T07_5ref": ("T07", ("T02", "T04", "T08", "T10", "T15")),
    "T11_5ref": ("T11", ("T09", "T12", "T14", "T16", "T17")),
}
# Prepost only: the screen does not run for toggle campaigns, so a toggle pass would
# record nothing at all rather than a toggle calibration.
_MODES = ("prepost",)


def default_output_root() -> Path:
    """Return the directory this driver writes under (``WIND_UP_BENCHMARKING_OUTPUT_DIR`` overrides)."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "screen_calibration"


def _coords(turbines: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Hill of Towie coordinates for ``turbines``."""
    metadata = load_hot_metadata()
    return {
        str(row.Name): (float(row.Latitude), float(row.Longitude))
        for row in metadata.itertuples()
        if str(row.Name) in set(turbines)
    }


def one_pass(
    *,
    mode: Literal["prepost", "toggle"],
    test_wtg: str,
    references: Sequence[str],
    scada_df: pd.DataFrame,
    era5_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run a single screening pass over a clean placebo pool and return its per-turbine deviations."""
    turbines = (test_wtg, *references)
    campaign = placebo_campaign(mode, upgraded=[test_wtg], turbines=list(turbines), coords=_coords(turbines))
    dataset = campaign.generate(scada_df)
    spec = campaign.spec()
    index = pd.DatetimeIndex(dataset.synthetic_df.index.unique()).sort_values()
    visible = north_scada(
        dataset.synthetic_df,
        columns=dataset.columns,
        north_offsets=spec.north_offsets,
        rated_power_kw=spec.rated_power_kw,
        era5_wd=era5_direction(era5_df, index),
        roles=DEFAULT_NORTHING_ROLES,
    )
    context = context_for(spec, turbine=test_wtg, scada_df=visible)
    method = PowerModelMethod(
        columns=HOT_COLUMNS,
        baseline_rated_power_kw=spec.rated_power_kw,
        era5_hourly_df=era5_df,
        conditions=(),
        availability_feature=False,
        era5_exclude=CURATED_ERA5_EXCLUDE,
        model_params=dict(TUNED_MODEL_PARAMS),
        # Infinite floor: nothing is ever dropped, so the loop stops after one pass and every
        # deviation recorded is the clean pool's own spread.
        screen_floor=float("inf"),
    )
    mi = MethodInput(scada_df=visible, test_wtg=test_wtg, campaign_context=context)
    result = method.screen_references(mi)
    passes = result.passes.copy()
    passes["mode"] = mode
    passes["test_wtg"] = test_wtg
    passes["pool"] = len(references)
    return passes


def run_calibration(
    *,
    modes: Sequence[str] = _MODES,
    pools: dict[str, tuple[str, tuple[str, ...]]] | None = None,
    out_root: str | Path | None = None,
) -> pd.DataFrame:
    """Run one clean screening pass per (mode, pool) and return every deviation observed."""
    pools = POOLS if pools is None else pools
    root = Path(out_root) if out_root is not None else default_output_root()
    run_dir = root / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_turbines = sorted({w for test, refs in pools.values() for w in (test, *refs)})
    era5_df = build_hot_v0_context(wtg_names=all_turbines).reanalysis_datasets[0].data
    frames: list[pd.DataFrame] = []
    for mode in modes:
        period = placebo_analysis_period(mode)  # type: ignore[arg-type]
        logger.info("loading Hill of Towie SCADA %s..%s for %s", *period, all_turbines)
        scada_df, _ = load_hot_scada(
            start_dt=period[0],
            end_dt_excl=period[1],
            wtg_numbers=[int(w[1:]) for w in all_turbines],
            wtg_names=all_turbines,
        )
        for label, (test_wtg, references) in pools.items():
            logger.info("screening %s %s (%d references)", mode, label, len(references))
            frames.append(
                one_pass(
                    mode=mode,  # type: ignore[arg-type]
                    test_wtg=test_wtg,
                    references=references,
                    scada_df=scada_df,
                    era5_df=era5_df,
                )
            )
    deviations = pd.concat(frames, ignore_index=True)
    deviations.to_csv(run_dir / "deviations.csv", index=False)
    summary = summarise(deviations)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info("wrote the calibration results to %s", run_dir)
    return deviations


def summarise(deviations: pd.DataFrame) -> pd.DataFrame:
    """Per (mode, test turbine, pool size): the spread of a clean pool's screening deviations."""
    if deviations.empty:
        return deviations
    grouped = deviations.groupby(["mode", "test_wtg", "pool"])["deviation"]
    return (
        pd.DataFrame(
            {
                "max_deviation_pp": grouped.max() * 100,
                "median_deviation_pp": grouped.median() * 100,
                "n": grouped.size(),
            }
        )
        .reset_index()
        .sort_values("max_deviation_pp", ascending=False)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    table = run_calibration()
    print(summarise(table).to_string(index=False))  # noqa: T201 - a driver's whole point is its printed summary
    worst = table["deviation"].max() * 100
    print(f"\nlargest clean-pool deviation: {worst:.3f} pp")  # noqa: T201
