"""Road-test the reference screen on several real farms, looking for what a fixture cannot show.

Runs the shipped `power_model` configuration, screen on, over **placebo** campaigns -- nothing
injected, so the truth is exactly zero -- on four reference sets:

* Hill of Towie, the whole farm;
* Hill of Towie west, ``T01``-``T15``, a geographically tighter pool;
* Kelmarsh (6 turbines) and Penmanshiel (14), a second and third site so the screen is not
  judged only where it was calibrated.

Nothing is injected, so **every reference the screen rules out is either a genuine find or a false
positive**, and both are worth seeing. The reported reference overall uplift should read near 0%
on a healthy campaign.

Run it::

    uv run python -m benchmarking.campaigns.screen_roadtest

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``screen_roadtest``/``<timestamp>/``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless

import pandas as pd

from benchmarking.baselines.power_model import CURATED_ERA5_EXCLUDE, TUNED_MODEL_PARAMS, PowerModelMethod
from benchmarking.baselines.power_model.method import reference_overall_uplift
from benchmarking.campaigns.context import context_for
from benchmarking.campaigns.declaration import SyntheticCampaign
from benchmarking.harness.method import MethodInput
from benchmarking.harness.northing import DEFAULT_NORTHING_ROLES, era5_direction, north_scada
from benchmarking.synthetic import HOT_COLUMNS, HOT_RATED_POWER_KW
from benchmarking.synthetic.sources.greenbyte import (
    GREENBYTE_COLUMNS,
    KELMARSH,
    PENMANSHIEL,
    load_greenbyte_metadata,
    load_greenbyte_scada,
)
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada
from wind_up_v0.era5 import get_era5_hourly_df

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)

# One year of baseline, then a year of campaign, matching the placebo shape.
BASELINE_START = pd.Timestamp("2017-01-01", tz="UTC")
CAMPAIGN_START = pd.Timestamp("2018-01-01", tz="UTC")
CAMPAIGN_END = pd.Timestamp("2019-01-01", tz="UTC")

HOT_ALL = tuple(f"T{n:02d}" for n in range(1, 22))
HOT_WEST = tuple(f"T{n:02d}" for n in range(1, 16))


@dataclass(frozen=True)
class RoadTest:
    """One farm and reference set to road-test.

    :param name: label in the output table
    :param source: ``"hot"``, ``"kelmarsh"`` or ``"penmanshiel"``
    :param turbines: every participating turbine
    :param test_wtgs: the turbines to estimate; the rest are candidate references
    """

    name: str
    source: str
    turbines: tuple[str, ...]
    test_wtgs: tuple[str, ...] = field(default_factory=tuple)


def _greenbyte_turbines(farm: object, limit: int | None = None) -> tuple[str, ...]:
    """Turbine names a Greenbyte farm publishes, in order."""
    metadata = load_greenbyte_metadata(farm)  # type: ignore[arg-type]
    names = [str(n) for n in metadata["Name"]]
    return tuple(names[:limit] if limit else names)


def road_tests() -> list[RoadTest]:
    """Return the farms and reference sets to road-test."""
    kelmarsh = _greenbyte_turbines(KELMARSH)
    penmanshiel = _greenbyte_turbines(PENMANSHIEL)
    return [
        RoadTest(name="hot_all", source="hot", turbines=HOT_ALL, test_wtgs=("T07", "T11")),
        RoadTest(name="hot_west", source="hot", turbines=HOT_WEST, test_wtgs=("T07", "T11")),
        RoadTest(name="kelmarsh", source="kelmarsh", turbines=kelmarsh, test_wtgs=kelmarsh[:2]),
        RoadTest(name="penmanshiel", source="penmanshiel", turbines=penmanshiel, test_wtgs=penmanshiel[:2]),
    ]


def _load(test: RoadTest) -> tuple[pd.DataFrame, dict[str, tuple[float, float]], ColumnSchema, float]:
    """Return SCADA, coordinates, the column schema and rated power for one road test."""
    if test.source == "hot":
        scada, _ = load_hot_scada(
            start_dt=BASELINE_START,
            end_dt_excl=CAMPAIGN_END,
            wtg_numbers=[int(w[1:]) for w in test.turbines],
            wtg_names=list(test.turbines),
        )
        metadata = load_hot_metadata()
        coords = {
            str(r.Name): (float(r.Latitude), float(r.Longitude))
            for r in metadata.itertuples()
            if str(r.Name) in set(test.turbines)
        }
        return scada, coords, HOT_COLUMNS, HOT_RATED_POWER_KW
    farm = KELMARSH if test.source == "kelmarsh" else PENMANSHIEL
    scada = load_greenbyte_scada(farm, years=[2017, 2018])
    scada = scada[scada[GREENBYTE_COLUMNS.turbine].isin(test.turbines)]
    metadata = load_greenbyte_metadata(farm)
    coords = {
        str(r.Name): (float(r.Latitude), float(r.Longitude))
        for r in metadata.itertuples()
        if str(r.Name) in set(test.turbines)
    }
    return scada, coords, GREENBYTE_COLUMNS, farm.rated_power_kw


def run_one(test: RoadTest, *, out_dir: Path) -> pd.DataFrame:
    """Run one road test and return a row per (test turbine, reference)."""
    scada, coords, columns, rated = _load(test)
    lat = sum(v[0] for v in coords.values()) / len(coords)
    lon = sum(v[1] for v in coords.values()) / len(coords)
    era5 = get_era5_hourly_df(lat=lat, lon=lon, start_date="2016-01-01", end_date="2019-06-01")

    campaign = SyntheticCampaign(
        upgraded_turbines=list(test.test_wtgs),
        upgrade_timing=CAMPAIGN_START,
        candidate_references=[w for w in test.turbines if w not in set(test.test_wtgs)],
        upgrades=[],
        coords=coords,
        north_offsets=None,
        rated_power_kw=rated,
        analysis_period=(BASELINE_START, CAMPAIGN_END),
        columns=columns,
    )
    dataset = campaign.generate(scada)
    spec = campaign.spec()
    index = pd.DatetimeIndex(dataset.synthetic_df.index.unique()).sort_values()
    visible = north_scada(
        dataset.synthetic_df,
        columns=columns,
        north_offsets=spec.north_offsets,
        rated_power_kw=rated,
        era5_wd=era5_direction(era5, index),
        roles=DEFAULT_NORTHING_ROLES,
    )

    rows: list[dict[str, object]] = []
    for wtg in test.test_wtgs:
        method = PowerModelMethod(
            columns=columns,
            baseline_rated_power_kw=rated,
            era5_hourly_df=era5,
            conditions=(),
            availability_feature=False,
            era5_exclude=CURATED_ERA5_EXCLUDE,
            model_params=dict(TUNED_MODEL_PARAMS),
            out_dir=out_dir / test.name / wtg,
        )
        context = context_for(spec, turbine=wtg, scada_df=visible)
        out = method.estimate(MethodInput(scada_df=visible, test_wtg=wtg, campaign_context=context))
        refs = out.reference_uplifts
        combined = reference_overall_uplift(refs, rated_power_kw=rated) if refs is not None else float("nan")
        logger.info(
            "%s %s: uplift=%+.3f%% (truth 0), reference overall=%+.3f%%, screened=%s",
            test.name,
            wtg,
            out.p50_overall * 100,
            combined * 100,
            sorted(refs.loc[refs["screened"], "turbine"]) if refs is not None else [],
        )
        if refs is not None:
            rows.extend(
                {
                    "farm": test.name,
                    "test_wtg": wtg,
                    "test_uplift_pct": out.p50_overall * 100,
                    "reference_overall_pct": combined * 100,
                    "reference": row.turbine,
                    "reference_uplift_pct": row.uplift * 100,
                    "screened": row.screened,
                }
                for row in refs.itertuples()
            )
    return pd.DataFrame(rows)


def default_output_root() -> Path:
    """Return the directory this driver writes under (``WIND_UP_BENCHMARKING_OUTPUT_DIR`` overrides)."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "screen_roadtest"


def run_roadtest(*, names: Sequence[str] | None = None, out_root: str | Path | None = None) -> pd.DataFrame:
    """Run every road test (or the named subset) and write the results."""
    root = Path(out_root) if out_root is not None else default_output_root()
    run_dir = root / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for test in road_tests():
        if names is not None and test.name not in names:
            continue
        logger.info("road-testing %s (%d turbines)", test.name, len(test.turbines))
        try:
            frames.append(run_one(test, out_dir=run_dir))
        except Exception:
            logger.exception("road test %s failed", test.name)
    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    results.to_csv(run_dir / "roadtest.csv", index=False)
    logger.info("wrote the road-test results to %s", run_dir)
    return results


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Per (farm, test turbine): the headline, the reference sanity check and what was screened."""
    if results.empty:
        return results
    grouped = results.groupby(["farm", "test_wtg"])
    return pd.DataFrame(
        {
            "test_uplift_pct": grouped["test_uplift_pct"].first(),
            "reference_overall_pct": grouped["reference_overall_pct"].first(),
            "n_references": grouped["reference"].nunique(),
            "n_screened": grouped["screened"].sum(),
            "screened": grouped.apply(
                lambda g: ",".join(sorted(g.loc[g["screened"], "reference"])) or "-", include_groups=False
            ),
        }
    ).reset_index()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    table = run_roadtest()
    print(summarise(table).to_string(index=False))  # noqa: T201 - a driver's whole point is its printed summary
