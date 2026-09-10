"""The composed ``wind-up``: one name, one declaration, one answer.

``wind-up`` is campaign-level rather than a :class:`~benchmarking.harness.Method`. At method
level it would relabel ``power_model``, whose reference-validity screen is already its own and
whose northing is applied farm-wide upstream. What this module composes is the shared northing
step, one ``power_model`` built from the accepted defaults, and the truth-free report.

Run one::

    python -m benchmarking.campaigns run campaign.yaml --out DIR

The module name is a placeholder: ``wind_up.py`` inside ``benchmarking/campaigns/`` reads badly
next to the real ``wind_up`` package.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.baselines.power_model import CURATED_ERA5_EXCLUDE, TUNED_MODEL_PARAMS, PowerModelMethod
from benchmarking.campaigns.loader import load_declaration
from benchmarking.campaigns.report import write_report
from benchmarking.campaigns.run import estimate_campaign
from benchmarking.harness.northing import era5_direction
from wind_up_v0.era5 import get_era5_hourly_df

if TYPE_CHECKING:
    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.campaigns.loader import Declaration
    from benchmarking.campaigns.run import CampaignReport
    from benchmarking.harness import Method
    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)

WIND_UP = "wind-up"

# What the resolved declaration is echoed to, so a mis-declared timezone is visible after a run.
RESOLVED_FILENAME = "resolved_campaign.json"

OUTPUT_DIR_ENV = "WIND_UP_BENCHMARKING_OUTPUT_DIR"


def wind_up_method(
    spec: CampaignSpec, *, columns: ColumnSchema, out_dir: Path, era5_hourly_df: pd.DataFrame | None
) -> Method:
    """Build ``wind-up``'s estimator for one campaign.

    The accepted ``power_model`` defaults, under one name. The reference-validity screen and the
    reference-stability table are its own defaults, so nothing is turned on here.

    :param spec: the campaign being run; supplies the turbine rating
    :param columns: the source-native schema the SCADA is keyed by
    :param out_dir: where the method writes its own diagnostics
    :param era5_hourly_df: reanalysis; without it the per-condition estimates are not reported
    """
    return PowerModelMethod(
        name=WIND_UP,
        columns=columns,
        baseline_rated_power_kw=spec.rated_power_kw,
        era5_hourly_df=era5_hourly_df,
        conditions=PowerModelMethod.conditions if era5_hourly_df is not None else (),
        availability_feature=False,
        era5_exclude=CURATED_ERA5_EXCLUDE,
        model_params=dict(TUNED_MODEL_PARAMS),
        out_dir=out_dir,
        save_plots=True,
    )


def default_out_dir(name: str) -> Path:
    """Return the directory a run of ``name`` writes to when none is given."""
    root = Path(os.getenv(OUTPUT_DIR_ENV, Path.home() / "temp" / "wind-up-benchmarking"))
    return root / name


def run_declaration(
    path: str | Path, *, out_dir: Path | None = None, era5_hourly_df: pd.DataFrame | None = None
) -> CampaignReport:
    """Run the campaign declared at ``path`` and write its report.

    :param path: the campaign declaration
    :param out_dir: where to write; defaults to :func:`default_out_dir` of the campaign's name
    :param era5_hourly_df: reanalysis to use instead of self-serving it from the farm centroid,
        for a caller that already holds it
    :return: the truth-free campaign report, which is also written under ``out_dir``
    """
    declaration = load_declaration(path)
    out_dir = out_dir if out_dir is not None else default_out_dir(declaration.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / RESOLVED_FILENAME).write_text(json.dumps(declaration.resolved(), indent=2))
    logger.info("Running campaign %r into %s", declaration.name, out_dir)

    scada_df = pd.read_parquet(declaration.scada_path)
    reanalysis = era5_hourly_df if era5_hourly_df is not None else _fetch_era5(declaration)
    index = pd.DatetimeIndex(scada_df.index.unique()).sort_values()

    report = estimate_campaign(
        declaration.spec,
        scada_df,
        build_methods=lambda wtg: [
            wind_up_method(
                declaration.spec,
                columns=declaration.columns,
                out_dir=out_dir / wtg / WIND_UP,
                era5_hourly_df=reanalysis,
            )
        ],
        columns=declaration.columns,
        era5_wd=era5_direction(reanalysis, index),
        northing_out_dir=out_dir / "northing",
    )
    write_report(report, out_dir=out_dir)
    logger.info("Wrote the %r report to %s", declaration.name, out_dir)
    return report


def _fetch_era5(declaration: Declaration) -> pd.DataFrame:
    """Fetch reanalysis for the farm centroid over the declaration's whole-year window.

    The fetch itself needs the optional ``era5`` dependency group and network on a cache miss.
    """
    lat, lon = declaration.centroid
    start_date, end_date = declaration.era5_window
    logger.info("Fetching reanalysis for (%.4f, %.4f) over %s..%s", lat, lon, start_date, end_date)
    return get_era5_hourly_df(lat=lat, lon=lon, start_date=start_date, end_date=end_date)
