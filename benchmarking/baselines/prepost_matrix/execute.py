"""Run one cell of the prepost campaign matrix: a full shipped ``wind-up`` run, and what it recorded.

Each cell reads the SCADA and reanalysis :func:`prefetch` put in the study's ``sources/`` folder and
the reanalysis cache, so cells run offline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from benchmarking.baselines.prepost_matrix.cells import REAL_SITE, Cell, campaign_seed, draw_exclusions
from benchmarking.campaigns.composed import WIND_UP, wind_up_method
from benchmarking.campaigns.loader import centroid, era5_window, load_declaration
from benchmarking.campaigns.real import (
    HOT_AEROUP_T13_END,
    HOT_AEROUP_T13_START,
    HOT_AEROUP_WORKS,
    write_hot_aeroup_t13,
)
from benchmarking.campaigns.rollout import draw_rollout, hot_site, kelmarsh_site, penmanshiel_site, rollout_campaign
from benchmarking.campaigns.run import estimate_campaign
from benchmarking.campaigns.runner import CampaignRunner, per_turbine_table
from benchmarking.diagnostics.context import era5_source_label
from benchmarking.harness.northing import NORTH_TABLE_YAML, era5_direction
from benchmarking.synthetic.sources.greenbyte import (
    KELMARSH,
    PENMANSHIEL,
    ensure_greenbyte_data,
    load_greenbyte_scada,
)
from benchmarking.synthetic.sources.hill_of_towie import HOT_COORDINATES, ensure_hot_data_files, load_hot_scada
from wind_up.analysis_period import PlanSettings
from wind_up_v0.era5 import get_era5_hourly_df

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from benchmarking.baselines.prepost_matrix.cells import MatrixSettings
    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.campaigns.rollout import RolloutSite
    from benchmarking.harness import Method
    from benchmarking.synthetic import ColumnSchema
    from wind_up.analysis_period import AnalysisPlan

logger = logging.getLogger(__name__)

SITES: dict[str, Callable[[], RolloutSite]] = {"hot": hot_site, "pen": penmanshiel_site, "kel": kelmarsh_site}
GREENBYTE_FARMS = {"pen": PENMANSHIEL, "kel": KELMARSH}
HOT_METADATA = "Hill_of_Towie_turbine_metadata.csv"
SOURCES_DIRNAME = "sources"
METHOD_DIRNAME = "method"
NORTHING_DIRNAME = "northing"
REAL_T13 = "T13"


def sources_dir(study_dir: Path) -> Path:
    """Return where a study keeps the SCADA its cells read."""
    return study_dir / SOURCES_DIRNAME


def _site_scada_path(study_dir: Path, site: str) -> Path:
    return sources_dir(study_dir) / f"{site}.parquet"


def _real_declaration_path(study_dir: Path) -> Path:
    return sources_dir(study_dir) / REAL_SITE / "campaign.yaml"


def download_sources(cells: Iterable[Cell]) -> None:
    """Download from Zenodo the open data every site ``cells`` need, and cache their reanalysis.

    Files already present are not fetched again, so this makes no network call once complete.
    """
    for site_key in sorted({cell.site for cell in cells}):
        if site_key == REAL_SITE:
            years = range(HOT_AEROUP_T13_START.year, HOT_AEROUP_T13_END.year)
            ensure_hot_data_files([*(f"{y}.zip" for y in years), HOT_AEROUP_WORKS, HOT_METADATA])
            _real_reanalysis(centroid(dict(HOT_COORDINATES)))
            continue
        if site_key in GREENBYTE_FARMS:
            ensure_greenbyte_data(GREENBYTE_FARMS[site_key])
        site = SITES[site_key]()
        if site_key == "hot":
            ensure_hot_data_files(
                [*(f"{y}.zip" for y in range(site.data_start.year, site.data_end.year)), HOT_METADATA]
            )
        _site_reanalysis(site)


def prefetch(study_dir: Path, cells: Iterable[Cell]) -> None:
    """Download what ``cells`` need, then write each site's SCADA under ``sources/``.

    A site already written is not reloaded. After this the cells run offline.
    """
    cells = list(cells)
    download_sources(cells)
    sources_dir(study_dir).mkdir(parents=True, exist_ok=True)
    for site_key in sorted({cell.site for cell in cells}):
        if site_key == REAL_SITE:
            path = _real_declaration_path(study_dir)
            if not path.exists():
                write_hot_aeroup_t13(path.parent)
            continue
        path = _site_scada_path(study_dir, site_key)
        if not path.exists():
            site = SITES[site_key]()
            logger.info("Loading %s SCADA %s..%s", site.name, site.data_start, site.data_end)
            partial = path.with_suffix(".partial")
            _load_site_scada(site_key, site).to_parquet(partial)
            partial.replace(path)


def _load_site_scada(site_key: str, site: RolloutSite) -> pd.DataFrame:
    if site_key == "hot":
        scada, _ = load_hot_scada(start_dt=site.data_start, end_dt_excl=site.data_end)
        return scada
    farm = GREENBYTE_FARMS[site_key]
    return load_greenbyte_scada(farm, years=farm.years)


def _site_reanalysis(site: RolloutSite) -> pd.DataFrame:
    lat, lon = centroid(dict(site.coords))
    start, end = era5_window(site.data_start, site.data_end)
    return get_era5_hourly_df(lat=lat, lon=lon, start_date=start, end_date=end)


def _real_reanalysis(site_centroid: tuple[float, float]) -> pd.DataFrame:
    start, end = era5_window(HOT_AEROUP_T13_START, HOT_AEROUP_T13_END)
    return get_era5_hourly_df(lat=site_centroid[0], lon=site_centroid[1], start_date=start, end_date=end)


def _read_scada(path: Path, *, start: pd.Timestamp | None = None, end: pd.Timestamp) -> pd.DataFrame:
    """Read the SCADA in ``[start, end)`` alone, rather than reading everything and cutting it."""
    index = json.loads(pq.read_schema(path).metadata[b"pandas"])["index_columns"][0]
    filters = [(index, "<", end), *([(index, ">=", start)] if start is not None else [])]
    return pd.read_parquet(path, filters=filters)


def execute_cell(cell: Cell, *, study_dir: Path, cell_dir: Path, settings: MatrixSettings) -> dict[str, Any]:
    """Run ``cell`` and return what it recorded: estimates, truths, reference readings and diagnostics.

    :param cell: the cell to run
    :param study_dir: the study, whose ``sources/`` :func:`prefetch` has filled
    :param cell_dir: where the run writes its method diagnostics and discovered north table
    :param settings: the matrix, for its master seed
    """
    if cell.arm == "real":
        return _execute_real(cell, study_dir=study_dir, cell_dir=cell_dir, settings=settings)
    return _execute_synthetic(cell, study_dir=study_dir, cell_dir=cell_dir, settings=settings)


def _execute_synthetic(cell: Cell, *, study_dir: Path, cell_dir: Path, settings: MatrixSettings) -> dict[str, Any]:
    site = SITES[cell.site]()
    seed = campaign_seed(settings.master_seed, site=cell.site, seed_index=cell.seed_index)  # type: ignore[arg-type]
    draw = draw_rollout(site, seed=seed, full_rollout=cell.full_rollout)
    campaign = rollout_campaign(draw, multiplier=float(cell.multiplier), post_months=cell.post_months)  # type: ignore[arg-type]
    if cell.exclusion_channels is not None:
        exclusions = draw_exclusions(
            [t for t in site.coords if t not in draw.trial], start=site.data_start, end=site.data_end, seed=seed
        )
        campaign = replace(campaign, exclusions=exclusions)
    start, end = draw.data_window(cell.post_months)
    scada = _read_scada(_site_scada_path(study_dir, cell.site), start=start, end=end)
    dataset = campaign.generate(scada)
    spec = campaign.spec()
    era5 = _site_reanalysis(site)
    index = pd.DatetimeIndex(dataset.synthetic_df.index.unique()).sort_values()
    screen_cache: dict = {}
    label = era5_source_label(*centroid(dict(site.coords)))
    result = CampaignRunner(
        spec,
        dataset,
        build_methods=lambda wtg: [
            _matrix_method(
                spec,
                columns=site.columns,
                out_dir=cell_dir / METHOD_DIRNAME / wtg,
                era5=era5,
                screen_cache=screen_cache,
                era5_label=label,
                exclusion_channels=cell.exclusion_channels,
            )
        ],
        era5_wd=era5_direction(era5, index),
        northing_out_dir=cell_dir / NORTHING_DIRNAME,
        northing_plots=False,
        plan_settings=plan_settings(cell, settings=settings),
    ).run()

    per_turbine = per_turbine_table(result)
    per_turbine = per_turbine[per_turbine["method"] == WIND_UP]
    farm = result.farm.set_index("method").loc[WIND_UP]
    return {
        "draw": {
            "campaign_seed": seed,
            "full_rollout": cell.full_rollout,
            "trial": list(draw.trial),
            "rollout": list(draw.rollout),
            "works": {t: [str(s), str(e)] for t, (s, e) in draw.works.items()},
            "exclusions": [[t, str(s), str(e)] for t, s, e in spec.exclusions],
            "data_window": [str(start), str(end)],
        },
        "turbines": [
            {"turbine": str(r.test_wtg), "estimate": float(r.estimate), "truth": float(r.truth)}
            for r in per_turbine.itertuples()
        ],
        "farm": {"estimate": float(farm["estimate"]), "truth": float(farm["truth"])},
        **_diagnostics(result.report.reference_stability, plans=result.report.plans, cell_dir=cell_dir),
    }


def _execute_real(cell: Cell, *, study_dir: Path, cell_dir: Path, settings: MatrixSettings) -> dict[str, Any]:
    declaration = load_declaration(_real_declaration_path(study_dir))
    spec = declaration.spec
    end = pd.Timestamp(spec.timing_for(REAL_T13)) + pd.DateOffset(months=cell.post_months)  # type: ignore[arg-type]
    scada = _read_scada(declaration.scada_path, end=end)
    era5 = _real_reanalysis(declaration.centroid)
    index = pd.DatetimeIndex(scada.index.unique()).sort_values()
    screen_cache: dict = {}
    label = era5_source_label(*declaration.centroid)
    report = estimate_campaign(
        spec,
        scada,
        build_methods=lambda wtg: [
            _matrix_method(
                spec,
                columns=declaration.columns,
                out_dir=cell_dir / METHOD_DIRNAME / wtg,
                era5=era5,
                screen_cache=screen_cache,
                era5_label=label,
                exclusion_channels=None,
            )
        ],
        columns=declaration.columns,
        era5_wd=era5_direction(era5, index),
        northing_out_dir=cell_dir / NORTHING_DIRNAME,
        northing_plots=False,
        plan_settings=plan_settings(cell, settings=settings),
    )
    per_turbine = report.per_turbine[report.per_turbine["method"] == WIND_UP]
    return {
        "draw": {"data_end": str(end)},
        "turbines": [
            {"turbine": str(r.test_wtg), "estimate": float(r.estimate), "truth": None} for r in per_turbine.itertuples()
        ],
        "farm": {"estimate": float(report.farm.set_index("method").loc[WIND_UP, "estimate"]), "truth": None},
        **_diagnostics(report.reference_stability, plans=report.plans, cell_dir=cell_dir),
    }


def plan_settings(cell: Cell, *, settings: MatrixSettings) -> PlanSettings:
    """Return the period selector's settings for ``cell``: its K and the matrix's shortest side."""
    return PlanSettings(k=cell.k, min_side=pd.Timedelta(days=settings.min_side_days))


def _matrix_method(
    spec: CampaignSpec,
    *,
    columns: ColumnSchema,
    out_dir: Path,
    era5: pd.DataFrame,
    screen_cache: dict,
    era5_label: str,
    exclusion_channels: str | None,
) -> Method:
    """Return shipped ``wind-up`` on one thread, without plots, with the cell's exclusion channels."""
    method = wind_up_method(
        spec, columns=columns, out_dir=out_dir, era5_hourly_df=era5, screen_cache=screen_cache, era5_label=era5_label
    )
    overrides: dict[str, Any] = {"save_plots": False, "model_params": {**method.model_params, "n_jobs": 1}}  # type: ignore[attr-defined]
    if exclusion_channels is not None:
        overrides["exclusion_channels"] = exclusion_channels
    return replace(method, **overrides)  # type: ignore[type-var]


def _diagnostics(references: pd.DataFrame, *, plans: Mapping[str, AnalysisPlan], cell_dir: Path) -> dict[str, Any]:
    """Return the reference readings, the plans, the screen's ejections and the northing found."""
    references = references[references["method"] == WIND_UP]
    readings = [
        {
            "test_wtg": str(r.test_wtg),
            "turbine": str(r.turbine),
            "uplift": float(r.uplift),
            "n_records": int(r.n_records),
            "screened": bool(r.screened),
            "unjudged": bool(r.unjudged),
        }
        for r in references.itertuples()
    ]
    table_path = cell_dir / NORTHING_DIRNAME / NORTH_TABLE_YAML
    table = yaml.safe_load(table_path.read_text()) or [] if table_path.exists() else []
    northing = [[str(device), str(stamp), float(offset)] for device, stamp, offset in table]
    return {
        "references": readings,
        "screen_ejections": {
            wtg: sorted(r["turbine"] for r in readings if r["test_wtg"] == wtg and r["screened"])
            for wtg in sorted(plans)
        },
        "plans": {turbine: _plan_record(plan) for turbine, plan in sorted(plans.items())},
        "northing": northing,
        "n_northing_changepoints": len(northing) - len({device for device, _, _ in northing}),
    }


def _plan_record(plan: AnalysisPlan) -> dict[str, object]:
    return {
        **plan.summary(),
        "power_reference_distances_d": {r: round(plan.distances_d[r], 3) for r in plan.power_references},
        "waking_only": dict(plan.waking_only),
    }
