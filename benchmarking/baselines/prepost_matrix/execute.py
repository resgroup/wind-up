"""Run one cell of the prepost campaign matrix: a full shipped ``wind-up`` run, and what it recorded.

Each cell reads the SCADA and reanalysis :func:`prefetch` put in the study's detail directory's
``sources/`` folder and the reanalysis cache, so cells run offline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from benchmarking.baselines.power_model.method import is_reading_column
from benchmarking.baselines.prepost_matrix.cells import REAL_SITE, Cell, campaign_seed, draw_exclusions
from benchmarking.campaigns.composed import DATA_PREPARATION_DIRNAME, WIND_UP, wind_up_method
from benchmarking.campaigns.loader import centroid, era5_window, load_declaration, operating_state_config
from benchmarking.campaigns.preparation import prepare_data
from benchmarking.campaigns.real import (
    HOT_AEROUP_T13_END,
    HOT_AEROUP_T13_START,
    HOT_AEROUP_WORKS,
    HOT_OPERATING_STATE,
    HOT_STATE_COL,
    label_hot_site_states,
    write_hot_aeroup_t13,
)
from benchmarking.campaigns.rollout import draw_rollout, hot_site, kelmarsh_site, penmanshiel_site, rollout_campaign
from benchmarking.campaigns.run import estimate_campaign
from benchmarking.campaigns.runner import CampaignRunner, per_turbine_table
from benchmarking.diagnostics import stages
from benchmarking.diagnostics.context import era5_source_label
from benchmarking.harness.northing import NORTH_TABLE_YAML
from benchmarking.harness.operating_state import OperatingStateConfig
from benchmarking.harness.reanalysis import ERA5_WS_RAW, normalise_timestamps
from benchmarking.synthetic.ground_truth import ERA5_WIND_SPEED_COL
from benchmarking.synthetic.sources.greenbyte import (
    GREENBYTE_OPERATING_STATE,
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
    from benchmarking.campaigns.data_screen import DataScreen
    from benchmarking.campaigns.declaration import CampaignSpec, Exclusion
    from benchmarking.campaigns.rollout import RolloutDraw, RolloutSite
    from benchmarking.harness import Method, MethodOutput
    from benchmarking.synthetic import ColumnSchema
    from wind_up.analysis_period import AnalysisPlan

logger = logging.getLogger(__name__)

SITES: dict[str, Callable[[], RolloutSite]] = {"hot": hot_site, "pen": penmanshiel_site, "kel": kelmarsh_site}
GREENBYTE_FARMS = {"pen": PENMANSHIEL, "kel": KELMARSH}
HOT_METADATA = "Hill_of_Towie_turbine_metadata.csv"
SOURCES_DIRNAME = "sources"
METHOD_DIRNAME = "method"
# The test turbine's measurement folder holds one ``*_record_losses_*.csv``.
RECORD_LOSSES_STEM = "_record_losses_"
REAL_T13 = "T13"


def sources_dir(detail_dir: Path) -> Path:
    """Return where a study's detail directory keeps the SCADA its cells read."""
    return detail_dir / SOURCES_DIRNAME


def _site_scada_path(detail_dir: Path, site: str) -> Path:
    return sources_dir(detail_dir) / f"{site}.parquet"


def _real_declaration_path(detail_dir: Path) -> Path:
    return sources_dir(detail_dir) / REAL_SITE / "campaign.yaml"


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


def prefetch(detail_dir: Path, cells: Iterable[Cell]) -> None:
    """Download what ``cells`` need, then write each site's SCADA under the detail directory's ``sources/``.

    A site already written is not reloaded. After this the cells run offline.
    """
    cells = list(cells)
    download_sources(cells)
    sources_dir(detail_dir).mkdir(parents=True, exist_ok=True)
    for site_key in sorted({cell.site for cell in cells}):
        if site_key == REAL_SITE:
            path = _real_declaration_path(detail_dir)
            if not path.exists():
                write_hot_aeroup_t13(path.parent)
            continue
        path = _site_scada_path(detail_dir, site_key)
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


def execute_cell(
    cell: Cell,
    *,
    detail_dir: Path,
    cell_dir: Path,
    settings: MatrixSettings,
    method_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run ``cell`` and return what it recorded: estimates, truths, reference readings and diagnostics.

    :param cell: the cell to run
    :param detail_dir: the study's detail directory, whose ``sources/`` :func:`prefetch` has filled
    :param cell_dir: where the run writes its method diagnostics and discovered north table
    :param settings: the matrix, for its master seed
    :param method_overrides: fields replaced on every turbine's method after the matrix's own, for a
        probe that needs the method configured differently (the level probe's ``row_dump_dir``)
    """
    run = _execute_real if cell.arm == "real" else _execute_synthetic
    return run(cell, detail_dir=detail_dir, cell_dir=cell_dir, settings=settings, method_overrides=method_overrides)


def describe_cell(cell: Cell, *, settings: MatrixSettings) -> dict[str, Any]:
    """Return ``cell``'s campaign draw as the record carries it; empty for the real campaign.

    Deterministic and cheap, so it is recorded before the cell runs and a failed cell keeps it.
    """
    if cell.arm == "real":
        return {}
    _, seed, draw, exclusions = _draw(cell, settings=settings)
    start, end = draw.data_window(cell.post_months)
    return {
        "draw": {
            "campaign_seed": seed,
            "full_rollout": cell.full_rollout,
            "trial": list(draw.trial),
            "rollout": list(draw.rollout),
            "works": {t: [str(s), str(e)] for t, (s, e) in draw.works.items()},
            "exclusions": [[t, str(s), str(e)] for t, s, e in exclusions],
            "data_window": [str(start), str(end)],
        }
    }


def _draw(cell: Cell, *, settings: MatrixSettings) -> tuple[RolloutSite, int, RolloutDraw, list[Exclusion]]:
    """Return a synthetic cell's site, campaign seed, rollout draw and reference exclusions."""
    site = SITES[cell.site]()
    seed = campaign_seed(settings.master_seed, site=cell.site, seed_index=cell.seed_index)  # type: ignore[arg-type]
    draw = draw_rollout(site, seed=seed, full_rollout=cell.full_rollout)
    exclusions: list[Exclusion] = []
    if cell.has_exclusions:
        exclusions = draw_exclusions(
            [t for t in site.coords if t not in draw.trial], start=site.data_start, end=site.data_end, seed=seed
        )
    return site, seed, draw, exclusions


def _execute_synthetic(
    cell: Cell,
    *,
    detail_dir: Path,
    cell_dir: Path,
    settings: MatrixSettings,
    method_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    site, _, draw, exclusions = _draw(cell, settings=settings)
    campaign = rollout_campaign(draw, multiplier=float(cell.multiplier), post_months=cell.post_months)  # type: ignore[arg-type]
    if cell.has_exclusions:
        campaign = replace(campaign, exclusions=exclusions)
    start, end = draw.data_window(cell.post_months)
    scada = _read_scada(_site_scada_path(detail_dir, cell.site), start=start, end=end)
    dataset = campaign.generate(scada)
    labelled, operating_state = site_operating_state(cell.site, dataset.synthetic_df)
    prepared = prepare_data(
        labelled,
        spec=campaign.spec(),
        columns=site.columns,
        era5_hourly_df=_site_reanalysis(site),
        operating_state=operating_state,
        out_dir=cell_dir / DATA_PREPARATION_DIRNAME,
        plots=False,
    )
    spec = prepared.with_north_offsets(campaign.spec())
    screen_cache: dict = {}
    label = era5_source_label(*centroid(dict(site.coords)))
    result = CampaignRunner(
        spec,
        replace(
            dataset,
            synthetic_df=prepared.scada_df,
            original_df=with_reanalysis_wind_speed(dataset.original_df, prepared.aligned_era5),
        ),
        build_methods=lambda wtg: [
            _matrix_method(
                spec,
                columns=site.columns,
                out_dir=cell_dir / METHOD_DIRNAME / wtg,
                era5=prepared.aligned_era5,
                screen_cache=screen_cache,
                era5_label=label,
                method_overrides=method_overrides,
            )
        ],
        northing_plots=False,
        plan_settings=plan_settings(cell, settings=settings),
        possible_changes=prepared.possible_changes_by_turbine(),
    ).run()

    per_turbine = per_turbine_table(result)
    per_turbine = per_turbine[per_turbine["method"] == WIND_UP]
    farm = result.farm.set_index("method").loc[WIND_UP]
    return {
        **describe_cell(cell, settings=settings),
        "turbines": [
            {
                "turbine": str(r.test_wtg),
                "estimate": float(r.estimate),
                "truth": float(r.truth),
                **sigmas(result.outputs[WIND_UP, str(r.test_wtg)]),
            }
            for r in per_turbine.itertuples()
        ],
        "farm": {"estimate": float(farm["estimate"]), "truth": float(farm["truth"]), **combined_references(farm)},
        "bins": scored_bins(result.scores),
        **diagnostics(
            result.report.reference_stability,
            plans=result.report.plans,
            unplanned=result.report.unplanned,
            cell_dir=cell_dir,
        ),
        **measurement_details(result.report.data_screens, cell_dir=cell_dir),
    }


def _execute_real(
    cell: Cell,
    *,
    detail_dir: Path,
    cell_dir: Path,
    settings: MatrixSettings,
    method_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    declaration = load_declaration(_real_declaration_path(detail_dir))
    end = pd.Timestamp(declaration.spec.timing_for(REAL_T13)) + pd.DateOffset(months=cell.post_months)  # type: ignore[arg-type]
    scada = _read_scada(declaration.scada_path, end=end)
    timebase = pd.Timedelta(np.median(np.diff(pd.DatetimeIndex(scada.index.unique()).sort_values().to_numpy())))
    scada = normalise_timestamps(scada, timestamps=declaration.timestamps, timebase=timebase)
    prepared = prepare_data(
        scada,
        spec=declaration.spec,
        columns=declaration.columns,
        era5_hourly_df=_real_reanalysis(declaration.centroid),
        operating_state=declaration.operating_state,
        cell_selection=declaration.cell_selection,
        out_dir=cell_dir / DATA_PREPARATION_DIRNAME,
        plots=False,
    )
    spec = prepared.with_north_offsets(declaration.spec)
    screen_cache: dict = {}
    label = era5_source_label(*declaration.centroid)
    report = estimate_campaign(
        spec,
        prepared.scada_df,
        build_methods=lambda wtg: [
            _matrix_method(
                spec,
                columns=declaration.columns,
                out_dir=cell_dir / METHOD_DIRNAME / wtg,
                era5=prepared.aligned_era5,
                screen_cache=screen_cache,
                era5_label=label,
                method_overrides=method_overrides,
            )
        ],
        columns=declaration.columns,
        northing_plots=False,
        plan_settings=plan_settings(cell, settings=settings),
        possible_changes=prepared.possible_changes_by_turbine(),
    )
    per_turbine = report.per_turbine[report.per_turbine["method"] == WIND_UP]
    return {
        "draw": {"data_end": str(end)},
        "turbines": [
            {
                "turbine": str(r.test_wtg),
                "estimate": float(r.estimate),
                "truth": None,
                **sigmas(report.outputs[WIND_UP, str(r.test_wtg)]),
            }
            for r in per_turbine.itertuples()
        ],
        "farm": {
            "estimate": float(report.farm.set_index("method").loc[WIND_UP, "estimate"]),
            "truth": None,
            **combined_references(report.farm.set_index("method").loc[WIND_UP]),
        },
        "bins": reported_bins(report.outputs),
        **diagnostics(report.reference_stability, plans=report.plans, unplanned=report.unplanned, cell_dir=cell_dir),
        **measurement_details(report.data_screens, cell_dir=cell_dir),
    }


def site_operating_state(site_key: str, scada: pd.DataFrame) -> tuple[pd.DataFrame, OperatingStateConfig]:
    """Return ``scada`` with the site's operating-state labels, and the labels' validity.

    Hill of Towie gets its site labels; Kelmarsh and Penmanshiel carry theirs from the loader.
    """
    if site_key == "hot":
        labelled = scada.assign(**{HOT_STATE_COL: label_hot_site_states(scada)})
        return labelled, operating_state_config(HOT_OPERATING_STATE)
    if site_key in GREENBYTE_FARMS:
        return scada, operating_state_config(GREENBYTE_OPERATING_STATE)
    return scada, OperatingStateConfig()


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
    method_overrides: Mapping[str, Any] | None = None,
) -> Method:
    """Return shipped ``wind-up`` on one thread, without plots, recording the reverse-direction reading.

    ``method_overrides`` is applied last, so a probe's field wins over the matrix's own.
    """
    method = wind_up_method(
        spec, columns=columns, out_dir=out_dir, era5_hourly_df=era5, screen_cache=screen_cache, era5_label=era5_label
    )
    overrides: dict[str, Any] = {
        "save_plots": False,
        "model_params": {**method.model_params, "n_jobs": 1},  # type: ignore[attr-defined]
        "record_reversal": True,
    }
    overrides.update(method_overrides or {})
    return replace(method, **overrides)  # type: ignore[type-var]


def diagnostics(
    references: pd.DataFrame,
    *,
    plans: Mapping[str, AnalysisPlan],
    unplanned: Mapping[str, str],
    cell_dir: Path,
) -> dict[str, Any]:
    """Return the reference readings, the plans and unplanned turbines, the screen's ejections and the northing."""
    references = references[references["method"] == WIND_UP]
    reading_columns = [c for c in references.columns if is_reading_column(c)]
    readings = [
        {
            "test_wtg": str(row["test_wtg"]),
            "turbine": str(row["turbine"]),
            "uplift": float(row["uplift"]),
            "n_records": int(row["n_records"]),
            "screened": bool(row["screened"]),
            "unjudged": bool(row["unjudged"]),
            **{c: float(row[c]) for c in reading_columns},
        }
        for _, row in references.iterrows()
    ]
    table_path = cell_dir / DATA_PREPARATION_DIRNAME / stages.NORTHING / NORTH_TABLE_YAML
    table = yaml.safe_load(table_path.read_text()) or [] if table_path.exists() else []
    northing = [[str(device), str(stamp), float(offset)] for device, stamp, offset in table]
    return {
        "references": readings,
        "screen_ejections": {
            wtg: sorted(r["turbine"] for r in readings if r["test_wtg"] == wtg and r["screened"])
            for wtg in sorted(plans)
        },
        "plans": {turbine: _plan_record(plan) for turbine, plan in sorted(plans.items())},
        "unplanned": dict(sorted(unplanned.items())),
        "northing": northing,
        "n_northing_changepoints": len(northing) - len({device for device, _, _ in northing}),
    }


def measurement_details(data_screens: Mapping[str, DataScreen], *, cell_dir: Path) -> dict[str, Any]:
    """Return each test turbine's hours lost per reason and period, and its data screen.

    ``record_losses`` is read from the test turbine's measurement folder under ``cell_dir``; a
    turbine without one is left out. ``data_screen`` holds the candidates' table and the joint usable
    hours against the number of references.
    """
    losses: dict[str, list[dict[str, Any]]] = {}
    for wtg in sorted(data_screens):
        found = sorted((cell_dir / METHOD_DIRNAME / wtg).glob(f"*/*{RECORD_LOSSES_STEM}*.csv"))
        if found:
            losses[wtg] = _plain_rows(pd.read_csv(found[-1]))
    screens = {
        wtg: {"candidates": _plain_rows(screen.table), "hours": _plain_rows(screen.hours)}
        for wtg, screen in sorted(data_screens.items())
    }
    return {"record_losses": losses, "data_screen": screens}


def _plain_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return ``frame``'s rows as JSON-ready dicts: numbers as floats, NaN as None, the rest as strings."""
    rows = []
    for row in frame.to_dict(orient="records"):
        plain: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (bool, np.bool_)):
                plain[str(key)] = bool(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                plain[str(key)] = None if pd.isna(value) else float(value)
            else:
                plain[str(key)] = None if value is None else str(value)
        rows.append(plain)
    return rows


def with_reanalysis_wind_speed(original_df: pd.DataFrame, aligned_era5: pd.DataFrame | None) -> pd.DataFrame:
    """Return ``original_df`` with the aligned ERA5 100 m wind speed joined on by timestamp, for the per-bin truth."""
    if aligned_era5 is None or ERA5_WS_RAW not in aligned_era5.columns:
        return original_df
    return original_df.join(aligned_era5[[ERA5_WS_RAW]].rename(columns={ERA5_WS_RAW: ERA5_WIND_SPEED_COL}), how="left")


def scored_bins(scores: pd.DataFrame) -> list[dict[str, Any]]:
    """Return ``wind-up``'s per-bin rows of the harness scores: estimate, truth and sigma per condition bin."""
    rows = scores[(scores["method"] == WIND_UP) & (scores["condition"] != "overall")]
    return [
        {
            "test_wtg": str(r.test_wtg),
            "condition": str(r.condition),
            "condition_bin": str(r.condition_bin),
            "estimate": float(r.estimate),
            "truth": float(r.truth),
            "sigma": float(r.sigma),
        }
        for r in rows.itertuples()
    ]


def reported_bins(outputs: Mapping[tuple[str, str], MethodOutput]) -> list[dict[str, Any]]:
    """Return ``wind-up``'s per-bin rows from its outputs, with no truth."""
    rows: list[dict[str, Any]] = []
    for (method, wtg), output in sorted(outputs.items()):
        if method != WIND_UP or output.p50_by_condition is None:
            continue
        frame = output.p50_by_condition
        sigma = frame["sigma_uplift"] if "sigma_uplift" in frame.columns else pd.Series(np.nan, index=frame.index)
        rows.extend(
            {
                "test_wtg": wtg,
                "condition": str(c),
                "condition_bin": str(b),
                "estimate": float(e),
                "truth": None,
                "sigma": float(s),
            }
            for c, b, e, s in zip(frame["condition"], frame["condition_bin"], frame["p50_uplift"], sigma, strict=True)
        )
    return rows


def sigmas(output: MethodOutput) -> dict[str, float]:
    """Return a measurement's ``sigma``, its ``sigma_block_<h>h`` sweep and its reverse-direction reading.

    NaN where it reports none.
    """
    out = {"sigma": float("nan") if output.sigma_overall is None else float(output.sigma_overall)}
    diagnostics = output.uncertainty_diagnostics
    if diagnostics is not None and not diagnostics.empty:
        row = diagnostics.iloc[0]
        out.update({str(c): float(row[c]) for c in diagnostics.columns if is_reading_column(c) and c != "sigma"})
    return out


def combined_references(farm: pd.Series) -> dict[str, float]:
    """Return the farm row's combined reference uplift before and after the screen, and the reference spread."""
    return {
        key: float(farm[key])
        for key in ("reference_uplift_before_screen", "reference_uplift_after_screen", "reference_uplift_spread")
        if key in farm.index
    }


def _plan_record(plan: AnalysisPlan) -> dict[str, object]:
    return {
        **plan.summary(),
        "power_reference_distances_d": {r: round(plan.distances_d[r], 3) for r in plan.power_references},
        "waking_only": dict(plan.waking_only),
    }
