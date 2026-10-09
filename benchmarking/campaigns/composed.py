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

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

from benchmarking.baselines.power_model import CURATED_ERA5_EXCLUDE, TUNED_MODEL_PARAMS, PowerModelMethod
from benchmarking.campaigns.declaration import layout_coords
from benchmarking.campaigns.loader import era5_window, load_declaration
from benchmarking.campaigns.report import write_report
from benchmarking.campaigns.run import estimate_campaign
from benchmarking.diagnostics import stages
from benchmarking.diagnostics.context import ERA5_UNLOCATED, era5_source_label, infer_timebase
from benchmarking.diagnostics.input_data import write_input_data_plots
from benchmarking.diagnostics.operating_states import (
    NORTHING_VIEWS,
    WAKING_VIEWS,
    write_operating_state_plots,
    write_validity_plots,
)
from benchmarking.diagnostics.reanalysis import write_reanalysis_outputs
from benchmarking.harness.operating_state import label_operating_states
from benchmarking.harness.reanalysis import ERA5_WD_RAW, normalise_timestamps, prepare_reanalysis
from wind_up.analysis_period import DEFAULT_PLAN_SETTINGS
from wind_up_v0.era5 import get_era5_hourly_df

if TYPE_CHECKING:
    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.campaigns.loader import Declaration
    from benchmarking.campaigns.run import CampaignReport
    from benchmarking.harness import Method
    from benchmarking.synthetic import ColumnSchema
    from wind_up.analysis_period import PlanSettings

logger = logging.getLogger(__name__)

WIND_UP = "wind-up"

# The declaration as the run understood it, echoed beside the report: what was declared, what was
# defaulted, and what each timestamp resolved to. YAML, and shaped like campaign.yaml, so it reads
# as the analyst's own file filled in rather than as a separate dialect.
RESOLVED_FILENAME = "campaign_resolved.yaml"

_RESOLVED_HEADER = """\
# The campaign as wind-up understood it: campaign.yaml with every default filled in and every
# timestamp resolved to UTC. Values you did not declare appear here anyway -- references defaults
# to every other turbine in the turbines file, excluded to none. `reanalysis` is derived, not
# declared: where the weather series was drawn from and the window it was fetched over.
"""

# Where the run's log is kept. Some of what a run decides -- the reference screen's thresholds, its
# pool size, whether it stopped early -- is reported only in the log.
LOG_FILENAME = "run.log"
_LOG_HANDLER_NAME = "campaign_run_log"

OUTPUT_DIR_ENV = "WIND_UP_BENCHMARKING_OUTPUT_DIR"

# The run's output is laid out by the parts of docs/v1/method.md: data preparation for every turbine,
# the estimator once per test turbine, and the campaign's results across turbines. Inside each, a
# folder is named for the method step it shows.
DATA_PREPARATION_DIRNAME = "A_data_preparation"
ESTIMATOR_DIRNAME = "B_uplift_estimator"
CAMPAIGN_DIRNAME = "C_campaign"
README_FILENAME = "README.md"

_README = f"""\
# Campaign run output

Laid out by the parts and steps of wind-up's method (docs/v1/method.md).

- `{RESOLVED_FILENAME}`: the campaign as wind-up understood it.
- `{LOG_FILENAME}`: the run's log.
- `{DATA_PREPARATION_DIRNAME}/`: part A, every turbine over every record provided.
  - `{stages.CHANGES}/`: step 1, operating relationships, coverage and power factor.
  - `{stages.OPERATING_STATES}/`: step 2, the operating-state labels and hours per state.
  - `{stages.REANALYSIS}/`: step 3, the reanalysis time-shift check, ERA5 against the site wind speed,
    and what the reanalysis covers.
  - `{stages.NORTHING}/`: step 4, the northing corrections, and each turbine's records used and not
    used for northing, coloured by operating state.
  - `{stages.WAKING}/`: step 5, each turbine's records considered waking, part waking and not waking,
    coloured by operating state.
- `{ESTIMATOR_DIRNAME}/<test turbine>/`: part B, one folder per test turbine, over its span and with
  its power references. Its results are in CSVs; its plots are under `plots/`, one folder per step.
  `plots/{stages.VALID_RECORDS}/` has the records used and not used for uplift, coloured by state.
- `{CAMPAIGN_DIRNAME}/`: part C, the results across turbines: analysis plans (steps 7 and 11),
  reference stability (step 11), per-turbine uplift (step 12), conditional uplift (step 13) and farm
  uplift (step 14).
"""


def wind_up_method(
    spec: CampaignSpec,
    *,
    columns: ColumnSchema,
    out_dir: Path,
    era5_hourly_df: pd.DataFrame | None,
    screen_cache: dict | None = None,
    era5_label: str = ERA5_UNLOCATED,
    run_subdir: bool = True,
) -> Method:
    """Build ``wind-up``'s estimator for one campaign.

    The accepted ``power_model`` defaults, under one name. The reference-validity screen and the
    reference-stability table are its own defaults, so nothing is turned on here.

    :param spec: the campaign being run; supplies the turbine rating
    :param columns: the source-native schema the SCADA is keyed by
    :param out_dir: where the method writes its own diagnostics
    :param era5_hourly_df: reanalysis; without it the per-condition estimates are not reported
    :param screen_cache: one dict shared across the campaign's turbines, so the reference screen
        runs once rather than once per test turbine
    :param era5_label: how reanalysis is named in this run's output; the campaign passes the point
        it fetched from
    :param run_subdir: write into a named, timestamped folder under ``out_dir``, as a study comparing
        many runs wants; a campaign gives each test turbine its own ``out_dir`` and passes ``False``
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
        screen_cache=screen_cache,
        era5_label=era5_label,
        run_subdir=run_subdir,
    )


def output_root() -> Path:
    """Return the directory runs write under: ``WIND_UP_BENCHMARKING_OUTPUT_DIR``, or a default."""
    return Path(os.getenv(OUTPUT_DIR_ENV, Path.home() / "temp" / "wind-up-benchmarking"))


def default_out_dir(name: str) -> Path:
    """Return the directory a run of ``name`` writes to when none is given."""
    return output_root() / name


def log_to_file(out_dir: Path) -> Path:
    """Send the root logger to ``out_dir``/``run.log`` as well as wherever it already goes.

    Replaces the file this added for any previous run in the same process, so a second run writes
    to its own directory and not also to the first one's. Records reach the file at the root
    logger's level, which the command line sets to INFO.

    :param out_dir: the run's output directory, which must exist
    :return: the log file's path
    """
    root = logging.getLogger()
    for stale in [h for h in root.handlers if getattr(h, "name", None) == _LOG_HANDLER_NAME]:
        root.removeHandler(stale)
        stale.close()
    path = out_dir / LOG_FILENAME
    handler = logging.FileHandler(path, mode="w")
    handler.name = _LOG_HANDLER_NAME
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    return path


def run_declaration(
    path: str | Path,
    *,
    out_dir: Path | None = None,
    era5_hourly_df: pd.DataFrame | None = None,
    plan_settings: PlanSettings = DEFAULT_PLAN_SETTINGS,
    input_plots_dir: Path | None = None,
) -> CampaignReport:
    """Run the campaign declared at ``path`` and write its report.

    :param path: the campaign declaration
    :param out_dir: where to write; defaults to :func:`default_out_dir` of the campaign's name
    :param era5_hourly_df: reanalysis to use instead of self-serving it from the farm centroid,
        for a caller that already holds it
    :param plan_settings: how a planned campaign's spans and power references are chosen
    :param input_plots_dir: where the plots of every turbine and every record go (steps 1 and 2, and
        the northing and waking validity), in a folder per step; defaults to the run's data-preparation folder
    :return: the truth-free campaign report, which is also written under ``out_dir``
    """
    declaration = load_declaration(path)
    out_dir = (out_dir if out_dir is not None else default_out_dir(declaration.name)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_to_file(out_dir)
    resolved = yaml.safe_dump(declaration.resolved(), sort_keys=False, default_flow_style=False)
    (out_dir / RESOLVED_FILENAME).write_text(_RESOLVED_HEADER + resolved)
    (out_dir / README_FILENAME).write_text(_README)
    preparation = out_dir / DATA_PREPARATION_DIRNAME
    input_plots = input_plots_dir if input_plots_dir is not None else preparation
    logger.info(
        "Running campaign %r declared in %s\n  writing to %s\n  logging to %s",
        declaration.name,
        Path(path).resolve(),
        out_dir,
        log_path,
    )

    scada_df = pd.read_parquet(declaration.scada_path)
    timebase = infer_timebase(pd.DatetimeIndex(scada_df.index.unique()).sort_values())
    scada_df = normalise_timestamps(scada_df, timestamps=declaration.timestamps, timebase=timebase)
    index = pd.DatetimeIndex(scada_df.index.unique()).sort_values()
    scada_df = label_operating_states(
        scada_df,
        columns=declaration.columns,
        config=declaration.operating_state,
        timebase=timebase,
        rated_power_kw=declaration.spec.rated_power_kw,
    )
    changeovers = declaration.spec.changeovers()
    write_input_data_plots(
        scada_df, columns=declaration.columns, out_dir=input_plots / stages.CHANGES, changeovers=changeovers
    )
    write_operating_state_plots(
        scada_df,
        columns=declaration.columns,
        timebase=timebase,
        out_dir=input_plots / stages.OPERATING_STATES,
        changeovers=changeovers,
    )
    for views, stage in ((NORTHING_VIEWS, stages.NORTHING), (WAKING_VIEWS, stages.WAKING)):
        write_validity_plots(
            scada_df, views=views, columns=declaration.columns, timebase=timebase, out_dir=input_plots / stage
        )
    reanalysis = era5_hourly_df if era5_hourly_df is not None else _fetch_era5(declaration, index=index)
    spec = declaration.spec
    prepared = prepare_reanalysis(
        reanalysis,
        scada_df=scada_df,
        columns=declaration.columns,
        unchanged=[
            w for w in layout_coords(spec.layout) if w not in set(spec.upgraded_turbines) | set(spec.excluded_turbines)
        ],
        timebase=timebase,
    )
    write_reanalysis_outputs(
        prepared,
        era5_hourly_df=reanalysis,
        cell_selection=declaration.cell_selection,
        out_dir=input_plots / stages.REANALYSIS,
    )
    # One screen verdict for the campaign: every test turbine is judged against the same references.
    screen_cache: dict = {}

    report = estimate_campaign(
        declaration.spec,
        scada_df,
        build_methods=lambda wtg: [
            wind_up_method(
                declaration.spec,
                columns=declaration.columns,
                out_dir=out_dir / ESTIMATOR_DIRNAME / wtg,
                era5_hourly_df=reanalysis,
                screen_cache=screen_cache,
                era5_label=era5_source_label(*declaration.centroid),
                run_subdir=False,
            )
        ],
        columns=declaration.columns,
        era5_wd=prepared.aligned[ERA5_WD_RAW],
        northing_out_dir=preparation / stages.NORTHING,
        plan_settings=plan_settings,
    )
    write_report(report, out_dir=out_dir / CAMPAIGN_DIRNAME)
    logger.info("Wrote the %r report to %s", declaration.name, out_dir)
    return report


def reanalysis_window(index: pd.DatetimeIndex) -> tuple[str, str]:
    """Return the reanalysis window: the whole SCADA record, rounded out to whole calendar years.

    The window runs to 1 January after the last year, so the final hour of the record has an hour
    after it to interpolate towards.
    """
    start, end = era5_window(index.min(), index.max() + pd.Timedelta(nanoseconds=1))
    return start, (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _fetch_era5(declaration: Declaration, *, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Fetch reanalysis for the farm centroid over the whole SCADA record, from a sea cell when offshore.

    The fetch itself needs the optional ``era5`` dependency group and network on a cache miss.
    """
    lat, lon = declaration.centroid
    start_date, end_date = reanalysis_window(index)
    logger.info(
        "Fetching reanalysis for (%.4f, %.4f) over %s..%s, %s cell",
        lat,
        lon,
        start_date,
        end_date,
        declaration.cell_selection,
    )
    return get_era5_hourly_df(
        lat=lat, lon=lon, start_date=start_date, end_date=end_date, cell_selection=declaration.cell_selection
    )
