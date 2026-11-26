"""Part A of the method, data preparation (steps 1 to 6), done once per dataset.

:func:`prepare_data` gives every turbine a record at every timestamp, labels the operating states with the
campaign's operating state overrides and exclusions applied, aligns the reanalysis, norths the whole farm over the whole
record, finds the possible yaw-alignment changes, adds the waking columns and, when asked, writes the outputs
of each step. The composed run and the benchmark
studies share it, so a method is judged on the same prepared data wherever it runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from benchmarking.baselines.power_model.features import solar_elevation_frame
from benchmarking.campaigns.context import campaign_upgrades
from benchmarking.campaigns.declaration import layout_coords
from benchmarking.campaigns.loader import centroid
from benchmarking.diagnostics.features import write_feature_outputs
from benchmarking.diagnostics.reanalysis import write_reanalysis_outputs
from benchmarking.diagnostics.yaw_changes import possible_yaw_change_report, write_possible_yaw_changes
from benchmarking.harness.label_checks import LABEL_CHECKS_CSV, label_checks, warn_label_checks
from wind_up.diagnostics import stages
from wind_up.diagnostics.context import infer_timebase
from wind_up.diagnostics.input_data import write_input_data_plots
from wind_up.diagnostics.operating_states import (
    NORTHING_VIEWS,
    WAKING_VIEWS,
    write_operating_state_plots,
    write_validity_plots,
)
from wind_up.diagnostics.waking_scenarios import write_waking_outputs
from wind_up.labels import NORTHING, UPLIFT
from wind_up.northing_step import (
    DEFAULT_NORTHING_ROLES,
    apply_north_tables,
    north_offsets_of,
    north_tables,
    northing_rows,
)
from wind_up.operating_state import (
    VALID_NORTHING_COL,
    VALID_UPLIFT_COL,
    OperatingStateConfig,
    apply_declarations,
    declared_records,
    fill_grid,
    label_operating_states,
)
from wind_up.reanalysis import ERA5_WD_RAW, prepare_reanalysis
from wind_up.waking import WAKING_COLUMNS, add_waking_scenarios, waking_directions

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy.typing as npt

    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.synthetic import ColumnSchema
    from wind_up.labels import Label
    from wind_up.reanalysis import ReanalysisResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedData:
    """What Part A produced.

    :param scada_df: every turbine at every timestamp of the record, a row not provided filled with NaN, with
        the operating-state columns of step 2 and the waking columns of step 5
    :param reanalysis: the aligned reanalysis of step 3 and its checks
    :param north_tables: each turbine's north table of step 4, over the whole record
    :param possible_changes: the possible yaw-alignment changes of step 4
    :param timebase: the records' period
    """

    scada_df: pd.DataFrame
    reanalysis: ReanalysisResult
    north_tables: dict[str, pd.DataFrame]
    possible_changes: pd.DataFrame
    timebase: pd.Timedelta

    @property
    def aligned_era5(self) -> pd.DataFrame:
        """The reanalysis on the SCADA index."""
        return self.reanalysis.aligned

    def with_north_offsets(self, spec: CampaignSpec) -> CampaignSpec:
        """Return ``spec`` declaring the north tables found here, so the measurement applies them as given."""
        return replace(spec, north_offsets=north_offsets_of(self.north_tables))

    def possible_changes_by_turbine(self) -> dict[str, list[pd.Timestamp]]:
        """Each turbine's possible yaw-alignment change dates, for the warnings of step 8."""
        return {str(t): list(rows["timestamp"]) for t, rows in self.possible_changes.groupby("turbine", sort=True)}


def prepare_data(
    scada_df: pd.DataFrame,
    *,
    spec: CampaignSpec,
    columns: ColumnSchema,
    era5_hourly_df: pd.DataFrame,
    operating_state: OperatingStateConfig | None = None,
    cell_selection: Literal["land", "sea"] = "land",
    out_dir: Path | None = None,
    plots: bool = True,
    labels: Sequence[Label] = (),
) -> PreparedData:
    """Run Part A on ``scada_df`` and return what it produced.

    :param scada_df: long-format SCADA of every turbine over every record provided, indexed by
        period-start UTC timestamps
    :param spec: the campaign, for its layout, rating, test turbines and changeovers
    :param columns: the schema ``scada_df`` is keyed by
    :param era5_hourly_df: hourly reanalysis covering the record, or one already aligned to it
    :param operating_state: the site's operating-state labels; the generic states alone when ``None``
    :param cell_selection: the reanalysis grid cell asked for, named in the reanalysis summary
    :param out_dir: where each step's outputs go, in a folder per step; nothing is written when ``None``
    :param plots: write the plots as well as the tables
    :param labels: the analyst's labels, columns of ``scada_df``; a record holding one is not valid for the
        uses it withholds
    """
    index = pd.DatetimeIndex(scada_df.index.unique()).sort_values()
    timebase = infer_timebase(index)
    config = operating_state if operating_state is not None else OperatingStateConfig()
    supplied = pd.MultiIndex.from_arrays([pd.DatetimeIndex(scada_df.index), scada_df[columns.turbine].to_numpy()])
    scada_df = fill_grid(scada_df, turbine_col=columns.turbine, timebase=timebase)
    scada_df = _unlabel_filled(scada_df, supplied=supplied, turbine_col=columns.turbine, labels=labels)
    scada_df = label_operating_states(
        scada_df, columns=columns, config=config, timebase=timebase, rated_power_kw=spec.rated_power_kw
    )
    scada_df = apply_declarations(
        scada_df,
        turbine_col=columns.turbine,
        config=config,
        exclusions=spec.exclusions,
        overrides=spec.operating_state_overrides,
    )
    scada_df = withhold_labels(scada_df, labels=labels)
    scada_df = withhold_settling(scada_df, spec=spec)
    declared = declared_records(
        scada_df, turbine_col=columns.turbine, windows=[*spec.exclusions, *spec.operating_state_overrides]
    )
    checks = label_checks(
        scada_df, columns=columns, timebase=timebase, rated_power_kw=spec.rated_power_kw, declared=declared
    )
    warn_label_checks(checks)
    if out_dir is not None:
        (out_dir / stages.OPERATING_STATES).mkdir(parents=True, exist_ok=True)
        checks.to_csv(out_dir / stages.OPERATING_STATES / LABEL_CHECKS_CSV, index=False)
    changeovers = spec.changeovers()
    draw = out_dir is not None and plots
    if out_dir is not None and draw:
        write_input_data_plots(
            scada_df,
            columns=columns,
            out_dir=out_dir / stages.CHANGES,
            changeovers=changeovers,
            rated_power_kw=spec.rated_power_kw,
        )
        write_operating_state_plots(
            scada_df,
            columns=columns,
            timebase=timebase,
            out_dir=out_dir / stages.OPERATING_STATES,
            changeovers=changeovers,
        )
        write_validity_plots(
            scada_df, views=WAKING_VIEWS, columns=columns, timebase=timebase, out_dir=out_dir / stages.WAKING
        )

    unchanged = [
        w for w in layout_coords(spec.layout) if w not in set(spec.upgraded_turbines) | set(spec.excluded_turbines)
    ]
    reanalysis = prepare_reanalysis(
        era5_hourly_df, scada_df=scada_df, columns=columns, unchanged=unchanged, timebase=timebase
    )
    if out_dir is not None and draw:
        write_reanalysis_outputs(
            reanalysis,
            era5_hourly_df=era5_hourly_df,
            cell_selection=cell_selection,
            out_dir=out_dir / stages.REANALYSIS,
        )
        learned = northing_learning_set(scada_df, spec=spec, columns=columns, era5_wd=reanalysis.aligned[ERA5_WD_RAW])
        write_validity_plots(
            scada_df.assign(**{VALID_NORTHING_COL: learned}),
            views=NORTHING_VIEWS,
            columns=columns,
            timebase=timebase,
            out_dir=out_dir / stages.NORTHING,
        )

    tables = north_tables(
        scada_df,
        columns=columns,
        north_offsets=spec.north_offsets,
        rated_power_kw=spec.rated_power_kw,
        layout=spec.layout,
        era5_wd=reanalysis.aligned[ERA5_WD_RAW],
        out_dir=out_dir / stages.NORTHING if out_dir is not None else None,
        plots=plots,
    )
    yaw = {
        "tables": tables,
        "columns": columns,
        "rated_power_kw": spec.rated_power_kw,
        "timebase": timebase,
        "changeovers": changeovers,
        "analysis_period": spec.period_bounds(),
    }
    if out_dir is not None:
        possible_changes = write_possible_yaw_changes(scada_df, out_dir=out_dir / stages.NORTHING, plots=plots, **yaw)
    else:
        possible_changes, _ = possible_yaw_change_report(scada_df, **yaw)

    scada_df = _add_waking(
        scada_df,
        spec=spec,
        tables=tables,
        columns=columns,
        timebase=timebase,
        out_dir=out_dir / stages.WAKING if draw and out_dir is not None else None,
    )
    if out_dir is not None and draw:
        latitude, longitude = centroid(layout_coords(spec.layout))
        write_feature_outputs(
            scada_df,
            site_features=solar_elevation_frame(
                pd.DatetimeIndex(scada_df.index.unique()).sort_values(),
                timebase=timebase,
                latitude=latitude,
                longitude=longitude,
            ),
            turbine_col=columns.turbine,
            changeovers=changeovers,
            out_dir=out_dir / stages.FEATURES,
        )
    return PreparedData(
        scada_df=scada_df,
        reanalysis=reanalysis,
        north_tables=tables,
        possible_changes=possible_changes,
        timebase=timebase,
    )


def withhold_labels(scada_df: pd.DataFrame, *, labels: Sequence[Label]) -> pd.DataFrame:
    """Return ``scada_df`` with each record holding a label not valid for the uses that label withholds.

    A record whose label column is not true, a filled record's NaN included, keeps its validity.
    """
    withholding = [label for label in labels if label.withholds]
    if not withholding:
        return scada_df
    out = scada_df.copy()
    for label in withholding:
        held = out[label.column].eq(True).to_numpy(dtype=bool)  # noqa: FBT003
        for use, column in ((UPLIFT, VALID_UPLIFT_COL), (NORTHING, VALID_NORTHING_COL)):
            if use in label.withholds:
                out[column] = out[column].to_numpy(dtype=bool) & ~held
    return out


def withhold_settling(scada_df: pd.DataFrame, *, spec: CampaignSpec) -> pd.DataFrame:
    """Return ``scada_df`` with each record settling after a toggle change not valid for uplift or northing."""
    index = pd.DatetimeIndex(scada_df.index)
    turbines = scada_df[spec.turbine_col].astype(str).to_numpy()
    settling = np.zeros(len(scada_df), dtype=bool)
    for upgrade in campaign_upgrades(spec):
        for name in upgrade.turbines:
            mine = turbines == name
            settling[mine] |= upgrade.settling(index[mine], turbine=name)
    if not settling.any():
        return scada_df
    out = scada_df.copy()
    for column in (VALID_UPLIFT_COL, VALID_NORTHING_COL):
        out[column] = out[column].to_numpy(dtype=bool) & ~settling
    return out


def _unlabel_filled(
    filled: pd.DataFrame, *, supplied: pd.MultiIndex, turbine_col: str, labels: Sequence[Label]
) -> pd.DataFrame:
    """Return ``filled`` with every label false on each record whose ``(timestamp, turbine)`` is not in ``supplied``."""
    pairs = pd.MultiIndex.from_arrays([pd.DatetimeIndex(filled.index), filled[turbine_col].to_numpy()])
    added = ~pairs.isin(supplied)
    if not labels or not added.any():
        return filled
    out = filled.copy()
    for column in dict.fromkeys(label.column for label in labels):
        values = out[column].to_numpy(dtype=object).copy()
        values[added] = False
        out[column] = pd.Series(values).infer_objects().to_numpy()
    return out


def northing_learning_set(
    scada_df: pd.DataFrame, *, spec: CampaignSpec, columns: ColumnSchema, era5_wd: pd.Series
) -> npt.NDArray[np.bool_]:
    """Rows of a step-2 labelled frame that northing discovery learns from.

    None when the campaign declares its corrections or turns discovery off.
    """
    if spec.north_offsets is not None:
        return np.zeros(len(scada_df), dtype=bool)
    return northing_rows(scada_df, columns=columns, rated_power_kw=spec.rated_power_kw, era5_wd=era5_wd)


def _add_waking(
    scada_df: pd.DataFrame,
    *,
    spec: CampaignSpec,
    tables: dict[str, pd.DataFrame],
    columns: ColumnSchema,
    timebase: pd.Timedelta,
    out_dir: Path | None,
    roles: Sequence[str] = DEFAULT_NORTHING_ROLES,
) -> pd.DataFrame:
    """Return ``scada_df`` with the step 5 waking columns, found from the northed direction.

    Returns ``scada_df`` unchanged when it has no nacelle position. Writes the step's outputs under
    ``out_dir`` when given.
    """
    if columns.nacelle_position is None or columns.nacelle_position not in scada_df.columns:
        logger.warning("no waking scenarios: the SCADA has no nacelle position")
        return scada_df
    northed = apply_north_tables(scada_df, tables=tables, columns=columns, roles=roles)
    directions = waking_directions(northed, columns=columns, layout=spec.layout, test_turbines=spec.upgraded_turbines)
    waking = add_waking_scenarios(
        northed,
        columns=columns,
        layout=spec.layout,
        test_turbines=spec.upgraded_turbines,
        directions=directions,
    )
    if out_dir is not None:
        write_waking_outputs(
            waking, directions=directions, layout=spec.layout, columns=columns, timebase=timebase, out_dir=out_dir
        )
    return scada_df.assign(**{col: waking[col].to_numpy() for col in WAKING_COLUMNS})
